"""
Task Evaluation Framework for TerminalBench

This module provides comprehensive evaluation capabilities for terminal tasks,
including correctness checking, performance measurement, and safety validation.
"""

import json
import logging
import os
import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timezone
from docker.models.containers import Container
from terminal_bench.handlers.trial_handler import Task, TaskPaths

@dataclass
class EvaluationCriteria:
    """Criteria for evaluating task completion"""
    success_conditions: List[str]  # Commands or checks that must succeed
    file_requirements: Dict[str, str]  # Files that must exist with specific content
    output_requirements: List[str]  # Required output patterns
    forbidden_commands: List[str]  # Commands that should not be executed
    max_execution_time: float = 300.0  # Maximum total execution time
    max_commands: int = 50  # Maximum number of commands allowed
    max_tokens: Optional[int] = None  # Maximum total tokens
    max_tokens_per_request: Optional[float] = None  # Maximum tokens per API request


@dataclass
class EvaluationResult:
    """Result of task evaluation"""
    task_id: str
    passed: bool
    score: float  # 0.0 to 1.0
    correctness: Dict[str, Any]

    details: Dict[str, Any]
    timestamp: str


class TaskEvaluator:
    """Evaluates task completion against success criteria"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def evaluate(
        self, task_id: str, 
        task_paths: TaskPaths,
        container: Container,
        white_agent_response: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Comprehensive task evaluation.
        
        Args:
            task_id: ID of the task being evaluated
            task_paths: Terminal Bench TaskPaths object
            container: Docker container for running tests
            white_agent_response: Optional white agent response containing token/request metadata
            
        Returns:
            EvaluationResult with detailed evaluation
        """
        self.logger.info(f"Evaluating task {task_id}")
        
        pytest_results = self._run_terminal_bench_tests(container, task_paths)
        
        if not pytest_results or pytest_results["tests_run"] == 0:
            raise ValueError(f"No pytest tests found or executed for task {task_id}")
        
        # Use pytest results as source of truth for correctness
        correctness_result = {
            "passed": pytest_results["passed"],
            "unit_tests": pytest_results
        }
        
        # Extract token/request counts from white agent response
        token_request_data = self._extract_token_and_request_counts(white_agent_response)
        
        # Binary scoring: 1.0 if all tests pass, 0.0 otherwise
        overall_score = 1.0 if correctness_result["passed"] else 0.0
        
        # Task passed if all tests passed
        passed = correctness_result["passed"]
        
        return EvaluationResult(
            task_id=task_id,
            passed=passed,
            score=overall_score,
            correctness=correctness_result,
            details={
                "total_tokens": token_request_data.get("total_tokens"),
                "total_requests": token_request_data.get("total_requests"),
                "tokens_per_request": (
                    token_request_data["total_tokens"] / token_request_data["total_requests"]
                    if token_request_data.get("total_tokens") is not None 
                    and token_request_data.get("total_requests") is not None
                    and token_request_data["total_requests"] > 0
                    else None
                ),
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    def _extract_criteria(self, task_spec: Dict[str, Any]) -> EvaluationCriteria:
        """Extract evaluation criteria from task specification."""
        # Default criteria
        criteria = EvaluationCriteria(
            success_conditions=[],
            file_requirements={},
            output_requirements=[],
            forbidden_commands=["rm -rf /", "sudo", "chmod 777", "dd if=/dev/zero"],
            max_execution_time=300.0,
            max_commands=50
        )
        
        # Extract from task specification
        if "test" in task_spec:
            test_description = task_spec["test"]
            criteria.success_conditions.append(test_description)
        
        if "environment" in task_spec:
            env = task_spec["environment"]
            if "working_directory" in env:
                # Check if specific files should be created
                if "expected_files" in env:
                    criteria.file_requirements.update(env["expected_files"])
        
        # Parse test description for specific requirements
        if "test" in task_spec:
            test_text = task_spec["test"].lower()
            
            # Look for file existence requirements
            if "solution.txt" in test_text:
                criteria.file_requirements["solution.txt"] = ".*"
            if "certificate" in test_text or "cert" in test_text:
                criteria.file_requirements.update({
                    "ssl/server.crt": ".*",
                    "ssl/server.key": ".*"
                })
        
        # Extract token/request limits from metadata
        if "metadata" in task_spec:
            metadata = task_spec["metadata"]
            
            # Check validation section first
            if "validation" in metadata:
                validation = metadata["validation"]
                if "max_tokens" in validation:
                    criteria.max_tokens = validation["max_tokens"]
                if "max_tokens_per_request" in validation:
                    criteria.max_tokens_per_request = validation["max_tokens_per_request"]
            
            # Also check top level metadata
            if "max_tokens" in metadata:
                criteria.max_tokens = metadata["max_tokens"]
            if "max_tokens_per_request" in metadata:
                criteria.max_tokens_per_request = metadata["max_tokens_per_request"]
        
        return criteria
    
    def _run_terminal_bench_tests(self, container: Container, task_paths: TaskPaths) -> Dict[str, Any]:
        """
        Run Terminal-Bench pytest tests using Terminal Bench's test infrastructure.
        
        Args:
            container: Docker container to run tests in
            task_paths: Terminal Bench TaskPaths object
            
        Returns:
            Dictionary with test results
        """
        from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
        
        self.logger.info(f"Copying test files to container")
        
        try:
            # Use Terminal Bench's built-in copy_to_container method
            paths = [task_paths.run_tests_path]
            if task_paths.test_dir.exists():
                paths.append(task_paths.test_dir)
            
            DockerComposeManager.copy_to_container(
                container=container,
                paths=paths,
                container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR)
            )
            
            # Run the test script (same pattern as Terminal Bench harness)
            test_script_path = DockerComposeManager.CONTAINER_TEST_DIR / task_paths.run_tests_path.name
            self.logger.info(f"Running Terminal Bench test script: {test_script_path}")
            
            exec_result = container.exec_run(
                ["bash", str(test_script_path)],
                workdir="/app"
            )
            
            exit_code = exec_result.exit_code
            output = exec_result.output.decode('utf-8', errors='replace') if exec_result.output else ""
            
            if exit_code != 0:
                self.logger.warning(f"Test execution completed with exit code {exit_code}")
            
            self.logger.info(f"Test output:\n{output}...")
            
            # Parse pytest output
            return self._parse_pytest_output(output, exit_code)
            
        except Exception as e:
            self.logger.error(f"Failed to run tests: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {"passed": False, "tests_run": 0, "error": str(e)}
    
    def _parse_pytest_output(self, output: str, exit_code: int) -> Dict[str, Any]:
        """
        Parse pytest output using Terminal Bench's official parser.
        
        Terminal Bench uses the PytestParser which:
        - Looks for the "short test summary info" section
        - Parses individual test results
        - Returns dict[test_name -> UnitTestStatus]
        
        This is more reliable than regex counting.
        """
        from terminal_bench.parsers.pytest_parser import PytestParser
        from terminal_bench.parsers.base_parser import UnitTestStatus
        
        parser = PytestParser()
        
        # Use Terminal Bench's parser to get per-test results
        test_results = parser.parse(output)
        
        # Calculate aggregates
        tests_run = len(test_results)
        tests_passed = sum(1 for status in test_results.values() if status == UnitTestStatus.PASSED)
        tests_failed = sum(1 for status in test_results.values() if status == UnitTestStatus.FAILED)
        
        # Task passes if all tests passed (tests_failed == 0)
        passed = tests_failed == 0 and tests_run > 0
        
        self.logger.info(f"Pytest results: {tests_passed}/{tests_run} tests passed, {tests_failed} failed")
        
        return {
            "passed": passed,
            "tests_run": tests_run,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "details": f"Pytest: {tests_passed}/{tests_run} passed",
            "test_results": {name: status.value for name, status in test_results.items()},  # Per-test details
            "raw_output": output
        }
    
    def _extract_token_and_request_counts(self, white_agent_response: Optional[Dict[str, Any]]) -> Dict[str, Optional[int]]:
        """
        Extract token and request counts from white agent response.
        
        Args:
            white_agent_response: Response from white agent
            
        Returns:
            Dictionary with 'total_tokens' and 'total_requests' (None if not available)
        """
        result = {
            "total_tokens": None,
            "total_requests": None
        }
        
        if not white_agent_response:
            return result
        
        # Log response structure for debugging
        self.logger.debug(f"White agent response structure: {json.dumps(white_agent_response, indent=2)[:500]}")
        
        try:
            # Try to extract token usage from response metadata
            # Check multiple possible paths
            token_data = None
            
            # Path 1: result.metadata.usage
            if "result" in white_agent_response:
                result_data = white_agent_response["result"]
                if isinstance(result_data, dict):
                    if "metadata" in result_data:
                        metadata = result_data["metadata"]
                        if isinstance(metadata, dict):
                            if "usage" in metadata:
                                token_data = metadata["usage"]
                            elif "total_tokens" in metadata:
                                token_data = metadata
            
            # Path 2: metadata.usage (top level)
            if not token_data and "metadata" in white_agent_response:
                metadata = white_agent_response["metadata"]
                if isinstance(metadata, dict):
                    if "usage" in metadata:
                        token_data = metadata["usage"]
                    elif "total_tokens" in metadata:
                        token_data = metadata
            
            # Extract total_tokens if found
            if token_data:
                if isinstance(token_data, dict):
                    if "total_tokens" in token_data:
                        result["total_tokens"] = token_data["total_tokens"]
                    elif "prompt_tokens" in token_data and "completion_tokens" in token_data:
                        # Calculate total if not provided
                        result["total_tokens"] = token_data["prompt_tokens"] + token_data["completion_tokens"]
                elif isinstance(token_data, int):
                    result["total_tokens"] = token_data
            
            # Extract request count
            # Each send_task_to_white_agent call = 1 request
            # For now, if response exists, count as 1 request
            # Could be enhanced to track multiple internal API calls
            if "result" in white_agent_response or "error" not in white_agent_response:
                result["total_requests"] = 1
            
            # Check if request count is explicitly provided in metadata
            if "result" in white_agent_response:
                result_data = white_agent_response["result"]
                if isinstance(result_data, dict) and "metadata" in result_data:
                    metadata = result_data["metadata"]
                    if isinstance(metadata, dict):
                        if "request_count" in metadata:
                            result["total_requests"] = metadata["request_count"]
                        elif "num_requests" in metadata:
                            result["total_requests"] = metadata["num_requests"]
            
            if result["total_tokens"] is not None:
                self.logger.info(f"Extracted token count: {result['total_tokens']}")
            if result["total_requests"] is not None:
                self.logger.info(f"Extracted request count: {result['total_requests']}")
                
        except Exception as e:
            self.logger.warning(f"Failed to extract token/request counts from white agent response: {e}")
        
        return result
