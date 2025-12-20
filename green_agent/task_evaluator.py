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
    max_tokens: Optional[int] = None  # Maximum total tokens
    max_tokens_per_request: Optional[float] = None  # Maximum tokens per API request


@dataclass
class EvaluationResult:
    """Result of task evaluation"""
    task_id: str
    passed: bool  # True if all tests passed
    accuracy: float  # Fraction of tests passed (0.0 to 1.0)
    passed_test_cases: int  # Number of tests that passed
    total_test_cases: int  # Total number of tests run
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
        task: Optional[Any] = None,
        session: Optional[Any] = None,
    ) -> EvaluationResult:
        """
        Comprehensive task evaluation.
        
        Args:
            task_id: ID of the task being evaluated
            task_paths: Terminal Bench TaskPaths object
            container: Docker container for running tests
            white_agent_response: Optional white agent response containing token/request metadata
            task: Task object with run_tests_in_same_shell flag
            session: TmuxSession from agent execution (used if run_tests_in_same_shell is true)
            
        Returns:
            EvaluationResult with detailed evaluation
        """
        self.logger.info(f"Evaluating task {task_id}")
        
        pytest_results = self._run_terminal_bench_tests(container, task_paths, task, session)
        
        if not pytest_results or pytest_results["tests_run"] == 0:
            raise ValueError(f"No pytest tests found or executed for task {task_id}")
        
        # Extract test metrics from pytest results
        tests_passed = pytest_results.get("tests_passed", 0)
        tests_run = pytest_results.get("tests_run", 1)
        tests_failed = pytest_results.get("tests_failed", 0)
        
        # Calculate accuracy (fraction of tests passed)
        accuracy = tests_passed / tests_run if tests_run > 0 else 0.0
        
        # Task passed if all tests passed (no failures)
        passed = pytest_results.get("passed", False)
        
        # Extract token count from white agent response
        total_tokens = self._extract_token_count(white_agent_response)
        
        return EvaluationResult(
            task_id=task_id,
            passed=passed,
            accuracy=accuracy,
            passed_test_cases=tests_passed,
            total_test_cases=tests_run,
            details={
                "total_tokens": total_tokens,
                "unit_tests": pytest_results  # Keep full details for debugging
            } if total_tokens else {
                "unit_tests": pytest_results
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
    
    def _run_terminal_bench_tests(self, container: Container, task_paths: TaskPaths, 
                                   task: Optional[Any] = None, session: Optional[Any] = None) -> Dict[str, Any]:
        """
        Run Terminal-Bench pytest tests using tmux sessions (matching Terminal-Bench's approach).
        
        Args:
            container: Docker container to run tests in
            task_paths: Terminal Bench TaskPaths object
            task: Task object with run_tests_in_same_shell flag
            session: TmuxSession from agent execution (used if run_tests_in_same_shell is true)
            
        Returns:
            Dictionary with test results
        """
        from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
        from terminal_bench.terminal.tmux_session import TmuxSession
        
        self.logger.info(f"Copying test files to container")
        
        try:
            # Copy test files to container
            paths = [task_paths.run_tests_path]
            if task_paths.test_dir.exists():
                paths.append(task_paths.test_dir)
            
            DockerComposeManager.copy_to_container(
                container=container,
                paths=paths,
                container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR)
            )
            
            # Determine which session to use based on run_tests_in_same_shell flag
            test_session = session
            created_new_session = False
            
            if task and not getattr(task, 'run_tests_in_same_shell', False):
                # Create a new tmux session for tests (like Terminal-Bench does)
                self.logger.info("Creating new tmux session for tests (run_tests_in_same_shell=false)")
                test_session = TmuxSession(
                    session_name="tests",
                    container=container,
                    commands_path=None,
                    disable_recording=True,
                    user=""  # Run as default user
                )
                test_session.start()
                created_new_session = True
            else:
                # Use the same session as the agent (like Terminal-Bench does for run_tests_in_same_shell=true)
                self.logger.info("Using existing tmux session for tests (run_tests_in_same_shell=true)")
            
            # Run the test script using tmux (same as Terminal-Bench harness)
            test_script_path = DockerComposeManager.CONTAINER_TEST_DIR / task_paths.run_tests_path.name
            self.logger.info(f"Running test script in tmux: {test_script_path}")
            
            # Get test timeout
            test_timeout_sec = getattr(task, 'max_test_timeout_sec', 180.0) if task else 180.0
            
            try:
                # Send keys to run the test script (blocking with timeout)
                test_session.send_keys(
                    ["bash ", str(test_script_path), "Enter"],
                    block=True,
                    max_timeout_sec=test_timeout_sec
                )
            except TimeoutError:
                self.logger.warning(f"Test command timed out after {test_timeout_sec}s")
                # Capture whatever output we have
                output = test_session.capture_pane(capture_entire=False)
                return self._parse_pytest_output(output, -1)
            
            # Capture the output after tests complete
            output = test_session.capture_pane(capture_entire=False)
            
            self.logger.info(f"Test output:\n{output[:500]}...")
            
            # Clean up the test session if we created it
            if created_new_session:
                try:
                    test_session.stop()
                except:
                    pass
            
            # Parse pytest output (exit code not available from tmux)
            return self._parse_pytest_output(output, 0)
            
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
    
    def _extract_token_count(self, white_agent_response: Optional[Dict[str, Any]]) -> Optional[int]:
        """
        Extract cumulative token count from white agent response.
        
        Args:
            white_agent_response: Response from white agent
            
        Returns:
            Total tokens used (cumulative), or None if not available
        """
        if not white_agent_response:
            return None
        
        try:
            # Path 1: result.message.metadata.cumulative_tokens (our implementation)
            if "result" in white_agent_response:
                result_data = white_agent_response["result"]
                if isinstance(result_data, dict):
                    # Check message.metadata for cumulative_tokens
                    if "message" in result_data:
                        message = result_data["message"]
                        if isinstance(message, dict) and "metadata" in message:
                            metadata = message["metadata"]
                            if isinstance(metadata, dict) and "cumulative_tokens" in metadata:
                                tokens = metadata["cumulative_tokens"]
                                self.logger.info(f"Extracted cumulative_tokens: {tokens}")
                                return tokens
                    
                    # Fallback: Check result.metadata directly
                    if "metadata" in result_data:
                        metadata = result_data["metadata"]
                        if isinstance(metadata, dict):
                            if "cumulative_tokens" in metadata:
                                return metadata["cumulative_tokens"]
                            elif "total_tokens" in metadata:
                                return metadata["total_tokens"]
                
        except Exception as e:
            self.logger.warning(f"Failed to extract token count: {e}")
        
        return None
