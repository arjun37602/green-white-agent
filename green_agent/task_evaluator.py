"""
Task Evaluation Framework for TerminalBench

This module provides comprehensive evaluation capabilities for terminal tasks,
including correctness checking, performance measurement, and safety validation.
"""

import json
import logging
import os
import re
import subprocess
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from .sandbox_manager import CommandResult, SandboxState
from .attempt_store import AttemptStore, Attempt


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
    performance: Dict[str, Any]
    safety: Dict[str, Any]
    efficiency: Dict[str, Any]
    details: Dict[str, Any]
    timestamp: str


class TaskEvaluator:
    """Evaluates task completion against success criteria"""
    
    def __init__(self, attempt_store: Optional[AttemptStore] = None):
        self.logger = logging.getLogger(__name__)
        self.attempt_store = attempt_store
    
    def evaluate(self, task_id: str, task_spec: Dict[str, Any], 
                 command_history: List[CommandResult], 
                 sandbox_manager=None, sandbox_id: str = None,
                 white_agent_response: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """
        Comprehensive task evaluation.
        
        Args:
            task_id: ID of the task being evaluated
            task_spec: Task specification with success criteria
            command_history: History of commands executed
            sandbox_state: Current state of the sandbox
            sandbox_manager: Optional sandbox manager instance
            sandbox_id: Optional sandbox ID
            white_agent_response: Optional white agent response containing token/request metadata
            
        Returns:
            EvaluationResult with detailed evaluation
        """
        self.logger.info(f"Evaluating task {task_id}")
        
        # Extract evaluation criteria
        criteria = self._extract_criteria(task_spec)
        
        # Run Terminal-Bench pytest tests (required)
        if not sandbox_manager or not sandbox_id or "metadata" not in task_spec:
            raise ValueError(f"Cannot evaluate task {task_id}: missing sandbox_manager, sandbox_id, or metadata")
        
        pytest_results = self._run_terminal_bench_tests(sandbox_manager, sandbox_id, task_spec["metadata"])
        
        if not pytest_results or pytest_results["tests_run"] == 0:
            raise ValueError(f"No pytest tests found or executed for task {task_id}")
        
        # Use pytest results as source of truth for correctness
        correctness_result = {
            "passed": pytest_results["passed"],
            "unit_tests": pytest_results
        }
        
        # Extract token/request counts from white agent response
        token_request_data = self._extract_token_and_request_counts(white_agent_response)
        
        performance_result = self._evaluate_performance(
            command_history, 
            total_tokens=token_request_data.get("total_tokens"),
            total_requests=token_request_data.get("total_requests")
        )
        safety_result = self._evaluate_safety(criteria, command_history)
        efficiency_result = self._evaluate_efficiency(criteria, command_history)
        
        # Binary scoring: 1.0 if all tests pass, 0.0 otherwise
        overall_score = 1.0 if correctness_result["passed"] else 0.0
        
        # Task passed if all tests passed
        passed = correctness_result["passed"]
        
        # Save attempt if store available
        if self.attempt_store:
            self._save_attempt(task_id, task_spec, correctness_result, performance_result)
        
        return EvaluationResult(
            task_id=task_id,
            passed=passed,
            score=overall_score,
            correctness=correctness_result,
            performance=performance_result,
            safety=safety_result,
            efficiency=efficiency_result,
            details={
                "total_commands": len(command_history),
                "total_execution_time": sum(cmd.execution_time for cmd in command_history),
                "successful_commands": sum(1 for cmd in command_history if cmd.success),
                "total_tokens": token_request_data.get("total_tokens"),
                "total_requests": token_request_data.get("total_requests"),
                "tokens_per_request": (
                    token_request_data["total_tokens"] / token_request_data["total_requests"]
                    if token_request_data.get("total_tokens") is not None 
                    and token_request_data.get("total_requests") is not None
                    and token_request_data["total_requests"] > 0
                    else None
                ),
                "evaluation_timestamp": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow().isoformat()
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
    
    def _evaluate_performance(self, command_history: List[CommandResult], 
                             total_tokens: Optional[int] = None,
                             total_requests: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate performance metrics based on tokens per request."""
        result = {
            "total_time": 0.0,
            "avg_time": 0.0,
            "slowest_command": None,
            "command_count": len(command_history) if command_history else 0,
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "tokens_per_request": None,
            "performance_score": 0.0
        }
        
        if command_history:
            total_time = sum(cmd.execution_time for cmd in command_history)
            avg_time = total_time / len(command_history)
            slowest_command = max(command_history, key=lambda cmd: cmd.execution_time)
            
            result["total_time"] = total_time
            result["avg_time"] = avg_time
            result["slowest_command"] = {
                "command": slowest_command.command,
                "execution_time": slowest_command.execution_time
            }
        
        # Calculate tokens per request if both metrics available
        if total_tokens is not None and total_requests is not None and total_requests > 0:
            tokens_per_request = total_tokens / total_requests
            result["tokens_per_request"] = tokens_per_request
            
            # Calculate performance score based on tokens per request
            if tokens_per_request < 500:
                result["performance_score"] = 0.2
            elif tokens_per_request < 2000:
                result["performance_score"] = 0.1
            else:
                result["performance_score"] = 0.0
            
            self.logger.info(f"Performance evaluation: {total_tokens} tokens, {total_requests} requests, "
                           f"{tokens_per_request:.2f} tokens/request, score: {result['performance_score']}")
        else:
            result["performance_score"] = 0.0
            if total_tokens is None or total_requests is None:
                self.logger.warning("Token/request counts not available for performance evaluation")
        
        return result
    
    def _evaluate_safety(self, criteria: EvaluationCriteria, 
                        command_history: List[CommandResult]) -> Dict[str, Any]:
        """Evaluate safety of executed commands."""
        safety_violations = []
        
        for cmd in command_history:
            command_lower = cmd.command.lower()
            
            # Check for forbidden commands
            for forbidden in criteria.forbidden_commands:
                if forbidden.lower() in command_lower:
                    safety_violations.append({
                        "command": cmd.command,
                        "violation": f"Forbidden command pattern: {forbidden}",
                        "severity": "high"
                    })
            
            # Check for potentially dangerous patterns
            dangerous_patterns = [
                r"rm\s+-rf\s+/",
                r"sudo\s+",
                r"chmod\s+777",
                r"dd\s+if=/dev/zero",
                r">\s*/dev/sda",
                r"mkfs\s+"
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, command_lower):
                    safety_violations.append({
                        "command": cmd.command,
                        "violation": f"Dangerous command pattern: {pattern}",
                        "severity": "high"
                    })
        
        return {
            "violations": safety_violations,
            "safe": len(safety_violations) == 0,
            "violation_count": len(safety_violations)
        }
    
    def _evaluate_efficiency(self, criteria: EvaluationCriteria, 
                           command_history: List[CommandResult]) -> Dict[str, Any]:
        """Evaluate efficiency of task completion."""
        if not command_history:
            return {"efficient": True, "command_count": 0, "within_limits": True}
        
        command_count = len(command_history)
        total_time = sum(cmd.execution_time for cmd in command_history)
        
        # Check if within limits
        within_command_limit = command_count <= criteria.max_commands
        within_time_limit = total_time <= criteria.max_execution_time
        
        # Check for redundant commands
        redundant_commands = self._detect_redundant_commands(command_history)
        
        return {
            "command_count": command_count,
            "total_time": total_time,
            "within_command_limit": within_command_limit,
            "within_time_limit": within_time_limit,
            "redundant_commands": redundant_commands,
            "efficient": within_command_limit and within_time_limit and len(redundant_commands) == 0
        }
    
    def _run_terminal_bench_tests(self, sandbox_manager, sandbox_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Terminal-Bench pytest tests in the Docker container.
        
        Args:
            sandbox_manager: Sandbox manager instance
            sandbox_id: ID of the sandbox
            metadata: Task metadata containing test file path
            
        Returns:
            Dictionary with test results
        """
        try:
            test_file_path = metadata.get("test_file_path")
            if not test_file_path or not os.path.exists(test_file_path):
                self.logger.warning("No valid test file found in metadata")
                return {"passed": False, "tests_run": 0, "tests_passed": 0, "details": "No test file found"}
            
            sandbox_info = sandbox_manager.get_sandbox_info(sandbox_id)
            if not sandbox_info:
                return {"passed": False, "tests_run": 0, "tests_passed": 0, "details": "Sandbox not found"}
            
            return self._run_tests_in_docker(sandbox_manager, sandbox_id, test_file_path, sandbox_info)
                
        except Exception as e:
            self.logger.error(f"Error running Terminal-Bench tests: {e}")
            return {"passed": False, "tests_run": 0, "tests_passed": 0, "details": f"Error: {e}"}
    
    def _run_tests_in_docker(self, sandbox_manager, sandbox_id: str, test_file_path: str, 
                            sandbox_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run tests inside a Docker container (Terminal Bench style)."""
        try:
            container = sandbox_info.get("container")
            container_name = sandbox_info.get("container_name")
            environment_spec = sandbox_info.get("environment_spec", {})
            
            if not container or not container_name:
                return {"passed": False, "tests_run": 0, "tests_passed": 0, "details": "No container found"}
            
            # Get the main run-tests.sh script path from metadata
            run_tests_script_path = environment_spec.get("run_tests_script_path")
            if not run_tests_script_path or not os.path.exists(run_tests_script_path):
                self.logger.error(f"run-tests.sh not found at: {run_tests_script_path}")
                return {"passed": False, "tests_run": 0, "tests_passed": 0, "details": "run-tests.sh not found"}
            
            # Copy the main run-tests.sh script to container
            self.logger.info(f"Copying run-tests.sh from {run_tests_script_path} to container")
            import subprocess
            copy_cmd = ["docker", "cp", run_tests_script_path, f"{container_name}:/run-tests.sh"]
            result = subprocess.run(copy_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to copy run-tests.sh: {result.stderr}")
                return {"passed": False, "tests_run": 0, "tests_passed": 0, "details": "Failed to copy run-tests.sh"}
            
            # Copy test directory into container
            test_dir = os.path.dirname(test_file_path)
            self.logger.info(f"Copying tests from {test_dir} to container")
            
            copy_cmd = ["docker", "cp", test_dir, f"{container_name}:/"]
            result = subprocess.run(copy_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Failed to copy tests: {result.stderr}")
                return {"passed": False, "tests_run": 0, "tests_passed": 0, "details": "Failed to copy tests"}
            
            # Run the main run-tests.sh script
            # This script handles all dependencies and test execution
            test_dir_name = os.path.basename(test_dir)
            self.logger.info(f"Running Terminal Bench test script: /run-tests.sh")
            pytest_result = sandbox_manager.execute_command(
                sandbox_id,
                f"cd /app && TEST_DIR=/{test_dir_name} bash /run-tests.sh",
                timeout=180  # Tests can take time (including dependency installation)
            )
            
            if not pytest_result.success:
                self.logger.warning(
                    f"Test execution completed with exit code {pytest_result.returncode}"
                )
            
            # Log full pytest output for debugging
            self.logger.info(f"Test stdout:\n{pytest_result.stdout}")
            self.logger.info(f"Test stderr:\n{pytest_result.stderr}")
            
            return self._parse_pytest_output(pytest_result)
            
        except Exception as e:
            self.logger.error(f"Error running tests in Docker: {e}")
            import traceback
            traceback.print_exc()
            return {"passed": False, "tests_run": 0, "tests_passed": 0, "details": f"Docker test error: {e}"}
    
    
    def _parse_pytest_output(self, pytest_result) -> Dict[str, Any]:
        """Parse pytest output and return test results."""
        import re
        
        output = pytest_result.stdout + pytest_result.stderr
        
        # Count tests using pytest's summary line (e.g., "2 passed in 0.5s" or "1 passed, 1 failed in 0.5s")
        summary_match = re.search(r'(\d+)\s+passed', output)
        tests_passed = int(summary_match.group(1)) if summary_match else 0
        
        failed_match = re.search(r'(\d+)\s+failed', output)
        tests_failed = int(failed_match.group(1)) if failed_match else 0
        
        tests_run = tests_passed + tests_failed
        
        # If no summary found, fall back to counting PASSED/FAILED markers
        if tests_run == 0:
            passed_matches = re.findall(r"PASSED", output)
            tests_passed = len(passed_matches)
            failed_matches = re.findall(r"FAILED", output)
            tests_failed = len(failed_matches)
            tests_run = tests_passed + tests_failed
        
        passed = pytest_result.success and tests_failed == 0
        
        self.logger.info(f"Pytest results: {tests_passed}/{tests_run} tests passed, {tests_failed} failed")
        
        return {
            "passed": passed,
            "tests_run": tests_run,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "details": f"Pytest: {tests_passed}/{tests_run} passed",
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
    
    def _detect_redundant_commands(self, command_history: List[CommandResult]) -> List[str]:
        """Detect redundant or inefficient commands."""
        redundant = []
        commands = [cmd.command for cmd in command_history]
        
        # Check for repeated identical commands
        command_counts = {}
        for cmd in commands:
            command_counts[cmd] = command_counts.get(cmd, 0) + 1
        
        for cmd, count in command_counts.items():
            if count > 1:
                redundant.append(f"Repeated command '{cmd}' {count} times")
        
        # Check for inefficient patterns
        inefficient_patterns = [
            ("ls", "ls -la"),  # ls followed by ls -la
            ("cat", "head"),   # cat followed by head
            ("find", "grep"),  # find followed by grep on same data
        ]
        
        for i in range(len(commands) - 1):
            current = commands[i].lower()
            next_cmd = commands[i + 1].lower()
            
            for pattern, follow_up in inefficient_patterns:
                if pattern in current and follow_up in next_cmd:
                    redundant.append(f"Inefficient sequence: '{current}' followed by '{next_cmd}'")
        
        return redundant
    
    def _save_attempt(
        self,
        task_id: str,
        task_spec: Dict[str, Any],
        correctness_result: Dict[str, Any],
        performance_result: Dict[str, Any]
    ) -> None:
        """Save attempt to store"""
        try:
            # Extract model ID
            model_id = task_spec.get("model_id", "default_model")
            if "metadata" in task_spec and "model_id" in task_spec["metadata"]:
                model_id = task_spec["metadata"]["model_id"]
            
            # Extract attempt ID
            attempt_id = task_spec.get("attempt_id", 0)
            if "metadata" in task_spec and "attempt_id" in task_spec["metadata"]:
                attempt_id = task_spec["metadata"]["attempt_id"]
            
            # Calculate accuracy
            if "unit_tests" in correctness_result and correctness_result["unit_tests"].get("tests_run", 0) > 0:
                tests_run = correctness_result["unit_tests"]["tests_run"]
                tests_passed = correctness_result["unit_tests"]["tests_passed"]
                accuracy = tests_passed / tests_run if tests_run > 0 else 0.0
            else:
                accuracy = 1.0 if correctness_result["passed"] else 0.0
            
            # Create attempt
            attempt = Attempt(
                attempt_id=attempt_id,
                accuracy=accuracy,
                num_tokens=performance_result.get("total_tokens") or 0,
                num_turns=performance_result.get("total_requests") or 1,
                timestamp=datetime.utcnow().isoformat(),
                metadata={"correctness": correctness_result, "performance": performance_result}
            )
            
            self.attempt_store.save_attempt(model_id, task_id, attempt)
            self.logger.info(f"Saved attempt for {model_id} on {task_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save attempt: {e}")
