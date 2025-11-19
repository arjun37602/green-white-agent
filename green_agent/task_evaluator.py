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
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self, task_id: str, task_spec: Dict[str, Any], 
                 command_history: List[CommandResult], sandbox_state: SandboxState, 
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
        
        # Perform different types of evaluation
        # Try to run Terminal-Bench pytest tests first
        pytest_results = None
        if sandbox_manager and sandbox_id and "metadata" in task_spec:
            pytest_results = self._run_terminal_bench_tests(sandbox_manager, sandbox_id, task_spec["metadata"])
        
        unit_test_correctness = {"passed": True, "tests_run": 0, "tests_passed": 0, "details": "No pytest tests run"}
        if pytest_results:
            unit_test_correctness = pytest_results
        
        correctness_result = self._evaluate_correctness(criteria, command_history, sandbox_state)
        
        # Extract token/request counts from white agent response
        token_request_data = self._extract_token_and_request_counts(white_agent_response)
        
        performance_result = self._evaluate_performance(
            command_history, 
            total_tokens=token_request_data.get("total_tokens"),
            total_requests=token_request_data.get("total_requests")
        )
        safety_result = self._evaluate_safety(criteria, command_history)
        efficiency_result = self._evaluate_efficiency(criteria, command_history)
        
        # Override correctness with pytest results if available
        if pytest_results and pytest_results["tests_run"] > 0:
            correctness_result["passed"] = pytest_results["passed"]
            correctness_result["unit_tests"] = pytest_results
        
        # Calculate overall score
        overall_score = self._calculate_score(correctness_result, performance_result, safety_result, efficiency_result)
        
        # Determine if task passed
        passed = overall_score >= 0.7 and correctness_result["passed"]
        
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
    
    def _evaluate_correctness(self, criteria: EvaluationCriteria, 
                            command_history: List[CommandResult], 
                            sandbox_state: SandboxState) -> Dict[str, Any]:
        """Evaluate correctness of task completion."""
        result = {
            "passed": True,
            "file_checks": {},
            "output_checks": {},
            "command_checks": {},
            "errors": []
        }
        
        # Check file requirements
        for file_path, expected_content in criteria.file_requirements.items():
            file_check = self._check_file_requirement(file_path, expected_content, sandbox_state)
            result["file_checks"][file_path] = file_check
            
            if not file_check["exists"]:
                result["passed"] = False
                result["errors"].append(f"Required file missing: {file_path}")
            elif expected_content != ".*" and not file_check["content_matches"]:
                result["passed"] = False
                result["errors"].append(f"File content doesn't match requirement: {file_path}")
        
        # Check output requirements
        for pattern in criteria.output_requirements:
            output_check = self._check_output_requirement(pattern, command_history)
            result["output_checks"][pattern] = output_check
            
            if not output_check["found"]:
                result["passed"] = False
                result["errors"].append(f"Required output pattern not found: {pattern}")
        
        # Check success conditions
        for condition in criteria.success_conditions:
            condition_check = self._check_success_condition(condition, command_history, sandbox_state)
            result["command_checks"][condition] = condition_check
            
            if not condition_check["satisfied"]:
                result["passed"] = False
                result["errors"].append(f"Success condition not met: {condition}")
        
        return result
    
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
    
    def _check_file_requirement(self, file_path: str, expected_content: str, 
                              sandbox_state: SandboxState) -> Dict[str, Any]:
        """Check if a file requirement is met."""
        file_snapshot = sandbox_state.file_system_snapshot
        
        # Check if file exists with exact path
        if file_path in file_snapshot["files"]:
            actual_content = file_snapshot["files"][file_path]
            
            if expected_content == ".*":
                return {"exists": True, "content_matches": True, "content": actual_content}
            else:
                content_matches = expected_content in actual_content
                return {"exists": True, "content_matches": content_matches, "content": actual_content}
        
        # Check if file exists with app/ prefix (sandbox working directory)
        app_file_path = f"app/{file_path}"
        if app_file_path in file_snapshot["files"]:
            actual_content = file_snapshot["files"][app_file_path]
            
            if expected_content == ".*":
                return {"exists": True, "content_matches": True, "content": actual_content}
            else:
                content_matches = expected_content in actual_content
                return {"exists": True, "content_matches": content_matches, "content": actual_content}
        
        # Check if file exists anywhere in the sandbox (fallback)
        for existing_file_path, content in file_snapshot["files"].items():
            if file_path in existing_file_path or existing_file_path.endswith(file_path):
                if expected_content == ".*":
                    return {"exists": True, "content_matches": True, "content": content}
                else:
                    content_matches = expected_content in content
                    return {"exists": True, "content_matches": content_matches, "content": content}
        
        return {"exists": False, "content_matches": False, "content": None}
    
    def _run_terminal_bench_tests(self, sandbox_manager, sandbox_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Terminal-Bench pytest tests in the sandbox.
        
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
            
            # Copy test file to sandbox
            test_filename = os.path.basename(test_file_path)
            sandbox_test_path = f"app/{test_filename}"
            
            # Create app directory first
            mkdir_result = sandbox_manager.execute_command(
                sandbox_id,
                "mkdir -p app"
            )
            
            if not mkdir_result.success:
                self.logger.warning(f"Failed to create app directory: {mkdir_result.stderr}")
            
            # Copy test file to sandbox
            copy_result = sandbox_manager.execute_command(
                sandbox_id, 
                f"cp {test_file_path} {sandbox_test_path}"
            )
            
            if not copy_result.success:
                self.logger.error(f"Failed to copy test file to sandbox: {copy_result.stderr}")
                return {"passed": False, "tests_run": 0, "tests_passed": 0, "details": "Failed to copy test file"}
            
            # Install pytest in sandbox
            install_result = sandbox_manager.execute_command(
                sandbox_id,
                "pip install pytest"
            )
            
            if not install_result.success:
                self.logger.warning(f"Failed to install pytest: {install_result.stderr}")
                # Try to continue anyway - pytest might already be installed
            
            # Create symlink so pytest tests can find /app/* files
            # Tests expect /app/file.txt, but we created ./app/file.txt in sandbox
            symlink_result = sandbox_manager.execute_command(
                sandbox_id,
                "mkdir -p /tmp/test_app && cp -r app/* /tmp/test_app/ 2>/dev/null || true"
            )
            
            # Modify test file to use /tmp/test_app instead of /app
            # This is a workaround since we can't create /app
            sandbox_manager.execute_command(
                sandbox_id,
                f"sed -i.bak 's|/app|/tmp/test_app|g' {sandbox_test_path} 2>/dev/null || sed -i '' 's|/app|/tmp/test_app|g' {sandbox_test_path} 2>/dev/null || true"
            )
            
            # Also copy files to /tmp/test_app
            sandbox_manager.execute_command(
                sandbox_id,
                "cp -r ./app/* /tmp/test_app/ 2>/dev/null || true"
            )
            
            # Run pytest
            pytest_result = sandbox_manager.execute_command(
                sandbox_id,
                f"python -m pytest {sandbox_test_path} -v"
            )
            
            # Parse pytest output
            output = pytest_result.stdout + pytest_result.stderr
            
            # Count tests using pytest's summary line (e.g., "2 passed in 0.5s" or "1 passed, 1 failed in 0.5s")
            import re
            
            # Try to parse the summary line first
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
            
        except Exception as e:
            self.logger.error(f"Error running Terminal-Bench tests: {e}")
            return {"passed": False, "tests_run": 0, "tests_passed": 0, "details": f"Error: {e}"}
    
    def _check_output_requirement(self, pattern: str, command_history: List[CommandResult]) -> Dict[str, Any]:
        """Check if an output requirement is met."""
        all_output = ""
        for cmd in command_history:
            all_output += cmd.stdout + "\n" + cmd.stderr + "\n"
        
        found = pattern.lower() in all_output.lower()
        return {"found": found, "pattern": pattern, "searched_output": all_output[:500]}
    
    def _check_success_condition(self, condition: str, command_history: List[CommandResult], 
                               sandbox_state: SandboxState) -> Dict[str, Any]:
        """Check if a success condition is met."""
        condition_lower = condition.lower()
        
        # Simple keyword-based checking
        if "file exists" in condition_lower or "exists" in condition_lower:
            # Check if any files OR directories were created
            files_created = len(sandbox_state.file_system_snapshot["files"]) > 0
            directories_created = len(sandbox_state.file_system_snapshot["directories"]) > 0
            return {"satisfied": files_created or directories_created, "condition": condition, "details": "File/directory existence check"}
        
        if "directory" in condition_lower and "exists" in condition_lower:
            # Check specifically for directories
            directories_created = len(sandbox_state.file_system_snapshot["directories"]) > 0
            return {"satisfied": directories_created, "condition": condition, "details": "Directory existence check"}
        
        if "navigate" in condition_lower or "cd" in condition_lower:
            # Check if cd commands were executed successfully
            cd_commands = [cmd for cmd in command_history if cmd.command.startswith("cd")]
            cd_success = any(cmd.success for cmd in cd_commands)
            return {"satisfied": cd_success, "condition": condition, "details": "Navigation check"}
        
        if "success" in condition_lower or "completed" in condition_lower:
            # Check if last command was successful
            if command_history:
                last_command_success = command_history[-1].success
                return {"satisfied": last_command_success, "condition": condition, "details": "Last command success"}
        
        # Default: be strict - only pass if we can verify the condition
        return {"satisfied": False, "condition": condition, "details": "No specific condition matched - strict evaluation"}
    
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
    
    def _calculate_score(self, correctness: Dict[str, Any], performance: Dict[str, Any], 
                        safety: Dict[str, Any], efficiency: Dict[str, Any]) -> float:
        """
        Calculate overall evaluation score.
        
        Scoring breakdown:
        - Correctness: 80% (Partial credit based on checkpoint progress)
        - Turns (API requests): 10% (Proportional - fewer = better)
        - Token usage: 10% (Proportional - less = better)
        """
        score = 0.0
        
        # Maximum expected values (reasonable budgets for tasks)
        MAX_REQUESTS = 10  # Most tasks should complete in ≤10 requests
        MAX_TOKENS = 16384  # Reasonable token budget for complex tasks
        
        # 1. CORRECTNESS: 80% (Partial credit for checkpoint progress)
        # If we have unit test results, give partial credit based on tests passed
        if "unit_tests" in correctness and correctness["unit_tests"].get("tests_run", 0) > 0:
            tests_run = correctness["unit_tests"]["tests_run"]
            tests_passed = correctness["unit_tests"]["tests_passed"]
            # Proportional credit: (tests_passed / tests_run) * 0.80
            correctness_score = (tests_passed / tests_run) * 0.80
            score += correctness_score
            self.logger.info(f"✅ Partial correctness: {tests_passed}/{tests_run} tests passed → {correctness_score:.2f}/0.80")
        elif correctness["passed"]:
            # Binary pass/fail if no unit tests
            score += 0.80
        
        # 2. TURNS (API Requests): 10% - Proportional scoring
        # Formula: 0.10 * (1 - requests_used / max_requests)
        # Examples: 1 request = 0.09, 2 = 0.08, 5 = 0.05, 10 = 0.00
        total_requests = performance.get("total_requests")
        if total_requests is not None:
            requests_ratio = min(total_requests / MAX_REQUESTS, 1.0)
            turns_score = 0.10 * (1.0 - requests_ratio)
            score += turns_score
        
        # 3. TOKEN USAGE: 10% - Proportional scoring
        # Formula: 0.10 * (1 - tokens_used / max_tokens)
        # Examples: 1000 tokens = 0.095, 5000 = 0.075, 10000 = 0.05, 16384 = 0.00
        total_tokens = performance.get("total_tokens")
        if total_tokens is not None:
            tokens_ratio = min(total_tokens / MAX_TOKENS, 1.0)
            tokens_score = 0.10 * (1.0 - tokens_ratio)
            score += tokens_score
        
        return min(score, 1.0)
