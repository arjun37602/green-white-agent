#!/usr/bin/env python3
"""
Green Agent: Terminal Bench Task Runner and Evaluator
Loads Terminal Bench tasks and sends them to the white agent for evaluation.
Uses Terminal Bench's official Docker and container management.
"""

import json
import logging
import os
import sys
import time

# Suppress verbose a2a logging
logging.getLogger("a2a.client.card_resolver").setLevel(logging.WARNING)
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.handlers.trial_handler import Task, TaskPaths
from docker.models.containers import Container

from .task_evaluator import TaskEvaluator, EvaluationResult
from .dataset_loaders.terminal_bench_loader import TerminalBenchTaskLoader
from .results_store import ResultsStore, TaskResult

# A2A imports for proper message handling
from utils import send_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of a command execution."""
    command: str
    stdout: str
    stderr: str
    returncode: int
    execution_time: float
    timestamp: str
    success: bool


@dataclass
class TaskExecutionResult:
    """Result of a complete task execution"""
    task_id: str
    success: bool
    execution_time: float
    evaluation_result: EvaluationResult
    sandbox_id: str
    white_agent_response: Dict[str, Any]
    timestamp: str


class GreenAgentTerminalBench:
    """Green agent for running Terminal Bench tasks against the white agent."""
    
    # Tool definitions 
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "execute_bash_command",
                "description": "Execute a bash command in the task container and return the output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute"
                        }
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "stop",
                "description": "Stop the agent loop and proceed to task evaluation. Call this when you believe the task is complete.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Reason for stopping (optional)"
                        }
                    },
                    "required": []
                }
            }
        }
    ]
    
    def __init__(self, white_agent_url: str = "http://localhost:8002", 
                 sandbox_base_path: Optional[str] = None,
                 terminal_bench_dataset_path: Optional[str] = None,
                 model_id: str = "default_model",
                 results_dir: str = "./results"):
        self.white_agent_url = white_agent_url
        self.model_id = model_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize task evaluator
        self.task_evaluator = TaskEvaluator()
        
        # Initialize results store
        self.results_store = ResultsStore(results_dir)
        
        # Load completed tasks for caching
        self.completed_tasks = self.results_store.load_completed_tasks(model_id)
        self.logger.info(f"Loaded {len(self.completed_tasks)} completed tasks for model {model_id}")
        
        # Initialize Terminal-Bench loader if dataset path provided
        self.tb_loader = None
        if terminal_bench_dataset_path:
            try:
                self.tb_loader = TerminalBenchTaskLoader(terminal_bench_dataset_path)
                self.logger.info(f"Terminal-Bench dataset loaded from: {terminal_bench_dataset_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load Terminal-Bench dataset: {e}")
                raise e
        
        # Task execution tracking
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[TaskExecutionResult] = []
        
        # Current container for tool execution
        self.current_container: Optional[Container] = None
        self.current_container_manager: Optional[DockerComposeManager] = None
    
    def is_task_completed(self, task_id: str) -> bool:
        """Check if a task has already been completed (for caching)"""
        return task_id in self.completed_tasks
    
    def load_terminal_bench_tasks(self, task_ids: List[str] = None, skip_completed: bool = True) -> List[Tuple[Task, TaskPaths]]:
        """
        Load Terminal Bench tasks from Terminal-Bench dataset.
        
        Args:
            task_ids: Specific task IDs to load (None = load all)
            skip_completed: If True, skip tasks that are already completed
            
        Returns:
            List of (Task, TaskPaths) tuples from Terminal Bench
            
        Raises:
            RuntimeError: If Terminal-Bench dataset is not available
        """
        if not self.tb_loader:
            raise RuntimeError(
                "Terminal-Bench dataset loader not initialized. "
                "Please provide a valid terminal_bench_dataset_path when creating the GreenAgent."
            )
        
        # Load from Terminal-Bench dataset - returns (Task, TaskPaths) tuples
        self.logger.info("Loading tasks from Terminal-Bench dataset")
        tasks = self.tb_loader.load_tasks_from_dataset(task_ids)
        
        # Filter out completed tasks if requested
        if skip_completed:
            original_count = len(tasks)
            tasks = [(task, task_paths) for task, task_paths in tasks 
                    if task_paths.input_path.name not in self.completed_tasks]
            skipped = original_count - len(tasks)
            if skipped > 0:
                self.logger.info(f"Skipped {skipped} already completed tasks")
        
        self.logger.info(f"Loaded {len(tasks)} tasks from Terminal-Bench dataset")
        return tasks
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call from the white agent.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if tool_name == "execute_bash_command":
            return self._execute_bash_command_tool(arguments)
        elif tool_name == "stop":
            return self._stop_tool(arguments)
        else:
            self.logger.warning(f"Unknown tool: {tool_name}")
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }
    
    def _execute_bash_command_tool(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bash command tool."""
        if not self.current_container:
            return {
                "success": False,
                "error": "No active container"
            }
        
        command = arguments.get("command", "")
        if not command:
            return {
                "success": False,
                "error": "No command provided"
            }
        
        self.logger.info(f"Tool: execute_bash_command - {command}")
        
        try:
            start_time = time.time()
            exec_result = self.current_container.exec_run(
                ["bash", "-c", command],
                workdir="/app"
            )
            execution_time = time.time() - start_time
            
            stdout_output = exec_result.output.decode('utf-8', errors='replace') if exec_result.output else ""
            
            return {
                "success": exec_result.exit_code == 0,
                "command": command,
                "stdout": stdout_output,
                "stderr": "",  # Docker exec_run combines stdout and stderr
                "returncode": exec_result.exit_code,
                "execution_time": execution_time,
                # Include error message when command fails
                "error": stdout_output if exec_result.exit_code != 0 else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _stop_tool(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Stop tool - signals early termination."""
        reason = arguments.get("reason", "Agent signaled completion")
        self.logger.info(f"Tool: stop - {reason}")
        return {
            "success": True,
            "message": "Agent loop will stop",
            "reason": reason
        }

    
    async def execute_task_with_sandbox(self, task: Task, task_paths: TaskPaths, output_directory: Path) -> TaskExecutionResult:
        """
        Execute a complete task lifecycle with sandbox isolation.
        
        Args:
            task: Terminal Bench Task object
            task_paths: Terminal Bench TaskPaths object
            output_directory: Directory for output logs
            
        Returns:
            TaskExecutionResult with complete execution details
        """
        task_id = task_paths.input_path.name
        
        # Check cache first
        if self.is_task_completed(task_id):
            self.logger.info(f"Task {task_id} already completed, skipping...")
            # Return a cached result indicator
            return TaskExecutionResult(
                task_id=task_id,
                success=True,  # Already completed means it was successful
                execution_time=0.0,
                evaluation_result=None,
                sandbox_id="cached",
                white_agent_response={"status": "cached"},
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        start_time = time.time()
        
        # Initialize trajectory logging for this task
        trajectory_data = {
            "task_id": task_id,
            "task": {
                "instruction": task.instruction,
                "difficulty": task.difficulty.value,
                "category": task.category,
            },
            "start_time": datetime.now(timezone.utc).isoformat(),
            "interactions": []
        }
        
        self.logger.info(f"Starting task execution: {task_id} (model: {self.model_id})")
        
        # Container names
        container_name = f"tbench_{task_id.replace('-', '_')}_{int(time.time())}_client"
        image_name = f"tbench_{task_id.replace('-', '_')}_image"
        
        sessions_logs_path = output_directory / "sessions"
        agent_logs_path = output_directory / "agent-logs"
        sessions_logs_path.mkdir(parents=True, exist_ok=True)
        agent_logs_path.mkdir(parents=True, exist_ok=True)
        
        # Create Docker Compose Manager with log paths to satisfy environment variables
        container_manager = DockerComposeManager(
            client_container_name=container_name,
            client_image_name=image_name,
            docker_compose_path=task_paths.docker_compose_path,
            docker_image_name_prefix=f"tbench_{task_id}",
            no_rebuild=False,
            cleanup=True,
            sessions_logs_path=sessions_logs_path,
            agent_logs_path=agent_logs_path
        )
        
        try:
            # Start container
            self.logger.info(f"Starting Docker container for task {task_id}")
            container = container_manager.start()
            self.logger.info(f"Container started: {container.name}")
            
            # Set current container for tool execution
            self.current_container = container
            self.current_container_manager = container_manager
            
            # Iteratively call white agent until task complete or max iterations reached
            max_iterations = 50
            iteration = 0
            should_stop = False
            context_id = None  # Track context ID for conversation
            
            self.logger.info(f"Starting agent loop (max {max_iterations} iterations)")
            
            # Track tool results from previous iteration only
            previous_tool_results = None
            
            while not should_stop and iteration < max_iterations:
                iteration += 1
                self.logger.info(f"Iteration {iteration}/{max_iterations}")
                
                # Call white agent
                try:
                    if iteration == 1:
                        # First iteration: send initial task with tool definitions
                        self.logger.info("Sending initial task with tool definitions")
                        white_agent_response, interaction = await self.send_initial_task(task, task_paths)
                        # Extract context_id from response
                        if "result" in white_agent_response and "context_id" in white_agent_response["result"]:
                            context_id = white_agent_response["result"]["context_id"]
                        trajectory_data["interactions"].append(interaction) #add initial task
                    else:
                        # Subsequent iterations: send tool results or error
                        if previous_tool_results is None or len(previous_tool_results) == 0:
                            # White agent didn't send tool calls - send error
                            self.logger.warning("White agent did not send tool calls in previous iteration")
                            interaction = {
                                "iteration": iteration,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "user": {
                                    "type": "error",
                                    "error_message": "You must send a tool command"
                                }
                            }
                            trajectory_data["interactions"].append(interaction) #add error
                            white_agent_response = await self.send_tool_results(
                                task, task_paths, error_message="You must send a tool command", context_id=context_id
                            )
                        else:
                            # Send tool results from previous iteration
                            white_agent_response = await self.send_tool_results(
                                task, task_paths, tool_results=previous_tool_results, context_id=context_id
                            )
                    
                    # Log white agent response
                    interaction = {
                        "iteration": iteration,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    interaction["assistant"] = white_agent_response
                    trajectory_data["interactions"].append(interaction) #add assistant response
                    
                except Exception as e:
                    self.logger.error(f"Error communicating with white agent: {e}")
                    self.logger.warning("Breaking loop due to white agent error")
                    interaction["error"] = str(e)
                    trajectory_data["interactions"].append(interaction)
                    break
                
                # Extract and execute tool calls from white agent response
                tool_calls = self._extract_tool_calls(white_agent_response)
                
                if not tool_calls:
                    self.logger.warning("No tool calls in response - will send error next iteration")
                    previous_tool_results = None
                    continue
                
                # Execute each tool call
                iteration_tool_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name")
                    tool_arguments = tool_call.get("arguments", {})
                    tool_call_id = tool_call.get("id", f"call_{iteration}")
                    
                    # Execute the tool
                    tool_result = self.execute_tool(tool_name, tool_arguments)
                    
                    # Log the tool result
                    if tool_result.get('success'):
                        output = tool_result.get('stdout', '').strip()
                        if output:
                            self.logger.info(f"Tool result: SUCCESS\n{output}{'...' if len(output) > 200 else ''}")
                        else:
                            self.logger.info(f"Tool result: SUCCESS (no output)")
                    else:
                        # Show both error and stdout for failed commands
                        error = tool_result.get('error', 'Unknown error')
                        stdout = tool_result.get('stdout', '').strip()
                        if stdout:
                            self.logger.error(f"Tool result: FAILED (exit code {tool_result.get('returncode', '?')})\nOutput:\n{stdout}")
                        else:
                            self.logger.error(f"Tool result: FAILED - {error}")
                    
                    # Check if it's the stop tool
                    if tool_name == "stop":
                        should_stop = True
                    

                    # Store tool result
                    iteration_tool_results.append({
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "arguments": tool_arguments,
                        "result": tool_result
                    })
                
                # Log tool execution results
                interaction["tool_executions"] = iteration_tool_results
                
                # Store results for next iteration and for history
                previous_tool_results = iteration_tool_results
                
                # Add interaction to trajectory
                trajectory_data["interactions"].append(interaction)
                
                if should_stop:
                    self.logger.info(f"Agent called stop tool - terminating after iteration {iteration}")
                    break
            
            if iteration >= max_iterations:
                self.logger.warning(f"Reached max iterations ({max_iterations}) without completion")
            
            self.logger.info(f"Starting task evaluation...")
            
            try:
                evaluation_result = self.task_evaluator.evaluate(
                    task_id, task_paths, container, white_agent_response
                )
                self.logger.info(f"Evaluation complete: {evaluation_result.passed if evaluation_result else 'N/A'}")
                
                # Add evaluation result to trajectory
                if evaluation_result:
                    trajectory_data["evaluation"] = {
                        "passed": evaluation_result.passed,
                        "score": evaluation_result.score,
                        "details": evaluation_result.details
                    }
            except Exception as e:
                self.logger.error(f"Evaluation failed: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                evaluation_result = None
                trajectory_data["evaluation"] = {
                    "error": str(e)
                }
            
            # Clean up container
            self.logger.info(f"Cleaning up container...")
            try:
                container_manager.stop()
                self.logger.info(f"Container cleaned up")
            except Exception as e:
                self.logger.warning(f"Error cleaning up container: {e}")
            
            execution_time = time.time() - start_time
            trajectory_data["end_time"] = datetime.now(timezone.utc).isoformat()
            trajectory_data["execution_time"] = execution_time
            
            # Calculate metrics from trajectory
            num_turns = len([i for i in trajectory_data["interactions"] if "assistant" in i])
            num_tokens = self._extract_tokens_from_trajectory(trajectory_data)
            
            # Extract test case metrics from evaluation
            if evaluation_result and evaluation_result.details:
                # Try to extract from evaluation details
                passed_test_cases = evaluation_result.details.get("passed_tests", 1 if evaluation_result.passed else 0)
                total_test_cases = evaluation_result.details.get("total_tests", 1)
            else:
                # Fallback: use passed as binary
                passed_test_cases = 1 if (evaluation_result and evaluation_result.passed) else 0
                total_test_cases = 1
            
            accuracy = evaluation_result.score if evaluation_result else 0.0
            
            # Save to results store
            task_result = TaskResult(
                task_id=task_id,
                attempt_id=0,  # Can be incremented if running multiple attempts
                num_tokens=num_tokens,
                num_turns=num_turns,
                passed_test_cases=passed_test_cases,
                total_test_cases=total_test_cases,
                accuracy=accuracy,
                timestamp=datetime.now(timezone.utc).isoformat(),
                execution_time=execution_time,
                message_history=trajectory_data["interactions"],
                metadata={
                    "model_id": self.model_id,
                    "difficulty": task.difficulty.value,
                    "category": task.category,
                    "sandbox_id": container.name,
                }
            )
            self.results_store.save_result(self.model_id, task_result)
            self.logger.info(f"Saved result to {self.results_store.get_model_file(self.model_id)}")
            
            # Add to completed tasks cache
            self.completed_tasks.add(task_id)
            
            # Create execution result
            result = TaskExecutionResult(
                task_id=task_id,
                success=evaluation_result.passed if evaluation_result else False,
                execution_time=execution_time,
                evaluation_result=evaluation_result,
                sandbox_id=container.name,
                white_agent_response=white_agent_response,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            # Store in history
            self.execution_history.append(result)
            
            self.logger.info(f"Task {task_id} completed in {execution_time:.2f}s - {'PASSED' if result.success else 'FAILED'}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task {task_id} failed: {e}")
            
            trajectory_data["error"] = str(e)
            trajectory_data["end_time"] = datetime.now(timezone.utc).isoformat()
            
            # Clean up container on error
            if 'container_manager' in locals():
                try:
                    container_manager.stop()
                except:
                    pass
            
            # Create failed result
            execution_time = time.time() - start_time
            return TaskExecutionResult(
                task_id=task_id,
                success=False,
                execution_time=execution_time,
                evaluation_result=None,
                sandbox_id=container.name if 'container' in locals() else None,
                white_agent_response={"error": str(e)},
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        finally:
            # Clear container references
            self.current_container = None
            self.current_container_manager = None
    
    def _extract_tokens_from_trajectory(self, trajectory_data: Dict[str, Any]) -> int:
        """
        Extract actual token count from trajectory data.
        Looks for <!--TOKENS:N--> markers in assistant responses.
        """
        # Look for the last token count in assistant responses
        last_token_count = 0
        
        for interaction in trajectory_data.get("interactions", []):
            if "assistant" in interaction:
                assistant_data = interaction["assistant"]
                
                # Try to extract from various response structures
                response_text = ""
                if isinstance(assistant_data, dict):
                    # Check in result.message or result.history
                    if "result" in assistant_data:
                        result = assistant_data["result"]
                        if isinstance(result, dict):
                            if "message" in result and isinstance(result["message"], dict):
                                if "parts" in result["message"]:
                                    # Extract text from parts
                                    for part in result["message"]["parts"]:
                                        if isinstance(part, dict) and part.get("kind") == "text":
                                            response_text += part.get("text", "")
                            if "history" in result:
                                for msg in result["history"]:
                                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                                        response_text += msg.get("content", "")
                    # Direct content field
                    if "content" in assistant_data:
                        response_text += str(assistant_data["content"])
                else:
                    response_text = str(assistant_data)
                
                # Extract token count from <!--TOKENS:N--> marker
                import re
                token_match = re.search(r'<!--TOKENS:(\d+)-->', response_text)
                if token_match:
                    last_token_count = int(token_match.group(1))
        
        return last_token_count
    
    def _extract_tool_calls(self, white_agent_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract tool calls from white agent response and normalize to simplified format.
        
        Accepts OpenAI format and converts to simplified format for internal use.
        
        Args:
            white_agent_response: Response from white agent
            
        Returns:
            List of tool calls in simplified format: 
            [{"id": "call_123", "name": "execute_bash_command", "arguments": {...}}]
            Note: arguments is a dict, not a JSON string (unlike OpenAI format)
        """
        tool_calls = []
        
        try:
            # Check for OpenAI-style tool_calls in message
            if "result" in white_agent_response:
                result = white_agent_response["result"]
                
                # Check for tool_calls in the history/message
                if "history" in result:
                    for message in result["history"]:
                        if isinstance(message, dict) and "tool_calls" in message:
                            for tc in message["tool_calls"]:
                                tool_calls.append({
                                    "id": tc.get("id", f"call_{len(tool_calls)}"),
                                    "name": tc.get("function", {}).get("name") or tc.get("name"),
                                    "arguments": json.loads(tc.get("function", {}).get("arguments", "{}")) if "function" in tc else tc.get("arguments", {})
                                })
                
                # Check for artifacts with tool_calls (A2A format)
                if "artifacts" in result:
                    for artifact in result["artifacts"]:
                        # Check if tool_calls are directly in artifact
                        if "tool_calls" in artifact:
                            for tc in artifact["tool_calls"]:
                                tool_calls.append({
                                    "id": tc.get("id", f"call_{len(tool_calls)}"),
                                    "name": tc.get("name"),
                                    "arguments": tc.get("arguments", {})
                                })
                        # Check if artifact contains JSON text with tool_calls
                        elif "parts" in artifact:
                            for part in artifact["parts"]:
                                if part.get("kind") == "text":
                                    try:
                                        text_data = json.loads(part.get("text", "{}"))
                                        if "tool_calls" in text_data:
                                            for tc in text_data["tool_calls"]:
                                                tool_calls.append({
                                                    "id": tc.get("id", f"call_{len(tool_calls)}"),
                                                    "name": tc.get("function", {}).get("name") or tc.get("name"),
                                                    "arguments": json.loads(tc.get("function", {}).get("arguments", "{}")) if "function" in tc else tc.get("arguments", {})
                                                })
                                    except json.JSONDecodeError:
                                        pass
            
            if tool_calls:
                self.logger.info(f"Extracted {len(tool_calls)} tool calls")
                for tc in tool_calls:
                    self.logger.info(f"  - {tc['name']}({list(tc.get('arguments', {}).keys())})")
            
        except Exception as e:
            self.logger.error(f"Error extracting tool calls: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        return tool_calls
    
    
    async def send_initial_task(self, task: Task, task_paths: TaskPaths) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Send initial task with tool definitions to the white agent using A2A SDK.
        
        Args:
            task: Terminal Bench Task object
            task_paths: Terminal Bench TaskPaths object
            
        Returns:
            Tuple of (white agent response, interaction dict with logged request)
        """
        try:
            # Construct interaction
            interaction = {
                "iteration": 1,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Build initial task message with tools in prompt (text-based, not native tool calling)
            task_id = task_paths.input_path.name
            tools_description = json.dumps(self.TOOLS, indent=2)
            task_message = f"""Terminal Bench Task: {task_id}

            Description: {task.category} task

            Instructions:
            You are a terminal agent that can execute bash commands.
            {task.instruction}

            Environment:
            Working Directory: /app

            You have access to the following tools:
            {tools_description}

            Please respond in JSON format. Wrap your response in <json>...</json> tags.
            The JSON should contain:
            - "name": the tool name (e.g., "execute_bash_command" or "stop")
            - "kwargs": object with the tool arguments

            Example response format:
            <json>
            {{"name": "execute_bash_command", "kwargs": {{"command": "ls -la"}}}}
            </json>

            Complete this task using the available tools."""
            
            # Log the request
            self.logger.info(f"=== SENDING TO WHITE AGENT (INITIAL TASK) ===")
            self.logger.info(f"URL: {self.white_agent_url}")
            self.logger.info(f"Tools provided: {len(self.TOOLS)} tools")
            self.logger.debug(f"Tools: {tools_description}")
            self.logger.info(f"Message preview: {task_message[:200]}...")
            
            # Send using A2A SDK (no tools parameter - all in text)
            response = await send_message(
                self.white_agent_url,
                task_message,
                context_id=None
            )
            
            # Convert response to dict for compatibility with existing code
            from a2a.types import SendMessageSuccessResponse, Message
            from utils import parse_tags
            res_root = response.root
            if isinstance(res_root, SendMessageSuccessResponse):
                # Extract the assistant's response text
                assistant_text = self._extract_text_from_message(res_root.result) if isinstance(res_root.result, Message) else ""
                
                # Parse tool calls from <json>...</json> tags (tau-bench format)
                tool_calls_list = []
                try:
                    tags = parse_tags(assistant_text)
                    if "json" in tags:
                        json_content = tags["json"].strip()
                        self.logger.info(f"Parsing JSON content: {json_content}")
                        
                        # Try to parse the JSON, if it fails due to extra closing braces, strip them
                        try:
                            response_data = json.loads(json_content)
                        except json.JSONDecodeError as e:
                            # Check if it's an "Extra data" error and try stripping trailing braces
                            if "Extra data" in str(e):
                                # Remove trailing } characters one by one until valid JSON
                                cleaned = json_content.rstrip()
                                while cleaned.endswith('}') and len(cleaned) > 0:
                                    try:
                                        response_data = json.loads(cleaned)
                                        self.logger.info(f"Successfully parsed after removing extra braces")
                                        break
                                    except json.JSONDecodeError:
                                        cleaned = cleaned[:-1].rstrip()
                                else:
                                    raise e
                            else:
                                raise
                        
                        if isinstance(response_data, dict):
                            # Tau-bench format: {"name": "tool_name", "kwargs": {...}}
                            if "name" in response_data and "kwargs" in response_data:
                                tool_calls_list = [{
                                    "name": response_data["name"],
                                    "arguments": response_data["kwargs"]
                                }]
                            # Also support array format for backwards compatibility
                            elif "tool_calls" in response_data:
                                tool_calls_list = response_data["tool_calls"]
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Failed to parse JSON from response: {e}")
                    self.logger.info(f"Raw assistant response: {assistant_text}")
                
                # Build the result in a format compatible with existing code
                result = {
                    "jsonrpc": "2.0",
                    "id": res_root.id,
                    "result": {
                        "message": res_root.result.model_dump() if isinstance(res_root.result, Message) else res_root.result,
                        "context_id": res_root.result.context_id if isinstance(res_root.result, Message) else None,
                        # Include tool_calls in history for extraction
                        "history": [{"role": "assistant", "content": assistant_text, "tool_calls": tool_calls_list}],
                        "artifacts": []
                    }
                }
                
                # Log the request to interaction
                interaction["user"] = {
                    "type": "initial_task",
                    "message": task_message,
                    "tools": self.TOOLS
                }
                
                self.logger.debug(f"White agent initial response received")
                return result, interaction
            else:
                error_msg = f"Unexpected response type from white agent: {type(res_root)}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            self.logger.error(f"Failed to send initial task to white agent: {e}")
            raise
    
    def _extract_text_from_message(self, message) -> str:
        """Extract text content from an A2A message."""
        if hasattr(message, 'parts'):
            from a2a.utils import get_text_parts
            text_parts = get_text_parts(message.parts)
            return "\n".join(text_parts) if text_parts else ""
        return str(message)
    
    async def send_tool_results(self, task: Task, task_paths: TaskPaths,
                         tool_results: Optional[List[Dict[str, Any]]] = None,
                         error_message: Optional[str] = None,
                         context_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Send tool results or error message to the white agent using plain text format.
        
        Args:
            task: Terminal Bench Task object
            task_paths: Terminal Bench TaskPaths object
            tool_results: Tool execution results from previous iteration
            error_message: Error message if white agent didn't send tool calls
            context_id: Context ID to maintain conversation
            
        Returns:
            White agent response
        """
        try:
            # Build message based on input - simple plain text like tau-bench
            if error_message:
                message_text = f"Error: {error_message}"
            elif tool_results:
                # Simple text format - just send the tool results
                results_text = []
                for tr in tool_results:
                    tool_name = tr.get('tool_name', 'unknown')
                    result = tr.get('result', {})
                    if result.get('success'):
                        output = result.get('stdout', '').strip()
                        results_text.append(f"Tool call result for {tool_name}:\n{output}")
                    else:
                        error = result.get('error', 'Unknown error').strip()
                        results_text.append(f"Tool call result for {tool_name}:\nError: {error}")
                message_text = "\n\n".join(results_text)
                self.logger.debug(f"Sending {len(tool_results)} tool results in plain text")
            else:
                raise ValueError("Must provide either tool_results or error_message")
            
            # Send using A2A SDK
            response = await send_message(
                self.white_agent_url,
                message_text,
                context_id=context_id
            )
            
            # Convert response to dict for compatibility
            from a2a.types import SendMessageSuccessResponse, Message
            from utils import parse_tags
            res_root = response.root
            if isinstance(res_root, SendMessageSuccessResponse):
                # Extract the assistant's response text
                assistant_text = self._extract_text_from_message(res_root.result) if isinstance(res_root.result, Message) else ""
                
                # Parse tool calls from <json>...</json> tags (tau-bench format)
                tool_calls_list = []
                try:
                    tags = parse_tags(assistant_text)
                    if "json" in tags:
                        json_content = tags["json"].strip()
                        self.logger.info(f"Parsing JSON content: {json_content}")
                        
                        # Try to parse the JSON, if it fails due to extra closing braces, strip them
                        try:
                            response_data = json.loads(json_content)
                        except json.JSONDecodeError as e:
                            # Check if it's an "Extra data" error and try stripping trailing braces
                            if "Extra data" in str(e):
                                # Remove trailing } characters one by one until valid JSON
                                cleaned = json_content.rstrip()
                                while cleaned.endswith('}') and len(cleaned) > 0:
                                    try:
                                        response_data = json.loads(cleaned)
                                        self.logger.info(f"Successfully parsed after removing extra braces")
                                        break
                                    except json.JSONDecodeError:
                                        cleaned = cleaned[:-1].rstrip()
                                else:
                                    raise e
                            else:
                                raise
                        
                        if isinstance(response_data, dict):
                            # Tau-bench format: {"name": "tool_name", "kwargs": {...}}
                            if "name" in response_data and "kwargs" in response_data:
                                tool_calls_list = [{
                                    "name": response_data["name"],
                                    "arguments": response_data["kwargs"]
                                }]
                            # Also support array format for backwards compatibility
                            elif "tool_calls" in response_data:
                                tool_calls_list = response_data["tool_calls"]
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Failed to parse JSON from response: {e}")
                    self.logger.info(f"Raw assistant response: {assistant_text}")
                
                result = {
                    "jsonrpc": "2.0",
                    "id": res_root.id,
                    "result": {
                        "message": res_root.result.model_dump() if isinstance(res_root.result, Message) else res_root.result,
                        "context_id": res_root.result.context_id if isinstance(res_root.result, Message) else context_id,
                        "history": [{"role": "assistant", "content": assistant_text, "tool_calls": tool_calls_list}],
                        "artifacts": []
                    }
                }
                
                self.logger.debug(f"White agent follow-up response received")
                return result
            else:
                error_msg = f"Unexpected response type from white agent: {type(res_root)}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            self.logger.error(f"Failed to send tool results to white agent: {e}")
            raise
    
    def get_execution_history(self) -> List[TaskExecutionResult]:
        """Get complete execution history."""
        return self.execution_history
    
    def cleanup_resources(self):
        """Clean up Docker resources (containers, images, etc)."""
        self.logger.info("Cleaning up Docker resources...")
        try:
            import docker
            client = docker.from_env()
            
            # Prune containers
            client.containers.prune()
            
            # Prune images (force remove unused)
            client.images.prune(filters={"dangling": False})
            
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")