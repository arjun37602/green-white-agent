#!/usr/bin/env python3
"""
Green Agent: Terminal Bench Task Runner and Evaluator
Loads Terminal Bench tasks and sends them to the white agent for evaluation.
Enhanced with sandbox management and comprehensive evaluation.
"""

import json
import logging
import os
import requests
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

from .sandbox_manager import SandboxManager, CommandResult, SandboxState
from .task_evaluator import TaskEvaluator, EvaluationResult
from .dataset_loaders.terminal_bench_loader import TerminalBenchTaskLoader, TerminalBenchTask
from .attempt_store import AttemptStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TaskExecutionResult:
    """Result of a complete task execution"""
    task_id: str
    success: bool
    execution_time: float
    commands_executed: int
    evaluation_result: EvaluationResult
    sandbox_id: str
    white_agent_response: Dict[str, Any]
    timestamp: str


class GreenAgentTerminalBench:
    """Green agent for running Terminal Bench tasks against the white agent."""
    
    # Tool definitions (MCP format)
    TOOLS = [
        {
            "name": "execute_bash_command",
            "description": "Execute a bash command in the task container and return the output",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    }
                },
                "required": ["command"]
            }
        },
        {
            "name": "stop",
            "description": "Stop the agent loop and proceed to task evaluation. Call this when you believe the task is complete.",
            "inputSchema": {
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
    ]
    
    def __init__(self, white_agent_url: str = "http://localhost:8002", 
                 sandbox_base_path: Optional[str] = None,
                 terminal_bench_dataset_path: Optional[str] = None,
                 attempt_store_path: Optional[str] = None,
                 model_id: str = "default_model",
                 trajectory_output_dir: Optional[str] = None):
        self.white_agent_url = white_agent_url
        self.model_id = model_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize trajectory output directory
        self.trajectory_output_dir = trajectory_output_dir
        if self.trajectory_output_dir:
            os.makedirs(self.trajectory_output_dir, exist_ok=True)
            self.logger.info(f"Trajectory output directory: {self.trajectory_output_dir}")
        
        # Initialize attempt store
        self.attempt_store = AttemptStore(store_path=attempt_store_path or "./evaluation_results")
        
        # Initialize Docker-based sandbox manager
        self.sandbox_manager = SandboxManager(base_path=sandbox_base_path)
        self.task_evaluator = TaskEvaluator(attempt_store=self.attempt_store)
        
        # Initialize Terminal-Bench loader if dataset path provided
        self.tb_loader = None
        if terminal_bench_dataset_path:
            try:
                self.tb_loader = TerminalBenchTaskLoader(terminal_bench_dataset_path)
                self.logger.info(f"Terminal-Bench dataset loaded from: {terminal_bench_dataset_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load Terminal-Bench dataset: {e}")
                self.tb_loader = None
        
        # Task execution tracking
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[TaskExecutionResult] = []
        
        # Current sandbox ID for tool execution
        self.current_sandbox_id: Optional[str] = None
    
    def load_terminal_bench_tasks(self, task_ids: List[str] = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        Load Terminal Bench tasks from Terminal-Bench dataset ONLY.
        
        Args:
            task_ids: Specific task IDs to load (None = load all)
            limit: Maximum number of tasks to load
            
        Returns:
            List of task dictionaries
            
        Raises:
            RuntimeError: If Terminal-Bench dataset is not available
        """
        if not self.tb_loader:
            raise RuntimeError(
                "Terminal-Bench dataset loader not initialized. "
                "Please provide a valid terminal_bench_dataset_path when creating the GreenAgent."
            )
        
        # Load from Terminal-Bench dataset
        self.logger.info("Loading tasks from Terminal-Bench dataset")
        tb_tasks = self.tb_loader.load_tasks_from_dataset(task_ids, limit)
        
        # Convert to internal format
        tasks = []
        for tb_task in tb_tasks:
            task_dict = self.tb_loader.convert_to_internal_format(tb_task)
            tasks.append(task_dict)
        
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
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }
    
    def _execute_bash_command_tool(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bash command tool."""
        if not self.current_sandbox_id:
            return {
                "success": False,
                "error": "No active sandbox"
            }
        
        command = arguments.get("command", "")
        if not command:
            return {
                "success": False,
                "error": "No command provided"
            }
        
        self.logger.info(f"Tool: execute_bash_command - {command}")
        
        try:
            result = self.sandbox_manager.execute_command(self.current_sandbox_id, command)
            return {
                "success": result.success,
                "command": result.command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "execution_time": result.execution_time
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

    
    def execute_task_with_sandbox(self, task: Dict[str, Any]) -> TaskExecutionResult:
        """
        Execute a complete task lifecycle with sandbox isolation.
        
        Args:
            task: Task specification
            
        Returns:
            TaskExecutionResult with complete execution details
        """
        task_id = task['id']
        start_time = time.time()
        
        # Initialize trajectory logging for this task
        trajectory_data = {
            "task_id": task_id,
            "task": task,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "interactions": []
        }
        
        # Add model_id to task spec for recording
        if 'metadata' not in task:
            task['metadata'] = {}
        task['metadata']['model_id'] = self.model_id
        task['metadata']['attempt_id'] = task.get('attempt_id', 0)
        
        self.logger.info(f"Starting task execution: {task_id} (model: {self.model_id})")
        
        try:
            # Create sandbox environment with Docker support
            environment_spec = task['environment']
            
            # Add Docker-related paths from metadata if available
            if 'metadata' in task:
                metadata = task['metadata']
                if 'dockerfile_path' in metadata:
                    environment_spec['dockerfile_path'] = metadata['dockerfile_path']
                if 'docker_compose_path' in metadata:
                    environment_spec['docker_compose_path'] = metadata['docker_compose_path']
                if 'task_path' in metadata:
                    environment_spec['task_path'] = metadata['task_path']
                if 'run_tests_script_path' in metadata:
                    environment_spec['run_tests_script_path'] = metadata['run_tests_script_path']
            
            sandbox_id = self.sandbox_manager.create_sandbox(task_id, environment_spec)
            self.logger.info(f"Created sandbox: {sandbox_id}")
            
            # Set current sandbox for tool execution
            self.current_sandbox_id = sandbox_id
            
            # Iteratively call white agent until task complete or max iterations reached
            command_history = []
            max_iterations = 50
            iteration = 0
            should_stop = False
            
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
                        white_agent_response, interaction = self.send_initial_task(task)
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
                            white_agent_response = self.send_tool_results(
                                task, error_message="You must send a tool command"
                            )
                        else:
                            # Send tool results from previous iteration
                            white_agent_response = self.send_tool_results(task, tool_results=previous_tool_results)
                    
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
                    
                    # Check if it's the stop tool
                    if tool_name == "stop":
                        should_stop = True
                    
                    # Record command if it was execute_bash_command
                    if tool_name == "execute_bash_command" and tool_result.get("success"):
                        command_result = CommandResult(
                            command=tool_result["command"],
                            stdout=tool_result["stdout"],
                            stderr=tool_result["stderr"],
                            returncode=tool_result["returncode"],
                            execution_time=tool_result["execution_time"],
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            success=tool_result["returncode"] == 0
                        )
                        command_history.append(command_result)
                    
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
            
            self.logger.info(f"Total commands executed: {len(command_history)}")
            self.logger.info(f"Starting task evaluation...")
            
            try:
                evaluation_result = self.task_evaluator.evaluate(
                    task_id, task, command_history, 
                    self.sandbox_manager, sandbox_id, white_agent_response
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
            
            # Save trajectory to file
            if self.trajectory_output_dir:
                self._save_trajectory(task_id, trajectory_data)
            
            # Clean up sandbox
            self.logger.info(f"Cleaning up sandbox {sandbox_id}...")
            try:
                self.sandbox_manager.destroy_sandbox(sandbox_id)
                self.logger.info(f"Sandbox cleaned up")
            except Exception as e:
                self.logger.warning(f"Error cleaning up sandbox: {e}")
            
            execution_time = time.time() - start_time
            trajectory_data["end_time"] = datetime.now(timezone.utc).isoformat()
            trajectory_data["execution_time"] = execution_time
            
            # Create execution result
            result = TaskExecutionResult(
                task_id=task_id,
                success=evaluation_result.passed,
                execution_time=execution_time,
                commands_executed=len(command_history),
                evaluation_result=evaluation_result,
                sandbox_id=sandbox_id,
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
            
            # Save trajectory even on error
            if self.trajectory_output_dir:
                self._save_trajectory(task_id, trajectory_data)
            
            # Clean up sandbox on error
            if 'sandbox_id' in locals():
                self.sandbox_manager.destroy_sandbox(sandbox_id)
            
            # Create failed result
            execution_time = time.time() - start_time
            return TaskExecutionResult(
                task_id=task_id,
                success=False,
                execution_time=execution_time,
                commands_executed=0,
                evaluation_result=None,
                sandbox_id=sandbox_id if 'sandbox_id' in locals() else None,
                white_agent_response={"error": str(e)},
                timestamp=datetime.utcnow().isoformat()
            )
    
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
    
    def _save_trajectory(self, task_id: str, trajectory_data: Dict[str, Any]) -> None:
        """
        Save trajectory data to a JSON file.
        
        Args:
            task_id: The task identifier
            trajectory_data: Complete trajectory data including interactions
        """
        try:
            # Create task-specific directory
            task_dir = Path(self.trajectory_output_dir) / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for filename
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            trajectory_file = task_dir / f"trajectory_{timestamp}.json"
            
            # Write trajectory data to file
            with open(trajectory_file, 'w') as f:
                json.dump(trajectory_data, f, indent=2, default=str)
            
            self.logger.info(f"Trajectory saved to: {trajectory_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save trajectory: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    
    def send_initial_task(self, task: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Send initial task with tool definitions to the white agent.
        
        Args:
            task: Task specification
            iteration: Current iteration number
            
        Returns:
            Tuple of (white agent response, interaction dict with logged request)
        """
        try:
            # Construct interaction
            interaction = {
                "iteration": 1,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Build initial task message
            task_message = f"""Terminal Bench Task: {task.get('id')}

Description: {task.get('description', 'No description')}

Instructions:
{task.get('instruction', '')}

Environment:
{json.dumps(task.get('environment', {}), indent=2)}

Complete this task using the available tools."""
            
            # Create A2A message with tool definitions
            rpc_request = {
                "jsonrpc": "2.0",
                "id": f"task-{task.get('id', 'unknown')}-init",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [
                            {
                                "kind": "text",
                                "text": task_message
                            }
                        ]
                    },
                    "tools": self.TOOLS  # Include tool definitions
                }
            }
            
            # Log the request to interaction
            interaction["user"] = {
                "type": "initial_task",
                "rpc_request": rpc_request
            }
            
            # Debug: log the request
            self.logger.info(f"=== SENDING TO WHITE AGENT (INITIAL TASK) ===")
            self.logger.info(f"URL: {self.white_agent_url}")
            self.logger.info(f"Tools provided: {len(self.TOOLS)} tools")
            self.logger.debug(f"Tools: {json.dumps(self.TOOLS, indent=2)}")
            self.logger.info(f"Message preview: {task_message[:200]}...")
            
            # Send to white agent
            response = requests.post(self.white_agent_url, json=rpc_request, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                self.logger.debug(f"White agent initial response: {json.dumps(result, indent=2)}")
                return result, interaction
            else:
                self.logger.error(f"White agent returned status {response.status_code}: {response.text}")
                raise Exception(f"White agent error: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to send initial task to white agent: {e}")
            raise
    
    def send_tool_results(self, task: Dict[str, Any], 
                         tool_results: Optional[List[Dict[str, Any]]] = None,
                         error_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Send tool results or error message to the white agent.
        
        Args:
            task: Task specification
            tool_results: Tool execution results from previous iteration
            error_message: Error message if white agent didn't send tool calls
            
        Returns:
            White agent response
        """
        try:
            # Build message based on input
            if error_message:
                # Send error message as user
                rpc_request = {
                    "jsonrpc": "2.0",
                    "id": f"task-{task.get('id', 'unknown')}-error",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [{"kind": "text", "text": error_message}]
                        },
                        "tools": self.TOOLS
                    }
                }
            elif tool_results:
                # Send tool results in OpenAI format
                # Convert to OpenAI tool message format
                tool_messages = []
                for tr in tool_results:
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tr['tool_call_id'],
                        "content": json.dumps(tr['result'])
                    })
                
                rpc_request = {
                    "jsonrpc": "2.0",
                    "id": f"task-{task.get('id', 'unknown')}-tool-results",
                    "method": "message/send",
                    "params": {
                        "tool_messages": tool_messages,  # Send as structured data
                        "tools": self.TOOLS
                    }
                }
            else:
                raise ValueError("Must provide either tool_results or error_message")
            
            # Send to white agent
            response = requests.post(self.white_agent_url, json=rpc_request, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                self.logger.debug(f"White agent follow-up response: {json.dumps(result, indent=2)}")
                return result
            else:
                self.logger.error(f"White agent returned status {response.status_code}: {response.text}")
                raise Exception(f"White agent error: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to send tool results to white agent: {e}")
            raise
    
    def run_evaluation(self, task_ids: List[str] = None, limit: int = 5):
        """
        Run full evaluation loop with sandbox-based execution and Terminal-Bench pytest evaluation.
        
        Args:
            task_ids: Specific task IDs to run
            limit: Maximum number of tasks
        """
        self.logger.info("Starting Terminal Bench evaluation")
        
        # Load tasks from Terminal-Bench dataset
        tasks = self.load_terminal_bench_tasks(task_ids, limit)
        self.logger.info(f"Loaded {len(tasks)} tasks")
        
        results = []
        
        for i, task in enumerate(tasks, 1):
            task_id = task.get('id', f'task_{i}')
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📝 Task {i}/{len(tasks)}: {task_id}")
            self.logger.info(f"{'='*60}")
            
            # Execute task with sandbox isolation and Terminal-Bench pytest evaluation
            execution_result = self.execute_task_with_sandbox(task)
            
            result = {
                "task_id": task_id,
                "success": execution_result.success,
                "execution_time": execution_result.execution_time,
                "commands_executed": execution_result.commands_executed,
                "evaluation_score": execution_result.evaluation_result.score if execution_result.evaluation_result else 0.0,
                "sandbox_id": execution_result.sandbox_id,
                "white_agent_response": execution_result.white_agent_response,
                "evaluation_details": execution_result.evaluation_result.details if execution_result.evaluation_result else {},
                "timestamp": execution_result.timestamp
            }
            
            results.append(result)
            
            status = "PASSED" if result["success"] else "FAILED"
            score = result["evaluation_score"]
            self.logger.info(f"{status}: {task_id} (Score: {score:.2f})")
        
        # Summary
        passed = sum(1 for r in results if r["success"])
        avg_score = sum(r["evaluation_score"] for r in results) / len(results) if results else 0.0
        total_time = sum(r["execution_time"] for r in results)
        total_commands = sum(r["commands_executed"] for r in results)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 EVALUATION SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total Tasks: {len(results)}")
        self.logger.info(f"Passed: {passed}")
        self.logger.info(f"Failed: {len(results) - passed}")
        self.logger.info(f"Success Rate: {passed/len(results)*100:.1f}%")
        self.logger.info(f"Average Score: {avg_score:.2f}")
        self.logger.info(f"Total Execution Time: {total_time:.2f}s")
        self.logger.info(f"Total Commands Executed: {total_commands}")
        self.logger.info(f"Sandbox Mode: Always Enabled")
        
        return results
    
    def get_execution_history(self) -> List[TaskExecutionResult]:
        """Get complete execution history."""
        return self.execution_history
    
    def cleanup_resources(self):
        """Clean up all sandbox resources."""
        self.logger.info("Cleaning up sandbox resources...")
        self.sandbox_manager.cleanup_all()
        self.logger.info("Cleanup completed")