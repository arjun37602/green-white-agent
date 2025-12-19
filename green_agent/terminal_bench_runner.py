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
import asyncio

# Suppress verbose a2a logging
logging.getLogger("a2a.client.card_resolver").setLevel(logging.WARNING)
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.terminal.tmux_session import TmuxSession
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
    
    # Prompt template for agent commands
    # Uses terminus-json-plain format
    PROMPT_TEMPLATE = """You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {{
      "keystrokes": "ls -la\n",
      "duration": 0.1
    }},
    {{
      "keystrokes": "cd project\n",
      "duration": 0.1
    }}
  ],
  "task_complete": false
}}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete (defaults to false if not present)

Command object structure:
- "keystrokes": String containing the exact keystrokes to send to the terminal (required)
- "duration": Number of seconds to wait for the command to complete before the next command will be executed (defaults to 1.0 if not present)

IMPORTANT: The text inside "keystrokes" will be used completely verbatim as keystrokes. Write commands exactly as you want them sent to the terminal:
- Most bash commands should end with a newline (\n) to cause them to execute
- For special key sequences, use tmux-style escape sequences:
  - C-c for Ctrl+C
  - C-d for Ctrl+D
  - Escape for ESC key
  - Up/Down/Left/Right for arrow keys

The "duration" attribute specifies the number of seconds to wait for the command to complete (default: 1.0) before the next command will be executed. On immediate tasks (e.g., cd, ls, echo, cat) set a duration of 0.1 seconds. On commands (e.g., gcc, find, rustc) set a duration of 1.0 seconds. On slow commands (e.g., make, python3 [long running script], wget [file]) set an appropriate duration as you determine necessary.

It is better to set a smaller duration than a longer duration. It is always possible to wait again if the prior output has not finished, by running {{"keystrokes": "", "duration": 10.0}} on subsequent requests to wait longer. Never wait longer than 60 seconds; prefer to poll to see intermediate result status.

Important notes:
- Each command's keystrokes are sent exactly as written to the terminal
- Do not include extra whitespace before or after the keystrokes unless it's part of the intended command
- The JSON must be valid - use proper escaping for quotes and special characters within strings
- Commands array can be empty if you want to wait without taking action
"""
    
    def __init__(self, white_agent_url: str = "http://localhost:8002", 
                 sandbox_base_path: Optional[str] = None,
                 terminal_bench_dataset_path: Optional[str] = None,
                 model_id: str = "default_model",
                 results_dir: str = "./results",
                 max_parallel_tasks: int = 5,
                 max_attempts: int = 1):
        self.white_agent_url = white_agent_url
        self.model_id = model_id
        self.max_attempts = max_attempts
        self.logger = logging.getLogger(__name__)
        
        # Initialize results store for caching
        self.results_store = ResultsStore(results_dir=results_dir)
        
        # Semaphore to limit parallel task execution
        self.task_semaphore = asyncio.Semaphore(max_parallel_tasks)
        
        # Initialize task evaluator
        self.task_evaluator = TaskEvaluator()
        
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
    
    def load_terminal_bench_tasks(self, task_ids: List[str] = None) -> List[Tuple[Task, TaskPaths]]:
        """
        Load Terminal Bench tasks from Terminal-Bench dataset.
        
        Args:
            task_ids: Specific task IDs to load (None = load all)
            
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
        
        self.logger.info(f"Loaded {len(tasks)} tasks from Terminal-Bench dataset")
        return tasks
    
    def _limit_output_length(self, output: str, max_bytes: int = 10000) -> str:
        """
        Limit output to specified byte length, keeping first and last portions.
        Prevents token overflow by truncating middle of large outputs.
        
        Args:
            output: The terminal output to potentially truncate
            max_bytes: Maximum allowed bytes (default 10000)
            
        Returns:
            str: Original output if under limit, or truncated with middle omitted
        """
        if len(output.encode("utf-8")) <= max_bytes:
            return output
        
        # Calculate portions (half each for first and last)
        portion_size = max_bytes // 2
        
        # Convert to bytes for accurate splitting
        output_bytes = output.encode("utf-8")
        
        # Get first portion
        first_portion = output_bytes[:portion_size].decode("utf-8", errors="ignore")
        
        # Get last portion
        last_portion = output_bytes[-portion_size:].decode("utf-8", errors="ignore")
        
        # Calculate omitted bytes
        omitted_bytes = (
            len(output_bytes)
            - len(first_portion.encode("utf-8"))
            - len(last_portion.encode("utf-8"))
        )
        
        return (
            f"{first_portion}\n[... output limited to {max_bytes} bytes; "
            f"{omitted_bytes} bytes omitted ...]\n{last_portion}"
        )
    
    def execute_commands_batch(self, commands: List[Dict[str, Any]], session: TmuxSession) -> Dict[str, Any]:
        """
        Execute a batch of commands in the tmux session.
        
        Process:
        1. Execute all commands in sequence with their durations
        2. Capture incremental output once at the end (accumulates all command output)
        
        Args:
            commands: List of command dicts with 'keystrokes' and 'duration'
            session: TmuxSession for execution
            
        Returns:
            Result dict with terminal output from all commands
        """
        if not session:
            return {
                "success": False,
                "error": "No active tmux session",
                "output": ""
            }
        
        try:
            # Execute all commands in sequence
            for cmd in commands:
                keystrokes = cmd.get("keystrokes", "")
                duration = cmd.get("duration", 1.0)
                
                # Cap duration at 60 seconds to prevent excessive waits
                actual_duration = min(duration, 60.0)
                
                # Process escape sequences: convert literal \n to actual newlines
                try:
                    # Use unicode_escape to convert literal escape sequences
                    keystrokes_processed = keystrokes.encode('utf-8').decode('unicode_escape')
                except (UnicodeDecodeError, AttributeError):
                    # If decoding fails, use original keystrokes
                    keystrokes_processed = keystrokes
                
                self.logger.info(f"Executing keystrokes: {repr(keystrokes_processed)[:100]} (duration: {actual_duration}s)")
                
                # Send keystrokes to tmux (non-blocking execution)
                session.send_keys(
                    keystrokes_processed,
                    block=False,
                    min_timeout_sec=actual_duration
                )
            
            # Capture incremental output once after all commands complete
            # This accumulates output from all commands in the batch
            terminal_output = session.get_incremental_output()
            
            # Limit output length to prevent token overflow
            terminal_output = self._limit_output_length(terminal_output)
            
            return {
                "success": True,
                "output": terminal_output,
                "num_commands": len(commands)
            }
        except TimeoutError as e:
            # If any command times out, still return the output so far
            self.logger.warning(f"Command timeout: {e}")
            try:
                terminal_output = session.get_incremental_output()
            except:
                terminal_output = session.capture_pane(capture_entire=False)
            
            # Limit output length
            terminal_output = self._limit_output_length(terminal_output)
            
            return {
                "success": False,
                "error": f"Timeout: {str(e)}",
                "output": terminal_output
            }
        except Exception as e:
            self.logger.error(f"Error executing commands: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": ""
            }
    

    
    async def execute_multiple_tasks_parallel(self, task_tuples: List[Tuple[Task, TaskPaths]], 
                                              output_directory: Path) -> List[TaskExecutionResult]:
        """
        Execute multiple tasks in parallel with semaphore-based rate limiting.
        Checks cache and skips tasks that have reached max_attempts.
        
        Args:
            task_tuples: List of (Task, TaskPaths) tuples
            output_directory: Directory for results
            
        Returns:
            List of TaskExecutionResult objects
        """
        # Check cache and filter out tasks that reached max_attempts
        completed_task_ids = self.results_store.load_completed_tasks(self.model_id, self.max_attempts)
        self.logger.info(f"Found {len(completed_task_ids)} fully attempted tasks (>= {self.max_attempts} attempts) for model {self.model_id}")
        
        tasks_to_run = []
        cached_results = []
        
        for task, task_paths in task_tuples:
            task_id = task_paths.input_path.name
            if task_id in completed_task_ids:
                self.logger.info(f"Task {task_id} already has {self.max_attempts}+ attempts, loading from cache")
                # Load cached result (use most recent)
                all_results = self.results_store.load_all_results(self.model_id)
                task_results = [r for r in all_results if r.task_id == task_id]
                cached_result = task_results[-1] if task_results else None
                
                if cached_result:
                    # Convert TaskResult to TaskExecutionResult
                    cached_results.append(TaskExecutionResult(
                        task_id=cached_result.task_id,
                        success=cached_result.success,
                        execution_time=cached_result.execution_time,
                        evaluation_result=EvaluationResult(
                            task_id=cached_result.task_id,
                            passed=cached_result.success,
                            accuracy=cached_result.accuracy,
                            passed_test_cases=cached_result.passed_test_cases,
                            total_test_cases=cached_result.total_test_cases,
                            details=cached_result.metadata.get("evaluation_details", {}) if cached_result.metadata else {},
                            timestamp=cached_result.timestamp
                        ),
                        sandbox_id=f"cached-{cached_result.model_id}",
                        white_agent_response={"cached": True, "model_id": cached_result.model_id, "attempts": len(task_results)},
                        timestamp=cached_result.timestamp
                    ))
                else:
                    tasks_to_run.append((task, task_paths))
            else:
                tasks_to_run.append((task, task_paths))
        
        self.logger.info(f"Running {len(tasks_to_run)} tasks (skipped {len(cached_results)} cached with {self.max_attempts}+ attempts)")
        
        # Execute tasks in parallel with semaphore
        async def execute_with_semaphore(task: Task, task_paths: TaskPaths):
            async with self.task_semaphore:
                return await self.execute_task_with_sandbox(task, task_paths, output_directory)
        
        # Run all tasks in parallel
        new_results = await asyncio.gather(
            *[execute_with_semaphore(task, task_paths) for task, task_paths in tasks_to_run],
            return_exceptions=True
        )
        
        # Filter out exceptions and log them
        valid_results = []
        for i, result in enumerate(new_results):
            if isinstance(result, Exception):
                task_id = tasks_to_run[i][1].input_path.name if i < len(tasks_to_run) else "unknown"
                self.logger.error(f"Task {task_id} failed with exception: {result}")
            else:
                valid_results.append(result)
        
        # Combine cached and new results
        all_results = cached_results + valid_results
        return all_results
    
    async def execute_task_with_sandbox(self, task: Task, task_paths: TaskPaths, output_directory: Path) -> TaskExecutionResult:
        """
        Execute a complete task lifecycle with sandbox isolation.
        
        Args:
            task: Terminal Bench Task object
            task_paths: Terminal Bench TaskPaths object
            output_directory: Directory for results
            
        Returns:
            TaskExecutionResult with complete execution details
        """
        task_id = task_paths.input_path.name
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
        
        # Track token usage
        total_tokens = 0
        
        self.logger.info(f"Starting task execution: {task_id} (model: {self.model_id})")
        
        # Container names - add random suffix for parallel execution
        # Sanitize task_id for Docker: replace dots and invalid chars with underscores
        # Docker Compose project names must be lowercase alphanumeric, hyphens, and underscores only
        sanitized_task_id = task_id.replace('-', '_').replace('.', '_').replace('/', '_').lower()
        container_name = f"tbench_{sanitized_task_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}_client"
        image_name = f"tbench_{sanitized_task_id}_image"
        
        sessions_logs_path = output_directory / "sessions"
        agent_logs_path = output_directory / "agent-logs"
        sessions_logs_path.mkdir(parents=True, exist_ok=True)
        agent_logs_path.mkdir(parents=True, exist_ok=True)
        
        # Create Docker Compose Manager with log paths to satisfy environment variables
        container_manager = DockerComposeManager(
            client_container_name=container_name,
            client_image_name=image_name,
            docker_compose_path=task_paths.docker_compose_path,
            docker_image_name_prefix=f"tbench_{sanitized_task_id}",
            no_rebuild=False,
            cleanup=True,
            sessions_logs_path=sessions_logs_path,
            agent_logs_path=agent_logs_path
        )
        
        # Each task has its own container and tmux session (for parallel execution)
        current_container = None
        tmux_session = None
        
        try:
            # Start container
            self.logger.info(f"Starting Docker container for task {task_id}")
            container = container_manager.start()
            current_container = container
            self.logger.info(f"Container started: {container.name}")
            
            # Create and start tmux session in the container
            self.logger.info(f"Creating tmux session for task {task_id}")
            tmux_session = TmuxSession(
                session_name=f"agent_{sanitized_task_id}",
                container=container,
                commands_path=None,  # Don't log commands to file
                disable_recording=True,  # Disable asciinema recording for performance
                user=""  # Run as default user
            )
            tmux_session.start()
            self.logger.info(f"Tmux session started")
            
            # Get task-specific timeout (default to 15 minutes if not specified)
            agent_timeout_sec = task.max_agent_timeout_sec if hasattr(task, 'max_agent_timeout_sec') else 900.0
            self.logger.info(f"[{task_id}] Agent timeout: {agent_timeout_sec}s")
            
            # Iteratively call white agent until task complete or max iterations reached
            max_iterations = 25
            iteration = 0
            should_stop = False
            context_id = f"ctx-{task_id}-{uuid.uuid4().hex[:8]}"  # Unique context ID for this task
            
            self.logger.info(f"Starting agent loop (max {max_iterations} iterations, timeout {agent_timeout_sec}s) with context_id={context_id}")
            
            # Track terminal output from previous iteration
            previous_terminal_output = None
            white_agent_response = None  # Initialize to avoid UnboundLocalError if loop breaks early
            agent_timed_out = False
            
            # Wrap agent loop in timeout
            try:
                agent_loop_start = time.time()
                while not should_stop and iteration < max_iterations and (time.time() - agent_loop_start) < agent_timeout_sec:
                    iteration += 1
                    self.logger.info(f"[{task_id}] Iteration {iteration}/{max_iterations}")
                    
                    # Initialize interaction for this iteration
                    interaction = {
                        "iteration": iteration,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Call white agent
                    try:
                        if iteration == 1:
                            # First iteration: send initial task with prompt
                            self.logger.info(f"[{task_id}] Sending initial task")
                            white_agent_response, initial_interaction = await self.send_initial_task(task, task_paths, context_id, tmux_session)
                            # Merge initial interaction data
                            interaction.update(initial_interaction)
                            trajectory_data["interactions"].append(interaction) #add initial task
                        else:
                            # Subsequent iterations: send terminal output
                            if previous_terminal_output is None:
                                # No terminal output from previous iteration
                                self.logger.warning(f"[{task_id}] No terminal output from previous iteration")
                                white_agent_response = await self.send_terminal_output(
                                    task, task_paths, error_message="No commands were executed in previous iteration", context_id=context_id
                                )
                            else:
                                # Send terminal output from previous iteration
                                white_agent_response = await self.send_terminal_output(
                                    task, task_paths, terminal_output=previous_terminal_output, context_id=context_id
                                )
                        
                        # Extract token usage from metadata
                        if "result" in white_agent_response and "message" in white_agent_response["result"]:
                            message = white_agent_response["result"]["message"]
                            if isinstance(message, dict) and "metadata" in message:
                                metadata = message["metadata"]
                                if "cumulative_tokens" in metadata:
                                    total_tokens = metadata["cumulative_tokens"]
                                    self.logger.debug(f"[{task_id}] Token usage: {total_tokens} cumulative")
                        
                        # Log white agent response (only if not iteration 1, since already logged above)
                        if iteration > 1:
                            interaction["assistant"] = white_agent_response
                            trajectory_data["interactions"].append(interaction) #add assistant response
                        
                    except Exception as e:
                        self.logger.error(f"[{task_id}] Error communicating with white agent: {e}")
                        self.logger.warning(f"[{task_id}] Breaking loop due to white agent error")
                        # interaction is already defined at the start of the loop
                        interaction["error"] = str(e)
                        trajectory_data["interactions"].append(interaction)
                        break
                    
                    # Extract commands from white agent response
                    commands, is_task_complete = self._extract_commands(white_agent_response)
                    
                    # Log extracted commands
                    self.logger.info(f"[{task_id}] Extracted {len(commands)} commands, task_complete={is_task_complete}")
                    for i, cmd in enumerate(commands):
                        self.logger.info(f"[{task_id}] Command {i+1}: keystrokes={repr(cmd.get('keystrokes', ''))[:100]}, duration={cmd.get('duration', 1.0)}s")
                    
                    # Check for no commands
                    if not commands:
                        # If task is complete and no commands, that's fine - exit cleanly
                        if is_task_complete:
                            should_stop = True
                            self.logger.info(f"[{task_id}] Agent marked task as complete with no final commands - terminating after iteration {iteration}")
                            break
                        else:
                            # No commands and task not complete - this is an error
                            self.logger.warning(f"[{task_id}] No commands in response - will send error next iteration")
                            previous_terminal_output = None
                            continue
                    
                    # Execute all commands as a batch
                    self.logger.info(f"[{task_id}] Executing batch of {len(commands)} commands")
                    batch_result = self.execute_commands_batch(commands, tmux_session)
                    
                    # Log the batch execution result
                    if batch_result.get('success'):
                        terminal_output = batch_result.get('output', '').strip()
                        if terminal_output:
                            self.logger.info(f"[{task_id}] Terminal output:\n{terminal_output}")
                        else:
                            self.logger.info(f"[{task_id}] Commands executed (no output)")
                    else:
                        error = batch_result.get('error', 'Unknown error')
                        self.logger.error(f"[{task_id}] Command batch failed: {error}")
                        # Still use the partial output if available
                        terminal_output = batch_result.get('output', '')
                    
                    # Store command execution info
                    interaction["command_executions"] = {
                        "commands": commands,
                        "result": batch_result
                    }
                    
                    # Add interaction to trajectory
                    trajectory_data["interactions"].append(interaction)
                    
                    # Store terminal output for next iteration
                    previous_terminal_output = batch_result.get('output', '')
                    
                    # Check if task is complete AFTER executing commands
                    if is_task_complete:
                        should_stop = True
                        self.logger.info(f"[{task_id}] Agent marked task as complete - terminating after iteration {iteration}")
                        break
            
            except Exception as e:
                self.logger.error(f"[{task_id}] Error in agent loop: {e}")
                # Continue to evaluation
            
            # Check why loop ended
            elapsed_time = time.time() - agent_loop_start
            if elapsed_time >= agent_timeout_sec:
                self.logger.warning(f"[{task_id}] Agent timed out after {elapsed_time:.2f}s (limit: {agent_timeout_sec}s)")
                agent_timed_out = True
                trajectory_data["agent_timeout"] = True
            elif iteration >= max_iterations:
                self.logger.warning(f"[{task_id}] Reached max iterations ({max_iterations}) without completion")
            
            self.logger.info(f"[{task_id}] Starting task evaluation...")
            
            try:
                # Only evaluate if we have a white agent response (loop didn't break immediately)
                if white_agent_response is not None:
                    evaluation_result = self.task_evaluator.evaluate(
                        task_id, task_paths, current_container, white_agent_response, task=task, session=tmux_session
                    )
                else:
                    self.logger.warning(f"[{task_id}] No white agent response received, skipping evaluation")
                    evaluation_result = None
                self.logger.info(f"[{task_id}] Evaluation complete: {evaluation_result.passed if evaluation_result else 'N/A'}")
                
                # Add evaluation result to trajectory
                if evaluation_result:
                    trajectory_data["evaluation"] = {
                        "passed": evaluation_result.passed,
                        "accuracy": evaluation_result.accuracy,
                        "passed_test_cases": evaluation_result.passed_test_cases,
                        "total_test_cases": evaluation_result.total_test_cases,
                        "details": evaluation_result.details
                    }
            except Exception as e:
                self.logger.error(f"[{task_id}] Evaluation failed: {e}")
                import traceback
                self.logger.debug(traceback.format_exc())
                evaluation_result = None
                trajectory_data["evaluation"] = {
                    "error": str(e)
                }
            
            # Clean up tmux session and container
            self.logger.info(f"[{task_id}] Cleaning up tmux session and container...")
            try:
                if tmux_session:
                    try:
                        tmux_session.stop()
                        self.logger.info(f"[{task_id}] Tmux session stopped")
                    except Exception as e:
                        self.logger.warning(f"[{task_id}] Error stopping tmux session: {e}")
                
                container_manager.stop()
                self.logger.info(f"[{task_id}] Container cleaned up")
            except Exception as e:
                self.logger.warning(f"[{task_id}] Error cleaning up container: {e}")
            
            execution_time = time.time() - start_time
            trajectory_data["end_time"] = datetime.now(timezone.utc).isoformat()
            trajectory_data["execution_time"] = execution_time
            
            # Create execution result
            result = TaskExecutionResult(
                task_id=task_id,
                success=evaluation_result.passed if evaluation_result else False,
                execution_time=execution_time,
                evaluation_result=evaluation_result,
                sandbox_id=current_container.name if current_container else "unknown",
                white_agent_response=white_agent_response,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            # Store in history
            self.execution_history.append(result)
            
            # Save to ResultsStore - extract metrics from evaluation result with defaults
            success = getattr(evaluation_result, 'passed', False) if evaluation_result else False
            passed_test_cases = getattr(evaluation_result, 'passed_test_cases', 0) if evaluation_result else 0
            total_test_cases = getattr(evaluation_result, 'total_test_cases', 1) if evaluation_result else 1
            accuracy = getattr(evaluation_result, 'accuracy', 0.0) if evaluation_result else 0.0
            
            # Get current attempt count for this task
            current_attempt = self.results_store.get_attempt_count(self.model_id, task_id) + 1
            
            task_result = TaskResult(
                task_id=task_id,
                model_id=self.model_id,
                attempt_id=current_attempt,
                success=success,
                num_tokens=total_tokens,
                num_turns=iteration,  # Use iteration count (same as num_turns)
                passed_test_cases=passed_test_cases,
                total_test_cases=total_test_cases,
                accuracy=accuracy,
                timestamp=datetime.now(timezone.utc).isoformat(),
                execution_time=execution_time,
                message_history=trajectory_data["interactions"],
                metadata={
                    "evaluation_details": evaluation_result.details if evaluation_result else {},
                    "task_category": task.category,
                    "task_difficulty": task.difficulty.value
                }
            )
            
            self.results_store.save_result(self.model_id, task_result)
            self.logger.info(f"[{task_id}] Result saved to {self.results_store.get_model_file(self.model_id)}")
            
            self.logger.info(f"[{task_id}] Task completed in {execution_time:.2f}s - {'PASSED' if result.success else 'FAILED'} - {total_tokens} tokens")
            
            return result
            
        except Exception as e:
            self.logger.error(f"[{task_id}] Task failed: {e}")
            
            trajectory_data["error"] = str(e)
            trajectory_data["end_time"] = datetime.now(timezone.utc).isoformat()
            
            # Clean up tmux session and container on error
            if 'tmux_session' in locals() and tmux_session:
                try:
                    tmux_session.stop()
                except:
                    pass
            
            if 'container_manager' in locals():
                try:
                    container_manager.stop()
                except:
                    pass
            
            # Create failed result
            execution_time = time.time() - start_time
            failed_result = TaskExecutionResult(
                task_id=task_id,
                success=False,
                execution_time=execution_time,
                evaluation_result=None,
                sandbox_id=current_container.name if current_container else None,
                white_agent_response={"error": str(e)},
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            # Save failed result to ResultsStore
            # Note: iteration may not be defined if error happened before loop
            num_turns = locals().get('iteration', 0)
            task_result = TaskResult(
                task_id=task_id,
                model_id=self.model_id,
                attempt_id=1,
                success=False,
                num_tokens=total_tokens,
                num_turns=num_turns,
                passed_test_cases=0,
                total_test_cases=1,
                accuracy=0.0,
                timestamp=datetime.now(timezone.utc).isoformat(),
                execution_time=execution_time,
                message_history=trajectory_data.get("interactions", []),
                metadata={
                    "error": str(e)
                }
            )
            self.results_store.save_result(self.model_id, task_result)
            
            return failed_result
    
    def _extract_commands(self, white_agent_response: Dict[str, Any]) -> tuple[List[Dict[str, Any]], bool]:
        """
        Extract commands from white agent response.
        
        Parses JSON response to extract:
        - commands: List of {keystrokes, duration} dicts
        - task_complete: Boolean flag
        
        Args:
            white_agent_response: Response from white agent
            
        Returns:
            Tuple of (commands_list, is_task_complete)
            commands_list: [{"keystrokes": "ls\n", "duration": 0.1}, ...]
            is_task_complete: True if task is marked complete
        """
        commands = []
        is_task_complete = False
        
        try:
            # Extract assistant's response text
            assistant_text = ""
            if "result" in white_agent_response:
                result = white_agent_response["result"]
                
                # Check history for assistant message
                if "history" in result:
                    for message in result["history"]:
                        if isinstance(message, dict) and message.get("role") == "assistant":
                            assistant_text = message.get("content", "")
                            break
                
                # Also check message directly
                if not assistant_text and "message" in result:
                    message = result["message"]
                    if isinstance(message, dict) and "parts" in message:
                        from a2a.utils import get_text_parts
                        text_parts = get_text_parts(message["parts"])
                        assistant_text = "\n".join(text_parts) if text_parts else ""
            
            if not assistant_text:
                self.logger.warning("No assistant text found in response")
                return [], False
            
            # Parse JSON from response (look for {...} pattern)
            # Try to extract JSON block
            json_match = None
            try:
                # Try parsing the whole thing as JSON first
                response_data = json.loads(assistant_text)
                json_match = response_data
            except json.JSONDecodeError:
                # Look for JSON in text
                import re
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                matches = re.findall(json_pattern, assistant_text, re.DOTALL)
                for match in matches:
                    try:
                        response_data = json.loads(match)
                        if "commands" in response_data:
                            json_match = response_data
                            break
                    except json.JSONDecodeError:
                        continue
            
            if not json_match:
                self.logger.warning(f"No valid JSON found in response: {assistant_text[:200]}")
                return [], False
            
            # Extract commands array
            if "commands" in json_match and isinstance(json_match["commands"], list):
                for cmd_data in json_match["commands"]:
                    if not isinstance(cmd_data, dict):
                        continue
                    
                    keystrokes = cmd_data.get("keystrokes", "")
                    duration = cmd_data.get("duration", 1.0)
                    
                    # Validate and sanitize
                    if not isinstance(keystrokes, str):
                        self.logger.warning(f"Invalid keystrokes type: {type(keystrokes)}")
                        continue
                    
                    try:
                        duration = float(duration)
                    except (ValueError, TypeError):
                        self.logger.warning(f"Invalid duration value: {duration}, using 1.0")
                        duration = 1.0
                    
                    commands.append({
                        "keystrokes": keystrokes,
                        "duration": duration
                    })
            
            # Extract task_complete flag
            is_task_complete = json_match.get("task_complete", False)
            
            if commands:
                self.logger.info(f"Extracted {len(commands)} commands, task_complete={is_task_complete}")
            
        except Exception as e:
            self.logger.error(f"Error extracting commands: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        return commands, is_task_complete
    
    
    async def send_initial_task(self, task: Task, task_paths: TaskPaths, context_id: str, session: TmuxSession) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Send initial task with tool definitions to the white agent using A2A SDK.
        
        Args:
            task: Terminal Bench Task object
            task_paths: Terminal Bench TaskPaths object
            context_id: Context ID for this conversation
            session: TmuxSession (unused but kept for signature consistency)
            
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
            task_id = task_paths.input_path.name
            
            # Get initial terminal state
            try:
                terminal_state = session.get_incremental_output()
            except:
                terminal_state = session.capture_pane(capture_entire=False)
            
            # Use prompt template
            task_message = self.PROMPT_TEMPLATE + f"""
Task Description:
{task.instruction}

Current terminal state:
{terminal_state}"""
            
            # Log the request
            self.logger.info(f"=== SENDING TO WHITE AGENT (INITIAL TASK) ===")
            self.logger.info(f"URL: {self.white_agent_url}")
            self.logger.info(f"Task: {task_id}")
            self.logger.info(f"Message preview: {task_message[:300]}...")
            
            # Send using A2A SDK (no tools parameter - all in text)
            response = await send_message(
                self.white_agent_url,
                task_message,
                context_id=context_id
            )
            
            # Convert response to dict for compatibility with existing code
            from a2a.types import SendMessageSuccessResponse, Message
            res_root = response.root
            if isinstance(res_root, SendMessageSuccessResponse):
                # Extract the assistant's response text
                assistant_text = self._extract_text_from_message(res_root.result) if isinstance(res_root.result, Message) else ""
                
                # Log full LLM response
                self.logger.info(f"=== WHITE AGENT RESPONSE (INITIAL) ===")
                self.logger.info(f"Full response:\n{assistant_text}")
                self.logger.info(f"=== END WHITE AGENT RESPONSE ===")
                
                # Build the result in a format compatible with _extract_commands
                result = {
                    "jsonrpc": "2.0",
                    "id": res_root.id,
                    "result": {
                        "message": res_root.result.model_dump() if isinstance(res_root.result, Message) else res_root.result,
                        "context_id": res_root.result.context_id if isinstance(res_root.result, Message) else None,
                        "history": [{"role": "assistant", "content": assistant_text}],
                        "artifacts": []
                    }
                }
                
                # Log the request to interaction
                interaction["user"] = {
                    "type": "initial_task",
                    "message": task_message,
                    "format": "terminus-json-plain"
                }
                
                self.logger.debug(f"White agent initial response received")
                return result, interaction
            else:
                # Handle error response from white agent
                error_msg = f"Unexpected response type from white agent: {type(res_root)}"
                if hasattr(res_root, 'error'):
                    error_details = res_root.error
                    self.logger.error(f"{error_msg}")
                    self.logger.error(f"Error details from white agent: {error_details}")
                    raise Exception(f"White agent error: {error_details}")
                else:
                    self.logger.error(f"{error_msg}")
                    self.logger.error(f"Response object: {res_root}")
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
    
    async def send_terminal_output(self, task: Task, task_paths: TaskPaths,
                         terminal_output: Optional[str] = None,
                         error_message: Optional[str] = None,
                         context_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Send terminal output to the white agent.
        
        After executing commands, the agent receives the terminal output text
        to understand what happened and decide the next steps.
        
        Args:
            task: Terminal Bench Task object
            task_paths: Terminal Bench TaskPaths object
            terminal_output: Terminal output text from previous iteration
            error_message: Error message if white agent didn't send commands
            context_id: Context ID to maintain conversation
            
        Returns:
            White agent response
        """
        try:
            # Build message with terminal output or error
            if error_message:
                message_text = f"Error: {error_message}"
            elif terminal_output is not None:
                # Send the terminal output directly
                message_text = terminal_output
                self.logger.debug(f"Sending terminal output ({len(terminal_output)} chars)")
            else:
                raise ValueError("Must provide either terminal_output or error_message")
            
            # Send using A2A SDK
            response = await send_message(
                self.white_agent_url,
                message_text,
                context_id=context_id
            )
            
            # Convert response to dict for compatibility with _extract_commands
            from a2a.types import SendMessageSuccessResponse, Message
            res_root = response.root
            if isinstance(res_root, SendMessageSuccessResponse):
                # Extract the assistant's response text
                assistant_text = self._extract_text_from_message(res_root.result) if isinstance(res_root.result, Message) else ""
                
                # Log full LLM response
                self.logger.info(f"=== WHITE AGENT RESPONSE (ITERATION) ===")
                self.logger.info(f"Full response:\n{assistant_text}")
                self.logger.info(f"=== END WHITE AGENT RESPONSE ===")
                
                result = {
                    "jsonrpc": "2.0",
                    "id": res_root.id,
                    "result": {
                        "message": res_root.result.model_dump() if isinstance(res_root.result, Message) else res_root.result,
                        "context_id": res_root.result.context_id if isinstance(res_root.result, Message) else context_id,
                        "history": [{"role": "assistant", "content": assistant_text}],
                        "artifacts": []
                    }
                }
                
                self.logger.debug(f"White agent response received")
                return result
            else:
                # Handle error response from white agent
                error_msg = f"Unexpected response type from white agent: {type(res_root)}"
                if hasattr(res_root, 'error'):
                    error_details = res_root.error
                    self.logger.error(f"{error_msg} - Error details: {error_details}")
                    raise Exception(f"{error_msg}: {error_details}")
                else:
                    self.logger.error(f"{error_msg} - Response: {res_root}")
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