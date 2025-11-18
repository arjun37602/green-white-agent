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
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .sandbox_manager import SandboxManager, CommandResult, SandboxState
from .task_evaluator import TaskEvaluator, EvaluationResult
from .dataset_loaders.terminal_bench_loader import TerminalBenchTaskLoader, TerminalBenchTask

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
    
    def __init__(self, white_agent_url: str = "http://localhost:8002", 
                 sandbox_base_path: Optional[str] = None,
                 terminal_bench_dataset_path: Optional[str] = None):
        self.white_agent_url = white_agent_url
        self.logger = logging.getLogger(__name__)
        
        # Initialize sandbox manager and task evaluator
        self.sandbox_manager = SandboxManager(base_path=sandbox_base_path)
        self.task_evaluator = TaskEvaluator()
        
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
    
    def execute_task_with_sandbox(self, task: Dict[str, Any]) -> TaskExecutionResult:
        """
        Execute a complete task lifecycle with sandbox isolation.
        
        Args:
            task: Task specification
            
        Returns:
            TaskExecutionResult with complete execution details
        """
        task_id = task.get('id', 'unknown')
        start_time = time.time()
        
        self.logger.info(f"🚀 Starting task execution: {task_id}")
        
        try:
            # Step 1: Create sandbox environment
            sandbox_id = self.sandbox_manager.create_sandbox(task_id, task.get('environment'))
            self.logger.info(f"📦 Created sandbox: {sandbox_id}")
            
            # Step 2: Send task to white agent
            white_agent_response = self.send_task_to_white_agent(task)
            
            # Step 3: Execute commands in sandbox
            command_history = self._execute_white_agent_commands(sandbox_id, white_agent_response)
            
            # Step 4: Capture final sandbox state
            final_sandbox_state = self.sandbox_manager.capture_state(sandbox_id)
            
            # Step 5: Evaluate task completion
            evaluation_result = self.task_evaluator.evaluate(
                task_id, task, command_history, final_sandbox_state, 
                self.sandbox_manager, sandbox_id, white_agent_response
            )
            
            # Step 6: Clean up sandbox
            self.sandbox_manager.destroy_sandbox(sandbox_id)
            
            execution_time = time.time() - start_time
            
            # Create execution result
            result = TaskExecutionResult(
                task_id=task_id,
                success=evaluation_result.passed,
                execution_time=execution_time,
                commands_executed=len(command_history),
                evaluation_result=evaluation_result,
                sandbox_id=sandbox_id,
                white_agent_response=white_agent_response,
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Store in history
            self.execution_history.append(result)
            
            self.logger.info(f"✅ Task {task_id} completed in {execution_time:.2f}s - {'PASSED' if result.success else 'FAILED'}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Task {task_id} failed: {e}")
            
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
    
    def _execute_white_agent_commands(self, sandbox_id: str, white_agent_response: Dict[str, Any]) -> List[CommandResult]:
        """
        Execute commands from white agent response in sandbox.
        
        Args:
            sandbox_id: ID of the sandbox to execute in
            white_agent_response: Response from white agent
            
        Returns:
            List of CommandResult objects
        """
        command_history = []
        
        try:
            # Extract commands from white agent response
            commands = self._extract_commands_from_response(white_agent_response)
            
            if not commands:
                self.logger.warning("No commands found in white agent response")
                return command_history
            
            self.logger.info(f"Executing {len(commands)} commands in sandbox {sandbox_id}")
            
            for i, command in enumerate(commands):
                # Rewrite /app paths to ./app subdirectory
                # Tests expect files at /app/*, so we create ./app/* and later symlink
                original_command = command
                import re
                # Replace /app/ and /app at end of string or before whitespace
                command = re.sub(r'/app/', './app/', command)
                command = re.sub(r'/app(\s|$)', r'./app\1', command)
                if command != original_command:
                    self.logger.info(f"🔧 Rewrote path: {original_command} → {command}")
                
                self.logger.info(f"Command {i+1}/{len(commands)}: {command}")
                
                # Execute command in sandbox
                result = self.sandbox_manager.execute_command(sandbox_id, command)
                command_history.append(result)
                
                # Log result
                if result.success:
                    self.logger.info(f"✅ Command succeeded: {result.stdout[:100]}...")
                else:
                    self.logger.warning(f"⚠️ Command failed: {result.stderr[:100]}...")
                
                # Add small delay between commands
                time.sleep(0.1)
            
            return command_history
            
        except Exception as e:
            self.logger.error(f"Error executing commands: {e}")
            return command_history
    
    def _extract_commands_from_response(self, white_agent_response: Dict[str, Any]) -> List[str]:
        """
        Extract bash commands from white agent response.
        
        Args:
            white_agent_response: Response from white agent
            
        Returns:
            List of bash commands to execute
        """
        commands = []
        
        try:
            # Look for artifacts in the response
            if "result" in white_agent_response and "artifacts" in white_agent_response["result"]:
                artifacts = white_agent_response["result"]["artifacts"]
                
                for artifact in artifacts:
                    if "parts" in artifact:
                        for part in artifact["parts"]:
                            if part.get("kind") == "text":
                                text_content = part.get("text", "")
                                # Extract bash commands from text
                                extracted_commands = self._parse_bash_commands(text_content)
                                commands.extend(extracted_commands)
            
            # Also check for direct command responses
            if "commands" in white_agent_response:
                commands.extend(white_agent_response["commands"])
            
            return commands
            
        except Exception as e:
            self.logger.error(f"Error extracting commands: {e}")
            return []
    
    def _parse_bash_commands(self, text: str) -> List[str]:
        """
        Parse bash commands from text content.
        
        Args:
            text: Text content that may contain bash commands
            
        Returns:
            List of bash commands
        """
        commands = []
        
        # Look for code blocks with bash
        import re
        
        # Find bash code blocks
        bash_pattern = r'```(?:bash|sh|shell)?\n(.*?)\n```'
        bash_blocks = re.findall(bash_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for block in bash_blocks:
            # Split by lines and clean up
            block_commands = [line.strip() for line in block.split('\n') if line.strip()]
            commands.extend(block_commands)
        
        # Also look for single-line commands starting with $ or #
        line_pattern = r'^\s*[$#]\s*(.+)$'
        for line in text.split('\n'):
            match = re.match(line_pattern, line)
            if match:
                commands.append(match.group(1).strip())
        
        # If no commands found yet, try parsing plain text before any separator like "===" 
        if not commands:
            # Look for commands before any "===" separator or other metadata
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                # Stop at separator lines or explanatory text
                if line.startswith('===') or line.startswith('Tool:') or line.startswith('Args:'):
                    break
                # Look for lines that look like bash commands
                if line and not line.startswith('#') and ('/' in line or '>' in line or any(cmd in line for cmd in ['mkdir', 'echo', 'printf', 'cat', 'touch', 'cd', 'ls', 'cp', 'mv', 'rm', 'chmod', 'chown'])):
                    commands.append(line)
        
        # Filter out comments and empty commands
        commands = [cmd for cmd in commands if cmd and not cmd.startswith('#')]
        
        return commands
    
    def send_task_to_white_agent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Send a task to the white agent via A2A protocol."""
        try:
            # Create A2A message
            rpc_request = {
                "jsonrpc": "2.0",
                "id": f"task-{task.get('id', 'unknown')}",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [
                            {
                                "kind": "text",
                                "text": f"""Terminal Bench Task: {task.get('id')}

Description: {task.get('description', 'No description')}

Instructions:
{task.get('instruction', '')}

Environment:
{json.dumps(task.get('environment', {}), indent=2)}

Complete this task and verify it passes the tests.
"""
                            }
                        ]
                    }
                }
            }
            
            # Send to white agent
            response = requests.post(self.white_agent_url, json=rpc_request, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                
                # Debug: Log response structure to inspect token/request data
                self.logger.debug(f"White agent response structure for task {task.get('id', 'unknown')}: "
                                f"{json.dumps(result, indent=2)[:1000]}")
                
                # Check if the task was completed successfully
                if "result" in result and result["result"]:
                    task_data = result["result"]
                    task_id = task.get('id', 'unknown')
                    if task_data.get("status", {}).get("state") == "completed":
                        self.logger.info(f"✅ Task {task_id} completed successfully")
                        return result
                    else:
                        self.logger.warning(f"⚠️ Task {task_id} did not complete successfully: {task_data.get('status', {}).get('state')}")
                        return result
                else:
                    task_id = task.get('id', 'unknown')
                    self.logger.error(f"❌ Task {task_id} failed: {result.get('error', 'Unknown error')}")
                    return result
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            self.logger.error(f"Error sending task to white agent: {e}")
            return {"error": str(e)}
    
    def run_evaluation(self, task_ids: List[str] = None, limit: int = 5):
        """
        Run full evaluation loop with sandbox-based execution and Terminal-Bench pytest evaluation.
        
        Args:
            task_ids: Specific task IDs to run
            limit: Maximum number of tasks
        """
        self.logger.info("🚀 Starting Terminal Bench evaluation")
        
        # Load tasks from Terminal-Bench dataset
        tasks = self.load_terminal_bench_tasks(task_ids, limit)
        self.logger.info(f"📋 Loaded {len(tasks)} tasks")
        
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
            
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
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
        self.logger.info("🧹 Cleaning up sandbox resources...")
        self.sandbox_manager.cleanup_all()
        self.logger.info("✅ Cleanup completed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Green Agent: Terminal Bench Evaluator")
    parser.add_argument("--agent-url", default="http://localhost:8002", help="White agent URL")
    parser.add_argument("--tasks", nargs="+", help="Specific task IDs to run")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of tasks")
    parser.add_argument("--no-sandbox", action="store_true", help="Disable sandbox isolation")
    parser.add_argument("--sandbox-path", help="Custom sandbox base path")
    parser.add_argument("--dataset-path", help="Path to Terminal-Bench dataset directory")
    parser.add_argument("--cleanup", action="store_true", help="Clean up sandbox resources and exit")
    
    args = parser.parse_args()
    
    try:
        green_agent = GreenAgentTerminalBench(args.agent_url, args.sandbox_path, args.dataset_path)
        
        if args.cleanup:
            green_agent.cleanup_resources()
            sys.exit(0)
        
        # Run evaluation
        results = green_agent.run_evaluation(
            args.tasks, 
            args.limit, 
            use_sandbox=not args.no_sandbox
        )
        
        # Print detailed results
        print("\n" + "="*80)
        print("📋 DETAILED RESULTS")
        print("="*80)
        
        for result in results:
            print(f"\nTask: {result['task_id']}")
            print(f"  Success: {'✅' if result['success'] else '❌'}")
            print(f"  Score: {result['evaluation_score']:.2f}")
            print(f"  Execution Time: {result['execution_time']:.2f}s")
            print(f"  Commands Executed: {result['commands_executed']}")
            if result['sandbox_id']:
                print(f"  Sandbox ID: {result['sandbox_id']}")
        
        # Cleanup resources
        green_agent.cleanup_resources()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user")
        if 'green_agent' in locals():
            green_agent.cleanup_resources()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if 'green_agent' in locals():
            green_agent.cleanup_resources()
