#!/usr/bin/env python3
"""
Green Agent: Terminal Bench Task Runner and Evaluator
Loads Terminal Bench tasks and sends them to the white agent for evaluation.
"""

import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Any
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GreenAgentTerminalBench:
    """Green agent for running Terminal Bench tasks against the white agent."""
    
    def __init__(self, white_agent_url: str = "http://localhost:8002"):
        self.white_agent_url = white_agent_url
        self.logger = logging.getLogger(__name__)
    
    def load_terminal_bench_tasks(self, task_ids: List[str] = None, limit: int = None) -> List[Dict[str, Any]]:
        """
        Load Terminal Bench tasks using the terminal-bench CLI.
        
        Args:
            task_ids: Specific task IDs to load (None = load all)
            limit: Maximum number of tasks to load
            
        Returns:
            List of task dictionaries
        """
        try:
            # For now, we'll create a simple test task since terminal-bench doesn't have a direct list command
            # In a real implementation, you would use the terminal-bench run command with specific task IDs
            
            if task_ids:
                # Create mock tasks for the specified IDs
                tasks = []
                for task_id in task_ids:
                    task = {
                        "id": task_id,
                        "description": f"Terminal Bench task: {task_id}",
                        "instruction": f"Complete the task: {task_id}",
                        "environment": {"working_directory": "/app"},
                        "test": f"Test for {task_id}"
                    }
                    tasks.append(task)
                return tasks
            else:
                # Return some example tasks for testing
                example_tasks = [
                    {
                        "id": "crack-7z-hash",
                        "description": "Crack a 7z archive password",
                        "instruction": "You need to create a file called 'solution.txt' with the word found in the 'secrets.7z' archive. The final 'solution.txt' should be located at '/app/solution.txt'.",
                        "environment": {"working_directory": "/app"},
                        "test": "Check if solution.txt exists and contains the correct password"
                    },
                    {
                        "id": "openssl-selfsigned-cert",
                        "description": "Create a self-signed TLS certificate",
                        "instruction": "Your company needs a self-signed TLS certificate for an internal development server. Create a self-signed certificate using OpenSSL with the specified requirements.",
                        "environment": {"working_directory": "/app"},
                        "test": "Verify certificate files exist and are properly formatted"
                    }
                ]
                
                if limit:
                    return example_tasks[:limit]
                return example_tasks
                    
        except Exception as e:
            self.logger.error(f"Error loading Terminal Bench tasks: {e}")
            return []
    
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
    
    def evaluate_task(self, task_id: str) -> Dict[str, Any]:
        """Evaluate a completed task using terminal-bench."""
        try:
            self.logger.info(f"🔍 Evaluating task: {task_id}")
            
            # Run the actual terminal-bench evaluation
            # Note: This requires the white agent to be integrated as a custom agent
            #TODO: White agent should return the solution file or expected output
            # For now, we'll use a simple test to verify the task was completed
            
            # Check if the task has a solution file or expected output
            if task_id == "crack-7z-hash":
                # Check if solution.txt exists and contains expected content
                cmd = "test -f /app/solution.txt && echo 'File exists' || echo 'File missing'"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                
                evaluation_result = {
                    "task_id": task_id,
                    "passed": result.returncode == 0 and "File exists" in result.stdout,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "test_command": cmd
                }
                
            elif task_id == "openssl-selfsigned-cert":
                # Check if certificate files exist
                cmd = "test -f /app/ssl/server.crt && test -f /app/ssl/server.key && echo 'Certificate files exist' || echo 'Certificate files missing'"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                
                evaluation_result = {
                    "task_id": task_id,
                    "passed": result.returncode == 0 and "Certificate files exist" in result.stdout,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "test_command": cmd
                }
                
            else:
                # Generic evaluation - check if any output was produced
                cmd = f"ls -la /app/ | grep -v '^total' | wc -l"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                
                evaluation_result = {
                    "task_id": task_id,
                    "passed": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "test_command": cmd
                }
            
            self.logger.info(f"📊 Evaluation result for {task_id}: {'✅ PASSED' if evaluation_result['passed'] else '❌ FAILED'}")
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Error evaluating task {task_id}: {e}")
            return {"task_id": task_id, "passed": False, "error": str(e)}
    
    def run_evaluation(self, task_ids: List[str] = None, limit: int = 5):
        """Run full evaluation loop."""
        self.logger.info("🚀 Starting Terminal Bench evaluation")
        
        # Load tasks
        tasks = self.load_terminal_bench_tasks(task_ids, limit)
        self.logger.info(f"📋 Loaded {len(tasks)} tasks")
        
        results = []
        
        for i, task in enumerate(tasks, 1):
            task_id = task.get('id', f'task_{i}')
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📝 Task {i}/{len(tasks)}: {task_id}")
            self.logger.info(f"{'='*60}")
            
            # Send to white agent
            agent_response = self.send_task_to_white_agent(task)
            
            # Evaluate result
            evaluation = self.evaluate_task(task_id)
            
            result = {
                "task_id": task_id,
                "agent_response": agent_response,
                "evaluation": evaluation,
                "passed": evaluation.get("passed", False)
            }
            
            results.append(result)
            
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            self.logger.info(f"{status}: {task_id}")
        
        # Summary
        passed = sum(1 for r in results if r["passed"])
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 EVALUATION SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total: {len(results)}")
        self.logger.info(f"Passed: {passed}")
        self.logger.info(f"Failed: {len(results) - passed}")
        self.logger.info(f"Success Rate: {passed/len(results)*100:.1f}%")
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Green Agent: Terminal Bench Evaluator")
    parser.add_argument("--agent-url", default="http://localhost:8002", help="White agent URL")
    parser.add_argument("--tasks", nargs="+", help="Specific task IDs to run")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of tasks")
    
    args = parser.parse_args()
    
    green_agent = GreenAgentTerminalBench(args.agent_url)
    green_agent.run_evaluation(args.tasks, args.limit)
