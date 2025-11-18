#!/usr/bin/env python3
"""
Terminal Bench Green Agent Tools

This module provides tools for the green agent to orchestrate Terminal Bench evaluations.
It integrates with the green_agent package from the green-white-agent directory.
"""

import os
import sys
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add the green-white-agent to the Python path
# File is now in: green-white-agent/agentbeats/scenarios/terminal_bench/agents/green_agent/tools.py
# green-white-agent root is at: parents[4]
GREEN_WHITE_AGENT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(GREEN_WHITE_AGENT_PATH))

try:
    from green_agent import GreenAgentTerminalBench, TaskExecutionResult
    from green_agent.sandbox_manager import SandboxManager, CommandResult
    from green_agent.task_evaluator import TaskEvaluator, EvaluationResult
except ImportError as e:
    logging.error(f"Failed to import green_agent modules: {e}")
    logging.error(f"Tried to import from: {GREEN_WHITE_AGENT_PATH}")
    raise

# Import AgentBeats tool decorator
try:
    # Try to find AgentBeats (may be in parent directory if green-white-agent is in llm_runners)
    LLM_RUNNERS = GREEN_WHITE_AGENT_PATH.parent
    AGENTBEATS_SRC = LLM_RUNNERS / "agentbeats" / "src"
    if AGENTBEATS_SRC.exists() and str(AGENTBEATS_SRC) not in sys.path:
        sys.path.insert(0, str(AGENTBEATS_SRC))
    
    from agentbeats import tool
except ImportError as e:
    # If AgentBeats isn't available, create a dummy decorator for testing
    logger = logging.getLogger(__name__)
    logger.warning(f"AgentBeats not available ({e}), using dummy @tool decorator")
    def tool(func):
        """Dummy tool decorator when AgentBeats is not available."""
        return func

# Import A2A client libraries
try:
    from a2a.client import A2ACardResolver, A2AClient
    from a2a.types import (
        Message, MessageSendParams, SendStreamingMessageRequest,
        SendStreamingMessageSuccessResponse, TaskArtifactUpdateEvent,
        TaskStatusUpdateEvent, Role, Part, TextPart, AgentCard
    )
    import httpx
    A2A_AVAILABLE = True
except ImportError as e:
    # A2A libraries may not be available
    A2A_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"A2A libraries not available ({e}), some functions will be limited")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
_green_agent = None
_httpx_client = None

# Initialize httpx client only if A2A is available
if A2A_AVAILABLE:
    _httpx_client = httpx.AsyncClient()

def get_green_agent(white_agent_url: str = "http://localhost:8341") -> GreenAgentTerminalBench:
    """Get or create the green agent instance."""
    global _green_agent
    if _green_agent is None:
        _green_agent = GreenAgentTerminalBench(
            white_agent_url=white_agent_url,
            sandbox_base_path=None,
            terminal_bench_dataset_path=None
        )
    return _green_agent


########################################################
# Task Loading and Management Tools
########################################################

@tool
def load_terminal_bench_tasks(
    task_ids: Optional[List[str]] = None,
    limit: int = 5
) -> Dict[str, Any]:
    """
    Load Terminal Bench tasks for evaluation.
    
    Args:
        task_ids: Optional list of specific task IDs to load. If None, loads all available tasks.
        limit: Maximum number of tasks to load (default: 5)
    
    Returns:
        Dictionary with:
        - tasks: List of task dictionaries with id, description, instruction, environment
        - count: Number of tasks loaded
        - available_categories: List of task categories
    """
    try:
        logger.info(f"Loading Terminal Bench tasks (task_ids={task_ids}, limit={limit})")
        green_agent = get_green_agent()
        
        tasks = green_agent.load_terminal_bench_tasks(task_ids=task_ids, limit=limit)
        
        # Group tasks by category if available
        categories = set()
        for task in tasks:
            if 'category' in task:
                categories.add(task['category'])
        
        result = {
            "tasks": tasks,
            "count": len(tasks),
            "available_categories": list(categories),
            "status": "success"
        }
        
        logger.info(f"Successfully loaded {len(tasks)} tasks")
        return result
        
    except Exception as e:
        logger.error(f"Error loading tasks: {e}")
        return {
            "tasks": [],
            "count": 0,
            "available_categories": [],
            "status": "error",
            "error": str(e)
        }


########################################################
# Sandbox Management Tools
########################################################

@tool
def create_sandbox(
    task_id: str,
    environment: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create an isolated sandbox environment for task execution.
    
    Args:
        task_id: Unique identifier for the task
        environment: Optional environment configuration (working_directory, env_vars, etc.)
    
    Returns:
        Dictionary with:
        - sandbox_id: Unique sandbox identifier
        - status: Creation status
        - working_directory: Sandbox working directory
    """
    try:
        logger.info(f"Creating sandbox for task: {task_id}")
        green_agent = get_green_agent()
        
        sandbox_id = green_agent.sandbox_manager.create_sandbox(task_id, environment)
        
        result = {
            "sandbox_id": sandbox_id,
            "status": "created",
            "task_id": task_id,
            "working_directory": environment.get('working_directory', '/app') if environment else '/app'
        }
        
        logger.info(f"Created sandbox: {sandbox_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error creating sandbox: {e}")
        return {
            "sandbox_id": None,
            "status": "error",
            "error": str(e)
        }


@tool
def execute_command_in_sandbox(
    sandbox_id: str,
    command: str,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Execute a command in the specified sandbox.
    
    Args:
        sandbox_id: ID of the sandbox to execute in
        command: Command to execute
        timeout: Command timeout in seconds (default: 30)
    
    Returns:
        Dictionary with:
        - success: Whether command succeeded
        - stdout: Standard output
        - stderr: Standard error
        - exit_code: Command exit code
        - execution_time: Time taken to execute
    """
    try:
        logger.info(f"Executing command in sandbox {sandbox_id}: {command}")
        green_agent = get_green_agent()
        
        result = green_agent.sandbox_manager.execute_command(sandbox_id, command, timeout=timeout)
        
        return {
            "success": result.success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "execution_time": result.execution_time,
            "command": command,
            "sandbox_id": sandbox_id
        }
        
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "execution_time": 0.0,
            "error": str(e)
        }


@tool
def destroy_sandbox(sandbox_id: str) -> Dict[str, Any]:
    """
    Destroy a sandbox and clean up resources.
    
    Args:
        sandbox_id: ID of the sandbox to destroy
    
    Returns:
        Dictionary with status of destruction
    """
    try:
        logger.info(f"Destroying sandbox: {sandbox_id}")
        green_agent = get_green_agent()
        
        green_agent.sandbox_manager.destroy_sandbox(sandbox_id)
        
        return {
            "sandbox_id": sandbox_id,
            "status": "destroyed",
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error destroying sandbox: {e}")
        return {
            "sandbox_id": sandbox_id,
            "status": "error",
            "success": False,
            "error": str(e)
        }


########################################################
# White Agent Communication Tools
########################################################

@tool
async def send_task_to_white_agent(
    task: Dict[str, Any],
    white_agent_url: str = "http://localhost:8341",
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Send a task to the white agent for solving.
    
    Args:
        task: Task dictionary with id, description, instruction, environment
        white_agent_url: URL of the white agent
        timeout: Timeout in seconds (default: 300)
    
    Returns:
        Dictionary with white agent's response including solution
    """
    try:
        logger.info(f"Sending task {task.get('id')} to white agent at {white_agent_url}")
        green_agent = get_green_agent(white_agent_url)
        
        # Update the white agent URL if it changed
        green_agent.white_agent_url = white_agent_url
        
        response = green_agent.send_task_to_white_agent(task)
        
        return {
            "status": "completed",
            "task_id": task.get('id'),
            "response": response,
            "white_agent_url": white_agent_url
        }
        
    except Exception as e:
        logger.error(f"Error sending task to white agent: {e}")
        return {
            "status": "error",
            "task_id": task.get('id'),
            "error": str(e)
        }


@tool
async def talk_to_white_agent(
    query: str,
    white_agent_url: str = "http://localhost:8341",
    timeout_seconds: float = 120.0
) -> str:
    """
    Send a query to the white agent and get a response.
    
    Args:
        query: Question or instruction to send
        white_agent_url: URL of the white agent
        timeout_seconds: Timeout in seconds
    
    Returns:
        String response from the white agent
    """
    try:
        logger.info(f"Talking to white agent: {query[:100]}...")
        
        # Create A2A client
        resolver = A2ACardResolver(
            httpx_client=_httpx_client,
            base_url=white_agent_url,
        )
        card: AgentCard | None = await resolver.get_agent_card(
            relative_card_path="/.well-known/agent.json"
        )
        if card is None:
            raise RuntimeError(f"Failed to resolve agent card from {white_agent_url}")
        
        client = A2AClient(httpx_client=_httpx_client, agent_card=card)
        
        # Send message
        params = MessageSendParams(
            message=Message(
                role=Role.user,
                parts=[Part(TextPart(text=query))],
                messageId=str(uuid.uuid4()),
                taskId=None,
            )
        )
        req = SendStreamingMessageRequest(id=str(uuid.uuid4()), params=params)
        
        chunks: List[str] = []
        import asyncio
        
        async def stream_reply():
            async for chunk in client.send_message_streaming(req):
                if not isinstance(chunk.root, SendStreamingMessageSuccessResponse):
                    continue
                event = chunk.root.result
                
                if isinstance(event, TaskArtifactUpdateEvent):
                    for p in event.artifact.parts:
                        if isinstance(p.root, TextPart):
                            chunks.append(p.root.text)
                elif isinstance(event, TaskStatusUpdateEvent):
                    msg = event.status.message
                    if msg:
                        for p in msg.parts:
                            if isinstance(p.root, TextPart):
                                chunks.append(p.root.text)
        
        await asyncio.wait_for(stream_reply(), timeout=timeout_seconds)
        
        response = "".join(chunks).strip() or "No response from agent."
        logger.info(f"Received response from white agent: {response[:100]}...")
        return response
        
    except asyncio.TimeoutError:
        logger.warning(f"Timeout waiting for white agent response")
        return f"[Timeout] No response from agent after {timeout_seconds} seconds"
    except Exception as e:
        logger.error(f"Error talking to white agent: {e}")
        return f"Error: {str(e)}"


########################################################
# Task Evaluation Tools
########################################################

@tool
def evaluate_task(
    task_id: str,
    task: Dict[str, Any],
    sandbox_id: str,
    command_history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Evaluate whether a task was completed successfully.
    
    Args:
        task_id: ID of the task to evaluate
        task: Original task specification
        sandbox_id: ID of the sandbox where task was executed
        command_history: Optional list of commands executed
    
    Returns:
        Dictionary with:
        - passed: Whether task passed evaluation
        - score: Numeric score (0.0-1.0)
        - details: Detailed evaluation results
        - criteria_met: Which evaluation criteria were met
    """
    try:
        logger.info(f"Evaluating task: {task_id}")
        green_agent = get_green_agent()
        
        # Capture final sandbox state
        final_state = green_agent.sandbox_manager.capture_state(sandbox_id)
        
        # Convert command history to CommandResult objects if needed
        cmd_results = []
        if command_history:
            for cmd in command_history:
                if isinstance(cmd, dict):
                    from datetime import datetime
                    cmd_results.append(CommandResult(
                        command=cmd.get('command', ''),
                        stdout=cmd.get('stdout', ''),
                        stderr=cmd.get('stderr', ''),
                        returncode=cmd.get('returncode', cmd.get('exit_code', 0)),  # Support both names
                        execution_time=cmd.get('execution_time', 0.0),
                        timestamp=cmd.get('timestamp', datetime.utcnow().isoformat()),
                        success=cmd.get('success', False)
                    ))
        
        # Evaluate using task evaluator
        evaluation = green_agent.task_evaluator.evaluate(
            task_id=task_id,
            task=task,
            command_history=cmd_results,
            sandbox_state=final_state,
            sandbox_manager=green_agent.sandbox_manager,
            sandbox_id=sandbox_id
        )
        
        return {
            "task_id": task_id,
            "passed": evaluation.passed,
            "score": evaluation.score,
            "details": evaluation.details,
            "criteria_met": evaluation.criteria_met if hasattr(evaluation, 'criteria_met') else [],
            "evaluation_time": evaluation.evaluation_time if hasattr(evaluation, 'evaluation_time') else 0.0
        }
        
    except Exception as e:
        logger.error(f"Error evaluating task: {e}")
        return {
            "task_id": task_id,
            "passed": False,
            "score": 0.0,
            "details": {"error": str(e)},
            "criteria_met": []
        }


########################################################
# Full Task Execution Tool
########################################################

@tool
def execute_task_with_sandbox(
    task: Dict[str, Any],
    white_agent_url: str = "http://localhost:8341"
) -> Dict[str, Any]:
    """
    Execute a complete task lifecycle with sandbox isolation.
    This is a high-level tool that:
    1. Creates a sandbox
    2. Sends task to white agent
    3. Executes agent's commands
    4. Evaluates results
    5. Cleans up sandbox
    
    Args:
        task: Task specification with id, description, instruction, environment
        white_agent_url: URL of the white agent to use
    
    Returns:
        Dictionary with complete execution results
    """
    try:
        logger.info(f"Executing task with sandbox: {task.get('id')}")
        green_agent = get_green_agent(white_agent_url)
        
        # Update URL if changed
        green_agent.white_agent_url = white_agent_url
        
        # Execute the full task lifecycle
        result = green_agent.execute_task_with_sandbox(task)
        
        return {
            "task_id": result.task_id,
            "success": result.success,
            "execution_time": result.execution_time,
            "commands_executed": result.commands_executed,
            "evaluation": {
                "passed": result.evaluation_result.passed if result.evaluation_result else False,
                "score": result.evaluation_result.score if result.evaluation_result else 0.0,
                "details": result.evaluation_result.details if result.evaluation_result else {}
            },
            "sandbox_id": result.sandbox_id,
            "timestamp": result.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error executing task: {e}")
        return {
            "task_id": task.get('id', 'unknown'),
            "success": False,
            "execution_time": 0.0,
            "commands_executed": 0,
            "evaluation": {"passed": False, "score": 0.0, "details": {"error": str(e)}},
            "sandbox_id": None,
            "error": str(e)
        }


########################################################
# Reporting and Progress Tools
########################################################

@tool
def report_task_result(
    task_id: str,
    success: bool,
    score: float,
    execution_time: float,
    details: Dict[str, Any]
) -> str:
    """
    Report the result of a task execution.
    
    Args:
        task_id: Task identifier
        success: Whether task succeeded
        score: Numeric score (0.0-1.0)
        execution_time: Time taken to execute
        details: Additional details about execution
    
    Returns:
        Formatted report string
    """
    status_emoji = "✅" if success else "❌"
    report = f"""
{status_emoji} Task Result: {task_id}
----------------------------------------
Status: {'PASSED' if success else 'FAILED'}
Score: {score:.2f}
Execution Time: {execution_time:.2f}s
Commands Executed: {details.get('commands_executed', 0)}

Details:
{json.dumps(details, indent=2)}
"""
    logger.info(report)
    return report


@tool
def get_execution_summary() -> Dict[str, Any]:
    """
    Get a summary of all task executions.
    
    Returns:
        Dictionary with:
        - total_tasks: Total number of tasks executed
        - passed: Number of passed tasks
        - failed: Number of failed tasks
        - average_score: Average score across all tasks
        - total_time: Total execution time
        - task_results: List of individual task results
    """
    try:
        green_agent = get_green_agent()
        history = green_agent.get_execution_history()
        
        total_tasks = len(history)
        passed = sum(1 for r in history if r.success)
        failed = total_tasks - passed
        avg_score = sum(r.evaluation_result.score for r in history if r.evaluation_result) / total_tasks if total_tasks > 0 else 0.0
        total_time = sum(r.execution_time for r in history)
        
        task_results = [
            {
                "task_id": r.task_id,
                "success": r.success,
                "score": r.evaluation_result.score if r.evaluation_result else 0.0,
                "execution_time": r.execution_time,
                "commands_executed": r.commands_executed,
                "timestamp": r.timestamp
            }
            for r in history
        ]
        
        return {
            "total_tasks": total_tasks,
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / total_tasks * 100) if total_tasks > 0 else 0.0,
            "average_score": avg_score,
            "total_time": total_time,
            "task_results": task_results
        }
        
    except Exception as e:
        logger.error(f"Error getting execution summary: {e}")
        return {
            "total_tasks": 0,
            "passed": 0,
            "failed": 0,
            "success_rate": 0.0,
            "average_score": 0.0,
            "total_time": 0.0,
            "task_results": [],
            "error": str(e)
        }


@tool
def cleanup_all_resources() -> str:
    """
    Clean up all sandbox resources.
    
    Returns:
        Status message
    """
    try:
        logger.info("Cleaning up all resources...")
        green_agent = get_green_agent()
        green_agent.cleanup_resources()
        return "✅ All resources cleaned up successfully"
    except Exception as e:
        logger.error(f"Error cleaning up resources: {e}")
        return f"❌ Error cleaning up resources: {e}"

