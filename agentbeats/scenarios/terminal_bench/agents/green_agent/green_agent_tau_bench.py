"""Terminal Bench Green Agent - Tau-Bench Pattern Implementation

This implements the green agent following the tau-bench example pattern:
- Uses AgentExecutor directly (not tutorial's GreenAgent base class)
- Receives task description via A2A message with tags
- Sets up Terminal Bench environment
- Sends tasks to white agent via A2A
- Executes solutions in sandbox
- Evaluates results
"""

import uvicorn
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Add paths for imports
# File is now in: green-white-agent/agentbeats/scenarios/terminal_bench/agents/green_agent/green_agent_tau_bench.py
# green-white-agent root is at: parents[4]
GREEN_WHITE_AGENT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(GREEN_WHITE_AGENT_PATH))

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.utils import new_agent_text_message, get_text_parts
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import Message, MessageSendParams, SendMessageRequest, Role, Part, TextPart
import httpx
import asyncio
import uuid

from green_agent import GreenAgentTerminalBench

load_dotenv()


def parse_tags(text: str) -> Dict[str, str]:
    """Parse XML-like tags from text."""
    import re
    tags = {}
    pattern = r'<(\w+)>(.*?)</\1>'
    for match in re.finditer(pattern, text, re.DOTALL):
        tag_name = match.group(1)
        tag_content = match.group(2).strip()
        tags[tag_name] = tag_content
    return tags


async def send_message_to_agent(url: str, message: str, context_id: str = None):
    """Send message to agent via A2A and get response."""
    httpx_client = httpx.AsyncClient(timeout=120.0)
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
    card = await resolver.get_agent_card()
    
    if card is None:
        raise RuntimeError(f"Failed to resolve agent card from {url}")
    
    client = A2AClient(httpx_client=httpx_client, agent_card=card)
    
    message_id = uuid.uuid4().hex
    params = MessageSendParams(
        message=Message(
            role=Role.user,
            parts=[Part(TextPart(text=message))],
            message_id=message_id,
            task_id=None,
            context_id=context_id,
        )
    )
    request_id = uuid.uuid4().hex
    req = SendMessageRequest(id=request_id, params=params)
    
    response = await client.send_message(request=req)
    return response


class TerminalBenchGreenAgentExecutor(AgentExecutor):
    """Green agent executor for Terminal Bench following tau-bench pattern."""
    
    def __init__(self):
        self._green_agent_impl = None
    
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute Terminal Bench evaluation."""
        print("Green agent: Received a task, parsing...")
        user_input = context.get_user_input()
        
        # Parse task description with tags
        tags = parse_tags(user_input)
        white_agent_url = tags.get("white_agent_url", "").strip()
        task_config_str = tags.get("task_config", "{}")
        
        if not white_agent_url:
            await event_queue.enqueue_event(
                new_agent_text_message("Error: white_agent_url tag not found in task description")
            )
            return
        
        try:
            task_config = json.loads(task_config_str) if task_config_str else {}
        except json.JSONDecodeError as e:
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error: Invalid task_config JSON: {e}")
            )
            return
        
        # Get configuration
        task_ids = task_config.get("task_ids", None)
        limit = task_config.get("limit", 5)
        
        print(f"Green agent: Setting up Terminal Bench environment...")
        print(f"  White agent URL: {white_agent_url}")
        print(f"  Task IDs: {task_ids}")
        print(f"  Limit: {limit}")
        
        # Initialize green agent implementation
        self._green_agent_impl = GreenAgentTerminalBench(
            white_agent_url=white_agent_url,
            sandbox_base_path=None,
            terminal_bench_dataset_path=None
        )
        
        await event_queue.enqueue_event(
            new_agent_text_message(f"Loading Terminal Bench tasks (limit={limit})...")
        )
        
        # Load tasks
        tasks = self._green_agent_impl.load_terminal_bench_tasks(
            task_ids=task_ids,
            limit=limit
        )
        
        await event_queue.enqueue_event(
            new_agent_text_message(f"Loaded {len(tasks)} tasks. Starting evaluation...")
        )
        
        # Run evaluation for each task
        results = []
        for i, task in enumerate(tasks, 1):
            task_id = task.get('id', f'task_{i}')
            
            await event_queue.enqueue_event(
                new_agent_text_message(f"Evaluating task {i}/{len(tasks)}: {task_id}")
            )
            
            print(f"Green agent: Evaluating task {task_id}...")
            timestamp_started = time.time()
            
            # Execute task with sandbox
            execution_result = self._green_agent_impl.execute_task_with_sandbox(task)
            
            time_used = time.time() - timestamp_started
            
            # Store result
            result = {
                "task_id": execution_result.task_id,
                "success": execution_result.success,
                "execution_time": execution_result.execution_time,
                "commands_executed": execution_result.commands_executed,
                "score": execution_result.evaluation_result.score if execution_result.evaluation_result else 0.0,
                "time_used": time_used,
            }
            results.append(result)
            
            status_emoji = "✅" if execution_result.success else "❌"
            status_text = f"{status_emoji} Task {task_id}: {'PASSED' if execution_result.success else 'FAILED'} (Score: {result['score']:.2f}, Time: {time_used:.2f}s)"
            
            await event_queue.enqueue_event(
                new_agent_text_message(status_text)
            )
            print(f"Green agent: {status_text}")
        
        # Calculate summary
        passed = sum(1 for r in results if r["success"])
        total = len(results)
        avg_score = sum(r["score"] for r in results) / total if total > 0 else 0.0
        total_time = sum(r["time_used"] for r in results)
        
        metrics = {
            "total_tasks": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": (passed / total * 100) if total > 0 else 0.0,
            "average_score": avg_score,
            "total_time": total_time,
        }
        
        result_emoji = "✅" if passed > total / 2 else "❌"
        summary_text = f"""
Finished. White agent evaluation: {result_emoji}
Metrics: {json.dumps(metrics, indent=2)}

Task Results:
{json.dumps(results, indent=2)}
"""
        
        print("Green agent: Evaluation complete.")
        await event_queue.enqueue_event(
            new_agent_text_message(summary_text)
        )
        
        # Cleanup
        if self._green_agent_impl:
            self._green_agent_impl.cleanup_resources()
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel execution."""
        if self._green_agent_impl:
            self._green_agent_impl.cleanup_resources()
        raise NotImplementedError


def prepare_agent_card(url: str) -> AgentCard:
    """Create agent card for Terminal Bench green agent."""
    skill = AgentSkill(
        id="evaluate_terminal_bench",
        name="Evaluate Terminal Bench Tasks",
        description="Orchestrate and evaluate agents on Terminal Bench tasks",
        tags=["terminal-bench", "evaluation", "orchestration"],
        examples=[],
    )
    card = AgentCard(
        name="TerminalBenchGreenAgent",
        description="""
Terminal Bench Green Agent - Orchestrates and evaluates agents on Terminal Bench tasks.

Receives task description with tags:
- <white_agent_url>: URL of the white agent to evaluate
- <task_config>: JSON config with task_ids (optional) and limit (optional)

For each task:
1. Sends task to white agent via A2A
2. Executes solution in isolated sandbox
3. Evaluates task completion
4. Reports results
""",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    return card


def start_green_agent(host="localhost", port=8340):
    """Start the Terminal Bench green agent server."""
    print("Starting Terminal Bench green agent...")
    url = f"http://{host}:{port}"
    card = prepare_agent_card(url)
    
    request_handler = DefaultRequestHandler(
        agent_executor=TerminalBenchGreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )
    
    uvicorn.run(app.build(), host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Terminal Bench Green Agent")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind")
    parser.add_argument("--port", type=int, default=8340, help="Port to bind")
    args = parser.parse_args()
    
    start_green_agent(host=args.host, port=args.port)

