#!/usr/bin/env python3
"""
Terminal Bench Green Agent - Tutorial Pattern Implementation

This implements the green agent following the tutorial pattern:
- Inherits from GreenAgent base class
- Implements run_eval() and validate_request()
- Uses GreenExecutor for A2A protocol handling
- Receives EvalRequest with participants and config
- Uses TaskUpdater for status updates and artifacts
"""

import argparse
import contextlib
import uvicorn
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# Add paths for imports
# File is now in: green-white-agent/agentbeats/scenarios/terminal_bench/agents/green_agent/green_agent_tutorial.py
# green-white-agent root is at: parents[4]
GREEN_WHITE_AGENT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(GREEN_WHITE_AGENT_PATH))

# Try to find tutorial/src (may be in parent directory if green-white-agent is in llm_runners)
LLM_RUNNERS = GREEN_WHITE_AGENT_PATH.parent
TUTORIAL_SRC = LLM_RUNNERS / "tutorial" / "src"
if TUTORIAL_SRC.exists():
    sys.path.insert(0, str(TUTORIAL_SRC))

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import TaskState, Part, TextPart
from a2a.utils import new_agent_text_message

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider

from green_agent import GreenAgentTerminalBench

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("terminal_bench_green_agent")


def terminal_bench_agent_card(name: str, url: str):
    """Create agent card for Terminal Bench green agent."""
    from a2a.types import AgentCard, AgentCapabilities, AgentSkill
    
    return AgentCard(
        name=name,
        description="""
## Your Role
You are the **Green Agent (Terminal Bench Orchestrator)** responsible for evaluating agents against Terminal Bench tasks.

You orchestrate Terminal Bench evaluations by:
1. Loading Terminal Bench tasks
2. Sending tasks to participant agents
3. Executing solutions in isolated sandboxes
4. Evaluating task completion
5. Reporting detailed results

## Assessment Flow
- Receive assessment request with participant agent URL
- Load tasks from Terminal Bench dataset
- For each task:
  - Send task to participant agent
  - Execute solution in sandbox
  - Evaluate completion
- Generate final assessment report
""",
        url=url,
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
        ),
        skills=[
            AgentSkill(
                id="evaluate_terminal_bench",
                name="Evaluate Terminal Bench Tasks",
                description="Orchestrate and evaluate agents on Terminal Bench tasks",
                tags=["terminal-bench", "evaluation", "orchestration"],
                examples=["Evaluate agent on terminal bench tasks"],
            )
        ],
    )


class TerminalBenchGreenAgent(GreenAgent):
    """Green agent for Terminal Bench evaluations following tutorial pattern."""
    
    def __init__(self):
        self._required_roles = ["white_agent"]  # Participant role
        self._required_config_keys = []  # Optional config keys
        self._tool_provider = ToolProvider()
        self._green_agent_impl = None  # Will be initialized per assessment
    
    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate the assessment request."""
        # Check required roles
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        
        # Optional: validate config if needed
        # For now, config is optional
        return True, "ok"
    
    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        """Run the Terminal Bench evaluation."""
        logger.info(f"Starting Terminal Bench evaluation: {req}")
        
        try:
            # Get participant agent URL
            white_agent_url = str(req.participants["white_agent"])
            
            # Initialize green agent implementation
            self._green_agent_impl = GreenAgentTerminalBench(
                white_agent_url=white_agent_url,
                sandbox_base_path=None,
                terminal_bench_dataset_path=None
            )
            
            # Get config (optional)
            task_ids = req.config.get("task_ids", None)
            limit = req.config.get("limit", 5)
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Loading Terminal Bench tasks (limit={limit})...")
            )
            
            # Load tasks
            tasks = self._green_agent_impl.load_terminal_bench_tasks(
                task_ids=task_ids,
                limit=limit
            )
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Loaded {len(tasks)} tasks. Starting evaluation...")
            )
            
            # Run evaluation for each task
            results = []
            for i, task in enumerate(tasks, 1):
                task_id = task.get('id', f'task_{i}')
                
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Evaluating task {i}/{len(tasks)}: {task_id}")
                )
                
                # Execute task with sandbox
                execution_result = self._green_agent_impl.execute_task_with_sandbox(task)
                
                # Store result
                result = {
                    "task_id": execution_result.task_id,
                    "success": execution_result.success,
                    "execution_time": execution_result.execution_time,
                    "commands_executed": execution_result.commands_executed,
                    "score": execution_result.evaluation_result.score if execution_result.evaluation_result else 0.0,
                }
                results.append(result)
                
                status_emoji = "✅" if execution_result.success else "❌"
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"{status_emoji} Task {task_id}: {'PASSED' if execution_result.success else 'FAILED'} "
                        f"(Score: {result['score']:.2f}, Time: {execution_result.execution_time:.2f}s)"
                    )
                )
            
            # Calculate summary
            passed = sum(1 for r in results if r["success"])
            total = len(results)
            avg_score = sum(r["score"] for r in results) / total if total > 0 else 0.0
            total_time = sum(r["execution_time"] for r in results)
            
            # Create evaluation result
            eval_result = EvalResult(
                winner="white_agent" if passed > total / 2 else "none",  # Simple winner logic
                detail={
                    "total_tasks": total,
                    "passed": passed,
                    "failed": total - passed,
                    "success_rate": (passed / total * 100) if total > 0 else 0.0,
                    "average_score": avg_score,
                    "total_time": total_time,
                    "task_results": results,
                }
            )
            
            # Generate summary text
            summary_text = f"""
Terminal Bench Evaluation Summary
==================================
Total Tasks: {total}
Passed: {passed}
Failed: {total - passed}
Success Rate: {passed/total*100:.1f}%
Average Score: {avg_score:.2f}
Total Execution Time: {total_time:.2f}s

Task Results:
"""
            for r in results:
                status = "✅ PASSED" if r["success"] else "❌ FAILED"
                summary_text += f"  {status}: {r['task_id']} (Score: {r['score']:.2f})\n"
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Evaluation complete. Generating report...")
            )
            
            # Add artifact with results
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=summary_text)),
                    Part(root=TextPart(text=eval_result.model_dump_json())),
                ],
                name="Terminal Bench Evaluation Results",
            )
            
            logger.info(f"Terminal Bench evaluation complete: {eval_result.model_dump()}")
            
        except Exception as e:
            logger.error(f"Error in Terminal Bench evaluation: {e}", exc_info=True)
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Error during evaluation: {str(e)}")
            )
            raise
        finally:
            # Cleanup
            if self._green_agent_impl:
                self._green_agent_impl.cleanup_resources()
            self._tool_provider.reset()


async def main():
    """Main entry point for the green agent server."""
    parser = argparse.ArgumentParser(description="Terminal Bench Green Agent")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8340, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--cloudflare-quick-tunnel", action="store_true", 
                       help="Use a Cloudflare quick tunnel. Requires cloudflared.")
    args = parser.parse_args()
    
    if args.cloudflare_quick_tunnel:
        from agentbeats.cloudflare import quick_tunnel
        agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
    else:
        agent_url_cm = contextlib.nullcontext(args.card_url or f"http://{args.host}:{args.port}/")
    
    async with agent_url_cm as agent_url:
        agent = TerminalBenchGreenAgent()
        executor = GreenExecutor(agent)
        agent_card = terminal_bench_agent_card("TerminalBenchGreenAgent", agent_url)
        
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )
        
        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )
        
        uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
        uvicorn_server = uvicorn.Server(uvicorn_config)
        await uvicorn_server.serve()


if __name__ == '__main__':
    asyncio.run(main())

