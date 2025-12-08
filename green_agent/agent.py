"""Green agent implementation - manages assessment and evaluation."""

import uvicorn
import tomllib
import dotenv
import json
import time
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
from a2a.utils import new_agent_text_message
from utils import parse_tags

# Terminal Bench imports (required)
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.handlers.trial_handler import Task, TaskPaths

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file in project root
env_path = Path(__file__).parent.parent / ".env"
dotenv.load_dotenv(dotenv_path=env_path)

# Verify API key is loaded (needed for white agent communication)
if not os.getenv("OPENAI_API_KEY"):
    logger.warning(
        "OPENAI_API_KEY not found in environment. "
        "The white agent will need this to function properly."
    )


def load_agent_card_toml(agent_name):
    current_dir = Path(__file__).parent
    with open(current_dir / f"{agent_name}.toml", "rb") as f:
        return tomllib.load(f)

class TerminalBenchGreenAgentExecutor(AgentExecutor):
    """Green agent executor using RequestContext for state management."""
    
    def __init__(self, terminal_bench_dataset_path: str = None):
        self.terminal_bench_dataset_path = terminal_bench_dataset_path

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Parse the task from user input
        logger.info("Green agent: Received a task, parsing...")
        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        
        # Support both single white agent (backward compat) and multiple white agents
        white_agent_urls_str = tags.get("white_agent_urls")
        if white_agent_urls_str:
            white_agent_urls = json.loads(white_agent_urls_str)
            logger.info(f"Green agent: Using {len(white_agent_urls)} white agents for parallel execution")
        else:
            # Fallback to single white agent
            white_agent_url = tags.get("white_agent_url")
            white_agent_urls = [white_agent_url] if white_agent_url else []
        
        task_config_str = tags.get("task_config", "{}")
        task_config = json.loads(task_config_str)
        
        logger.info(f"Green agent: White agent URLs: {white_agent_urls}")
        logger.info(f"Green agent: Task config: {task_config}")
        
        # Load and execute Terminal Bench tasks
        logger.info("Green agent: Starting Terminal Bench evaluation...")
        timestamp_started = time.time()
        
        try:
            # Import the terminal bench runner
            from green_agent.terminal_bench_runner import GreenAgentTerminalBench
            import asyncio
            
            # Extract config
            task_ids = task_config.get("task_ids")
            dataset_path = Path(task_config["dataset_path"])
            output_directory = Path(task_config.get("output_directory", "results"))
            model_id = task_config.get("model_id", "default_model")
            results_dir = task_config.get("results_dir", "./results")
            max_workers = task_config.get("max_workers", len(white_agent_urls))
            skip_completed = task_config.get("skip_completed", True)
            
            # Create a runner to load tasks (using first white agent)
            tb_runner = GreenAgentTerminalBench(
                white_agent_url=white_agent_urls[0],
                terminal_bench_dataset_path=str(dataset_path),
                model_id=model_id,
                results_dir=results_dir
            )
            
            # Load tasks
            task_tuples = tb_runner.load_terminal_bench_tasks(task_ids, skip_completed=skip_completed)
            
            if not task_tuples:
                logger.info("No tasks to evaluate (all completed or none found)")
                await event_queue.enqueue_event(
                    new_agent_text_message("No tasks to evaluate.")
                )
                return
            
            logger.info(f"Loaded {len(task_tuples)} tasks, distributing across {len(white_agent_urls)} white agents")
            
            # Execute tasks in parallel using asyncio.gather
            async def execute_single_task(task, task_paths, white_url, idx):
                """Execute a single task with a specific white agent"""
                runner = GreenAgentTerminalBench(
                    white_agent_url=white_url,
                    terminal_bench_dataset_path=str(dataset_path),
                    model_id=model_id,
                    results_dir=results_dir
                )
                try:
                    result = await runner.execute_task_with_sandbox(task, task_paths, output_directory)
                    logger.info(f"Task {idx+1}/{len(task_tuples)}: {result.task_id} - {'PASSED' if result.success else 'FAILED'}")
                    return result
                finally:
                    runner.cleanup_resources()
            
            # Create tasks with round-robin white agent assignment
            parallel_tasks = []
            for i, (task, task_paths) in enumerate(task_tuples):
                white_url = white_agent_urls[i % len(white_agent_urls)]
                parallel_tasks.append(execute_single_task(task, task_paths, white_url, i))
            
            # Execute all tasks in parallel with concurrency limit
            logger.info(f"Starting parallel execution with {max_workers} concurrent workers...")
            semaphore = asyncio.Semaphore(max_workers)
            
            async def bounded_task(task):
                async with semaphore:
                    return await task
            
            all_results = await asyncio.gather(*[bounded_task(t) for t in parallel_tasks], return_exceptions=True)
            
            # Filter out exceptions and log them
            successful_results = []
            for i, result in enumerate(all_results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed with exception: {result}")
                else:
                    successful_results.append(result)
            
            all_results = successful_results
                
            # Prepare summary of all results
            total_tasks = len(all_results)
            successful_tasks = sum(1 for r in all_results if r.success)
            total_score = sum(r.evaluation_result.score if r.evaluation_result else 0.0 for r in all_results)
            avg_score = total_score / total_tasks if total_tasks > 0 else 0.0
            total_execution_time = sum(r.execution_time for r in all_results)
            wall_time = time.time() - timestamp_started
            
            # Build detailed results message
            results_summary = []
            for result in all_results:
                result_emoji = "✓" if result.success else "✗"
                score = result.evaluation_result.score if result.evaluation_result else 0.0
                results_summary.append(
                    f"  {result_emoji} {result.task_id}: Score={score:.2f}, Time={result.execution_time:.2f}s"
                )
            
            metrics = {
                "wall_time": wall_time,
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "failed_tasks": total_tasks - successful_tasks,
                "average_score": avg_score,
                "total_execution_time": total_execution_time,
                "parallelization": len(white_agent_urls)
            }
            
            logger.info("Green agent: Parallel evaluation complete.")
            await event_queue.enqueue_event(
                new_agent_text_message(
                    f"Parallel Evaluation Complete!\n\n"
                    f"Summary:\n"
                    f"  Total Tasks: {total_tasks}\n"
                    f"  Successful: {successful_tasks}\n"
                    f"  Failed: {total_tasks - successful_tasks}\n"
                    f"  Average Score: {avg_score:.2f}\n"
                    f"  Wall Time: {wall_time:.2f}s\n"
                    f"  Total Execution Time: {total_execution_time:.2f}s\n"
                    f"  Parallel Workers: {len(white_agent_urls)}\n\n"
                    f"Task Results:\n" + "\n".join(results_summary) + "\n\n"
                    f"Metrics: {json.dumps(metrics, indent=2)}\n"
                )
            )
            
        except Exception as e:
            logger.error(f"Green agent: Evaluation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            await event_queue.enqueue_event(
                new_agent_text_message(
                    f"Evaluation failed: {str(e)}\n"
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def start_green_agent(agent_name="terminal_bench_green_agent", host="localhost", port=9001):
    print("Starting green agent...")
    agent_card_dict = load_agent_card_toml(agent_name)
    
    # Use public URL from environment if available (for AgentBeats/ngrok)
    base_url = os.getenv("AGENT_PUBLIC_URL", f"http://{host}:{port}")
    agent_card_dict["url"] = base_url  # complete all required card fields

    request_handler = DefaultRequestHandler(
        agent_executor=TerminalBenchGreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )
    
    # Build the Starlette app
    starlette_app = app.build()
    
    # Add .well-known/agent-card.json endpoint for AgentBeats compatibility
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    
    agent_card = AgentCard(**agent_card_dict)
    
    async def get_agent_card_json(request):
        return JSONResponse(agent_card.model_dump(mode='json', exclude_none=True))

    async def get_agent_status(request):
        return JSONResponse({"status": "healthy"})

    # Add the agent card route with .json extension
    starlette_app.routes.append(
        Route("/.well-known/agent-card.json", get_agent_card_json, methods=["GET"])
    )
    starlette_app.routes.append(
        Route("/.well-known/agent-card", get_agent_card_json, methods=["GET"])
    )
    starlette_app.routes.append(
        Route("/status", get_agent_status, methods=["GET"])
    )

    uvicorn.run(starlette_app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Green Agent Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9001, help="Port to bind to")
    args = parser.parse_args()
    
    start_green_agent(host=args.host, port=args.port)
