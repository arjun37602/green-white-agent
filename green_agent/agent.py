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
        white_agent_url = tags["white_agent_url"]
        task_config_str = tags.get("task_config", "{}")
        task_config = json.loads(task_config_str)
        
        logger.info(f"Green agent: White agent URL: {white_agent_url}")
        logger.info(f"Green agent: Task config: {task_config}")
        
        # Load and execute Terminal Bench tasks
        logger.info("Green agent: Starting Terminal Bench evaluation...")
        timestamp_started = time.time()
        
        try:
            # Import the terminal bench runner
            from green_agent.terminal_bench_runner import GreenAgentTerminalBench
            import tempfile
            
            # Extract config
            task_ids = task_config.get("task_ids", None)
            dataset_path = Path.home() / ".cache" / "terminal-bench" / "terminal-bench-core" / "0.1.1"
            output_directory = Path(task_config.get("output_directory", "results_base_ab/"))
            model_id = task_config.get("model_id", "gpt-5-nano")
            results_dir = task_config.get("results_dir", "./results_base_ab")
            max_parallel_tasks = task_config.get("max_parallel_tasks", 20)
            attempts = task_config.get("attempts", 1)
            limit = task_config.get("limit", 80)
            
            # Create terminal bench runner
            with tempfile.TemporaryDirectory() as temp_dir:
                tb_runner = GreenAgentTerminalBench(
                    white_agent_url=white_agent_url,
                    sandbox_base_path=temp_dir,
                    terminal_bench_dataset_path=dataset_path,
                    model_id=model_id,
                    results_dir=results_dir,
                    max_parallel_tasks=max_parallel_tasks,
                    attempts=attempts
                )
                
                # Load tasks
                task_tuples = tb_runner.load_terminal_bench_tasks(task_ids)
                # Apply limit if specified
                if limit is not None:
                    task_tuples = task_tuples[:limit]
                if not task_tuples:
                    raise ValueError(f"No tasks found")
                
                # Execute all tasks in parallel with caching
                all_results = await tb_runner.execute_multiple_tasks_parallel(task_tuples, output_directory)
                
                # Cleanup
                tb_runner.cleanup_resources()
                
                # Prepare summary of all results
                total_tasks = len(all_results)
                successful_tasks = sum(1 for r in all_results if r.success)
                failed_tasks = total_tasks - successful_tasks
                success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
                total_accuracy = sum(r.evaluation_result.accuracy if r.evaluation_result else 0.0 for r in all_results)
                avg_accuracy = total_accuracy / total_tasks if total_tasks > 0 else 0.0
                total_execution_time = sum(r.execution_time for r in all_results)
                
                # Build detailed results message
                results_summary = []
                for result in all_results:
                    result_emoji = "SUCCESS" if result.success else "FAILURE"
                    accuracy = result.evaluation_result.accuracy if result.evaluation_result else 0.0
                    results_summary.append(
                        f"  {result.task_id}: {result_emoji} (Accuracy: {accuracy:.2f}, Time: {result.execution_time:.2f}s)"
                    )
                
                metrics = {
                    "time_used": time.time() - timestamp_started,
                    "total_tasks": total_tasks,
                    "successful_tasks": successful_tasks,
                    "failed_tasks": failed_tasks,
                    "success_rate": success_rate,
                    "average_accuracy": avg_accuracy,
                    "total_execution_time": total_execution_time
                }
                
                logger.info("Green agent: Evaluation complete.")
                await event_queue.enqueue_event(
                    new_agent_text_message(
                        f"Evaluation Complete!\n\n"
                        f"Summary:\n"
                        f"  Total Tasks: {total_tasks}\n"
                        f"  Successful: {successful_tasks} ({success_rate:.1%})\n"
                        f"  Failed: {failed_tasks}\n"
                        f"  Average Accuracy: {avg_accuracy:.2f}\n"
                        f"  Total Execution Time: {total_execution_time:.2f}s\n\n"
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
    # Otherwise use local URL for local execution
    base_url = os.getenv("AGENT_URL") or f"http://{host}:{port}"
    agent_card_dict["url"] = base_url  # complete all required card fields

    request_handler = DefaultRequestHandler(
        agent_executor=TerminalBenchGreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Green Agent Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9001, help="Port to bind to")
    args = parser.parse_args()
    
    start_green_agent(host=args.host, port=args.port)
