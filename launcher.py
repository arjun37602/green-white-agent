"""Launcher module - initiates and coordinates parallel evaluation with multiple white agents."""

import multiprocessing
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from green_agent import start_green_agent
from white_agent import start_white_agent
from utils import send_message, wait_agent_ready


async def launch_evaluation(
    model: str = "gpt-4",
    task_ids: Optional[List[str]] = None,
    dataset_path: str = "data/tasks",
    results_dir: str = "./results",
    max_workers: int = 4,
    skip_completed: bool = True
):
    """
    Launch parallel evaluation with multiple white agents via A2A protocol.
    
    Architecture:
    - Starts N white agents on different ports (parallel)
    - Starts 1 green agent (A2A server)
    - Sends task config to green agent with list of white agent URLs
    - Green agent internally parallelizes task execution across white agents
    
    Args:
        model: Model name for white agents
        task_ids: List of task IDs (None = all tasks)
        dataset_path: Path to Terminal Bench dataset
        results_dir: Directory for JSONL results
        max_workers: Number of parallel white agents
        skip_completed: Skip already completed tasks
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(results_dir) / f"eval_{timestamp}_{model.replace('/', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Parallel Evaluation Setup")
    print(f"{'='*60}")
    print(f"  Model: {model}")
    print(f"  Workers: {max_workers}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Start white agents in parallel
    print(f"Starting {max_workers} white agents...")
    white_processes = []
    white_urls = []
    
    for i in range(max_workers):
        port = 9002 + i
        white_url = f"http://localhost:{port}"
        p = multiprocessing.Process(
            target=start_white_agent,
            args=("terminal_bench_white_agent", "localhost", port),
            kwargs={"model": model}
        )
        p.start()
        white_processes.append(p)
        white_urls.append(white_url)
    
    try:
        # Wait for white agents in parallel (don't block on each one sequentially)
        print("Waiting for white agents to be ready...")
        ready_tasks = [wait_agent_ready(url, timeout=30) for url in white_urls]
        ready_results = await asyncio.gather(*ready_tasks)
        
        if not all(ready_results):
            failed_indices = [i for i, r in enumerate(ready_results) if not r]
            raise RuntimeError(f"White agents {failed_indices} failed to start")
        
        print(f"✓ All {max_workers} white agents ready!\n")
        
        # Start green agent
        print("Starting green agent...")
        green_address = ("localhost", 9001)
        green_url = f"http://{green_address[0]}:{green_address[1]}"
        p_green = multiprocessing.Process(
            target=start_green_agent,
            args=("terminal_bench_green_agent", *green_address)
        )
        p_green.start()
        
        if not await wait_agent_ready(green_url, timeout=30):
            raise RuntimeError("Green agent not ready in time")
        print("✓ Green agent ready!\n")
        
        # Send task config to green agent with all white agent URLs
        print("Sending task configuration to green agent...")
        task_config = {
            "task_ids": task_ids,
            "dataset_path": dataset_path,
            "output_directory": str(output_dir),
            "white_agent_urls": white_urls,  # List of all white agent URLs
            "model_id": model,
            "results_dir": results_dir,
            "max_workers": max_workers,
            "skip_completed": skip_completed
        }
        
        task_text = f"""
Your task is to evaluate white agents in parallel.

<white_agent_urls>
{json.dumps(white_urls, indent=2)}
</white_agent_urls>

<task_config>
{json.dumps(task_config, indent=2)}
</task_config>

Use parallel execution to distribute tasks across the available white agents.
"""
        
        print("Sending to green agent...")
        response = await send_message(green_url, task_text)
        print("\n✓ Evaluation complete!")
        print(f"\nGreen agent response:")
        print(response)
        
        print(f"\nResults saved to: {output_dir}")
        
    finally:
        print("\nTerminating agents...")
        if 'p_green' in locals():
            p_green.terminate()
            p_green.join(timeout=5)
            print("  ✓ Green agent terminated")
        
        for i, p in enumerate(white_processes):
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
        print(f"  ✓ All {max_workers} white agents terminated\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel evaluation with multiple white agents via A2A"
    )
    parser.add_argument("--model", default="gpt-4", help="Model name")
    parser.add_argument("--task-ids", nargs="*", default=None, 
                       help="Task IDs to evaluate (default: all tasks)")
    parser.add_argument("--dataset-path", default="data/tasks",
                       help="Path to Terminal Bench dataset")
    parser.add_argument("--results-dir", default="./results",
                       help="Directory for results")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Number of parallel white agents")
    parser.add_argument("--no-skip-completed", action="store_true",
                       help="Don't skip completed tasks")
    parser.add_argument("--all-tasks", action="store_true",
                       help="Evaluate all tasks (overrides --task-ids)")
    
    args = parser.parse_args()
    
    # Handle --all-tasks flag
    if args.all_tasks:
        task_ids = None
    else:
        task_ids = args.task_ids
    
    asyncio.run(launch_evaluation(
        model=args.model,
        task_ids=task_ids,
        dataset_path=args.dataset_path,
        results_dir=args.results_dir,
        max_workers=args.max_workers,
        skip_completed=not args.no_skip_completed
    ))

