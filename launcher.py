"""Launcher module - initiates and coordinates the evaluation process with parallel execution."""

import multiprocessing
import json
import asyncio
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

from green_agent.terminal_bench_runner import GreenAgentTerminalBench
from green_agent.results_store import ResultsStore
from terminal_bench.handlers.trial_handler import Task, TaskPaths
from white_agent import start_white_agent
from utils import wait_agent_ready


async def evaluate_single_task(
    white_agent_url: str,
    model_id: str,
    task: Task,
    task_paths: TaskPaths,
    dataset_path: Path,
    results_dir: Path,
    output_dir: Path
):
    """Evaluate a single task (called in parallel)"""
    import tempfile
    
    # Create a new runner instance for this task
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = GreenAgentTerminalBench(
            white_agent_url=white_agent_url,
            sandbox_base_path=temp_dir,
            terminal_bench_dataset_path=str(dataset_path),
            model_id=model_id,
            results_dir=str(results_dir)
        )
        
        try:
            result = await runner.execute_task_with_sandbox(task, task_paths, output_dir)
            return result
        finally:
            runner.cleanup_resources()


def run_task_sync(args):
    """Wrapper to run async task in thread pool"""
    white_agent_url, model_id, task, task_paths, dataset_path, results_dir, output_dir = args
    return asyncio.run(
        evaluate_single_task(
            white_agent_url, model_id, task, task_paths, 
            dataset_path, results_dir, output_dir
        )
    )


async def launch_evaluation(
    task_ids: Optional[List[str]] = None,
    dataset_path: str = "data/tasks",
    model_id: str = "gpt-4",
    results_dir: str = "./results",
    output_dir: str = "./output",
    max_workers: int = 4,
    skip_completed: bool = True
):
    """
    Launch parallel evaluation of multiple tasks.
    
    Args:
        task_ids: List of task IDs to evaluate (None = all tasks)
        dataset_path: Path to Terminal Bench dataset
        model_id: Model identifier for results storage
        results_dir: Directory for JSONL results
        output_dir: Directory for execution logs
        max_workers: Maximum number of parallel workers
        skip_completed: Skip tasks that are already completed
    """
    dataset_path = Path(dataset_path)
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    
    # Ensure directories exist
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start white agent
    print(f"Launching white agent for model: {model_id}...")
    white_address = ("localhost", 9002)
    white_url = f"http://{white_address[0]}:{white_address[1]}"
    p_white = multiprocessing.Process(
        target=start_white_agent, args=("terminal_bench_white_agent", *white_address)
    )
    p_white.start()
    
    try:
        assert await wait_agent_ready(white_url), "White agent not ready in time"
        print("White agent is ready.")
        
        # Load tasks and check which ones are completed
        print(f"Loading tasks from {dataset_path}...")
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = GreenAgentTerminalBench(
                white_agent_url=white_url,
                sandbox_base_path=temp_dir,
                terminal_bench_dataset_path=str(dataset_path),
                model_id=model_id,
                results_dir=str(results_dir)
            )
            
            task_tuples = loader.load_terminal_bench_tasks(task_ids, skip_completed=skip_completed)
        
        if not task_tuples:
            print("No tasks to evaluate (all completed or none found).")
            return
        
        print(f"\n{'='*60}")
        print(f"Starting parallel evaluation:")
        print(f"  Model: {model_id}")
        print(f"  Tasks to evaluate: {len(task_tuples)}")
        print(f"  Max parallel workers: {max_workers}")
        print(f"  Results: {results_dir / f'{model_id}.jsonl'}")
        print(f"{'='*60}\n")
        
        # Prepare task arguments
        task_args = [
            (white_url, model_id, task, task_paths, dataset_path, results_dir, output_dir)
            for task, task_paths in task_tuples
        ]
        
        # Execute tasks in parallel using ThreadPoolExecutor
        results = []
        completed_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(run_task_sync, args): args[2]  # args[2] is the task
                for args in task_args
            }
            
            # Process as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        completed_count += 1
                        status = "✓ PASSED"
                    else:
                        failed_count += 1
                        status = "✗ FAILED"
                    
                    print(f"{status} | {result.task_id} | {result.execution_time:.2f}s | "
                          f"Progress: {completed_count + failed_count}/{len(task_tuples)}")
                    
                except Exception as e:
                    failed_count += 1
                    print(f"✗ ERROR | Task failed with exception: {e}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Evaluation Complete!")
        print(f"{'='*60}")
        print(f"  Total tasks: {len(task_tuples)}")
        print(f"  Passed: {completed_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Success rate: {completed_count / len(task_tuples) * 100:.1f}%")
        print(f"  Results saved to: {results_dir / f'{model_id}.jsonl'}")
        print(f"{'='*60}\n")
        
        # Show summary stats
        results_store = ResultsStore(str(results_dir))
        stats = results_store.get_summary_stats(model_id)
        print(f"Overall Statistics for {model_id}:")
        print(f"  Average Accuracy: {stats['avg_accuracy']:.2%}")
        print(f"  Average Tokens: {stats['avg_tokens']:.0f}")
        print(f"  Average Turns: {stats['avg_turns']:.1f}")
        print(f"  Total Execution Time: {stats['total_execution_time']:.2f}s")
        
    finally:
        print("\nTerminating white agent...")
        p_white.terminate()
        p_white.join()
        print("White agent terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel evaluation launcher for Terminal Bench")
    parser.add_argument("--task-ids", nargs="*", default=None, 
                       help="Specific task IDs to evaluate (default: all tasks)")
    parser.add_argument("--dataset-path", default="data/tasks",
                       help="Path to Terminal Bench dataset")
    parser.add_argument("--model-id", default="gpt-4",
                       help="Model identifier for results")
    parser.add_argument("--results-dir", default="./results",
                       help="Directory for JSONL results")
    parser.add_argument("--output-dir", default="./output",
                       help="Directory for execution logs")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum number of parallel workers")
    parser.add_argument("--no-skip-completed", action="store_true",
                       help="Don't skip completed tasks (re-evaluate)")
    
    args = parser.parse_args()
    
    asyncio.run(launch_evaluation(
        task_ids=args.task_ids,
        dataset_path=args.dataset_path,
        model_id=args.model_id,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        skip_completed=not args.no_skip_completed
    ))

