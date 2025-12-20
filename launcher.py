"""Launcher module - initiates and coordinates the evaluation process."""

import multiprocessing
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
import httpx
from green_agent import start_green_agent
from utils import send_message, wait_agent_ready


async def launch_evaluation(model="gpt-5", task_ids=None, results_dir="./results", max_parallel_tasks=5, max_attempts=1, limit=None, agent_type="evolved"):
    """
    Launch evaluation with configurable settings.
    
    Args:
        model: Model name to use (e.g., "gpt-5-nano", "gpt-5", "gpt-4o")
        task_ids: List of task IDs to evaluate (None = all tasks, [] = default to ["hello-world"])
        results_dir: Directory for JSONL results and outputs (default: "./results")
        max_parallel_tasks: Maximum number of parallel tasks (default: 5)
        max_attempts: Maximum attempts per task before caching (default: 1)
        limit: Limit the number of tasks to run (None = all tasks)
        agent_type: Type of white agent to use ("evolved" or "basic") (default: "evolved")
    """
        
    # Stable results directory for JSONL cache (no timestamp for caching)
    results_base = Path(results_dir)
    results_base.mkdir(parents=True, exist_ok=True)
    
    # Create run-specific output directory with timestamp + UUID
    import uuid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())[:8]
    run_output_dir = results_base / f"{timestamp}_{run_id}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary goes inside the run directory
    summary_dir = run_output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Stable cache directory: {results_base}")
    print(f"JSONL cache: {results_base}/{model.replace('/', '_')}.jsonl")
    print(f"Run output directory: {run_output_dir}")
    print(f"Run summary directory: {summary_dir}")
    
    # Find available ports
    import socket
    
    def find_free_port(start_port=9001, max_tries=10):
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + max_tries):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("localhost", port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"No free ports found in range {start_port}-{start_port+max_tries}")
    
    # start green agent
    print("Launching green agent...")
    green_port = find_free_port(9001)
    green_address = ("localhost", green_port)
    green_url = f"http://{green_address[0]}:{green_address[1]}"
    print(f"Green agent port: {green_port}")
    p_green = multiprocessing.Process(
        target=start_green_agent, args=("terminal_bench_green_agent", *green_address)
    )
    p_green.start()
    assert await wait_agent_ready(green_url), "Green agent not ready in time"
    print("Green agent is ready.")

    # start white agent
    print(f"Launching white agent ({agent_type}) with model={model}...")
    white_port = find_free_port(9011)
    white_address = ("localhost", white_port)
    white_url = f"http://{white_address[0]}:{white_address[1]}"
    print(f"White agent port: {white_port}")
    
    # Import the appropriate white agent based on agent_type
    if agent_type == "evolved":
        from evolved_white_agent import start_white_agent
    elif agent_type == "basic":
        from base_white_agent import start_white_agent
    else:
        raise ValueError(f"Invalid agent_type: {agent_type}. Must be 'evolved' or 'basic'")
    
    if agent_type == "evolved":
        p_white = multiprocessing.Process(
            target=start_white_agent, 
            args=("terminal_bench_white_agent", *white_address),
            kwargs={"model": model, "results_dir": str(results_base)}
        )
    else:
        p_white = multiprocessing.Process(
            target=start_white_agent, 
            args=("terminal_bench_white_agent", *white_address),
            kwargs={"model": model}
        )
        
    p_white.start()
    assert await wait_agent_ready(white_url), "White agent not ready in time"
    print("White agent is ready.")

    # send the task description
    print("Sending task description to green agent...")
    # Pass None through to load all tasks, or pass the list of specific task IDs
    # None means all tasks, list means specific tasks
    
    # Get Terminal Bench dataset path (use cache location for complete dataset)
    tb_cache_path = Path.home() / ".cache" / "terminal-bench" / "terminal-bench-core" / "0.1.1"
    dataset_path = str(tb_cache_path) if tb_cache_path.exists() else "data/tasks"
    
    task_config = {
        "dataset_path": dataset_path,
        "output_directory": str(run_output_dir),  # Run-specific: sessions, agent-logs
        "model_id": model,
        "results_dir": str(results_base),  # Stable: JSONL cache
        "max_parallel_tasks": max_parallel_tasks,
        "max_attempts": max_attempts,
        "limit": limit
    }
    
    if task_ids:
        task_config["task_ids"] = task_ids
        
    task_text = f"""
Your task is to evaluate the white agent located at:
<white_agent_url>
http://{white_address[0]}:{white_address[1]}/
</white_agent_url>
You should use the following task configuration:
<task_config>
{json.dumps(task_config, indent=2)}
</task_config>
    """
    print("Task description:")
    print(task_text)
    print("Sending...")
    # Use very large timeout for launcher since green agent may process multiple tasks
    # 4 hours should be more than enough for any reasonable workload
    response = await send_message(green_url, task_text, timeout=14400.0)
    print("Response from green agent:")
    print(response)

    # Fetch trajectories from white agent before terminating (optional, only for evolved agent)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            trajectories_response = await client.get(f"{white_url}/trajectories")
            if trajectories_response.status_code == 200:
                trajectories_data = trajectories_response.json()
                trajectories = trajectories_data.get("trajectories", {})
                
                # Save trajectories to run-specific directory
                trajectories_file = run_output_dir / "trajectories.json"
                with open(trajectories_file, "w") as f:
                    json.dump(trajectories, f, indent=2)
                print(f"\nSaved trajectories to {trajectories_file}")
                print(f"   Found {len(trajectories)} context IDs with message histories")
            elif trajectories_response.status_code == 404:
                # Base agent doesn't have trajectories endpoint, that's fine
                pass
            else:
                print(f"\nWarning: Failed to fetch trajectories (status {trajectories_response.status_code})")
    except Exception as e:
        # Silently ignore trajectory fetch errors (not critical)
        pass

    # Save evaluation summary
    # Convert response to dict if it has model_dump, otherwise use string representation
    response_dict = None
    if hasattr(response, 'model_dump'):
        try:
            response_dict = response.model_dump(mode='json')
        except:
            response_dict = str(response)
    else:
        response_dict = str(response)
    
    summary = {
        "model": model,
        "task_ids": task_ids,
        "timestamp": timestamp,
        "run_id": run_id,
        "cache_directory": str(results_base),
        "run_output_directory": str(run_output_dir),
        "results_jsonl_cache": str(results_base / f"{model.replace('/', '_')}.jsonl"),
        "green_agent_url": green_url,
        "white_agent_url": white_url,
        "response": response_dict
    }
    summary_file = summary_dir / "evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary to {summary_file}")

    print("\nEvaluation complete. Terminating agents...")
    p_green.terminate()
    p_green.join()
    p_white.terminate()
    p_white.join()
    print("Agents terminated.")
    print(f"\nJSONL cache: {results_base}/{model.replace('/', '_')}.jsonl")
    print(f"Run outputs: {run_output_dir}")
    print(f"   - Sessions: {run_output_dir}/sessions")
    print(f"   - Agent logs: {run_output_dir}/agent-logs")
    print(f"   - Trajectories: {run_output_dir}/trajectories.json")
    print(f"   - Summary: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch terminal bench evaluation")
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-5",
        help="Model name to use (e.g., 'gpt-5-nano', 'gpt-5', 'gpt-4o'). Default: gpt-5"
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        default=[],
        help="Task IDs to evaluate. Default: ['hello-world']. Use --all-tasks to evaluate all tasks."
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Evaluate all tasks in the dataset (overrides --task-ids)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Stable directory for JSONL results cache (e.g., ./results/model.jsonl). Default: ./results"
    )
    parser.add_argument(
        "--max-parallel-tasks",
        type=int,
        default=5,
        help="Maximum number of parallel tasks. Default: 5"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Maximum attempts per task before caching (allows retries). Default: 1"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of tasks to run (e.g., 80 for first 80 tasks). Default: None (all tasks)"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="evolved",
        choices=["evolved", "basic"],
        help="Type of white agent to use: 'evolved' (with reflection) or 'basic' (simple). Default: evolved"
    )
    
    args = parser.parse_args()
    
    # Handle --all-tasks flag
    if args.all_tasks:
        task_ids = None  # None means load all tasks
        print("Will evaluate ALL tasks in the dataset")
    else:
        task_ids = args.task_ids if args.task_ids else ["hello-world"]
    
    asyncio.run(launch_evaluation(
        model=args.model, 
        task_ids=task_ids, 
        results_dir=args.results_dir,
        max_parallel_tasks=args.max_parallel_tasks,
        max_attempts=args.max_attempts,
        limit=args.limit,
        agent_type=args.agent
    ))

