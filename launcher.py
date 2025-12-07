"""Launcher module - initiates and coordinates the evaluation process."""

import multiprocessing
import json
import asyncio
import argparse
from green_agent import start_green_agent
from white_agent import start_white_agent
from utils import send_message, wait_agent_ready


async def launch_evaluation(model="gpt-5", task_ids=None):
    """
    Launch evaluation with configurable settings.
    
    Args:
        model: Model name to use (e.g., "gpt-5-nano", "gpt-5", "gpt-4o")
        task_ids: List of task IDs to evaluate (None = all tasks, [] = default to ["hello-world"])
    """
    # Don't override None here - let it pass through to load all tasks
    # Only set default if it's an empty list (which shouldn't happen, but be safe)
    if task_ids == []:
        task_ids = ["hello-world"]
    # start green agent
    print("Launching green agent...")
    green_address = ("localhost", 9001)
    green_url = f"http://{green_address[0]}:{green_address[1]}"
    p_green = multiprocessing.Process(
        target=start_green_agent, args=("terminal_bench_green_agent", *green_address)
    )
    p_green.start()
    assert await wait_agent_ready(green_url), "Green agent not ready in time"
    print("Green agent is ready.")

    # start white agent
    print(f"Launching white agent with model={model}...")
    white_address = ("localhost", 9002)
    white_url = f"http://{white_address[0]}:{white_address[1]}"
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
    task_config = {
        "task_ids": task_ids,  # None = all tasks, list = specific tasks
        "dataset_path": "data/tasks"
    }
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
    response = await send_message(green_url, task_text)
    print("Response from green agent:")
    print(response)

    print("Evaluation complete. Terminating agents...")
    p_green.terminate()
    p_green.join()
    p_white.terminate()
    p_white.join()
    print("Agents terminated.")


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
        default=None,
        help="Task IDs to evaluate. Default: ['hello-world']. Use --all-tasks to evaluate all tasks."
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Evaluate all tasks in the dataset (overrides --task-ids)"
    )
    
    args = parser.parse_args()
    
    # Handle --all-tasks flag
    if args.all_tasks:
        task_ids = None  # None means load all tasks
        print("Will evaluate ALL tasks in the dataset")
    else:
        task_ids = args.task_ids if args.task_ids else ["hello-world"]
    
    asyncio.run(launch_evaluation(model=args.model, task_ids=task_ids))

