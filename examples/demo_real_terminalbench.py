#!/usr/bin/env python3
"""
Real Terminal-Bench Integration Demo - Runs 3 sample tasks
"""

import sys
import os
import argparse
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock
import requests
from green_agent import GreenAgentTerminalBench

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the problematic imports 
harness_mock = MagicMock()
harness_models_mock = MagicMock()
base_agent_mock = MagicMock()
sys.modules['terminal_bench.harness'] = harness_mock
sys.modules['terminal_bench.harness.harness'] = MagicMock()
sys.modules['terminal_bench.harness.models'] = harness_models_mock
sys.modules['terminal_bench.agents'] = MagicMock()
sys.modules['terminal_bench.agents.base_agent'] = base_agent_mock
sys.modules['terminal_bench.db'] = MagicMock()
harness_mock.Harness = MagicMock()
harness_models_mock.BenchmarkResults = MagicMock()
base_agent_mock.BaseAgent = MagicMock()

# Now we can safely import Dataset
from terminal_bench.dataset.dataset import Dataset

def main(output_directory: str):
    """Run 3 Terminal-Bench sample tasks.
    
    Args:
        output_directory: Directory to store task execution trajectories (default: "trajectories")
    """
    # Get dataset path using Dataset object (like in test_dataset_import.py)
    dataset_instance = Dataset.__new__(Dataset)  # Create instance without calling __init__
    dataset_path = dataset_instance._get_cache_path("terminal-bench-core", "0.1.1")

    if not dataset_path.exists():
        print(f"Dataset not found at: {dataset_path}")
        return
    
    sample_tasks = ["hello-world"]
    
    # Check if white agent server is running
    white_agent_url = "http://localhost:8001"
    try:
        response = requests.get(f"{white_agent_url}/health", timeout=5)
        if response.status_code != 200:
            raise ValueError(
                f"White agent server at {white_agent_url} returned status {response.status_code}. "
                f"Make sure the server is running with: python -m white_agent.agent --server --port 8001"
            )
        health_data = response.json()
        if health_data.get("status") != "healthy":
            raise ValueError(
                f"White agent server at {white_agent_url} is not healthy: {health_data}"
            )
        print(f"White agent server is running: {health_data.get('agent', 'unknown')}")
    except requests.exceptions.ConnectionError:
        raise ValueError(
            f"Cannot connect to white agent server at {white_agent_url}. "
            f"Make sure the server is running with: python -m white_agent.agent --server --port 8001"
        )
    except requests.exceptions.Timeout:
        raise ValueError(
            f"White agent server at {white_agent_url} timed out. "
            f"The server may be unresponsive."
        )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        green_agent = GreenAgentTerminalBench(
            white_agent_url=white_agent_url,
            sandbox_base_path=temp_dir,
            terminal_bench_dataset_path=dataset_path,
            trajectory_output_dir=output_directory
        )
        
        try:
            for task_id in sample_tasks:
                tasks = green_agent.load_terminal_bench_tasks([task_id], limit=1)
                if not tasks:
                    raise ValueError(f"No tasks found for: {task_id}")
                
                task = tasks[0]
                
                result = green_agent.execute_task_with_sandbox(task)
                
                if result.success:
                    print("PASSED")
                else:
                    print("FAILED")
                
                if result.evaluation_result:
                    print(f"Score: {result.evaluation_result.score:.2f}")
                else:
                    print(f"Score: N/A (task failed to complete)")
                
                print(f"Time: {result.execution_time:.2f}s")
                print(f"Commands executed: {result.commands_executed}")
        
        finally:
            # Cleanup all resources (containers and images)
            print(f"Cleaning up Docker resources...")
            green_agent.cleanup_resources()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Terminal-Bench sample tasks with trajectory logging"
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        default="trajectories",
        help="Directory to store task execution trajectories (default: trajectories)"
    )
    args = parser.parse_args()
    
    main(output_directory=args.output_directory)
