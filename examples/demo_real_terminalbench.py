#!/usr/bin/env python3
"""
Real Terminal-Bench Integration Demo - Runs 3 sample tasks
"""

import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
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

def main():
    """Run 3 Terminal-Bench sample tasks."""
    # Get dataset path using Dataset object (like in test_dataset_import.py)
    dataset_instance = Dataset.__new__(Dataset)  # Create instance without calling __init__
    dataset_path = dataset_instance._get_cache_path("terminal-bench-core", "0.1.1")
    
    if not dataset_path.exists():
        print(f"Dataset not found at: {dataset_path}")
        return
    
    sample_tasks = ["hello-world", "crack-7z-hash", "openssl-selfsigned-cert", "chess-best-move"]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        green_agent = GreenAgentTerminalBench(
            white_agent_url="http://localhost:8001",
            sandbox_base_path=temp_dir,
            terminal_bench_dataset_path=dataset_path
        )
        
        for task_id in sample_tasks:
            tasks = green_agent.load_terminal_bench_tasks([task_id], limit=1)
            if not tasks:
                continue
            
            task = tasks[0]
            result = green_agent.execute_task_with_sandbox(task)
            
            print(f"\nTask: {task_id}")
            print(f"Status: {'PASSED' if result.success else 'FAILED'}")
            print(f"Score: {result.evaluation_result.score:.2f}")
            print(f"Time: {result.execution_time:.2f}s")

if __name__ == "__main__":
    main()
