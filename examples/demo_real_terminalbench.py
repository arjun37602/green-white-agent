#!/usr/bin/env python3
"""
Real Terminal-Bench Integration Demo - Runs 3 sample tasks
"""

import sys
import os
import tempfile
from pathlib import Path
from green_agent import GreenAgentTerminalBench

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Run 3 Terminal-Bench sample tasks."""
    dataset_path = "/Users/arjun/.cache/terminal-bench/terminal-bench-core/head"
    if not Path(dataset_path).exists():
        print(f"Dataset not found at: {dataset_path}")
        return
    
    sample_tasks = ["hello-world", "crack-7z-hash", "openssl-selfsigned-cert"]
    
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
