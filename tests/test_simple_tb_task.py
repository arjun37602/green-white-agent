#!/usr/bin/env python3
"""
Simple test script to run a Terminal Bench task with token tracking
"""

import sys
import os
import tempfile
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green_agent import GreenAgentTerminalBench

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_simple_tb_task():
    """Test with a simple Terminal Bench task."""
    print("=" * 80)
    print("Testing Simple Terminal Bench Task")
    print("=" * 80)
    
    # Check if Terminal-Bench dataset is available
    dataset_paths = [
        os.path.expanduser("~/.cache/terminal-bench/terminal-bench-core/head"),
        os.path.expanduser("~/.cache/terminal-bench/terminal-bench-core"),
        "/tmp/terminal-bench-core/head",
    ]
    
    dataset_path = None
    for path in dataset_paths:
        if Path(path).exists():
            dataset_path = path
            break
    
    if not dataset_path:
        print(f"❌ Terminal-Bench dataset not found.")
        print(f"   Checked paths:")
        for path in dataset_paths:
            print(f"   - {path}")
        print(f"\n   To download: tb datasets download terminal-bench-core")
        print(f"   Or manually download from: https://github.com/tianjianlu/terminal-bench")
        
        # Create a minimal test task instead
        print(f"\n📝 Creating a simple test task instead...")
        return test_with_demo_task()
    
    print(f"✅ Found Terminal-Bench dataset at: {dataset_path}")
    
    # Simple task IDs to try (these are typically the simplest)
    simple_task_ids = ["hello-world", "echo-hello", "create-file", "file-creation"]
    
    # Initialize green agent with dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        green_agent = GreenAgentTerminalBench(
            white_agent_url="http://localhost:8002",
            sandbox_base_path=temp_dir,
            terminal_bench_dataset_path=dataset_path
        )
        
        print(f"\n📋 Looking for simple tasks...")
        available_tasks = green_agent.tb_loader.list_available_tasks() if green_agent.tb_loader else []
        print(f"   Found {len(available_tasks)} available tasks")
        
        # Try to find a simple task
        task_to_run = None
        for task_id in simple_task_ids:
            if task_id in available_tasks:
                task_to_run = task_id
                break
        
        # If no simple task found, use the first available one
        if not task_to_run and available_tasks:
            task_to_run = available_tasks[0]
        
        if not task_to_run:
            print("❌ No tasks found in dataset")
            return test_with_demo_task()
        
        print(f"\n🚀 Running task: {task_to_run}")
        
        # Load the specific task
        tasks = green_agent.load_terminal_bench_tasks(task_ids=[task_to_run])
        
        if not tasks:
            print(f"❌ Failed to load task {task_to_run}")
            return test_with_demo_task()
        
        task = tasks[0]
        print(f"   Instruction: {task['instruction'][:100]}...")
        
        # Execute the task
        print(f"\n📤 Executing task...")
        result = green_agent.execute_task_with_sandbox(task)
        
        # Display results
        print(f"\n📊 Results:")
        print(f"   ✅ Success: {result.success}")
        print(f"   📈 Score: {result.evaluation_result.score:.2f}")
        print(f"   ⏱️  Performance Score: {result.evaluation_result.performance.get('performance_score', 0.0):.2f}")
        print(f"   🔢 Tokens Per Request: {result.evaluation_result.details.get('tokens_per_request', 'N/A')}")
        print(f"   📈 Total Tokens: {result.evaluation_result.details.get('total_tokens', 'N/A')}")
        print(f"   📞 Total Requests: {result.evaluation_result.details.get('total_requests', 'N/A')}")
        print(f"   ⏱️  Execution Time: {result.execution_time:.2f}s")
        print(f"   🔧 Commands Executed: {result.commands_executed}")
        
        if result.evaluation_result:
            print(f"\n   Correctness: {'✅' if result.evaluation_result.correctness.get('passed') else '❌'}")
            print(f"   Safety: {'✅' if result.evaluation_result.safety.get('safe') else '❌'}")
            print(f"   Efficiency: {'✅' if result.evaluation_result.efficiency.get('efficient') else '❌'}")
        
        print(f"\n✅ Test completed!")
        return result


def test_with_demo_task():
    """Test with a demo task if Terminal Bench dataset is not available."""
    print("=" * 80)
    print("Testing with Demo Task (Terminal Bench dataset not available)")
    print("=" * 80)
    
    # Create a simple demo task
    demo_task = {
        "id": "simple-demo-task",
        "description": "Simple demo task for testing",
        "instruction": "Create a file called 'solution.txt' with the content 'hello world'",
        "test": "Check if solution.txt exists and contains 'hello world'",
        "environment": {
            "working_directory": "/app"
        },
        "metadata": {
            "validation": {
                "max_tokens": 5000,
                "max_tokens_per_request": 2000
            }
        }
    }
    
    print(f"\n📋 Demo Task: {demo_task['id']}")
    print(f"   Instruction: {demo_task['instruction']}")
    
    # Initialize green agent
    with tempfile.TemporaryDirectory() as temp_dir:
        green_agent = GreenAgentTerminalBench(
            white_agent_url="http://localhost:8002",
            sandbox_base_path=temp_dir
        )
        
        print(f"\n🚀 Executing demo task...")
        result = green_agent.execute_task_with_sandbox(demo_task)
        
        # Display results
        print(f"\n📊 Results:")
        print(f"   ✅ Success: {result.success}")
        print(f"   📈 Score: {result.evaluation_result.score:.2f}")
        print(f"   ⏱️  Performance Score: {result.evaluation_result.performance.get('performance_score', 0.0):.2f}")
        print(f"   🔢 Tokens Per Request: {result.evaluation_result.details.get('tokens_per_request', 'N/A')}")
        print(f"   📈 Total Tokens: {result.evaluation_result.details.get('total_tokens', 'N/A')}")
        print(f"   📞 Total Requests: {result.evaluation_result.details.get('total_requests', 'N/A')}")
        print(f"   ⏱️  Execution Time: {result.execution_time:.2f}s")
        print(f"   🔧 Commands Executed: {result.commands_executed}")
        
        print(f"\n✅ Test completed!")
        return result


if __name__ == "__main__":
    try:
        result = test_simple_tb_task()
        if result and result.success:
            print("\n🎉 Test passed!")
        else:
            print("\n⚠️  Test completed but task did not pass")
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

