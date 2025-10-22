#!/usr/bin/env python3
"""
Green Agent Demo Script

This script demonstrates the enhanced Green Agent capabilities including:
- Sandbox environment management
- Task execution and evaluation
- Comprehensive reporting
"""

import sys
import os
import tempfile
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from green_agent import GreenAgentTerminalBench

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_sandbox_isolation():
    """Demonstrate sandbox isolation capabilities."""
    print("🔒 DEMO: Sandbox Isolation")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        green_agent = GreenAgentTerminalBench(sandbox_base_path=temp_dir)
        
        # Create a sandbox
        sandbox_id = green_agent.sandbox_manager.create_sandbox("demo_task")
        print(f"✅ Created isolated sandbox: {sandbox_id}")
        
        # Execute commands in sandbox
        commands = [
            "echo 'Hello from sandbox!'",
            "mkdir demo_files",
            "echo 'This is a test file' > demo_files/test.txt",
            "ls -la demo_files",
            "cat demo_files/test.txt"
        ]
        
        print("\n📝 Executing commands in sandbox:")
        for cmd in commands:
            result = green_agent.sandbox_manager.execute_command(sandbox_id, cmd)
            status = "✅" if result.success else "❌"
            print(f"  {status} {cmd}")
            if result.stdout.strip():
                print(f"    Output: {result.stdout.strip()}")
        
        # Capture sandbox state
        state = green_agent.sandbox_manager.capture_state(sandbox_id)
        print(f"\n📊 Sandbox state captured:")
        print(f"  Files: {len(state.file_system_snapshot['files'])}")
        print(f"  Directories: {len(state.file_system_snapshot['directories'])}")
        
        # Clean up
        green_agent.sandbox_manager.destroy_sandbox(sandbox_id)
        print(f"\n🧹 Sandbox cleaned up")
        
        print("✅ Sandbox isolation demo completed!\n")


def demo_task_evaluation():
    """Demonstrate task evaluation capabilities."""
    print("📊 DEMO: Task Evaluation")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        green_agent = GreenAgentTerminalBench(sandbox_base_path=temp_dir)
        
        # Create a mock task
        task = {
            "id": "demo_evaluation_task",
            "instruction": "Create a solution.txt file with specific content",
            "test": "Check if solution.txt exists and contains 'Hello World'",
            "environment": {
                "working_directory": "/app"
            }
        }
        
        print(f"📋 Task: {task['id']}")
        print(f"   Instruction: {task['instruction']}")
        print(f"   Test: {task['test']}")
        
        # Create sandbox and execute task
        sandbox_id = green_agent.sandbox_manager.create_sandbox(task['id'], task['environment'])
        
        # Simulate task execution
        commands = [
            "echo 'Hello World' > solution.txt",
            "cat solution.txt",
            "ls -la solution.txt"
        ]
        
        print(f"\n🔧 Executing task commands:")
        command_results = []
        for cmd in commands:
            result = green_agent.sandbox_manager.execute_command(sandbox_id, cmd)
            command_results.append(result)
            status = "✅" if result.success else "❌"
            print(f"  {status} {cmd}")
            if result.stdout.strip():
                print(f"    Output: {result.stdout.strip()}")
        
        # Capture final state
        final_state = green_agent.sandbox_manager.capture_state(sandbox_id)
        
        # Evaluate task
        evaluation_result = green_agent.task_evaluator.evaluate(
            task['id'], task, command_results, final_state
        )
        
        print(f"\n📊 Evaluation Results:")
        print(f"  Passed: {'✅' if evaluation_result.passed else '❌'}")
        print(f"  Score: {evaluation_result.score:.2f}")
        print(f"  Correctness: {'✅' if evaluation_result.correctness['passed'] else '❌'}")
        print(f"  Performance: {evaluation_result.performance['total_time']:.2f}s")
        print(f"  Safety: {'✅' if evaluation_result.safety['safe'] else '❌'}")
        print(f"  Efficiency: {'✅' if evaluation_result.efficiency['efficient'] else '❌'}")
        
        # Clean up
        green_agent.sandbox_manager.destroy_sandbox(sandbox_id)
        
        print("✅ Task evaluation demo completed!\n")


def demo_complete_workflow():
    """Demonstrate complete Green Agent workflow."""
    print("🚀 DEMO: Complete Workflow")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        green_agent = GreenAgentTerminalBench(sandbox_base_path=temp_dir)
        
        # Load sample tasks
        tasks = green_agent.load_terminal_bench_tasks(limit=2)
        print(f"📋 Loaded {len(tasks)} sample tasks")
        
        for i, task in enumerate(tasks, 1):
            print(f"\n📝 Task {i}: {task['id']}")
            print(f"   Description: {task['description']}")
            
            # Execute task with sandbox
            execution_result = green_agent.execute_task_with_sandbox(task)
            
            print(f"   Result: {'✅ PASSED' if execution_result.success else '❌ FAILED'}")
            print(f"   Score: {execution_result.evaluation_result.score:.2f}")
            print(f"   Execution Time: {execution_result.execution_time:.2f}s")
            print(f"   Commands Executed: {execution_result.commands_executed}")
        
        # Get execution history
        history = green_agent.get_execution_history()
        print(f"\n📊 Execution History:")
        print(f"   Total Tasks Executed: {len(history)}")
        print(f"   Success Rate: {sum(1 for h in history if h.success) / len(history) * 100:.1f}%")
        
        # Clean up
        green_agent.cleanup_resources()
        
        print("✅ Complete workflow demo completed!\n")


def main():
    """Run all demos."""
    print("🎯 Green Agent Enhanced Capabilities Demo")
    print("=" * 60)
    print("This demo showcases the enhanced Green Agent with:")
    print("• Sandbox environment isolation")
    print("• Comprehensive task evaluation")
    print("• Complete workflow management")
    print("• Resource cleanup and monitoring")
    print("=" * 60)
    
    try:
        # Run demos
        demo_sandbox_isolation()
        demo_task_evaluation()
        demo_complete_workflow()
        
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("✅ Green Agent is fully operational and ready for TerminalBench evaluation")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
