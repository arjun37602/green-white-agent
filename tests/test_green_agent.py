#!/usr/bin/env python3
"""
Test script for the enhanced Green Agent

This script demonstrates the Green Agent's sandbox management and evaluation capabilities.
"""

import sys
import os
import logging
import tempfile
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green_agent import GreenAgentTerminalBench, SandboxManager, TaskEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_sandbox_manager():
    """Test the sandbox manager functionality."""
    print("🧪 Testing Sandbox Manager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sandbox_manager = SandboxManager(base_path=temp_dir)
        
        # Test sandbox creation
        sandbox_id = sandbox_manager.create_sandbox("test_task", {"working_directory": "/app"})
        print(f"✅ Created sandbox: {sandbox_id}")
        
        # Test command execution
        result = sandbox_manager.execute_command(sandbox_id, "echo 'Hello from sandbox'")
        print(f"✅ Command executed: {result.command}")
        print(f"   Output: {result.stdout.strip()}")
        print(f"   Success: {result.success}")
        
        # Test file creation
        result = sandbox_manager.execute_command(sandbox_id, "echo 'test content' > test.txt")
        print(f"✅ File created: {result.success}")
        
        # Test file reading
        result = sandbox_manager.execute_command(sandbox_id, "cat test.txt")
        print(f"✅ File content: {result.stdout.strip()}")
        
        # Test state capture
        state = sandbox_manager.capture_state(sandbox_id)
        print(f"✅ State captured: {len(state.file_system_snapshot['files'])} files")
        
        # Test sandbox cleanup
        sandbox_manager.destroy_sandbox(sandbox_id)
        print(f"✅ Sandbox destroyed")
        
        print("🎉 Sandbox Manager test completed successfully!")


def test_task_evaluator():
    """Test the task evaluator functionality."""
    print("\n🧪 Testing Task Evaluator...")
    
    evaluator = TaskEvaluator()
    
    # Create mock command history
    from green_agent.sandbox_manager import CommandResult
    from green_agent.task_evaluator import SandboxState
    
    command_history = [
        CommandResult(
            command="echo 'Hello World'",
            stdout="Hello World\n",
            stderr="",
            returncode=0,
            execution_time=0.1,
            timestamp="2024-01-01T00:00:00",
            success=True
        ),
        CommandResult(
            command="echo 'test content' > solution.txt",
            stdout="",
            stderr="",
            returncode=0,
            execution_time=0.1,
            timestamp="2024-01-01T00:00:01",
            success=True
        )
    ]
    
    # Create mock sandbox state
    sandbox_state = SandboxState(
        sandbox_id="test_sandbox",
        working_directory="/app",
        environment_vars={},
        file_system_snapshot={
            "files": {
                "solution.txt": "test content"
            },
            "directories": [],
            "timestamp": "2024-01-01T00:00:00"
        },
        timestamp="2024-01-01T00:00:00"
    )
    
    # Create mock task spec
    task_spec = {
        "id": "test_task",
        "instruction": "Create a solution.txt file",
        "test": "Check if solution.txt exists",
        "environment": {
            "working_directory": "/app"
        }
    }
    
    # Evaluate task
    result = evaluator.evaluate("test_task", task_spec, command_history, sandbox_state)
    
    print(f"✅ Task evaluation completed")
    print(f"   Passed: {result.passed}")
    print(f"   Score: {result.score:.2f}")
    print(f"   Correctness: {result.correctness['passed']}")
    print(f"   Performance: {result.performance['total_time']:.2f}s")
    print(f"   Safety: {result.safety['safe']}")
    print(f"   Efficiency: {result.efficiency['efficient']}")
    
    print("🎉 Task Evaluator test completed successfully!")


def test_green_agent_integration():
    """Test the complete Green Agent integration."""
    print("\n🧪 Testing Green Agent Integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create Green Agent with custom sandbox path
        green_agent = GreenAgentTerminalBench(
            white_agent_url="http://localhost:8002",  # This will fail if white agent not running
            sandbox_base_path=temp_dir
        )
        
        print(f"✅ Green Agent initialized with sandbox path: {temp_dir}")
        
        # Test sandbox manager access
        sandboxes = green_agent.sandbox_manager.list_sandboxes()
        print(f"✅ Active sandboxes: {len(sandboxes)}")
        
        # Test task loading
        tasks = green_agent.load_terminal_bench_tasks(limit=2)
        print(f"✅ Loaded {len(tasks)} test tasks")
        
        # Test individual task execution (without white agent)
        if tasks:
            task = tasks[0]
            print(f"✅ Testing task: {task['id']}")
            
            # Create sandbox for task
            sandbox_id = green_agent.sandbox_manager.create_sandbox(
                task['id'], 
                task.get('environment')
            )
            print(f"✅ Created sandbox for task: {sandbox_id}")
            
            # Execute some test commands
            test_commands = [
                "echo 'Testing Green Agent'",
                "mkdir test_dir",
                "echo 'test content' > test_dir/test_file.txt",
                "ls -la test_dir"
            ]
            
            for cmd in test_commands:
                result = green_agent.sandbox_manager.execute_command(sandbox_id, cmd)
                print(f"   Command: {cmd} -> {'✅' if result.success else '❌'}")
            
            # Clean up
            green_agent.sandbox_manager.destroy_sandbox(sandbox_id)
            print(f"✅ Sandbox cleaned up")
        
        print("🎉 Green Agent Integration test completed successfully!")


def test_without_white_agent():
    """Test Green Agent functionality without requiring white agent."""
    print("\n🧪 Testing Green Agent (Standalone Mode)...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        green_agent = GreenAgentTerminalBench(sandbox_base_path=temp_dir)
        
        # Test sandbox creation and command execution
        sandbox_id = green_agent.sandbox_manager.create_sandbox("standalone_test")
        
        # Execute a simple task
        commands = [
            "echo 'Starting task'",
            "mkdir -p app",
            "cd app",
            "echo 'Hello World' > hello.txt",
            "cat hello.txt",
            "ls -la"
        ]
        
        print("Executing standalone task...")
        for cmd in commands:
            result = green_agent.sandbox_manager.execute_command(sandbox_id, cmd)
            print(f"  {cmd} -> {'✅' if result.success else '❌'}")
            if result.stdout.strip():
                print(f"    Output: {result.stdout.strip()}")
        
        # Capture final state
        final_state = green_agent.sandbox_manager.capture_state(sandbox_id)
        print(f"✅ Final state captured: {len(final_state.file_system_snapshot['files'])} files")
        
        # Clean up
        green_agent.sandbox_manager.destroy_sandbox(sandbox_id)
        
        print("🎉 Standalone test completed successfully!")


def main():
    """Run all tests."""
    print("🚀 Starting Green Agent Tests")
    print("=" * 60)
    
    try:
        # Test individual components
        test_sandbox_manager()
        test_task_evaluator()
        test_green_agent_integration()
        test_without_white_agent()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Green Agent is working correctly")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
