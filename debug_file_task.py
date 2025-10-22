#!/usr/bin/env python3
"""
Debug File Creation Task
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from green_agent import GreenAgentTerminalBench

def debug_file_creation_task():
    """Debug the file creation task specifically."""
    print("🔍 DEBUGGING FILE CREATION TASK")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        green_agent = GreenAgentTerminalBench(
            white_agent_url="http://localhost:8002",
            sandbox_base_path=temp_dir
        )
        
        # File creation task
        task = {
            "id": "debug-file-creation",
            "description": "Create a solution file",
            "instruction": "Create a file called 'solution.txt' with the answer 'demo123'",
            "test": "Check if solution.txt exists and contains 'demo123'",
            "environment": {"working_directory": "/app"}
        }
        
        print(f"📝 Task: {task['id']}")
        print(f"   Instruction: {task['instruction']}")
        print(f"   Test: {task['test']}")
        
        # Execute task with detailed logging
        print("\n🚀 Executing task...")
        result = green_agent.execute_task_with_sandbox(task)
        
        print(f"\n📊 Result:")
        print(f"   Success: {result.success}")
        print(f"   Score: {result.evaluation_result.score}")
        print(f"   Commands Executed: {result.commands_executed}")
        
        if result.evaluation_result:
            eval_result = result.evaluation_result
            print(f"\n🔍 Detailed Evaluation:")
            print(f"   Correctness: {eval_result.correctness}")
            
            # Show what went wrong
            if not eval_result.correctness['passed']:
                print(f"\n❌ Correctness Issues:")
                for error in eval_result.correctness.get('errors', []):
                    print(f"   - {error}")
                
                print(f"\n📁 File Checks:")
                for file_path, check in eval_result.correctness.get('file_checks', {}).items():
                    print(f"   - {file_path}: {check}")
                
                print(f"\n📋 Command Checks:")
                for condition, check in eval_result.correctness.get('command_checks', {}).items():
                    print(f"   - {condition}: {check}")

if __name__ == "__main__":
    debug_file_creation_task()
