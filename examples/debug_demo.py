#!/usr/bin/env python3
"""
Debug Demo Script - Detailed logging to understand evaluation failures
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green_agent import GreenAgentTerminalBench

def debug_single_task():
    """Debug a single task to see what's happening."""
    print("🔍 DEBUGGING SINGLE TASK")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        green_agent = GreenAgentTerminalBench(
            white_agent_url="http://localhost:8002",
            sandbox_base_path=temp_dir
        )
        
        # Simple directory task
        task = {
            "id": "debug-directory",
            "description": "Create and navigate directories",
            "instruction": "Create a directory called 'test_dir' and navigate into it",
            "test": "Check if test_dir exists and you can navigate into it",
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
            print(f"   Performance: {eval_result.performance}")
            print(f"   Safety: {eval_result.safety}")
            print(f"   Efficiency: {eval_result.efficiency}")
            
            # Show what went wrong
            if not eval_result.correctness['passed']:
                print(f"\n❌ Correctness Issues:")
                for error in eval_result.correctness.get('errors', []):
                    print(f"   - {error}")
            
            if not eval_result.safety['safe']:
                print(f"\n⚠️ Safety Issues:")
                for violation in eval_result.safety.get('violations', []):
                    print(f"   - {violation}")
            
            if not eval_result.efficiency['efficient']:
                print(f"\n⚠️ Efficiency Issues:")
                print(f"   - {eval_result.efficiency}")

def debug_evaluation_logic():
    """Debug the evaluation logic directly."""
    print("\n🔍 DEBUGGING EVALUATION LOGIC")
    print("="*60)
    
    from green_agent.task_evaluator import TaskEvaluator
    from green_agent.sandbox_manager import CommandResult, SandboxState
    
    evaluator = TaskEvaluator()
    
    # Mock command history for directory task
    command_history = [
        CommandResult(
            command="mkdir test_dir",
            stdout="",
            stderr="",
            returncode=0,
            execution_time=0.01,
            timestamp="2024-01-01T00:00:00",
            success=True
        ),
        CommandResult(
            command="cd test_dir",
            stdout="",
            stderr="",
            returncode=0,
            execution_time=0.00,
            timestamp="2024-01-01T00:00:01",
            success=True
        ),
        CommandResult(
            command="pwd",
            stdout="/app/test_dir\n",
            stderr="",
            returncode=0,
            execution_time=0.00,
            timestamp="2024-01-01T00:00:02",
            success=True
        )
    ]
    
    # Mock sandbox state
    sandbox_state = SandboxState(
        sandbox_id="debug_sandbox",
        working_directory="/app",
        environment_vars={},
        file_system_snapshot={
            "files": {},
            "directories": ["test_dir"],
            "timestamp": "2024-01-01T00:00:00"
        },
        timestamp="2024-01-01T00:00:00"
    )
    
    # Mock task spec
    task_spec = {
        "id": "debug-directory",
        "instruction": "Create a directory called 'test_dir' and navigate into it",
        "test": "Check if test_dir exists and you can navigate into it",
        "environment": {"working_directory": "/app"}
    }
    
    print("📋 Mock Data:")
    print(f"   Commands: {[cmd.command for cmd in command_history]}")
    print(f"   Directories: {sandbox_state.file_system_snapshot['directories']}")
    print(f"   Test: {task_spec['test']}")
    
    # Extract criteria
    criteria = evaluator._extract_criteria(task_spec)
    print(f"\n🔍 Extracted Criteria:")
    print(f"   Success Conditions: {criteria.success_conditions}")
    print(f"   File Requirements: {criteria.file_requirements}")
    print(f"   Output Requirements: {criteria.output_requirements}")
    print(f"   Forbidden Commands: {criteria.forbidden_commands}")
    
    # Evaluate correctness
    correctness_result = evaluator._evaluate_correctness(criteria, command_history, sandbox_state)
    print(f"\n🔍 Correctness Evaluation:")
    print(f"   Passed: {correctness_result['passed']}")
    print(f"   File Checks: {correctness_result['file_checks']}")
    print(f"   Output Checks: {correctness_result['output_checks']}")
    print(f"   Command Checks: {correctness_result['command_checks']}")
    print(f"   Errors: {correctness_result['errors']}")
    
    # Check success condition logic
    for condition in criteria.success_conditions:
        print(f"\n🔍 Checking Success Condition: '{condition}'")
        condition_result = evaluator._check_success_condition(condition, command_history, sandbox_state)
        print(f"   Result: {condition_result}")

if __name__ == "__main__":
    debug_single_task()
    debug_evaluation_logic()
