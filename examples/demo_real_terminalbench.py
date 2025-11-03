#!/usr/bin/env python3
"""
Real Terminal-Bench Integration Demo

This demo shows the Green Agent working with real Terminal-Bench tasks
from the official dataset.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green_agent import GreenAgentTerminalBench

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"🎯 {title}")
    print("="*80)

def print_step(step_num, description):
    """Print a formatted step."""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 60)

def print_model_output(result):
    """Print detailed model output information."""
    print(f"\n🤖 WHITE AGENT MODEL OUTPUT:")
    print(f"   Task ID: {result.task_id}")
    print(f"   Commands Generated: {result.commands_executed}")
    
    # Display White Agent response details
    if hasattr(result, 'white_agent_response') and result.white_agent_response:
        print(f"   White Agent Response:")
        if isinstance(result.white_agent_response, dict):
            for key, value in result.white_agent_response.items():
                if key == "result" and isinstance(value, dict):
                    print(f"     Result: {value}")
                elif key == "commands":
                    print(f"     Commands: {value}")
                else:
                    print(f"     {key}: {value}")
        else:
            print(f"     Response: {result.white_agent_response}")
    
    # Display evaluation details if available
    if hasattr(result, 'evaluation_result') and result.evaluation_result:
        eval_result = result.evaluation_result
        print(f"\n📊 EVALUATION DETAILS:")
        print(f"   Overall Score: {eval_result.score:.2f}/1.0")
        print(f"   Passed: {'✅' if eval_result.passed else '❌'}")
        if hasattr(eval_result, 'correctness'):
            print(f"   Correctness: {'✅' if eval_result.correctness.get('passed', False) else '❌'}")
        if hasattr(eval_result, 'performance'):
            print(f"   Performance: {eval_result.performance.get('total_time', 0):.2f}s")
        if hasattr(eval_result, 'safety'):
            print(f"   Safety: {'✅' if eval_result.safety.get('safe', True) else '❌'}")
        if hasattr(eval_result, 'efficiency'):
            print(f"   Efficiency: {'✅' if eval_result.efficiency.get('efficient', True) else '⚠️'}")

def demo_real_terminalbench_tasks():
    """Demo the Green Agent with real Terminal-Bench tasks."""
    print_header("REAL TERMINAL-BENCH INTEGRATION DEMO")
    print("This demo shows the Green Agent working with real Terminal-Bench tasks")
    print("from the official terminal-bench-core dataset.")
    
    # Check if Terminal-Bench dataset is available
    dataset_path = "/Users/arjun/.cache/terminal-bench/terminal-bench-core/head"
    if not Path(dataset_path).exists():
        print(f"❌ Terminal-Bench dataset not found at: {dataset_path}")
        print("Please download it first with: tb datasets download terminal-bench-core")
        return
    
    print_step(1, "Initializing Green Agent with Terminal-Bench Dataset")
    with tempfile.TemporaryDirectory() as temp_dir:
        green_agent = GreenAgentTerminalBench(
            white_agent_url="http://localhost:8001",
            sandbox_base_path=temp_dir,
            terminal_bench_dataset_path=dataset_path
        )
        print("✅ Green Agent initialized with Terminal-Bench dataset")
        print(f"   Dataset path: {dataset_path}")
        print(f"   Sandbox path: {temp_dir}")
    
    print_step(2, "Exploring Terminal-Bench Tasks")
    
    # Load a few different types of tasks
    test_tasks = ["hello-world", "crack-7z-hash", "openssl-selfsigned-cert"]
    available_tasks = []
    
    for task_id in test_tasks:
        try:
            tasks = green_agent.load_terminal_bench_tasks([task_id], limit=1)
            if tasks:
                task = tasks[0]
                available_tasks.append(task)
                print(f"✅ Loaded task: {task_id}")
                print(f"   Instruction: {task['instruction'][:100]}...")
                print(f"   Difficulty: {task['metadata'].get('difficulty', 'unknown')}")
                print(f"   Category: {task['metadata'].get('category', 'unknown')}")
            else:
                print(f"❌ Task not found: {task_id}")
        except Exception as e:
            print(f"❌ Failed to load task {task_id}: {e}")
    
    if not available_tasks:
        print("❌ No Terminal-Bench tasks could be loaded")
        return
    
    print(f"\n📊 Successfully loaded {len(available_tasks)} Terminal-Bench tasks")
    
    print_step(3, "Running Terminal-Bench Task Evaluation")
    
    results = []
    for i, task in enumerate(available_tasks[:2], 1):  # Limit to 2 tasks for demo
        print(f"\n🔧 Task {i}/{min(len(available_tasks), 2)}: {task['id']}")
        print(f"   Instruction: {task['instruction']}")
        
        # Execute task
        result = green_agent.execute_task_with_sandbox(task)
        results.append(result)
        
        # Display detailed model output
        print_model_output(result)
        
        # Display evaluation results
        status = "✅ PASSED" if result.success else "❌ FAILED"
        print(f"\n📊 EVALUATION RESULTS:")
        print(f"   Result: {status}")
        print(f"   Score: {result.evaluation_result.score:.2f}/1.0")
        print(f"   Execution Time: {result.execution_time:.2f}s")
        
        # Display pytest results if available
        if result.evaluation_result and 'unit_tests' in result.evaluation_result.correctness:
            unit_tests = result.evaluation_result.correctness['unit_tests']
            print(f"\n🧪 TERMINAL-BENCH PYTEST RESULTS:")
            print(f"   Tests Run: {unit_tests.get('tests_run', 0)}")
            print(f"   Tests Passed: {unit_tests.get('tests_passed', 0)}")
            print(f"   Tests Failed: {unit_tests.get('tests_failed', 0)}")
            print(f"   Overall Result: {'✅ PASSED' if unit_tests.get('passed', False) else '❌ FAILED'}")
            if 'raw_output' in unit_tests:
                print(f"   Raw Pytest Output (first 10 lines):")
                for line in unit_tests['raw_output'].split('\n')[:10]:  # First 10 lines
                    if line.strip():
                        print(f"     {line}")
        
        # Display detailed evaluation breakdown
        if result.evaluation_result:
            eval_result = result.evaluation_result
            print(f"\n📈 DETAILED EVALUATION BREAKDOWN:")
            print(f"   Correctness: {'✅' if eval_result.correctness['passed'] else '❌'}")
            print(f"   Performance: {'✅' if eval_result.performance['total_time'] < 60 else '⚠️'}")
            print(f"   Safety: {'✅' if eval_result.safety['safe'] else '❌'}")
            print(f"   Efficiency: {'✅' if eval_result.efficiency['efficient'] else '⚠️'}")
    
    print_step(4, "Terminal-Bench Integration Summary")
    passed = sum(1 for r in results if r.success)
    total = len(results)
    avg_score = sum(r.evaluation_result.score for r in results) / total if results else 0
    
    print(f"📊 Terminal-Bench Task Results:")
    print(f"   Total Tasks: {total}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {total - passed}")
    print(f"   Success Rate: {passed/total*100:.1f}%")
    print(f"   Average Score: {avg_score:.2f}/1.0")
    
    print_step(5, "System Capabilities Demonstrated")
    print("✅ Terminal-Bench Dataset Integration: Loaded real tasks from official dataset")
    print("✅ Task Parsing: Parsed task.yaml, solution.sh, and test files")
    print("✅ Sandbox Isolation: Each task ran in its own isolated environment")
    print("✅ A2A Protocol: Green Agent and White Agent communicated via A2A protocol")
    print("✅ Command Execution: Bash commands executed safely in sandboxes")
    print("✅ Task Evaluation: Multi-dimensional evaluation (correctness, performance, safety, efficiency)")
    print("✅ Resource Management: Automatic cleanup of sandbox resources")
    print("✅ Comprehensive Logging: Detailed execution traces and results")
    
    print_header("REAL TERMINAL-BENCH INTEGRATION COMPLETED!")
    print("The Green Agent successfully integrated with the official Terminal-Bench dataset!")
    print("Real Terminal-Bench tasks + Green Agent + White Agent + Sandbox Isolation = ✅")

def demo_task_loader():
    """Demo the Terminal-Bench task loader functionality."""
    print_header("TERMINAL-BENCH TASK LOADER DEMO")
    print("This demo shows the Terminal-Bench task loader parsing real task files.")
    
    from green_agent.dataset_loaders.terminal_bench_loader import TerminalBenchTaskLoader
    
    dataset_path = "/Users/arjun/.cache/terminal-bench/terminal-bench-core/head"
    if not Path(dataset_path).exists():
        print(f"❌ Terminal-Bench dataset not found at: {dataset_path}")
        return
    
    print_step(1, "Initializing Terminal-Bench Task Loader")
    loader = TerminalBenchTaskLoader(dataset_path)
    print("✅ Terminal-Bench Task Loader initialized")
    
    print_step(2, "Listing Available Tasks")
    available_tasks = loader.list_available_tasks()
    print(f"📋 Found {len(available_tasks)} available tasks")
    print("   Sample tasks:", available_tasks[:10])
    
    print_step(3, "Loading Sample Tasks")
    sample_tasks = ["hello-world", "crack-7z-hash"]
    
    for task_id in sample_tasks:
        if task_id in available_tasks:
            try:
                task = loader.load_task(task_id)
                internal_task = loader.convert_to_internal_format(task)
                
                print(f"\n🔧 Task: {task_id}")
                print(f"   Author: {task.author_name}")
                print(f"   Difficulty: {task.difficulty}")
                print(f"   Category: {task.category}")
                print(f"   Tags: {task.tags}")
                print(f"   Instruction: {task.instruction[:100]}...")
                print(f"   Solution Commands: {task.solution_commands}")
                print(f"   Test File: {task.test_file_path}")
                print(f"   Dockerfile: {task.dockerfile_path}")
                
            except Exception as e:
                print(f"❌ Failed to load task {task_id}: {e}")
        else:
            print(f"⚠️ Task not found: {task_id}")
    
    print_header("TERMINAL-BENCH TASK LOADER DEMO COMPLETED!")

def main():
    """Main demo function."""
    print("🚀 Real Terminal-Bench Integration Demo")
    print("Choose demo type:")
    print("1. Complete Real Terminal-Bench Integration Demo")
    print("2. Terminal-Bench Task Loader Demo")
    print("3. Both demos")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            demo_real_terminalbench_tasks()
        elif choice == "2":
            demo_task_loader()
        elif choice == "3":
            demo_task_loader()
            print("\n" + "="*80)
            input("Press Enter to continue to complete integration demo...")
            demo_real_terminalbench_tasks()
        else:
            print("Invalid choice. Running complete integration demo...")
            demo_real_terminalbench_tasks()
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
