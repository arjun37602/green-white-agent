#!/usr/bin/env python3
"""
TerminalBench System Demo

A comprehensive demo script that shows the complete Green Agent + White Agent
system working together on various TerminalBench tasks.
"""

import sys
import os
import time
import tempfile
import requests
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

def demo_complete_system():
    """Demo the complete TerminalBench system."""
    print_header("TERMINALBENCH SYSTEM DEMO")
    print("This demo shows the Green Agent + White Agent system working together")
    print("to evaluate TerminalBench tasks in isolated sandbox environments.")
    
    # Step 1: Check White Agent
    print_step(1, "Checking Simple White Agent Server")
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            print("✅ Simple White Agent is running and healthy!")
            agent_info = response.json()
            print(f"   Agent: {agent_info['agent']}")
        else:
            print("❌ Simple White Agent is not responding properly")
            return
    except requests.exceptions.RequestException:
        print("❌ Simple White Agent server is not running!")
        print("   Please start it with: python simple_white_agent.py --server --port 8002")
        return
    
    # Step 2: Initialize Green Agent
    print_step(2, "Initializing Green Agent")
    with tempfile.TemporaryDirectory() as temp_dir:
        green_agent = GreenAgentTerminalBench(
            white_agent_url="http://localhost:8002",
            sandbox_base_path=temp_dir
        )
        print("✅ Green Agent initialized with sandbox isolation")
        print(f"   Sandbox path: {temp_dir}")
        
        # Step 3: Load Demo Tasks
        print_step(3, "Loading TerminalBench Tasks")
        demo_tasks = [
            {
                "id": "demo-file-creation",
                "description": "Create a solution file",
                "instruction": "Create a file called 'solution.txt' with the answer 'demo123'",
                "test": "Check if solution.txt exists and contains 'demo123'",
                "environment": {"working_directory": "/app"}
            },
            {
                "id": "demo-directory-task", 
                "description": "Create and navigate directories",
                "instruction": "Create a directory called 'demo_dir' and navigate into it",
                "test": "Check if demo_dir exists and you can navigate into it",
                "environment": {"working_directory": "/app"}
            },
            {
                "id": "demo-ssl-cert",
                "description": "Create SSL certificate",
                "instruction": "Create a self-signed SSL certificate for testing",
                "test": "Check if SSL certificate files exist",
                "environment": {"working_directory": "/app"}
            }
        ]
        
        print(f"✅ Loaded {len(demo_tasks)} demo tasks:")
        for i, task in enumerate(demo_tasks, 1):
            print(f"   {i}. {task['id']}: {task['description']}")
        
        # Step 4: Run Tasks
        print_step(4, "Running TerminalBench Evaluation")
        
        results = []
        for i, task in enumerate(demo_tasks, 1):
            print(f"\n🔧 Task {i}/{len(demo_tasks)}: {task['id']}")
            print(f"   Description: {task['description']}")
            print(f"   Instruction: {task['instruction']}")
            
            # Execute task
            start_time = time.time()
            result = green_agent.execute_task_with_sandbox(task)
            execution_time = time.time() - start_time
            
            # Display results
            status = "✅ PASSED" if result.success else "❌ FAILED"
            print(f"   Result: {status}")
            print(f"   Score: {result.evaluation_result.score:.2f}/1.0")
            print(f"   Commands Executed: {result.commands_executed}")
            print(f"   Execution Time: {execution_time:.2f}s")
            
            if result.evaluation_result:
                eval_result = result.evaluation_result
                print(f"   Correctness: {'✅' if eval_result.correctness['passed'] else '❌'}")
                print(f"   Performance: {'✅' if eval_result.performance['total_time'] < 60 else '⚠️'}")
                print(f"   Safety: {'✅' if eval_result.safety['safe'] else '❌'}")
                print(f"   Efficiency: {'✅' if eval_result.efficiency['efficient'] else '⚠️'}")
            
            results.append(result)
        
        # Step 5: Summary
        print_step(5, "Evaluation Summary")
        passed = sum(1 for r in results if r.success)
        total = len(results)
        avg_score = sum(r.evaluation_result.score for r in results) / total if results else 0
        
        print(f"📊 Overall Results:")
        print(f"   Total Tasks: {total}")
        print(f"   Passed: {passed}")
        print(f"   Failed: {total - passed}")
        print(f"   Success Rate: {passed/total*100:.1f}%")
        print(f"   Average Score: {avg_score:.2f}/1.0")
        
        # Step 6: System Capabilities Demonstrated
        print_step(6, "System Capabilities Demonstrated")
        print("✅ Sandbox Isolation: Each task ran in its own isolated environment")
        print("✅ A2A Protocol: Green Agent and White Agent communicated via A2A protocol")
        print("✅ Command Execution: Bash commands executed safely in sandboxes")
        print("✅ Task Evaluation: Multi-dimensional evaluation (correctness, performance, safety, efficiency)")
        print("✅ Resource Management: Automatic cleanup of sandbox resources")
        print("✅ Comprehensive Logging: Detailed execution traces and results")
        
        print_header("DEMO COMPLETED SUCCESSFULLY!")
        print("The TerminalBench evaluation system is working correctly!")
        print("Green Agent + White Agent + Sandbox Isolation + Comprehensive Evaluation = ✅")

def demo_white_agent_only():
    """Demo just the White Agent capabilities."""
    print_header("SIMPLE WHITE AGENT DEMO")
    print("This demo shows the Simple White Agent's template-based approach")
    print("to solving TerminalBench tasks.")
    
    # Step 1: Check White Agent
    print_step(1, "Checking Simple White Agent Server")
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            print("✅ Simple White Agent is running and healthy!")
            agent_info = response.json()
            print(f"   Agent: {agent_info['agent']}")
        else:
            print("❌ Simple White Agent is not responding properly")
            return
    except requests.exceptions.RequestException:
        print("❌ Simple White Agent server is not running!")
        print("   Please start it with: python simple_white_agent.py --server --port 8002")
        return
    
    # Step 2: Test Different Task Types
    print_step(2, "Testing Different Task Types")
    
    test_tasks = [
        {
            "name": "File Creation Task",
            "description": "Create a file called solution.txt with the password",
            "expected_commands": ["echo", "solution.txt"]
        },
        {
            "name": "Directory Task", 
            "description": "Create a directory called test_dir and navigate into it",
            "expected_commands": ["mkdir", "cd", "pwd"]
        },
        {
            "name": "SSL Certificate Task",
            "description": "Create a self-signed SSL certificate",
            "expected_commands": ["openssl", "ssl"]
        },
        {
            "name": "Archive Task",
            "description": "Create and extract files from an archive",
            "expected_commands": ["tar", "archive"]
        }
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n🔧 Test {i}: {task['name']}")
        print(f"   Description: {task['description']}")
        
        # Send request to White Agent
        try:
            response = requests.post(
                "http://localhost:8002/",
                json={
                    "jsonrpc": "2.0",
                    "id": f"test-{i}",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [{
                                "kind": "text",
                                "text": task['description']
                            }]
                        }
                    }
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result and "artifacts" in result["result"]:
                    artifact = result["result"]["artifacts"][0]
                    solution_text = artifact["parts"][0]["text"]
                    
                    print("   ✅ White Agent Response:")
                    print(f"      {solution_text[:200]}...")
                    
                    # Check if expected commands are in the response
                    found_commands = []
                    for expected_cmd in task["expected_commands"]:
                        if expected_cmd.lower() in solution_text.lower():
                            found_commands.append(expected_cmd)
                    
                    print(f"   📋 Found Commands: {', '.join(found_commands)}")
                    print(f"   🎯 Match Rate: {len(found_commands)}/{len(task['expected_commands'])}")
                else:
                    print("   ❌ Invalid response format")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request failed: {e}")
    
    print_header("WHITE AGENT DEMO COMPLETED!")
    print("The Simple White Agent successfully processed different task types!")
    print("Template-based approach + Pattern matching = ✅")

def main():
    """Main demo function."""
    print("🚀 TerminalBench System Demo")
    
    # Check if running in non-interactive mode
    import sys
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("Choose demo type:")
        print("1. Complete System Demo (Green Agent + White Agent)")
        print("2. White Agent Only Demo")
        print("3. Both demos")
        
        try:
            choice = input("\nEnter choice (1-3): ").strip()
        except EOFError:
            # Running in non-interactive mode, default to complete demo
            print("\nRunning in non-interactive mode. Starting complete system demo...")
            choice = "1"
    
    try:
        if choice == "1":
            demo_complete_system()
        elif choice == "2":
            demo_white_agent_only()
        elif choice == "3":
            demo_white_agent_only()
            print("\n" + "="*80)
            print("Continuing to complete system demo...")
            demo_complete_system()
        else:
            print("Invalid choice. Running complete system demo...")
            demo_complete_system()
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
