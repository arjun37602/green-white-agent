#!/usr/bin/env python3
"""
Test script to see White Agent debugging output
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from white_agent.agent import TerminalBenchAgent

def test_white_agent():
    """Test the White Agent with debugging output"""
    print("🚀 Testing White Agent with Debugging Output")
    print("=" * 60)
    
    # Initialize the agent
    agent = TerminalBenchAgent(model="gpt-4o-mini")
    
    # Test with a simple task
    task_description = "Create a file called hello.txt with content Hello World"
    print(f"\n📝 Task: {task_description}")
    print("-" * 60)
    
    # Run the task
    result = agent.solve_problem(task_description)
    
    print(f"\n✅ Task completed!")
    print(f"Result: {result}")

if __name__ == "__main__":
    test_white_agent()
