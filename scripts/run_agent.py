#!/usr/bin/env python3
"""
Run the white agent - actually invoke the model and get responses

This script demonstrates the agent actually working by sending requests to the model.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_agent_interaction():
    """Run the agent and get actual model responses."""
    try:
        import white_agent.agent as wa
        
        print("🤖 White Agent is ready!")
        print(f"Agent: {wa.basic_white_agent.name}")
        print(f"Model: {wa.model.model}")
        print("-" * 50)
        
        # Test questions
        test_questions = [
            "What is 2 + 2?",
            "Write a simple Python function to calculate the factorial of a number.",
            "What is the capital of France?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🔵 Question {i}: {question}")
            print("⏳ Getting response from model...")
            
            try:
                # Actually run the agent
                response = wa.basic_white_agent.run(question)
                print(f"✅ Response: {response}")
                
            except Exception as e:
                print(f"❌ Error getting response: {e}")
                print("This might be due to API key or authentication issues.")
                
        return True
        
    except Exception as e:
        print(f"❌ Error setting up agent: {e}")
        return False

def interactive_mode():
    """Run the agent in interactive mode."""
    try:
        import white_agent.agent as wa
        
        print("🤖 White Agent Interactive Mode")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                continue
                
            print("⏳ Thinking...")
            try:
                response = wa.basic_white_agent.run(user_input)
                print(f"🤖 Agent: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Error in interactive mode: {e}")

if __name__ == "__main__":
    print("🚀 Running White Agent...")
    print("=" * 50)
    
    # First, try the test questions
    success = run_agent_interaction()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 Agent is working! Starting interactive mode...")
        print("=" * 50)
        
        # Then start interactive mode
        interactive_mode()
    else:
        print("💥 Failed to run agent. Check your setup.")
        sys.exit(1)
