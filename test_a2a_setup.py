#!/usr/bin/env python3
"""
Test script for A2A setup
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_a2a_imports():
    """Test if A2A dependencies are available."""
    print("🔍 Testing A2A imports...")
    
    try:
        from white_agent.a2a_protocol import (
            AgentCard, AgentSkill, AgentCapabilities, AgentProvider,
            Message, Task, TaskStatus, Artifact, Part, TextPart,
            JsonRpcRequest, JsonRpcResponse, JsonRpcError
        )
        print("✅ A2A protocol imports successful")
        print("   Using internal A2A implementation (v0.3.0 compliant)")
        return True
    except ImportError as e:
        print(f"❌ A2A import failed: {e}")
        print("💡 Check that white_agent/a2a_protocol.py exists")
        return False

def test_openai_setup():
    """Test OpenAI setup."""
    print("🔍 Testing OpenAI setup...")
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    print("✅ OpenAI API key found")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client created successfully")
        return True
    except Exception as e:
        print(f"❌ OpenAI client creation failed: {e}")
        return False

def test_agent_creation():
    """Test agent creation."""
    print("🔍 Testing agent creation...")
    
    try:
        import white_agent.agent as wa
        
        # Test simple agent
        agent = wa.Agent()
        print(f"✅ Simple agent created: {agent.name}")
        
        # Test A2A agent if available
        if wa.A2A_AVAILABLE:
            a2a_agent = wa.TerminalBenchAgent()
            print(f"✅ A2A agent created: {a2a_agent.name}")
            print(f"   Skills: {[skill.name for skill in a2a_agent.skills]}")
        else:
            print("⚠️  A2A not available, using simple agent")
        
        return True
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

def test_terminal_bench_loader():
    """Test terminal bench loader."""
    print("🔍 Testing terminal bench loader...")
    
    try:
        from green_agent.dataset_loaders.terminal_bench_loader import TerminalBenchTaskLoader
        # Note: We need a valid dataset path, so we'll just test the import
        print("✅ Terminal bench loader class imported successfully")
        print("   Use TerminalBenchTaskLoader(dataset_path) to create an instance")
        return True
    except Exception as e:
        print(f"❌ Terminal bench loader failed: {e}")
        return False

def test_a2a_converter():
    """Test A2A converter."""
    print("🔍 Testing A2A converter...")
    
    try:
        from terminal_bench_to_a2a_converter import TerminalBenchToA2AConverter
        converter = TerminalBenchToA2AConverter()
        print("✅ A2A converter created")
        
        # Test sample problem conversion
        sample_problem = {
            "id": "test_problem",
            "instruction": "Write a bash script to find all files larger than 100MB",
            "environment": {"os": "linux"},
            "test": "Script should output file paths and sizes",
            "difficulty": "medium",
            "category": "file_operations"
        }
        
        a2a_task = converter.convert_problem_to_a2a_task(sample_problem)
        print("✅ Problem to A2A task conversion successful")
        print(f"   Task has {len(a2a_task['artifacts'])} artifacts")
        
        return True
    except Exception as e:
        print(f"❌ A2A converter failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing A2A Setup")
    print("=" * 50)
    
    tests = [
        test_a2a_imports,
        test_openai_setup,
        test_terminal_bench_loader,
        test_agent_creation,
        test_a2a_converter
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    print("=" * 50)
    print("📊 Test Results:")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} tests passed!")
        print("\n🚀 Ready to run A2A agent:")
        print("   python white_agent/agent.py")
        print("\n🔧 To convert terminal bench problems:")
        print("   python terminal_bench_to_a2a_converter.py <dataset_path>")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        print("\n💡 Install missing dependencies:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
