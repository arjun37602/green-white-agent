#!/usr/bin/env python3
"""
Test script for A2A integration
"""

import os
import sys
import json
import requests
import time
import subprocess
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_agent_startup():
    """Test starting the A2A agent."""
    print("🚀 Testing A2A agent startup...")
    
    try:
        # Import the agent
        from white_agent.agent import TerminalBenchAgent, A2ATerminalBenchServer
        
        # Test agent creation
        agent = TerminalBenchAgent()
        print(f"✅ Agent created: {agent.name}")
        
        # Test agent card
        card = agent.get_agent_card()
        print(f"✅ Agent card: {card.name} v{card.version}")
        
        return True
    except Exception as e:
        print(f"❌ Agent startup failed: {e}")
        return False

def test_a2a_task_handling():
    """Test A2A task handling."""
    print("🔧 Testing A2A task handling...")
    
    try:
        from white_agent.agent import TerminalBenchAgent, create_text_message
        
        agent = TerminalBenchAgent()
        
        # Create a test message using the A2A protocol
        test_message = create_text_message(
            text="Write a simple bash script to list all .txt files",
            role="user"
        )
        
        # Handle the message (returns a Task)
        task = agent.handle_message(test_message)
        
        print(f"✅ Task handled successfully")
        print(f"   Status: {task.status.state}")
        print(f"   Task ID: {task.id}")
        print(f"   Response length: {len(task.artifacts[0].parts[0].text) if task.artifacts else 0} characters")
        
        return True
    except Exception as e:
        print(f"❌ A2A task handling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_server_startup():
    """Test starting the A2A server."""
    print("🌐 Testing A2A server startup...")
    
    try:
        from white_agent.agent import A2ATerminalBenchServer
        
        # Create server (don't start it yet)
        server = A2ATerminalBenchServer(port=8002)  # Use different port for testing
        print(f"✅ Server created on port {server.port}")
        
        # Test server routes
        app = server.app
        print(f"✅ FastAPI app created with {len(app.routes)} routes")
        
        return True
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        return False

def test_converter():
    """Test the A2A converter."""
    print("🔄 Testing A2A converter...")
    
    try:
        from terminal_bench_to_a2a_converter import TerminalBenchToA2AConverter
        
        converter = TerminalBenchToA2AConverter("http://localhost:8002")
        
        # Test problem conversion
        sample_problem = {
            "id": "test_problem",
            "instruction": "Write a bash script to find all files larger than 100MB",
            "environment": {"os": "linux"},
            "test": "Script should output file paths and sizes",
            "difficulty": "medium",
            "category": "file_operations"
        }
        
        a2a_task = converter.convert_problem_to_a2a_task(sample_problem)
        print(f"✅ Problem converted to A2A task")
        print(f"   Task has {len(a2a_task['artifacts'])} artifacts")
        
        return True
    except Exception as e:
        print(f"❌ Converter test failed: {e}")
        return False

def test_end_to_end():
    """Test end-to-end A2A integration."""
    print("🔄 Testing end-to-end A2A integration...")
    
    try:
        # Start server in background
        print("   Starting server...")
        server_process = subprocess.Popen([
            sys.executable, "white_agent/agent.py", "--server", "--port", "8002"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(3)
        
        # Test health check
        try:
            response = requests.get("http://localhost:8002/health", timeout=5)
            if response.status_code == 200:
                print("   ✅ Server health check passed")
            else:
                print(f"   ❌ Server health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Server not responding: {e}")
            return False
        
        # Test agent card
        try:
            response = requests.get("http://localhost:8002/agent-card", timeout=5)
            if response.status_code == 200:
                card = response.json()
                print(f"   ✅ Agent card retrieved: {card['name']}")
            else:
                print(f"   ❌ Agent card failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Agent card request failed: {e}")
            return False
        
        # Test task handling using JSON-RPC 2.0 (A2A protocol)
        try:
            # Create a JSON-RPC 2.0 request with message/send method
            jsonrpc_request = {
                "jsonrpc": "2.0",
                "id": "test-1",
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{
                            "type": "text",
                            "text": "Write a simple bash script to list all .txt files"
                        }]
                    }
                }
            }
            
            response = requests.post(
                "http://localhost:8002/",  # JSON-RPC endpoint
                json=jsonrpc_request,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    task = result["result"]
                    print(f"   ✅ Task handled successfully: {task.get('status', {}).get('state', 'unknown')}")
                elif "error" in result:
                    print(f"   ❌ Task handling returned error: {result['error']}")
                    return False
            else:
                print(f"   ❌ Task handling failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Task request failed: {e}")
            return False
        
        # Stop server
        server_process.terminate()
        server_process.wait()
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        # Make sure to stop server if it's running
        try:
            server_process.terminate()
        except:
            pass
        return False

def main():
    """Run all A2A integration tests."""
    print("🧪 A2A Integration Tests")
    print("=" * 50)
    
    tests = [
        test_agent_startup,
        test_a2a_task_handling,
        test_server_startup,
        test_converter,
        test_end_to_end
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
        print("\n🚀 A2A integration is working correctly!")
        print("\n📋 Usage:")
        print("   Start A2A server: python white_agent/agent.py --server")
        print("   Test agent: python white_agent/agent.py --test")
        print("   Convert problems: python terminal_bench_to_a2a_converter.py sample_terminal_bench.json")
        print("   Check agent: python terminal_bench_to_a2a_converter.py --check-agent")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        print("\n💡 Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
