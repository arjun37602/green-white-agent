#!/usr/bin/env python3
"""
Test A2A Protocol Compliance

This script tests the actual A2A protocol implementation with HTTP requests.
"""

import requests
import json
import sys

SERVER_URL = "http://localhost:8002"

def test_agent_card():
    """Test 1: Agent Card Discovery (RFC 8615 Well-Known URI)"""
    print("=" * 60)
    print("TEST 1: Agent Card Discovery")
    print("=" * 60)
    
    # Test well-known URI
    print(f"\n📋 GET {SERVER_URL}/.well-known/agent-card")
    response = requests.get(f"{SERVER_URL}/.well-known/agent-card")
    
    if response.status_code == 200:
        agent_card = response.json()
        print("✅ Agent card retrieved successfully")
        print(f"   Agent: {agent_card.get('name')}")
        print(f"   Version: {agent_card.get('version')}")
        print(f"   Provider: {agent_card.get('provider')}")
        print(f"   Skills: {[s['name'] for s in agent_card.get('skills', [])]}")
        print(f"   Capabilities: streaming={agent_card.get('capabilities', {}).get('streaming')}, "
              f"LRO={agent_card.get('capabilities', {}).get('longRunningTasks')}")
        return True
    else:
        print(f"❌ Failed: {response.status_code}")
        return False

def test_message_send():
    """Test 2: message.send RPC method"""
    print("\n" + "=" * 60)
    print("TEST 2: message.send (JSON-RPC 2.0)")
    print("=" * 60)
    
    rpc_request = {
        "jsonrpc": "2.0",
        "id": "test-001",
        "method": "message.send",
        "params": {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "Write a simple bash script to list all .txt files in the current directory"
                    }
                ]
            }
        }
    }
    
    print(f"\n📤 POST {SERVER_URL}/")
    print(f"   Method: message.send")
    print(f"   Message: {rpc_request['params']['message']['parts'][0]['text'][:50]}...")
    
    response = requests.post(f"{SERVER_URL}/", json=rpc_request)
    
    if response.status_code == 200:
        rpc_response = response.json()
        
        if "result" in rpc_response:
            task = rpc_response["result"]
            print("✅ Message sent successfully")
            print(f"   JSON-RPC ID: {rpc_response.get('id')}")
            
            # Debug: check if task is None
            if task is None:
                print(f"❌ Error: result is None")
                print(f"   Full response: {rpc_response}")
                return None
            
            print(f"   Task ID: {task.get('id')}")
            print(f"   Context ID: {task.get('contextId')}")
            print(f"   Status: {task.get('status', {}).get('state')}")
            print(f"   Artifacts: {len(task.get('artifacts', []))}")
            
            if task.get('artifacts'):
                artifact = task['artifacts'][0]
                print(f"   Artifact Name: {artifact.get('name')}")
                solution_text = artifact.get('parts', [{}])[0].get('text', '')
                print(f"   Solution Length: {len(solution_text)} characters")
                print(f"   Solution Preview: {solution_text[:100]}...")
            
            return task
        else:
            print(f"❌ Error in response: {rpc_response.get('error')}")
            return None
    else:
        print(f"❌ Failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return None

def test_task_get(task_id):
    """Test 3: task.get RPC method"""
    print("\n" + "=" * 60)
    print("TEST 3: task.get (JSON-RPC 2.0)")
    print("=" * 60)
    
    rpc_request = {
        "jsonrpc": "2.0",
        "id": "test-002",
        "method": "tasks/get",
        "params": {
            "id": task_id
        }
    }
    
    print(f"\n📤 POST {SERVER_URL}/")
    print(f"   Method: tasks/get")
    print(f"   Task ID: {task_id}")
    
    response = requests.post(f"{SERVER_URL}/", json=rpc_request)
    
    if response.status_code == 200:
        rpc_response = response.json()
        
        if "result" in rpc_response:
            task = rpc_response["result"]
            print("✅ Task retrieved successfully")
            print(f"   Task ID: {task.get('id')}")
            print(f"   Status: {task.get('status', {}).get('state')}")
            print(f"   Created: {task.get('createdAt')}")
            print(f"   Updated: {task.get('updatedAt')}")
            return True
        else:
            print(f"❌ Error in response: {rpc_response.get('error')}")
            return False
    else:
        print(f"❌ Failed: {response.status_code}")
        return False

def test_context_continuation(context_id, task_id):
    """Test 4: Context continuation with referenceTaskIds"""
    print("\n" + "=" * 60)
    print("TEST 4: Context Continuation (Follow-up)")
    print("=" * 60)
    
    rpc_request = {
        "jsonrpc": "2.0",
        "id": "test-003",
        "method": "message.send",
        "params": {
            "message": {
                "role": "user",
                "contextId": context_id,
                "referenceTaskIds": [task_id],
                "parts": [
                    {
                        "kind": "text",
                        "text": "Now modify that script to also show file sizes"
                    }
                ]
            }
        }
    }
    
    print(f"\n📤 POST {SERVER_URL}/")
    print(f"   Method: message.send")
    print(f"   Context ID: {context_id}")
    print(f"   Reference Task: {task_id}")
    print(f"   Message: Follow-up request")
    
    response = requests.post(f"{SERVER_URL}/", json=rpc_request)
    
    if response.status_code == 200:
        rpc_response = response.json()
        
        if "result" in rpc_response:
            task = rpc_response["result"]
            print("✅ Follow-up message sent successfully")
            print(f"   New Task ID: {task.get('id')}")
            print(f"   Same Context ID: {task.get('contextId') == context_id}")
            print(f"   Status: {task.get('status', {}).get('state')}")
            return True
        else:
            print(f"❌ Error in response: {rpc_response.get('error')}")
            return False
    else:
        print(f"❌ Failed: {response.status_code}")
        return False

def test_invalid_method():
    """Test 5: Invalid RPC method (error handling)"""
    print("\n" + "=" * 60)
    print("TEST 5: Invalid Method (Error Handling)")
    print("=" * 60)
    
    rpc_request = {
        "jsonrpc": "2.0",
        "id": "test-004",
        "method": "invalid.method",
        "params": {}
    }
    
    print(f"\n📤 POST {SERVER_URL}/")
    print(f"   Method: invalid.method (should fail)")
    
    response = requests.post(f"{SERVER_URL}/", json=rpc_request)
    
    if response.status_code == 200:
        rpc_response = response.json()
        
        if "error" in rpc_response:
            error = rpc_response["error"]
            print("✅ Error handled correctly")
            print(f"   Error Code: {error.get('code')}")
            print(f"   Error Message: {error.get('message')}")
            return True
        else:
            print("❌ Expected error response, got result")
            return False
    else:
        print(f"❌ Unexpected HTTP status: {response.status_code}")
        return False

def test_health_check():
    """Test 6: Health check endpoint"""
    print("\n" + "=" * 60)
    print("TEST 6: Health Check")
    print("=" * 60)
    
    print(f"\n📋 GET {SERVER_URL}/health")
    response = requests.get(f"{SERVER_URL}/health")
    
    if response.status_code == 200:
        health = response.json()
        print("✅ Health check successful")
        print(f"   Status: {health.get('status')}")
        print(f"   Agent: {health.get('agent')}")
        return True
    else:
        print(f"❌ Failed: {response.status_code}")
        return False

def main():
    """Run all A2A protocol tests"""
    print("\n🧪 A2A Protocol Compliance Tests")
    print("Testing server at:", SERVER_URL)
    print()
    
    results = []
    
    # Test 1: Agent Card
    try:
        results.append(("Agent Card Discovery", test_agent_card()))
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        results.append(("Agent Card Discovery", False))
    
    # Test 2: message.send
    task = None
    try:
        task = test_message_send()
        results.append(("message.send", task is not None))
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        results.append(("message.send", False))
    
    # Test 3: task.get
    if task:
        try:
            results.append(("task.get", test_task_get(task.get('id'))))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append(("task.get", False))
    else:
        results.append(("task.get", False))
        print("\n⚠️  Skipping task.get test (no task available)")
    
    # Test 4: Context continuation
    if task:
        try:
            results.append(("Context Continuation", 
                          test_context_continuation(task.get('contextId'), task.get('id'))))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append(("Context Continuation", False))
    else:
        results.append(("Context Continuation", False))
        print("\n⚠️  Skipping context continuation test (no task available)")
    
    # Test 5: Invalid method
    try:
        results.append(("Error Handling", test_invalid_method()))
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        results.append(("Error Handling", False))
    
    # Test 6: Health check
    try:
        results.append(("Health Check", test_health_check()))
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        results.append(("Health Check", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print()
    print(f"Results: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All A2A protocol tests passed!")
        print("✅ The agent is fully A2A protocol compliant!")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Cannot connect to server at {SERVER_URL}")
        print("Make sure the server is running:")
        print("  python white_agent/agent.py --server --port 8002")
        sys.exit(1)

