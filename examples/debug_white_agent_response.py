#!/usr/bin/env python3
"""
Debug script to inspect white agent response structure
and verify token/request data extraction
"""

import sys
import os
import json
import logging
import tempfile

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from green_agent import GreenAgentTerminalBench

# Set up logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_white_agent_response():
    """Test white agent response and inspect its structure."""
    print("=" * 80)
    print("DEBUG: White Agent Response Structure Test")
    print("=" * 80)
    
    # Check if white agent is running
    white_agent_url = "http://localhost:8002"
    
    try:
        import requests
        response = requests.get(f"{white_agent_url}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ White agent at {white_agent_url} is not healthy")
            print(f"   Status: {response.status_code}")
            print(f"   Please start the white agent with: python simple_white_agent.py --server --port 8002")
            return
        print(f"✅ White agent is running at {white_agent_url}")
    except Exception as e:
        print(f"❌ Cannot connect to white agent at {white_agent_url}")
        print(f"   Error: {e}")
        print(f"   Please start the white agent with: python simple_white_agent.py --server --port 8002")
        return
    
    # Create a simple test task
    test_task = {
        "id": "debug_test_task",
        "description": "Debug test to inspect response structure",
        "instruction": "Create a file called test.txt with content 'hello world'",
        "test": "Check if test.txt exists",
        "environment": {
            "working_directory": "/app"
        }
    }
    
    print(f"\n📋 Test Task: {test_task['id']}")
    print(f"   Instruction: {test_task['instruction']}")
    
    # Create green agent
    with tempfile.TemporaryDirectory() as temp_dir:
        green_agent = GreenAgentTerminalBench(
            white_agent_url=white_agent_url,
            sandbox_base_path=temp_dir
        )
        
        print(f"\n📤 Sending task to white agent...")
        
        # Send task to white agent
        white_agent_response = green_agent.send_task_to_white_agent(test_task)
        
        print(f"\n📥 Received white agent response")
        print("=" * 80)
        print("FULL RESPONSE STRUCTURE:")
        print("=" * 80)
        print(json.dumps(white_agent_response, indent=2, default=str))
        print("=" * 80)
        
        # Extract token/request counts using the evaluator method
        print(f"\n🔍 Extracting token/request counts...")
        token_request_data = green_agent.task_evaluator._extract_token_and_request_counts(white_agent_response)
        
        print(f"\n📊 Extracted Data:")
        print(f"   Total Tokens: {token_request_data.get('total_tokens')}")
        print(f"   Total Requests: {token_request_data.get('total_requests')}")
        
        if token_request_data.get('total_tokens') and token_request_data.get('total_requests'):
            tokens_per_request = token_request_data['total_tokens'] / token_request_data['total_requests']
            print(f"   Tokens Per Request: {tokens_per_request:.2f}")
        else:
            print(f"   ⚠️  Token/request data not available")
        
        # Check specific paths
        print(f"\n🔎 Checking common paths in response:")
        
        paths_to_check = [
            ("result.metadata.usage", white_agent_response.get("result", {}).get("metadata", {}).get("usage")),
            ("result.metadata.total_tokens", white_agent_response.get("result", {}).get("metadata", {}).get("total_tokens")),
            ("result.metadata.request_count", white_agent_response.get("result", {}).get("metadata", {}).get("request_count")),
            ("metadata.usage", white_agent_response.get("metadata", {}).get("usage") if isinstance(white_agent_response.get("metadata"), dict) else None),
            ("metadata.total_tokens", white_agent_response.get("metadata", {}).get("total_tokens") if isinstance(white_agent_response.get("metadata"), dict) else None),
        ]
        
        for path, value in paths_to_check:
            if value is not None:
                print(f"   ✅ Found at '{path}': {value}")
            else:
                print(f"   ❌ Not found at '{path}'")
        
        print(f"\n✅ Debug test completed!")
        print("=" * 80)


if __name__ == "__main__":
    try:
        test_white_agent_response()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
