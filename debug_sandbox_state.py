#!/usr/bin/env python3
"""
Debug Sandbox State Capture
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from green_agent import GreenAgentTerminalBench

def debug_sandbox_state():
    """Debug sandbox state capture in detail."""
    print("🔍 DEBUGGING SANDBOX STATE CAPTURE")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        green_agent = GreenAgentTerminalBench(
            white_agent_url="http://localhost:8002",
            sandbox_base_path=temp_dir
        )
        
        # Create sandbox
        sandbox_id = green_agent.sandbox_manager.create_sandbox("debug-sandbox", {"working_directory": "/app"})
        print(f"📦 Created sandbox: {sandbox_id}")
        
        # Execute commands manually
        print("\n🔧 Executing commands...")
        
        # Command 1: Create file
        result1 = green_agent.sandbox_manager.execute_command(sandbox_id, "echo 'password123' > solution.txt")
        print(f"Command 1: echo 'password123' > solution.txt")
        print(f"   Success: {result1.success}")
        print(f"   Return code: {result1.returncode}")
        print(f"   Stdout: '{result1.stdout}'")
        print(f"   Stderr: '{result1.stderr}'")
        
        # Command 2: List files
        result2 = green_agent.sandbox_manager.execute_command(sandbox_id, "ls -la")
        print(f"\nCommand 2: ls -la")
        print(f"   Success: {result2.success}")
        print(f"   Stdout: '{result2.stdout}'")
        
        # Command 3: Check file content
        result3 = green_agent.sandbox_manager.execute_command(sandbox_id, "cat solution.txt")
        print(f"\nCommand 3: cat solution.txt")
        print(f"   Success: {result3.success}")
        print(f"   Stdout: '{result3.stdout}'")
        
        # Capture state
        print("\n📸 Capturing sandbox state...")
        state = green_agent.sandbox_manager.capture_state(sandbox_id)
        
        print(f"\n📊 Sandbox State:")
        print(f"   Sandbox ID: {state.sandbox_id}")
        print(f"   Working Directory: {state.working_directory}")
        print(f"   Files: {state.file_system_snapshot['files']}")
        print(f"   Directories: {state.file_system_snapshot['directories']}")
        
        # Check if solution.txt is in files
        if "solution.txt" in state.file_system_snapshot["files"]:
            print(f"✅ solution.txt found in files!")
            print(f"   Content: '{state.file_system_snapshot['files']['solution.txt']}'")
        else:
            print(f"❌ solution.txt NOT found in files!")
            print(f"   Available files: {list(state.file_system_snapshot['files'].keys())}")
        
        # Clean up
        green_agent.sandbox_manager.destroy_sandbox(sandbox_id)
        print(f"\n🧹 Sandbox cleaned up")

if __name__ == "__main__":
    debug_sandbox_state()
