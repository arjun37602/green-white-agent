#!/usr/bin/env python3
"""
Final AgentBeats Integration Verification

This script verifies the integration structure is correct.
Dependencies may need to be installed separately.
"""

import sys
from pathlib import Path

def check_structure():
    """Check that all files and paths are correct."""
    print("=" * 80)
    print("AGENTBEATS INTEGRATION STRUCTURE VERIFICATION")
    print("=" * 80)
    
    base = Path(__file__).resolve().parent
    # Use same logic as tools.py: from tools.py, parents[5] = llm_runners
    # verify_integration.py is at: agentbeats/scenarios/terminal_bench/verify_integration.py
    # parents[0] = terminal_bench
    # parents[1] = scenarios
    # parents[2] = agentbeats  
    # parents[3] = llm_runners
    llm_runners = base.parents[3]
    
    # But tools.py uses parents[5] from its location
    tools_path = base / "agents" / "green_agent" / "tools.py"
    if tools_path.exists():
        llm_runners_from_tools = tools_path.resolve().parents[5]
    else:
        llm_runners_from_tools = llm_runners
    
    checks = {
        "Scenario directory": base,
        "Green agent card": base / "agents" / "green_agent" / "agent_card.toml",
        "Green agent tools": base / "agents" / "green_agent" / "tools.py",
        "White agent card": base / "agents" / "white_agent" / "agent_card.toml",
        "Scenario config": base / "scenario.toml",
        "Green-white-agent (from tools.py logic)": llm_runners_from_tools / "green-white-agent",
        "AgentBeats src (from tools.py logic)": llm_runners_from_tools / "agentbeats" / "src",
    }
    
    all_pass = True
    for name, path in checks.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        if not exists:
            all_pass = False
    
    print("\n" + "=" * 80)
    print("PATH RESOLUTION TEST")
    print("=" * 80)
    
    # Test path resolution in tools.py
    tools_path = base / "agents" / "green_agent" / "tools.py"
    if tools_path.exists():
        # Calculate expected paths
        green_white_path = tools_path.resolve().parents[5] / "green-white-agent"
        agentbeats_src = tools_path.resolve().parents[5] / "agentbeats" / "src"
        
        print(f"Tools file: {tools_path}")
        print(f"Expected green-white-agent: {green_white_path}")
        print(f"  Exists: {'✅' if green_white_path.exists() else '❌'}")
        print(f"Expected agentbeats/src: {agentbeats_src}")
        print(f"  Exists: {'✅' if agentbeats_src.exists() else '❌'}")
    
    print("\n" + "=" * 80)
    print("CONFIGURATION VALIDATION")
    print("=" * 80)
    
    try:
        import toml
        scenario_file = base / "scenario.toml"
        if scenario_file.exists():
            with open(scenario_file) as f:
                config = toml.load(f)
            
            print(f"✅ Scenario TOML is valid")
            print(f"   Name: {config.get('scenario', {}).get('name', 'unknown')}")
            print(f"   Agents: {len(config.get('agents', []))}")
            
            # Check green agent
            for agent in config.get('agents', []):
                if agent.get('name') == 'Terminal Bench Green Agent':
                    print(f"   Green agent tools: {agent.get('tools', [])}")
                    break
    except ImportError:
        print("⚠️  toml not installed (optional for this check)")
    except Exception as e:
        print(f"❌ Config validation failed: {e}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if all_pass:
        print("✅ All structure checks passed!")
        print("\nThe integration structure is correct.")
        print("\nTo complete setup, install dependencies:")
        print("  1. pip install -r green-white-agent/requirements.txt")
        print("  2. pip install uvicorn fastapi httpx a2a-sdk openai-agents openai toml")
        print("  3. cd green-white-agent && pip install -e .")
        return 0
    else:
        print("❌ Some structure checks failed.")
        print("Please verify the file paths are correct.")
        return 1

if __name__ == "__main__":
    sys.exit(check_structure())

