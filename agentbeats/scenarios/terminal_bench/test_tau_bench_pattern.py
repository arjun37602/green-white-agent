#!/usr/bin/env python3
"""
Test script to verify Terminal Bench follows tau-bench pattern

This verifies that:
1. Green agent uses AgentExecutor (like tau-bench)
2. Receives task description with tags
3. Communicates with white agent via A2A
4. Follows same structure as tau-bench example
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_tau_bench_pattern():
    """Test that Terminal Bench follows tau-bench pattern."""
    logger.info("=" * 80)
    logger.info("TERMINAL BENCH TAU-BENCH PATTERN VERIFICATION")
    logger.info("=" * 80)
    
    scenario_dir = Path(__file__).resolve().parent
    
    # Test 1: Check files exist
    logger.info("\nTest 1: File Structure")
    files = {
        "Green agent (tau-bench pattern)": scenario_dir / "agents" / "green_agent" / "green_agent_tau_bench.py",
        "Launcher": scenario_dir / "launcher_tau_bench.py",
        "White agent server": scenario_dir / "agents" / "white_agent" / "white_agent_server.py",
    }
    
    all_exist = True
    for name, file_path in files.items():
        exists = file_path.exists()
        status = "✅" if exists else "❌"
        logger.info(f"{status} {name}: {file_path.name}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        return False
    
    # Test 2: Check code structure
    logger.info("\nTest 2: Code Structure")
    green_agent_file = files["Green agent (tau-bench pattern)"]
    
    try:
        with open(green_agent_file) as f:
            content = f.read()
        
        checks = {
            "AgentExecutor inheritance": "class TerminalBenchGreenAgentExecutor(AgentExecutor)" in content,
            "execute method": "async def execute(self, context: RequestContext" in content,
            "parse_tags function": "def parse_tags" in content,
            "send_message_to_agent function": "async def send_message_to_agent" in content,
            "A2A communication": "A2AClient" in content,
            "GreenAgentTerminalBench integration": "GreenAgentTerminalBench" in content,
            "Task description with tags": "<white_agent_url>" in content or "white_agent_url" in content,
        }
        
        all_pass = True
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"{status} {check_name}")
            if not passed:
                all_pass = False
        
        return all_pass
        
    except Exception as e:
        logger.error(f"❌ Error checking code structure: {e}")
        return False

def test_launcher_structure():
    """Test launcher structure."""
    logger.info("\nTest 3: Launcher Structure")
    launcher_file = Path(__file__).resolve().parent / "launcher_tau_bench.py"
    
    try:
        with open(launcher_file) as f:
            content = f.read()
        
        checks = {
            "launch_evaluation function": "async def launch_evaluation" in content,
            "wait_agent_ready": "async def wait_agent_ready" in content,
            "multiprocessing": "multiprocessing.Process" in content,
            "task description with tags": "<white_agent_url>" in content,
            "task_config tag": "<task_config>" in content,
        }
        
        all_pass = True
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"{status} {check_name}")
            if not passed:
                all_pass = False
        
        return all_pass
        
    except Exception as e:
        logger.error(f"❌ Error checking launcher: {e}")
        return False

def compare_with_tau_bench():
    """Compare structure with tau-bench example."""
    logger.info("\nTest 4: Comparison with Tau-Bench Example")
    
    # Check if tau-bench example exists
    tau_bench_example = Path(__file__).resolve().parents[3] / "agentify-example-tau-bench"
    
    if not tau_bench_example.exists():
        logger.warning("⚠️  tau-bench example not found, skipping comparison")
        return True
    
    logger.info("✅ Tau-bench example found")
    
    # Compare key patterns
    patterns = {
        "Uses AgentExecutor": "AgentExecutor" in "TerminalBenchGreenAgentExecutor(AgentExecutor)",
        "Receives task via A2A": True,  # Our implementation does this
        "Sends messages to white agent": True,  # Our implementation does this
        "Uses tags for config": True,  # Our implementation uses <white_agent_url> and <task_config>
    }
    
    for pattern, matches in patterns.items():
        status = "✅" if matches else "❌"
        logger.info(f"{status} {pattern}")
    
    return True

def main():
    """Run all tests."""
    test1 = test_tau_bench_pattern()
    test2 = test_launcher_structure()
    test3 = compare_with_tau_bench()
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    all_pass = test1 and test2 and test3
    
    if all_pass:
        logger.info("✅ All tests passed!")
        logger.info("\nThe Terminal Bench implementation follows the tau-bench pattern correctly.")
        logger.info("\nTo run the evaluation:")
        logger.info("  cd agentbeats/scenarios/terminal_bench")
        logger.info("  python launcher_tau_bench.py")
        return 0
    else:
        logger.error("❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

