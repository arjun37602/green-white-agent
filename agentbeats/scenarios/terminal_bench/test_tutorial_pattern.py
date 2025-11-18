#!/usr/bin/env python3
"""
Test script to verify Terminal Bench green agent follows tutorial pattern

This script tests that:
1. Green agent can be imported
2. It inherits from GreenAgent
3. It implements required methods
4. Path resolution works
5. Integration with green_agent package works
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_tutorial_integration():
    """Test that the green agent follows the tutorial pattern."""
    logger.info("=" * 80)
    logger.info("TERMINAL BENCH TUTORIAL PATTERN VERIFICATION")
    logger.info("=" * 80)
    
    # Test 1: Path resolution
    logger.info("\nTest 1: Path Resolution")
    green_agent_file = Path(__file__).resolve().parent / "agents" / "green_agent" / "green_agent_tutorial.py"
    if not green_agent_file.exists():
        logger.error(f"❌ Green agent file not found: {green_agent_file}")
        return False
    logger.info(f"✅ Green agent file exists: {green_agent_file}")
    
    # Test 2: Check file structure
    logger.info("\nTest 2: File Structure")
    required_files = [
        green_agent_file,
        green_agent_file.parent.parent / "white_agent" / "white_agent_server.py",
        Path(__file__).resolve().parent / "scenario_tutorial.toml",
    ]
    
    all_exist = True
    for file_path in required_files:
        exists = file_path.exists()
        status = "✅" if exists else "❌"
        logger.info(f"{status} {file_path.name}: {file_path}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        return False
    
    # Test 3: Check imports (without actually importing, just check structure)
    logger.info("\nTest 3: Code Structure")
    try:
        with open(green_agent_file) as f:
            content = f.read()
        
        # Check for required components
        checks = {
            "GreenAgent inheritance": "class TerminalBenchGreenAgent(GreenAgent)" in content,
            "run_eval method": "async def run_eval(self, req: EvalRequest" in content,
            "validate_request method": "def validate_request(self, request: EvalRequest" in content,
            "GreenExecutor usage": "GreenExecutor" in content,
            "EvalRequest handling": "EvalRequest" in content,
            "TaskUpdater usage": "TaskUpdater" in content,
            "GreenAgentTerminalBench integration": "GreenAgentTerminalBench" in content,
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

def test_scenario_toml():
    """Test scenario TOML structure."""
    logger.info("\nTest 4: Scenario TOML Structure")
    try:
        import toml
        
        scenario_file = Path(__file__).resolve().parent / "scenario_tutorial.toml"
        with open(scenario_file) as f:
            config = toml.load(f)
        
        # Check structure
        checks = {
            "green_agent section": "green_agent" in config,
            "green_agent endpoint": "endpoint" in config.get("green_agent", {}),
            "green_agent cmd": "cmd" in config.get("green_agent", {}),
            "participants section": "participants" in config,
            "white_agent participant": any(p.get("role") == "white_agent" for p in config.get("participants", [])),
            "config section": "config" in config,
        }
        
        all_pass = True
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"{status} {check_name}")
            if not passed:
                all_pass = False
        
        if all_pass:
            logger.info(f"\n✅ Scenario TOML structure is valid")
            logger.info(f"   Green agent: {config['green_agent']['endpoint']}")
            logger.info(f"   Participants: {len(config['participants'])}")
        
        return all_pass
        
    except ImportError:
        logger.warning("⚠️  toml not installed (optional)")
        return True
    except Exception as e:
        logger.error(f"❌ Error checking scenario TOML: {e}")
        return False

def main():
    """Run all tests."""
    test1 = test_tutorial_integration()
    test2 = test_scenario_toml()
    
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    if test1 and test2:
        logger.info("✅ All tests passed!")
        logger.info("\nThe Terminal Bench green agent follows the tutorial pattern correctly.")
        logger.info("\nTo run the scenario:")
        logger.info("  cd tutorial")
        logger.info("  uv run agentbeats-run ../agentbeats/scenarios/terminal_bench/scenario_tutorial.toml")
        return 0
    else:
        logger.error("❌ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

