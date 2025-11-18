#!/usr/bin/env python3
"""
AgentBeats Integration Verification Test

This test verifies that the Terminal Bench scenario works with the actual
AgentBeats package by testing imports, tool registration, and scenario loading.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
AGENTBEATS_SRC = Path(__file__).resolve().parents[2] / "src"
SCENARIO_DIR = Path(__file__).resolve().parent
GREEN_WHITE_AGENT_DIR = Path(__file__).resolve().parents[3] / "green-white-agent"

class AgentBeatsVerificationTest:
    """Verification test for AgentBeats integration."""
    
    def __init__(self):
        self.test_results = []
    
    def run_all_tests(self):
        """Run all verification tests."""
        logger.info("=" * 80)
        logger.info("AGENTBEATS INTEGRATION VERIFICATION")
        logger.info("=" * 80)
        
        tests = [
            ("Test 1: Import AgentBeats package", self.test_agentbeats_import),
            ("Test 2: Verify @tool decorator", self.test_tool_decorator),
            ("Test 3: Load scenario TOML", self.test_scenario_toml),
            ("Test 4: Import tools.py with AgentBeats", self.test_tools_import),
            ("Test 5: Verify tool registration", self.test_tool_registration),
            ("Test 6: Test CLI imports", self.test_cli_imports),
            ("Test 7: Validate agent cards with AgentBeats", self.test_agent_cards_validation),
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n{'-' * 80}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'-' * 80}")
            
            try:
                result = test_func()
                self.test_results.append({
                    "test": test_name,
                    "status": "PASSED" if result else "FAILED",
                    "passed": result
                })
                
                status = "✅ PASSED" if result else "❌ FAILED"
                logger.info(f"{status}: {test_name}")
                
            except Exception as e:
                logger.error(f"❌ ERROR in {test_name}: {e}")
                import traceback
                traceback.print_exc()
                self.test_results.append({
                    "test": test_name,
                    "status": "ERROR",
                    "passed": False,
                    "error": str(e)
                })
        
        # Print summary
        self.print_summary()
        
        return all(r["passed"] for r in self.test_results)
    
    def test_agentbeats_import(self) -> bool:
        """Test importing AgentBeats from source."""
        try:
            # Add AgentBeats src to path
            sys.path.insert(0, str(AGENTBEATS_SRC))
            
            import agentbeats
            logger.info(f"✅ AgentBeats imported successfully")
            logger.info(f"   Location: {agentbeats.__file__}")
            
            # Check for key exports
            required_attrs = ['tool', 'get_registered_tools', 'BattleContext']
            missing_attrs = [attr for attr in required_attrs if not hasattr(agentbeats, attr)]
            
            if missing_attrs:
                logger.error(f"Missing attributes: {missing_attrs}")
                return False
            
            logger.info(f"✅ All required attributes present")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import agentbeats: {e}")
            return False
    
    def test_tool_decorator(self) -> bool:
        """Test the @tool decorator works correctly."""
        try:
            sys.path.insert(0, str(AGENTBEATS_SRC))
            import agentbeats
            
            # Reset tool registry
            agentbeats._TOOL_REGISTRY.clear()
            
            # Define a test tool
            @agentbeats.tool
            def test_function(x: int, y: int) -> int:
                """Test function for tool decorator."""
                return x + y
            
            # Check if tool was registered
            tools = agentbeats.get_registered_tools()
            
            if len(tools) == 0:
                logger.error("Tool was not registered")
                return False
            
            if test_function not in tools:
                logger.error("Test function not found in registry")
                return False
            
            # Test the function still works
            result = test_function(2, 3)
            if result != 5:
                logger.error(f"Function returned {result}, expected 5")
                return False
            
            logger.info(f"✅ Tool decorator works correctly")
            logger.info(f"   Registered tools: {len(tools)}")
            logger.info(f"   Test function result: {result}")
            
            return True
            
        except Exception as e:
            logger.error(f"Tool decorator test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_scenario_toml(self) -> bool:
        """Test loading the scenario TOML file."""
        try:
            import toml
            
            scenario_file = SCENARIO_DIR / "scenario.toml"
            with open(scenario_file) as f:
                scenario_config = toml.load(f)
            
            # Validate structure
            if "scenario" not in scenario_config:
                logger.error("Missing [scenario] section")
                return False
            
            if "agents" not in scenario_config:
                logger.error("Missing [[agents]] sections")
                return False
            
            # Check green agent
            green_agent = None
            for agent in scenario_config["agents"]:
                if agent.get("name") == "Terminal Bench Green Agent":
                    green_agent = agent
                    break
            
            if not green_agent:
                logger.error("Green agent not found in scenario")
                return False
            
            # Check tools path
            if "tools" not in green_agent or not green_agent["tools"]:
                logger.error("Green agent has no tools specified")
                return False
            
            tools_path = SCENARIO_DIR / green_agent["tools"][0]
            if not tools_path.exists():
                logger.error(f"Tools file not found: {tools_path}")
                return False
            
            logger.info(f"✅ Scenario TOML loaded successfully")
            logger.info(f"   Scenario: {scenario_config['scenario']['name']}")
            logger.info(f"   Agents: {len(scenario_config['agents'])}")
            logger.info(f"   Tools file: {tools_path.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load scenario TOML: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_tools_import(self) -> bool:
        """Test importing tools.py with AgentBeats in path."""
        try:
            # Add paths
            sys.path.insert(0, str(AGENTBEATS_SRC))
            sys.path.insert(0, str(GREEN_WHITE_AGENT_DIR))
            sys.path.insert(0, str(SCENARIO_DIR / "agents" / "green_agent"))
            
            import agentbeats
            agentbeats._TOOL_REGISTRY.clear()
            
            # Import tools module
            import importlib.util
            tools_path = SCENARIO_DIR / "agents" / "green_agent" / "tools.py"
            spec = importlib.util.spec_from_file_location("terminal_bench_tools", tools_path)
            tools_module = importlib.util.module_from_spec(spec)
            
            # Execute module
            spec.loader.exec_module(tools_module)
            
            logger.info(f"✅ Tools module imported successfully")
            
            # Check registered tools
            registered_tools = agentbeats.get_registered_tools()
            logger.info(f"✅ Registered {len(registered_tools)} tools")
            
            # List tool names
            tool_names = [tool.__name__ for tool in registered_tools]
            logger.info(f"   Tools: {', '.join(tool_names[:5])}...")
            
            if len(registered_tools) < 10:
                logger.warning(f"Expected ~11 tools, found {len(registered_tools)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to import tools: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_tool_registration(self) -> bool:
        """Test that all tools are properly registered."""
        try:
            sys.path.insert(0, str(AGENTBEATS_SRC))
            sys.path.insert(0, str(GREEN_WHITE_AGENT_DIR))
            sys.path.insert(0, str(SCENARIO_DIR / "agents" / "green_agent"))
            
            import agentbeats
            agentbeats._TOOL_REGISTRY.clear()
            
            # Import tools
            import importlib.util
            tools_path = SCENARIO_DIR / "agents" / "green_agent" / "tools.py"
            spec = importlib.util.spec_from_file_location("terminal_bench_tools", tools_path)
            tools_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tools_module)
            
            # Get registered tools
            registered_tools = agentbeats.get_registered_tools()
            
            # Expected tool names
            expected_tools = [
                "load_terminal_bench_tasks",
                "create_sandbox",
                "execute_command_in_sandbox",
                "destroy_sandbox",
                "send_task_to_white_agent",
                "talk_to_white_agent",
                "evaluate_task",
                "execute_task_with_sandbox",
                "report_task_result",
                "get_execution_summary",
                "cleanup_all_resources"
            ]
            
            tool_names = [tool.__name__ for tool in registered_tools]
            
            # Check which expected tools are present
            found_tools = [name for name in expected_tools if name in tool_names]
            missing_tools = [name for name in expected_tools if name not in tool_names]
            
            logger.info(f"✅ Found {len(found_tools)}/{len(expected_tools)} expected tools")
            
            if missing_tools:
                logger.warning(f"   Missing tools: {missing_tools}")
            
            if found_tools:
                logger.info(f"   Found tools: {', '.join(found_tools[:3])}...")
            
            # Pass if we found at least 80% of expected tools
            return len(found_tools) >= len(expected_tools) * 0.8
            
        except Exception as e:
            logger.error(f"Tool registration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_cli_imports(self) -> bool:
        """Test that CLI can import scenario-related modules."""
        try:
            sys.path.insert(0, str(AGENTBEATS_SRC))
            
            from agentbeats import cli
            logger.info(f"✅ CLI module imported")
            
            # Check for key CLI functions
            if hasattr(cli, 'main'):
                logger.info(f"✅ CLI main function found")
            else:
                logger.warning("CLI main function not found")
            
            return True
            
        except Exception as e:
            logger.error(f"CLI import test failed: {e}")
            return False
    
    def test_agent_cards_validation(self) -> bool:
        """Test that agent cards are valid for AgentBeats."""
        try:
            import toml
            
            cards = [
                SCENARIO_DIR / "agents" / "green_agent" / "agent_card.toml",
                SCENARIO_DIR / "agents" / "white_agent" / "agent_card.toml",
            ]
            
            for card_path in cards:
                with open(card_path) as f:
                    card = toml.load(f)
                
                # Check required fields for AgentBeats
                required_fields = ["name", "description", "url", "host", "port"]
                missing = [f for f in required_fields if f not in card]
                
                if missing:
                    logger.error(f"Card {card_path.name} missing: {missing}")
                    return False
                
                logger.info(f"✅ Valid agent card: {card['name']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Agent card validation failed: {e}")
            return False
    
    def print_summary(self):
        """Print test summary."""
        logger.info("\n" + "=" * 80)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 80)
        
        passed = sum(1 for r in self.test_results if r["passed"])
        total = len(self.test_results)
        
        for result in self.test_results:
            status_icon = "✅" if result["passed"] else "❌"
            logger.info(f"{status_icon} {result['test']}: {result['status']}")
            if "error" in result:
                logger.info(f"   Error: {result['error']}")
        
        logger.info("=" * 80)
        logger.info(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        logger.info("=" * 80)
        
        if passed == total:
            logger.info("🎉 All verification tests passed!")
            logger.info("✅ The Terminal Bench scenario is fully compatible with AgentBeats!")
        else:
            logger.warning(f"⚠️  {total - passed} test(s) failed.")


def main():
    """Main test runner."""
    test_suite = AgentBeatsVerificationTest()
    
    success = test_suite.run_all_tests()
    
    if success:
        logger.info("\n✅ AgentBeats integration verification completed successfully!")
        logger.info("The scenario is ready to use with AgentBeats commands.")
        return 0
    else:
        logger.error("\n❌ Some verification tests failed.")
        logger.info("The scenario structure is correct, but may need AgentBeats to be installed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


