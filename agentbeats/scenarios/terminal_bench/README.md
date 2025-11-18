# AgentBeats Integration for Terminal Bench

This directory contains the AgentBeats integration for the Terminal Bench green-white-agent system.

## Structure

```
agentbeats/scenarios/terminal_bench/
├── agents/
│   ├── green_agent/
│   │   ├── green_agent_tau_bench.py    # Tau-bench pattern implementation
│   │   ├── green_agent_tutorial.py     # Tutorial pattern implementation
│   │   ├── tools.py                     # AgentBeats tools pattern
│   │   └── agent_card.toml              # Agent card for tools pattern
│   └── white_agent/
│       ├── white_agent_server.py        # White agent server wrapper
│       └── agent_card.toml              # White agent card
├── launcher_tau_bench.py                # Launcher for tau-bench pattern
├── scenario.toml                         # AgentBeats scenario (tools pattern)
├── scenario_tutorial.toml                # Tutorial scenario
└── test_*.py                            # Test files
```

## Three Integration Patterns

### 1. Tau-Bench Pattern (`green_agent_tau_bench.py`)

Follows the tau-bench example pattern:
- Uses `AgentExecutor` directly
- Receives task via A2A message with tags (`<white_agent_url>`, `<task_config>`)
- Compatible with tau-bench-style launchers

**Usage:**
```bash
python agentbeats/scenarios/terminal_bench/launcher_tau_bench.py
```

### 2. Tutorial Pattern (`green_agent_tutorial.py`)

Follows the AgentBeats tutorial pattern:
- Uses `GreenAgent` base class
- Receives `EvalRequest` with participants and config
- Works with tutorial's `run_scenario.py`

**Usage:**
```bash
cd tutorial
uv run agentbeats-run ../green-white-agent/agentbeats/scenarios/terminal_bench/scenario_tutorial.toml
```

### 3. Tools Pattern (`tools.py`)

Original AgentBeats LLM-based approach:
- Uses `@tool` decorator
- Green agent is an LLM that decides which tools to use
- Works with AgentBeats CLI

**Usage:**
```bash
cd agentbeats
agentbeats run_scenario ../green-white-agent/agentbeats/scenarios/terminal_bench/scenario.toml
```

## Running Tests

```bash
# Test tau-bench pattern
python agentbeats/scenarios/terminal_bench/test_tau_bench_pattern.py

# Test tutorial pattern
python agentbeats/scenarios/terminal_bench/test_tutorial_pattern.py

# Test integration
python agentbeats/scenarios/terminal_bench/test_integration.py
```

## Path Resolution

All files use relative path resolution to find the green-white-agent root:
- Files in `agentbeats/scenarios/terminal_bench/` use `parents[4]` to reach green-white-agent root
- This allows the integration to work regardless of where green-white-agent is located

## Dependencies

The integration requires:
- `green-white-agent` package (available in parent directory)
- `a2a` protocol libraries (for A2A communication)
- `agentbeats` (optional, for tools pattern and tutorial pattern)
- `tutorial/src` (optional, for tutorial pattern)

All dependencies are handled gracefully with fallbacks when not available.

