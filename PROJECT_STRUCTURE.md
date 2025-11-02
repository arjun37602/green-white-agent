# Project Structure

This document describes the organization of the green-white-agent project.

## Directory Layout

```
green-white-agent/
├── green_agent/              # Green Agent (Evaluation System)
│   ├── __init__.py           # Package initialization
│   ├── terminal_bench_runner.py  # Main runner for terminal bench evaluation
│   ├── sandbox_manager.py        # Sandbox environment isolation
│   ├── task_evaluator.py         # Task evaluation logic
│   └── dataset_loaders/
│       └── terminal_bench_loader.py  # Terminal-Bench dataset loader
│
├── white_agent/              # White Agent (Problem Solving System)
│   ├── __init__.py           # Package initialization
│   ├── agent.py              # Full OpenAI-powered A2A agent
│   ├── simple_agent.py       # Template-based test agent
│   ├── a2a_protocol.py       # A2A protocol models and definitions
│   └── requirements.txt      # White agent dependencies
│
├── examples/                 # Example Scripts and Demonstrations
│   ├── __init__.py           # Package initialization
│   ├── demo_green_agent.py          # Green agent demo
│   ├── demo_real_terminalbench.py   # Real terminal-bench integration demo
│   ├── demo_terminalbench_system.py # Complete system demo
│   ├── debug_demo.py                # Debugging demo
│   ├── debug_file_task.py          # File task debugging
│   ├── debug_sandbox_state.py      # Sandbox state debugging
│   └── debug_white_agent_response.py # White agent response debugging
│
├── tests/                    # Test Suite
│   ├── __init__.py           # Package initialization
│   ├── test_a2a_protocol.py  # A2A protocol compliance tests
│   ├── test_green_agent.py   # Green agent tests
│   ├── test_simple_tb_task.py # Simple terminal bench task tests
│   └── test_white_agent_debug.py # White agent debugging tests
│
├── scripts/                  # Utility Scripts
│   ├── __init__.py           # Package initialization
│   ├── run_agent.py          # Agent runner script
│   └── terminal_bench_to_a2a_converter.py  # Terminal bench to A2A converter
│
├── data/                     # Sample Data and Artifacts
│   └── .gitkeep             # Keep directory in git
│
├── venv/                     # Python virtual environment (git-ignored)
├── requirements.txt          # Main project dependencies
├── setup.py                  # Package setup configuration
├── README.md                 # Main project documentation
├── PROJECT_STRUCTURE.md      # This file
└── .gitignore               # Git ignore rules
```

## Component Descriptions

### Green Agent (Evaluation)
The green agent is responsible for loading terminal bench tasks, sending them to white agents, executing commands in isolated sandboxes, and evaluating the results. Key components:

- **terminal_bench_runner.py**: Main entry point that orchestrates task execution
- **sandbox_manager.py**: Creates isolated environments for safe command execution
- **task_evaluator.py**: Evaluates task completion across multiple dimensions
- **dataset_loaders/**: Loads and parses terminal-bench tasks

### White Agent (Problem Solving)
The white agent solves terminal bench problems. Two implementations:

- **agent.py**: Full-featured OpenAI agent with tool calling
- **simple_agent.py**: Template-based agent for testing (no API calls)
- **a2a_protocol.py**: A2A protocol data models and helper functions

### Examples
Demonstration scripts showing various capabilities:
- System demos
- Integration examples
- Debugging tools

### Tests
Comprehensive test suite covering:
- A2A protocol compliance
- Green agent functionality
- White agent behavior
- End-to-end integration

### Scripts
Utility tools for:
- Running agents
- Converting terminal bench to A2A format
- Format conversions

## Running Code

All scripts use path manipulation to import from the project root:

```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

This allows them to be run from anywhere in the filesystem:

```bash
# From project root
python examples/demo_green_agent.py

# Or from examples directory
cd examples
python demo_green_agent.py
```

## Adding New Files

When adding new files:

1. **Code files** → Choose the appropriate directory based on purpose
2. **Tests** → Add to `tests/` directory with `test_` prefix
3. **Examples** → Add to `examples/` directory
4. **Utility scripts** → Add to `scripts/` directory
5. **Data files** → Add to `data/` directory (and update `.gitignore` if needed)

Make sure to update this document when adding major new components!

