# Green-White Agent: A2A Compatible Terminal Bench Agent


## 🚀 Features

- **A2A Protocol Support**: Full Agent-to-Agent protocol compatibility
- **Terminal Bench Integration**: Convert and process terminal bench problems
- **FastAPI Server**: RESTful API endpoints for A2A communication
- **Dual Agent Architecture**: Green agent (evaluation) + White agent (problem-solving)
- **Format Conversion**: Convert terminal bench problems to A2A format
- **Health Monitoring**: Built-in health checks and agent discovery
- **Comprehensive Testing**: Full test suite for A2A integration

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)

## 🛠️ Installation

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## 🎯 Usage

### Starting the A2A Server

```bash
# Start the simple template-based agent server (for testing)
python white_agent/simple_agent.py --server --port 8002

# Start the full OpenAI-powered agent server
python white_agent/agent.py --server --port 8001 --host 0.0.0.0
```

### Testing the Agent

```bash
# Run simple agent tests
python white_agent/simple_agent.py --test

# Run comprehensive A2A integration tests
python tests/test_a2a_protocol.py
```

### Converting Terminal Bench Problems

```bash
# Convert problems to A2A format and send to agent
python scripts/terminal_bench_to_a2a_converter.py sample_terminal_bench.json

# Create A2A test suite
python scripts/terminal_bench_to_a2a_converter.py sample_terminal_bench.json --create-test-suite

# Check agent health
python scripts/terminal_bench_to_a2a_converter.py --check-agent

# Get agent card
python scripts/terminal_bench_to_a2a_converter.py --agent-card
```

### Using the Green Agent

```python
from green_agent import GreenAgent

agent = GreenAgent()
agent.run_evaluation()
```

## 🌐 A2A API Endpoints

When running the server, the following endpoints are available:

- `GET /` - Server status
- `GET /health` - Health check
- `GET /agent-card` - Agent discovery information
- `POST /tasks` - Process A2A tasks
- `POST /tasks/` - Alternative task endpoint

### Example A2A Task

```json
{
  "artifacts": [
    {
      "parts": [
        {
          "type": "text",
          "text": "Write a bash script to find all files larger than 100MB"
        }
      ]
    }
  ],
  "metadata": {
    "problem_id": "find_large_files",
    "difficulty": "medium",
    "category": "file_operations"
  }
}
```

### Example A2A Response

```json
{
  "artifacts": [
    {
      "parts": [
        {
          "type": "text",
          "text": "Here's a bash script to find all files larger than 100MB:\n\n```bash\n#!/bin/bash\nfind . -type f -size +100M -exec ls -lh {} \\;\n```\n\nThis script uses the `find` command to locate files larger than 100MB and displays them with human-readable sizes."
        }
      ]
    }
  ],
  "status": "completed",
  "metadata": {
    "problem_id": "find_large_files",
    "difficulty": "medium",
    "category": "file_operations"
  }
}
```

## 🏗️ Project Structure

```
green-white-agent/
├── green_agent/              # Green Agent (Evaluation)
│   ├── __init__.py
│   ├── terminal_bench_runner.py  # Main runner
│   ├── sandbox_manager.py        # Sandbox isolation
│   ├── task_evaluator.py         # Task evaluation
│   └── dataset_loaders/
│       └── terminal_bench_loader.py
├── white_agent/              # White Agent (Problem Solving)
│   ├── __init__.py
│   ├── agent.py                 # A2A-compatible agent with OpenAI
│   ├── simple_agent.py          # Template-based test agent
│   ├── a2a_protocol.py          # A2A protocol models
│   └── requirements.txt
├── examples/                 # Example scripts and demos
│   ├── demo_green_agent.py
│   ├── demo_real_terminalbench.py
│   ├── demo_terminalbench_system.py
│   └── debug_*.py
├── tests/                    # Test suite
│   ├── test_a2a_protocol.py
│   ├── test_green_agent.py
│   └── test_*.py
├── scripts/                  # Utility scripts
│   ├── run_agent.py
│   └── terminal_bench_to_a2a_converter.py
├── data/                     # Sample data and artifacts
├── requirements.txt
├── setup.py
└── README.md
```

## 🧪 Testing

### Run All Tests

```bash
# Quick simple agent test
python white_agent/simple_agent.py --test

# A2A protocol compliance tests (requires server running)
# Terminal 1: Start simple agent server
python white_agent/simple_agent.py --server --port 8002

# Terminal 2: Run A2A protocol tests
python tests/test_a2a_protocol.py

# Green agent tests
python tests/test_green_agent.py

# Simple terminal bench task test
python tests/test_simple_tb_task.py
```

### Running Examples

```bash
# Demo green agent capabilities
python examples/demo_green_agent.py

# Demo complete terminal bench system
python examples/demo_terminalbench_system.py

# Demo with real terminal bench tasks
python examples/demo_real_terminalbench.py
```