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
# Start the A2A-compatible server
python white_agent/agent.py --server

# Start on custom port/host
python white_agent/agent.py --server --port 8001 --host 0.0.0.0
```

### Testing the Agent

```bash
# Run local tests
python white_agent/agent.py --test

# Run comprehensive A2A integration tests
python test_a2a_integration.py
```

### Converting Terminal Bench Problems

```bash
# Convert problems to A2A format and send to agent
python terminal_bench_to_a2a_converter.py sample_terminal_bench.json

# Create A2A test suite
python terminal_bench_to_a2a_converter.py sample_terminal_bench.json --create-test-suite

# Check agent health
python terminal_bench_to_a2a_converter.py --check-agent

# Get agent card
python terminal_bench_to_a2a_converter.py --agent-card
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
├── green_agent/
│   ├── __init__.py
│   ├── agent.py              # Main green agent implementation
│   └── dataset_loaders/      # Dataset loading utilities
│       └── terminal_bench_loader.py
├── white_agent/
│   ├── agent.py              # A2A-compatible white agent
│   └── requirements.txt      # White agent dependencies
├── terminal_bench_to_a2a_converter.py  # A2A format converter
├── test_a2a_integration.py   # A2A integration tests
├── test_a2a_setup.py         # Setup validation tests
├── sample_terminal_bench.json # Sample terminal bench problems
├── terminal_bench_a2a_test_suite.json # A2A test suite
├── requirements.txt          # Main dependencies
├── setup.py
└── README.md
```

## 🧪 Testing

### Run All Tests

```bash
# Quick agent test
python white_agent/agent.py --test

# A2A protocol compliance tests (requires server running)
# Terminal 1: Start server
python white_agent/agent.py --server --port 8002

# Terminal 2: Run tests
python test_a2a_protocol.py

# Legacy integration tests
python test_a2a_integration.py

# Setup validation
python test_a2a_setup.py
```