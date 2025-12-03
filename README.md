# Green-White Agent: A2A Compatible Terminal Bench Agent

## Features

- **A2A Protocol Support**: Full Agent-to-Agent protocol compatibility
- **Terminal Bench Integration**: Convert and process terminal bench problems
- **FastAPI Server**: RESTful API endpoints for A2A communication
- **Dual Agent Architecture**: Green agent (evaluation) + White agent (problem-solving)
- **OpenAI Tool Calling**: Uses OpenAI function calling format for tool execution
- **Trajectory Logging**: Complete interaction history saved in JSON format
- **Docker Sandbox**: Isolated task execution with Docker containers
- **Health Monitoring**: Built-in health checks and agent discovery

## Prerequisites

- Python 3.12+
- Docker (installed and running)
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Usage

### Starting the White Agent Server

First, set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Then start the server:
```bash
python -m white_agent.agent --server --port 8001
```

The white agent expects tools in OpenAI function calling format and maintains conversation history with tool calls and results.

### Running Terminal Bench Tasks

Run tasks with trajectory logging:
```bash
# Use default output directory (trajectories/)
python examples/demo_real_terminalbench.py

# Specify custom output directory
python examples/demo_real_terminalbench.py --output-directory my_logs
```

Trajectories are saved to `{output_directory}/{task_name}/trajectory_{timestamp}.json` and include:
- Initial task prompt and tool definitions
- All agent-to-agent interactions
- Tool execution requests and responses
- Evaluation results

### Using the Green Agent

```python
from green_agent import GreenAgentTerminalBench

# Create agent (requires white agent server running)
agent = GreenAgentTerminalBench(
    white_agent_url="http://localhost:8001",
    trajectory_output_dir="trajectories"  # Optional: enable trajectory logging
)

# Load and run Terminal Bench tasks
tasks = agent.load_terminal_bench_tasks(["hello-world"], limit=1)
result = agent.execute_task_with_sandbox(tasks[0])

print(f"Success: {result.success}")
print(f"Score: {result.evaluation_result.score}")
```

## Tool Calling Format

Both agents use OpenAI's function calling format. Tools are defined as:

```python
{
    "name": "execute_bash_command",
    "description": "Execute a bash command in the task container",
    "inputSchema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute"
            }
        },
        "required": ["command"]
    }
}
```

Tool calls follow OpenAI's format with `id`, `type`, and `function` fields. Tool results are returned as JSON-RPC tool messages.

## A2A API Endpoints

When running the server, the following endpoints are available:

- `GET /` - Server status
- `GET /health` - Health check
- `GET /.well-known/agent-card` - Agent discovery (A2A standard)
- `POST /` - JSON-RPC 2.0 endpoint for A2A messages

### JSON-RPC Methods

- `message/send` - Send a message or tool results to the agent
- `task/get` - Retrieve task by ID
- `task/cancel` - Cancel a running task

Make sure Docker is installed and running:
```bash
# Verify Docker works
docker ps

# If not running, start Docker Desktop (Mac/Windows)
# or start Docker daemon (Linux):
sudo systemctl start docker
```

## Project Structure

```
green-white-agent/
├── green_agent/              # Green Agent (Evaluation)
│   ├── __init__.py
│   ├── terminal_bench_runner.py  # Main runner with trajectory logging
│   ├── sandbox_manager.py        # Docker sandbox isolation
│   ├── task_evaluator.py         # Task evaluation with pytest
│   └── dataset_loaders/
│       └── terminal_bench_loader.py
├── white_agent/              # White Agent (Problem Solving)
│   ├── __init__.py
│   ├── agent.py                 # A2A-compatible agent with OpenAI
│   └── a2a_protocol.py          # A2A protocol models
├── examples/                 # Example scripts
│   ├── demo_real_terminalbench.py  # Run tasks with trajectory logging
│   └── demo_green_agent.py
├── tests/                    # Test suite
└── trajectories/             # Default trajectory output directory
```

## Testing

### Quick Start

1. Start the white agent server:
```bash
export OPENAI_API_KEY="your-api-key"
python -m white_agent.agent --server --port 8001
```

2. Run Terminal Bench tasks (in another terminal):
```bash
python examples/demo_real_terminalbench.py
```

3. View trajectories:
```bash
cat trajectories/hello-world/trajectory_*.json
```

### Docker Monitoring

```bash
# Watch Docker containers
docker ps

# View container logs
docker logs <container_name>

# Clean up Docker resources
docker system prune -a
```

## Trajectory Format

Each trajectory file contains:

```json
{
  "task_id": "hello-world",
  "task": { ... },
  "start_time": "2025-12-03T04:09:17.641333+00:00",
  "interactions": [
    {
      "iteration": 1,
      "timestamp": "...",
      "user": {
        "type": "initial_task",
        "rpc_request": { ... }
      },
      "assistant": { ... },
      "tool_executions": [ ... ]
    }
  ],
  "evaluation": {
    "passed": true,
    "score": 1.0
  },
  "end_time": "...",
  "execution_time": 12.34
}
```