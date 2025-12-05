# Green-White Agent: A2A Compatible Terminal Bench Agent

## Features

- **A2A Protocol Support**: Full Agent-to-Agent protocol compatibility
- **Terminal Bench Integration**: Evaluate agents on terminal bench tasks
- **Dual Agent Architecture**: Green agent (evaluation manager) + White agent (task executor)
- **Text-Based Tool Calling**: Uses JSON-wrapped tool calls (tau-bench style) instead of native OpenAI tool calling
- **Docker Sandbox**: Isolated task execution with Docker containers
- **Sequential Task Execution**: Evaluate multiple tasks in sequence
- **Health Monitoring**: Built-in health checks and agent discovery

## Prerequisites

- Python 3.10+
- Docker (installed and running)
- OpenAI API key (set in `.env` file)

## Installation

```bash
source setup_conda.sh
```

## Usage

### Quick Start - Launch Complete Evaluation

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your-api-key-here
```

Then run the launcher:
```bash
python launcher.py
```

This will:
1. Start the green agent on port 9001
2. Start the white agent on port 9002
3. Execute the tasks defined in `launcher.py`
4. Report results and terminate both agents

### Customizing Tasks

Edit `launcher.py` to change which tasks are evaluated:

```python
task_config = {
    "task_ids": ["hello-world", "csv-to-parquet"],  # Add task IDs here
    "dataset_path": "data/tasks"
}
```

## Tool Calling Format

The system uses **text-based tool calling** (tau-bench style) instead of native OpenAI function calling:

### Tools are defined in OpenAI format:

```python
{
    "type": "function",
    "function": {
        "name": "execute_bash_command",
        "description": "Execute a bash command in the task container",
        "parameters": {
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
}
```

### Agent responses use JSON tags:

```
<json>
{"name": "execute_bash_command", "kwargs": {"command": "ls -la"}}
</json>
```

### Tool results are sent as plain text:

```
Tool call result for execute_bash_command:
total 12
drwxr-xr-x 2 root root 4096 Jan 1 00:00 .
-rw-r--r-- 1 root root  101 Nov 19 00:33 data.csv
```

## A2A API Endpoints

When running as servers, both agents expose:

- `GET /.well-known/agent-card.json` - Agent discovery (A2A standard)
- `POST /` - JSON-RPC 2.0 endpoint for A2A messages

### JSON-RPC Methods

- `message/send` - Send a message or tool results to the agent
- `task/get` - Retrieve task by ID
- `task/cancel` - Cancel a running task

## Project Structure (simplified)

```
green-white-agent/
├── launcher.py              # Main entry point - launches evaluation
├── utils/                   # Shared utilities (A2A helpers, tag parsing)
│   └── utils.py
├── green_agent/             # Green Agent (evaluation manager)
│   ├── agent.py                     # A2A server executor
│   ├── terminal_bench_runner.py     # Task execution orchestrator
│   ├── task_evaluator.py            # pytest-based task evaluation
│   ├── sandbox_manager.py           # Docker sandbox management
│   └── dataset_loaders/
│       └── terminal_bench_loader.py # Load tasks from dataset
├── white_agent/             # White Agent (Task Executor)
│   ├── agent.py                     # Simple LLM agent with text-based tools
│   └── __init__.py
├── data/tasks/              # Terminal Bench task definitions
└── results/                 # Evaluation results
```

## Docker Requirements

Make sure Docker is installed and running:
```bash
# Verify Docker works
docker ps

# If not running, start Docker Desktop (Mac/Windows)
# or start Docker daemon (Linux):
sudo systemctl start docker
```

## Configuration

### Green Agent Configuration

Configure the green agent in `green_agent/terminal_bench_green_agent.toml`:
- Agent name, description, and capabilities
- Skills and tags for agent discovery

### White Agent Model

The white agent model can be configured in `white_agent/agent.py`:
```python
def __init__(self, model="gpt-4o"):
    self.model = model
```

### Timeout Settings

The A2A client timeout is set in `utils/utils.py` (default: 600 seconds).

## Architecture

### Green Agent (Evaluation Manager)
- Receives evaluation requests via A2A protocol
- Loads Terminal Bench tasks from dataset
- Creates Docker sandboxes for task execution
- Sends tasks to white agent with tool definitions
- Executes tools requested by white agent
- Sends tool results back to white agent
- Evaluates task completion using pytest
- Reports results and metrics

### White Agent (Task Executor)
- Simple LLM-based agent using litellm
- Receives tasks as plain text with embedded tool definitions
- Responds with tool calls in JSON format within `<json>` tags
- Receives tool results as plain text
- Maintains conversation history per context

### Text-Based Tool Calling Flow

1. **Green → White**: Task description + tools (as JSON text)
2. **White → Green**: Response with `<json>{"name": "tool", "kwargs": {...}}</json>`
3. **Green executes tool** in Docker container
4. **Green → White**: `"Tool call result for {tool}:\n{output}"`
5. Repeat until task complete or max iterations reached

## Troubleshooting

### Timeout Errors
If tasks take longer than 10 minutes, increase the timeout in `utils/utils.py`.

### Docker Container Issues
```bash
# List running containers
docker ps

# Stop specific container
docker stop <container_name>

# Clean up all stopped containers
docker container prune
```

### A2A Connection Issues
- Verify both agents are running (check ports 9001 and 9002)
- Check `.env` file has valid `OPENAI_API_KEY`
- Check firewall allows localhost connections
