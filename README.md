# Green-White Agent: Terminal Bench Evaluation System

A dual-agent system for evaluating LLM agents on Terminal Bench tasks. The **green agent** manages evaluation and task execution, while the **white agent** solves the tasks.

## Features

- **Dual Agent Architecture**: Green agent (evaluator) + White agent (task solver)
- **Two White Agent Types**: 
  - **Basic**: Simple LLM agent
  - **Evolved**: LLM agent with reflection and self-improvement
- **Docker Sandbox**: Isolated task execution
- **Parallel Execution**: Run multiple tasks concurrently
- **Result Caching**: JSONL-based result caching for efficient re-runs
- **AgentBeats Integration**: Run agents on AgentBeats evaluation platform

## Prerequisites

- Python 3.10+
- Docker (installed and running)
- Conda (for environment setup)
- OpenAI API key

## Setup

### 1. Create Conda Environment

```bash
source setup_conda.sh
conda activate green-white-agent
```

### 2. Cache Terminal Bench Dataset

Download and cache the Terminal Bench dataset:

```bash
python -c "from datasets import load_dataset; load_dataset('agentsea/terminal-bench-core', split='test')"
```

This downloads the dataset to `~/.cache/terminal-bench/terminal-bench-core/0.1.1`.

### 3. Setup Docker (Green Agent Only)

The green agent requires Docker for sandboxed task execution. Verify Docker is running:

```bash
docker ps
```

If Docker is not running:
- **Mac/Windows**: Start Docker Desktop
- **Linux**: `sudo systemctl start docker`

### 4. Create `.env` File

Create a `.env` file in the project root with your OpenAI API key:

```bash
OPENAI_API_KEY=your-api-key-here
```

## Running Evaluations

### Local Execution (via launcher.py)

The launcher starts both green and white agents locally and coordinates evaluation.

#### Run with Default Settings

```bash
python launcher.py
```

This runs the "hello-world" task with the evolved white agent using gpt-5.

#### Run with Custom Tasks

```bash
python launcher.py --task-ids hello-world csv-to-parquet fix-permissions --model gpt-4o --agent basic
```

#### Run All Tasks

```bash
python launcher.py --all-tasks --max-parallel-tasks 5
```

#### Common Options

- `--model`: Model to use (default: `gpt-5`)
  - Examples: `gpt-5`, `gpt-5-nano`, `gpt-4o`, `claude-3-5-sonnet-20241022`
- `--agent`: White agent type (default: `evolved`)
  - `evolved`: Agent with reflection capabilities
  - `basic`: Simple agent without reflection
- `--task-ids`: Specific tasks to evaluate (space-separated)
- `--all-tasks`: Evaluate all tasks in dataset
- `--max-parallel-tasks`: Max parallel tasks (default: 5)
- `--max-attempts`: Max retry attempts per task (default: 1)
- `--results-dir`: Results directory (default: `./results`)
- `--limit`: Limit number of tasks (e.g., `--limit 10` for first 10 tasks)

#### Output Structure

```
results/
├── gpt-5.jsonl                    # Cached results (JSONL format)
└── 20231218_143022_a1b2c3d4/      # Run-specific outputs
    ├── sessions/                   # Task execution sessions
    ├── agent-logs/                 # Agent interaction logs
    ├── trajectories.json           # Message histories (evolved agent only)
    └── summary/
        └── evaluation_summary.json # Evaluation summary
```

### AgentBeats Evaluation

Run agents on the AgentBeats platform for remote evaluation.

#### Basic White Agent

```bash
cd base_white_agent
./start_agentbeats.sh your-domain.ngrok.io
```

#### Evolved White Agent

```bash
cd evolved_white_agent
./start_agentbeats.sh your-domain.ngrok.io
```

#### Green Agent

```bash
cd green_agent
./start_agentbeats.sh your-green-domain.ngrok.io
```

The scripts will:
1. Load `OPENAI_API_KEY` from `.env` (white agents only)
2. Export `HTTPS_ENABLED=true`
3. Export `CLOUDRUN_HOST=<your-domain>`
4. Run `agentbeats run_ctrl`

## Analyzing Results

Use `analyze_results.py` to compute high-level metrics from cached results:

```bash
python analyze_results.py results/gpt-5.jsonl
```

Output:

```
Terminal Bench Results Analysis
==================================================
Total Tasks: 15

Average Metrics:
  Score (Pass Rate):  73.33% (11/15 passed)
  Accuracy:           0.8267
  Tokens per Task:    1234.5
  Time per Task:      45.23s
  Turns per Task:     12.3
```

Save to file:

```bash
python analyze_results.py results/gpt-5.jsonl results/analysis.txt
```

## Understanding Logs

### JSONL Cache (`results/gpt-5.jsonl`)

Each line is a JSON object with task results:

```json
{
  "task_id": "hello-world",
  "success": true,
  "accuracy": 1.0,
  "num_tokens": 1234,
  "execution_time": 45.2,
  "num_turns": 12,
  "timestamp": "2023-12-18T14:30:22.123456"
}
```

### Session Logs (`sessions/`)

Detailed task execution logs including:
- Container setup and teardown
- Command executions and outputs
- File system state
- Test results

### Agent Logs (`agent-logs/`)

Raw A2A protocol messages between green and white agents:
- Task instructions sent to white agent
- Tool call requests and responses
- Evaluation results

### Trajectories (`trajectories.json`)

Full message histories for each task (evolved agent only):
- System prompts (including learned improvements)
- User messages (tasks and tool results)
- Assistant responses

## Key Commands Reference

### Run White Agent (Complete Task)

**Local:**
```bash
python launcher.py --task-ids <task-id> --agent evolved
```

**AgentBeats:**
```bash
cd evolved_white_agent
./start_agentbeats.sh your-domain.ngrok.io
```

### Run White Agent + Green Agent (Evaluate)

**Local:**
```bash
python launcher.py --task-ids <task-id1> <task-id2> --agent basic
```

**AgentBeats:**
1. Start white agent (see above)
2. Start green agent:
   ```bash
   cd green_agent
   ./start_agentbeats.sh your-green-domain.ngrok.io
   ```

### Test Green Agent Evaluation

Test green agent's evaluation on specific tasks:

```bash
python launcher.py --task-ids hello-world --max-parallel-tasks 1
```

### Reproduce Benchmark Results

Run the full evaluation suite:

```bash
python launcher.py --all-tasks --model gpt-5 --agent evolved --max-parallel-tasks 5
```

Results are cached in `results/gpt-5.jsonl` for reproducibility.

## Project Structure

```
green-white-agent/
├── launcher.py                    # Main entry point
├── base_white_agent/              # Basic white agent
│   ├── agent.py
│   └── start_agentbeats.sh
├── evolved_white_agent/           # Evolved white agent with reflection
│   ├── agent.py
│   └── start_agentbeats.sh
├── green_agent/                   # Green agent (evaluator)
│   ├── agent.py
│   ├── terminal_bench_runner.py
│   ├── task_evaluator.py
│   ├── sandbox_manager.py
│   └── start_agentbeats.sh
├── utils/                         # Shared utilities
│   └── utils.py
├── analyze_results.py             # Results analysis tool
├── setup_conda.sh                 # Conda setup script
└── results/                       # Evaluation results
```

## Agent Types

### Basic White Agent

Simple LLM-based agent with:
- System prompt for terminal task solving
- Message history management
- Token usage tracking

### Evolved White Agent

Advanced agent with:
- All basic agent features
- **Reflection mechanism**: Analyzes performance every 10 turns
- **Self-improvement**: Updates system prompt with learned patterns
- **Trajectory tracking**: Records full conversation history

The evolved agent learns from experience and improves its strategy over time.

## Troubleshooting

### Docker Container Issues

```bash
# List running containers
docker ps

# Stop specific container
docker stop <container_name>

# Clean up all stopped containers
docker container prune
```

### OPENAI_API_KEY Not Found

Ensure `.env` file exists in project root with:
```
OPENAI_API_KEY=your-key-here
```

### Terminal Bench Dataset Not Found

Run the cache command again:
```bash
python -c "from datasets import load_dataset; load_dataset('agentsea/terminal-bench-core', split='test')"
```

### Port Already in Use

The launcher automatically finds free ports. If you encounter issues, check for processes using ports 9001+ and terminate them.

## License

See LICENSE file for details.
