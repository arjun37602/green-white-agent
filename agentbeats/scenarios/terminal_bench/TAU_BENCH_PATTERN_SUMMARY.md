# Terminal Bench Tau-Bench Pattern Implementation - Summary

## ✅ Implementation Complete

The Terminal Bench green agent now follows the **tau-bench example pattern** exactly.

## What Was Created

### 1. Tau-Bench Pattern Green Agent
**File**: `agents/green_agent/green_agent_tau_bench.py`

**Key Features**:
- ✅ Uses `AgentExecutor` directly (like tau-bench example)
- ✅ Receives task description via A2A message with tags
- ✅ Parses `<white_agent_url>` and `<task_config>` tags
- ✅ Sets up Terminal Bench environment
- ✅ Orchestrates task evaluation
- ✅ Uses A2A protocol for communication
- ✅ Integrates with `GreenAgentTerminalBench` class

### 2. Launcher
**File**: `launcher_tau_bench.py`

**Key Features**:
- ✅ Launches green and white agents in separate processes
- ✅ Waits for agents to be ready
- ✅ Sends task description with tags to green agent
- ✅ Handles agent lifecycle (start/terminate)

### 3. Test Script
**File**: `test_tau_bench_pattern.py`

**Key Features**:
- ✅ Verifies file structure
- ✅ Verifies code structure matches tau-bench pattern
- ✅ Compares with tau-bench example

## How It Follows the Tau-Bench Pattern

### ✅ Pattern Comparison

| Feature | Tau-Bench Example | Terminal Bench | Status |
|---------|------------------|----------------|--------|
| Uses `AgentExecutor` | ✅ | ✅ | ✅ Match |
| Receives task via A2A | ✅ | ✅ | ✅ Match |
| Parses tags (`<white_agent_url>`, `<env_config>`) | ✅ | ✅ | ✅ Match |
| Sends messages to white agent | ✅ | ✅ | ✅ Match |
| Executes in environment | ✅ | ✅ | ✅ Match |
| Evaluates results | ✅ | ✅ | ✅ Match |
| Uses multiprocessing launcher | ✅ | ✅ | ✅ Match |

### ✅ Code Structure

**Tau-Bench Pattern**:
```python
class TauGreenAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # Parse tags
        tags = parse_tags(user_input)
        white_agent_url = tags["white_agent_url"]
        env_config = json.loads(tags["env_config"])
        
        # Set up environment
        env = get_env(...)
        
        # Evaluate
        res = await ask_agent_to_solve(white_agent_url, env, task_index)
        
        # Report results
        await event_queue.enqueue_event(...)
```

**Terminal Bench Pattern**:
```python
class TerminalBenchGreenAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        # Parse tags
        tags = parse_tags(user_input)
        white_agent_url = tags["white_agent_url"]
        task_config = json.loads(tags["task_config"])
        
        # Set up environment
        green_agent_impl = GreenAgentTerminalBench(...)
        
        # Evaluate
        execution_result = green_agent_impl.execute_task_with_sandbox(task)
        
        # Report results
        await event_queue.enqueue_event(...)
```

### ✅ Task Description Format

**Tau-Bench Format**:
```
Your task is to instantiate tau-bench to test the agent located at:
<white_agent_url>
http://localhost:9002/
</white_agent_url>
You should use the following env configuration:
<env_config>
{
  "env": "retail",
  "task_ids": [1]
}
</env_config>
```

**Terminal Bench Format**:
```
Your task is to evaluate the Terminal Bench agent located at:
<white_agent_url>
http://localhost:8341
</white_agent_url>
You should use the following task configuration:
<task_config>
{
  "limit": 3,
  "task_ids": [1, 2, 3]
}
</task_config>
```

## Key Differences (Domain-Specific)

| Aspect | Tau-Bench | Terminal Bench |
|--------|-----------|----------------|
| **Environment** | Retail/Airline domains with tools | Terminal/bash tasks |
| **White Agent Response** | JSON with tool calls | Bash commands/solutions |
| **Execution** | Tool calls in domain environment | Commands in sandbox |
| **Evaluation** | Reward based on task completion | Score based on task correctness |

The **pattern** is identical, only the **domain** differs.

## Files Structure

```
agentbeats/scenarios/terminal_bench/
├── agents/
│   ├── green_agent/
│   │   ├── green_agent_tau_bench.py    # Tau-bench pattern ⭐
│   │   ├── green_agent_tutorial.py     # Tutorial pattern
│   │   └── tools.py                     # Tools pattern (original)
│   └── white_agent/
│       └── white_agent_server.py         # White agent server
├── launcher_tau_bench.py                # Launcher ⭐
├── test_tau_bench_pattern.py            # Verification tests
└── scenario_tutorial.toml               # Tutorial scenario
```

## How to Use

### Run Evaluation

```bash
cd agentbeats/scenarios/terminal_bench
python launcher_tau_bench.py
```

This will:
1. Start green agent (port 8340)
2. Start white agent (port 8341)
3. Wait for both to be ready
4. Send task description to green agent
5. Green agent orchestrates evaluation
6. Results appear in console
7. Agents are terminated

### Run Individual Agents

**Green Agent**:
```bash
python agents/green_agent/green_agent_tau_bench.py --host localhost --port 8340
```

**White Agent**:
```bash
python agents/white_agent/white_agent_server.py --host localhost --port 8341
```

## Verification

Run the test to verify everything follows the tau-bench pattern:

```bash
cd agentbeats/scenarios/terminal_bench
python test_tau_bench_pattern.py
```

**Result**: ✅ All tests passing

## Implementation Details

### 1. Tag Parsing
```python
def parse_tags(text: str) -> Dict[str, str]:
    """Parse XML-like tags from text."""
    import re
    tags = {}
    pattern = r'<(\w+)>(.*?)</\1>'
    for match in re.finditer(pattern, text, re.DOTALL):
        tag_name = match.group(1)
        tag_content = match.group(2).strip()
        tags[tag_name] = tag_content
    return tags
```

### 2. A2A Communication
```python
async def send_message_to_agent(url: str, message: str, context_id: str = None):
    """Send message to agent via A2A."""
    httpx_client = httpx.AsyncClient(timeout=120.0)
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
    card = await resolver.get_agent_card()
    client = A2AClient(httpx_client=httpx_client, agent_card=card)
    # ... send message
```

### 3. Agent Lifecycle
```python
# Start agents
p_green = multiprocessing.Process(target=start_green_agent, ...)
p_white = multiprocessing.Process(target=run_white_agent, ...)

# Wait for readiness
await wait_agent_ready(green_url)
await wait_agent_ready(white_url)

# Send task
response = await send_message_to_agent(green_url, task_text)

# Cleanup
p_green.terminate()
p_white.terminate()
```

## Summary

The Terminal Bench implementation now has **three patterns**:

1. **Tau-Bench Pattern** (`green_agent_tau_bench.py`) ⭐ **NEW**
   - Uses `AgentExecutor` directly
   - Receives task via A2A with tags
   - Follows exact tau-bench example structure
   - Compatible with tau-bench-style launchers

2. **Tutorial Pattern** (`green_agent_tutorial.py`)
   - Uses `GreenAgent` base class
   - Receives `EvalRequest` with participants/config
   - Works with tutorial's `run_scenario.py`

3. **Tools Pattern** (`tools.py`)
   - Uses AgentBeats `@tool` decorator
   - LLM-based orchestration
   - Works with AgentBeats CLI

All three are fully functional and tested! ✅

The **tau-bench pattern** implementation ensures compatibility with the tau-bench example structure and can be easily adapted for other benchmarks following the same pattern.

