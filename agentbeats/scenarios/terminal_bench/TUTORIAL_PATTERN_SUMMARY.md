# Terminal Bench Tutorial Pattern Implementation - Summary

## ✅ Implementation Complete

The Terminal Bench green agent now follows the **tutorial pattern** as described in the tutorial README.

## What Was Created

### 1. Tutorial Pattern Green Agent
**File**: `agents/green_agent/green_agent_tutorial.py`

**Key Features**:
- ✅ Inherits from `GreenAgent` base class
- ✅ Implements `run_eval(request: EvalRequest, updater: TaskUpdater)`
- ✅ Implements `validate_request(request: EvalRequest) -> tuple[bool, str]`
- ✅ Uses `GreenExecutor` for A2A protocol handling
- ✅ Receives `EvalRequest` with `participants` (role-endpoint mapping) and `config`
- ✅ Uses `TaskUpdater` for status updates and artifacts
- ✅ Integrates with your existing `GreenAgentTerminalBench` class

### 2. White Agent Server Wrapper
**File**: `agents/white_agent/white_agent_server.py`

**Key Features**:
- ✅ Wraps your existing `A2ATerminalBenchServer`
- ✅ Compatible with tutorial's run_scenario.py
- ✅ Supports command-line arguments (host, port, model, card-url)

### 3. Tutorial Scenario Configuration
**File**: `scenario_tutorial.toml`

**Key Features**:
- ✅ Follows tutorial's scenario.toml format
- ✅ Defines green_agent with endpoint and cmd
- ✅ Defines participants (white_agent) with endpoint and cmd
- ✅ Includes config section for assessment parameters

## How It Follows the Tutorial Pattern

### ✅ Green Agent Principles (from README)

1. **Lightweight Orchestrator**
   - Green agent doesn't do heavy computation
   - Uses your `GreenAgentTerminalBench` class for actual work
   - Only orchestrates and evaluates

2. **Receives Assessment Request**
   ```python
   async def run_eval(self, req: EvalRequest, updater: TaskUpdater):
       # req.participants = {"white_agent": "http://..."}
       # req.config = {"limit": 5, "task_ids": [...]}
   ```

3. **Uses A2A Protocol**
   - Uses `ToolProvider` for A2A communication
   - Sends tasks to white agent via A2A
   - Receives solutions via A2A

4. **Produces Task Updates**
   ```python
   await updater.update_status(
       TaskState.working,
       new_agent_text_message("Loading tasks...")
   )
   ```

5. **Produces Artifacts**
   ```python
   await updater.add_artifact(
       parts=[...],
       name="Terminal Bench Evaluation Results"
   )
   ```

6. **Starts Fresh**
   - Each assessment initializes new `GreenAgentTerminalBench`
   - Cleans up resources in `finally` block
   - Resets tool provider

### ✅ Assessment Flow

```
1. Tutorial's run_scenario.py starts agents
   ↓
2. Tutorial's client_cli.py sends EvalRequest to green agent
   ↓
3. GreenExecutor receives request
   ↓
4. TerminalBenchGreenAgent.validate_request() validates
   ↓
5. TerminalBenchGreenAgent.run_eval() orchestrates:
   - Loads tasks
   - For each task:
     - Sends to white agent via A2A
     - Executes in sandbox
     - Evaluates results
   - Generates summary
   ↓
6. TaskUpdater sends status updates (visible in UI)
   ↓
7. TaskUpdater adds artifact with results
   ↓
8. Assessment complete
```

## Key Differences from Tools Pattern

### Tools Pattern (Original)
- Uses `@tool` decorator
- Green agent is an LLM that decides which tools to use
- More flexible but requires LLM for orchestration

### Tutorial Pattern (New)
- Inherits from `GreenAgent` base class
- Implements `run_eval()` method
- Direct orchestration logic (no LLM needed for structure)
- Follows exact tutorial example pattern
- More structured and reproducible

## Files Structure

```
agentbeats/scenarios/terminal_bench/
├── scenario_tutorial.toml          # Tutorial format scenario
├── agents/
│   ├── green_agent/
│   │   ├── green_agent_tutorial.py  # Tutorial pattern implementation
│   │   ├── agent_card.toml         # Original (tools pattern)
│   │   └── tools.py                # Original (tools pattern)
│   └── white_agent/
│       ├── white_agent_server.py   # Tutorial pattern wrapper
│       └── agent_card.toml         # Original
└── test_tutorial_pattern.py        # Verification tests
```

## How to Use

### Option 1: Tutorial Pattern (Recommended for Tutorial)

```bash
cd tutorial
uv run agentbeats-run ../agentbeats/scenarios/terminal_bench/scenario_tutorial.toml
```

This will:
1. Start green agent server (port 8340)
2. Start white agent server (port 8341)
3. Send EvalRequest to green agent
4. Green agent orchestrates evaluation
5. Results appear in console and as artifacts

### Option 2: AgentBeats Tools Pattern (Original)

```bash
cd agentbeats
agentbeats run_scenario scenarios/terminal_bench/scenario.toml
```

This uses the LLM-based tools pattern.

## Verification

Run the test to verify everything follows the tutorial pattern:

```bash
cd agentbeats/scenarios/terminal_bench
python test_tutorial_pattern.py
```

**Result**: ✅ All tests passing

## Key Implementation Details

### 1. EvalRequest Structure
```python
EvalRequest(
    participants={
        "white_agent": "http://127.0.0.1:8341"
    },
    config={
        "limit": 5,
        "task_ids": ["task1", "task2"]  # optional
    }
)
```

### 2. Status Updates
```python
# Progress updates visible in UI
await updater.update_status(
    TaskState.working,
    new_agent_text_message("Evaluating task 1/5...")
)
```

### 3. Artifacts
```python
# Final results as artifact
await updater.add_artifact(
    parts=[
        Part(root=TextPart(text=summary_text)),
        Part(root=TextPart(text=json_results)),
    ],
    name="Terminal Bench Evaluation Results"
)
```

### 4. Error Handling
```python
try:
    # ... evaluation logic
except Exception as e:
    await updater.update_status(
        TaskState.working,
        new_agent_text_message(f"Error: {str(e)}")
    )
    raise
finally:
    # Cleanup
    self._green_agent_impl.cleanup_resources()
    self._tool_provider.reset()
```

## Compliance with Tutorial README

✅ **Green agent orchestrates evaluations** - Yes, via `run_eval()`
✅ **Receives assessment_request** - Yes, via `EvalRequest`
✅ **Creates A2A task** - Yes, via `GreenExecutor`
✅ **Uses A2A protocol** - Yes, via `ToolProvider`
✅ **Produces task updates** - Yes, via `TaskUpdater`
✅ **Produces artifacts** - Yes, via `updater.add_artifact()`
✅ **Lightweight orchestrator** - Yes, delegates to `GreenAgentTerminalBench`
✅ **Starts fresh** - Yes, initializes new instance per assessment
✅ **Isolates contexts** - Yes, uses task_id for sandbox isolation

## Summary

The Terminal Bench green agent now has **two implementations**:

1. **Tutorial Pattern** (`green_agent_tutorial.py`)
   - Follows exact tutorial example
   - Uses `GreenAgent` base class
   - Structured, reproducible
   - Works with tutorial's `run_scenario.py`

2. **Tools Pattern** (`tools.py` + `agent_card.toml`)
   - Uses AgentBeats `@tool` decorator
   - LLM-based orchestration
   - More flexible
   - Works with AgentBeats CLI

Both are fully functional and tested! ✅

