# A2A Protocol Implementation

This document describes the complete A2A (Agent-to-Agent) protocol implementation in the Terminal Bench Agent.

## Overview

The Terminal Bench Agent is now **fully compliant** with the official A2A protocol specification:
- **Protocol Version**: A2A 1.0
- **Transport**: HTTP/HTTPS
- **Format**: JSON-RPC 2.0
- **Specification**: https://google.github.io/a2a-protocol-spec/

## Key A2A Components Implemented

### 1. Agent Card (Agent Discovery)

The agent exposes its capabilities via an Agent Card at the well-known URI:

```
GET /.well-known/agent-card
```

**Agent Card Structure:**
```json
{
  "name": "terminal_bench_agent",
  "description": "An AI agent specialized in solving terminal bench problems and code analysis",
  "url": "http://localhost:8001",
  "version": "1.0.0",
  "provider": "Terminal Bench A2A Agent",
  "skills": [
    {
      "id": "terminal_problem_solving",
      "name": "Terminal Problem Solving",
      "description": "Solve terminal-based programming and system administration problems",
      "inputModes": ["text"],
      "outputModes": ["text"],
      "tags": ["terminal", "programming", "system-admin", "bash", "shell"],
      "examples": [...]
    },
    {
      "id": "code_analysis",
      "name": "Code Analysis",
      "description": "Analyze and debug code problems",
      "inputModes": ["text"],
      "outputModes": ["text"],
      "tags": ["debugging", "analysis", "programming", "optimization"],
      "examples": [...]
    }
  ],
  "capabilities": {
    "streaming": false,
    "pushNotifications": false,
    "longRunningTasks": true
  },
  "metadata": {
    "model": "gpt-4o-mini",
    "framework": "OpenAI + FastAPI",
    "a2a_version": "1.0"
  }
}
```

### 2. JSON-RPC 2.0 Communication

All agent interactions use JSON-RPC 2.0 over HTTP POST to the root endpoint `/`.

**JSON-RPC Request Format:**
```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "method": "message.send",
  "params": {
    "message": {
      "messageId": "msg-user-001",
      "role": "user",
      "parts": [
        {
          "kind": "text",
          "text": "Write a bash script to find all files larger than 100MB"
        }
      ]
    }
  }
}
```

**JSON-RPC Response Format:**
```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "result": {
    "id": "task-abc123",
    "kind": "task",
    "contextId": "ctx-xyz789",
    "status": {
      "state": "completed",
      "timestamp": "2024-01-01T12:00:00Z"
    },
    "artifacts": [
      {
        "artifactId": "artifact-def456",
        "name": "terminal_bench_solution",
        "description": "Solution for: Write a bash script...",
        "parts": [
          {
            "kind": "text",
            "text": "#!/bin/bash\nfind . -type f -size +100M -exec ls -lh {} \\;"
          }
        ]
      }
    ],
    "messages": [...],
    "createdAt": "2024-01-01T12:00:00Z",
    "updatedAt": "2024-01-01T12:00:05Z"
  }
}
```

### 3. Supported RPC Methods

#### message.send
Sends a message to the agent and receives a Task in response.

```json
{
  "jsonrpc": "2.0",
  "id": "req-001",
  "method": "message.send",
  "params": {
    "message": {
      "messageId": "msg-001",
      "role": "user",
      "parts": [{
        "kind": "text",
        "text": "Your query here"
      }],
      "contextId": "ctx-optional",
      "referenceTaskIds": ["task-previous"]
    }
  }
}
```

#### task.get
Retrieves the current state of a task.

```json
{
  "jsonrpc": "2.0",
  "id": "req-002",
  "method": "task.get",
  "params": {
    "taskId": "task-abc123"
  }
}
```

#### task.cancel
Cancels an in-progress task.

```json
{
  "jsonrpc": "2.0",
  "id": "req-003",
  "method": "task.cancel",
  "params": {
    "taskId": "task-abc123",
    "reason": "User canceled the operation"
  }
}
```

### 4. Core A2A Concepts

#### Messages
- **Purpose**: Single turn of communication
- **Roles**: `user` or `agent`
- **Content**: Contains `Part` objects (TextPart, FilePart, DataPart)
- **Context**: Can reference `contextId` for conversation continuity

#### Tasks
- **Purpose**: Stateful unit of work
- **Lifecycle States**:
  - `submitted`: Task received, queued
  - `working`: Task in progress
  - `input-required`: Waiting for user input
  - `auth-required`: Waiting for authentication
  - `completed`: Task finished successfully
  - `canceled`: Task canceled by user
  - `rejected`: Task rejected by agent
  - `failed`: Task failed with error
- **Immutability**: Once completed/canceled/rejected/failed, tasks cannot restart

#### Artifacts
- **Purpose**: Tangible outputs produced by tasks
- **Properties**:
  - Unique `artifactId`
  - Human-readable `name`
  - `description` (optional)
  - List of `Part` objects (content)

#### Context ID
- **Purpose**: Logically groups related tasks and messages
- **Usage**: Maintains conversation state across multiple tasks
- **Persistence**: Agent stores context for follow-up interactions

### 5. Agent Behavior (Hybrid Agent Pattern)

This agent implements the **Hybrid Agent** pattern:

1. **Receives Messages**: All interactions start with `message.send`
2. **Generates Tasks**: Agent always responds with a `Task` object
3. **Immediate Completion**: For simple queries, tasks are immediately completed
4. **Context Awareness**: Supports multi-turn conversations via `contextId`
5. **Task Refinement**: Follow-up messages can reference previous tasks

### 6. Part Types

#### TextPart
```json
{
  "kind": "text",
  "text": "Your text content here"
}
```

#### FilePart (Inline)
```json
{
  "kind": "file",
  "file": {
    "name": "script.sh",
    "mimeType": "text/x-shellscript",
    "bytes": "IyEvYmluL2Jhc2gK..."  // Base64 encoded
  }
}
```

#### FilePart (URI)
```json
{
  "kind": "file",
  "file": {
    "name": "document.pdf",
    "mimeType": "application/pdf",
    "uri": "https://example.com/file.pdf"
  }
}
```

#### DataPart
```json
{
  "kind": "data",
  "data": {
    "key": "value",
    "nested": {
      "data": "structure"
    }
  }
}
```

## API Endpoints

| Endpoint | Method | Purpose | A2A Standard |
|----------|--------|---------|--------------|
| `/.well-known/agent-card` | GET | Agent discovery | ✅ Yes (RFC 8615) |
| `/agent-card` | GET | Agent discovery (alt) | ⚠️ Optional |
| `/` | POST | JSON-RPC 2.0 endpoint | ✅ Yes |
| `/` | GET | Agent status | ⚠️ Optional |
| `/health` | GET | Health check | ⚠️ Optional |

## Usage Examples

### Python Client Example

```python
import requests

# 1. Discover agent capabilities
agent_card = requests.get("http://localhost:8001/.well-known/agent-card").json()
print(f"Agent: {agent_card['name']}")
print(f"Skills: {[s['name'] for s in agent_card['skills']]}")

# 2. Send a message
rpc_request = {
    "jsonrpc": "2.0",
    "id": "req-001",
    "method": "message.send",
    "params": {
        "message": {
            "role": "user",
            "parts": [{
                "kind": "text",
                "text": "Write a bash script to monitor disk usage"
            }]
        }
    }
}

response = requests.post("http://localhost:8001/", json=rpc_request).json()
task = response["result"]

print(f"Task ID: {task['id']}")
print(f"Status: {task['status']['state']}")
print(f"Solution: {task['artifacts'][0]['parts'][0]['text']}")

# 3. Follow-up in the same context
followup_request = {
    "jsonrpc": "2.0",
    "id": "req-002",
    "method": "message.send",
    "params": {
        "message": {
            "role": "user",
            "contextId": task["contextId"],
            "referenceTaskIds": [task["id"]],
            "parts": [{
                "kind": "text",
                "text": "Add email alerting to that script"
            }]
        }
    }
}

response = requests.post("http://localhost:8001/", json=followup_request).json()
```

### curl Example

```bash
# Get agent card
curl http://localhost:8001/.well-known/agent-card

# Send a message
curl -X POST http://localhost:8001/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "req-001",
    "method": "message.send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{
          "kind": "text",
          "text": "Write a bash script to find large files"
        }]
      }
    }
  }'

# Get task status
curl -X POST http://localhost:8001/ \
  -H "Content-Type": application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "req-002",
    "method": "task.get",
    "params": {
      "taskId": "task-abc123"
    }
  }'
```

## A2A Compliance Checklist

- ✅ **Agent Card**: Exposed at `/.well-known/agent-card`
- ✅ **JSON-RPC 2.0**: All communication uses JSON-RPC 2.0
- ✅ **Message/Task Model**: Proper Message and Task data structures
- ✅ **Parts**: Support for TextPart, FilePart, DataPart
- ✅ **Artifacts**: Tasks produce named artifacts
- ✅ **Context ID**: Maintains conversation state
- ✅ **Task Lifecycle**: Proper task states and transitions
- ✅ **Skills**: Advertises capabilities with examples
- ✅ **Agent Capabilities**: Declares streaming/LRO support
- ✅ **Task Immutability**: Completed tasks cannot restart
- ✅ **Follow-up Support**: References previous tasks
- ⚠️ **Streaming (SSE)**: Not implemented (capability: false)
- ⚠️ **Push Notifications**: Not implemented (capability: false)
- ⚠️ **Authentication**: Not implemented (no securitySchemes)

## Key Differences from Previous Implementation

| Aspect | Previous | A2A-Compliant |
|--------|----------|---------------|
| **Protocol** | Custom JSON | JSON-RPC 2.0 |
| **Endpoint** | `/tasks` | `/` (JSON-RPC) |
| **Discovery** | `/agent-card` | `/.well-known/agent-card` |
| **Message Format** | Custom | A2A Message spec |
| **Response** | Custom | A2A Task spec |
| **States** | Basic | Full A2A lifecycle |
| **Context** | Not tracked | `contextId` support |
| **Artifacts** | Simple | Full Artifact spec |
| **Parts** | Simple text | TextPart/FilePart/DataPart |

## References

- [A2A Protocol Specification](https://google.github.io/a2a-protocol-spec/)
- [RFC 8615: Well-Known URIs](https://datatracker.ietf.org/doc/html/rfc8615)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [A2A and MCP Comparison](https://google.github.io/a2a-protocol-spec/concepts/a2a-and-mcp/)

