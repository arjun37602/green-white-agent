"""
A2A Compatible Agent for Terminal Bench Problems

This module implements a fully compliant A2A (Agent-to-Agent) protocol agent
following the official specification at https://google.github.io/a2a-protocol-spec/
"""
import os
import sys
import logging
import json
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# Import A2A protocol models
from .a2a_protocol import (
    AgentCard, AgentSkill, AgentCapabilities, AgentProvider, TransportProtocol,
    Message, Task, TaskStatus, Artifact,
    TextPart, Part,
    JsonRpcRequest, JsonRpcResponse, JsonRpcError,
    SendMessageRequest, SendMessageResponse,
    GetTaskRequest, GetTaskResponse,
    CancelTaskRequest, CancelTaskResponse,
    create_text_message, create_completed_task, create_text_artifact
)

# A2A support - fully implemented following official spec
A2A_AVAILABLE = True

# Load environment variables
load_dotenv()

class TerminalBenchAgent:
    """A2A-compliant agent for handling terminal bench problems."""
    
    def __init__(self, model="gpt-4o-mini", name="terminal_bench_agent", base_url="http://localhost:8001"):
        self.model = model
        self.name = name
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=api_key)
        
        # Store for active tasks and contexts
        self.tasks: Dict[str, Task] = {}
        self.contexts: Dict[str, List[Message]] = {}
        
        # Define tool functions for terminal operations
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_bash_command",
                    "description": "Execute a bash command in the terminal and return the output",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The bash command to execute"
                            },
                            "working_directory": {
                                "type": "string",
                                "description": "The working directory to execute the command in (optional)"
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }
            }
        ]
        
        # Define agent skills following A2A spec
        self.skills = [
            AgentSkill(
                id="terminal_problem_solving",
                name="Terminal Problem Solving",
                description="Solve terminal-based programming and system administration problems",
                inputModes=["text"],
                outputModes=["text"],
                tags=["terminal", "programming", "system-admin", "bash", "shell"],
                examples=[
                    "Write a bash script to find all files larger than 100MB",
                    "Set up a web server with nginx",
                    "Debug a failing Python script",
                    "Create a monitoring script for disk usage"
                ]
            ),
            AgentSkill(
                id="code_analysis",
                name="Code Analysis",
                description="Analyze and debug code problems",
                inputModes=["text"],
                outputModes=["text"],
                tags=["debugging", "analysis", "programming", "optimization"],
                examples=[
                    "Why is this Python function failing?",
                    "How can I optimize this algorithm?",
                    "What are the security issues in this code?",
                    "Review this code for best practices"
                ]
            )
        ]
        
        # Create A2A Agent Card (v0.3.0 compliant)
        self.agent_card = AgentCard(
            protocolVersion="0.3.0",
            name=self.name,
            description="An AI agent specialized in solving terminal bench problems and code analysis",
            url=self.base_url,
            preferredTransport="JSONRPC",
            version="1.0.0",
            provider=AgentProvider(
                organization="Terminal Bench A2A Agent",
                url=self.base_url
            ),
            skills=self.skills,
            capabilities=AgentCapabilities(
                streaming=False,
                pushNotifications=False,
                longRunningTasks=True
            ),
            defaultInputModes=["text/plain", "application/json"],
            defaultOutputModes=["text/plain", "application/json"],
            additionalInterfaces=[
                TransportProtocol(url=self.base_url, transport="JSONRPC")
            ],
            supportsAuthenticatedExtendedCard=False,
            metadata={
                "model": self.model,
                "framework": "OpenAI + FastAPI",
                "a2a_spec_version": "0.3.0"
            }
        )
    
    def execute_bash_command(self, command: str, working_directory: str = None) -> Dict[str, Any]:
        """Execute a bash command and return the result."""
        try:
            sys.stderr.write(f"   💻 Executing: {command}\n")
            sys.stderr.flush()
            self.logger.info(f"🔧 Executing command: {command}")
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=working_directory,
                timeout=30
            )
            
            output = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0
            }
            
            if result.stdout:
                sys.stderr.write(f"   📤 stdout: {result.stdout[:200]}{'...' if len(result.stdout) > 200 else ''}\n")
            if result.stderr:
                sys.stderr.write(f"   ⚠️  stderr: {result.stderr[:200]}{'...' if len(result.stderr) > 200 else ''}\n")
            sys.stderr.write(f"   📊 returncode: {result.returncode}\n")
            sys.stderr.flush()
            
            self.logger.info(f"✅ Command output: {output['stdout'][:200]}")
            return output
        except subprocess.TimeoutExpired:
            return {"error": "Command timeout", "success": False}
        except Exception as e:
            self.logger.error(f"Error executing command: {e}")
            return {"error": str(e), "success": False}
    
    def read_file(self, file_path: str) -> Dict[str, Any]:
        """Read a file and return its contents."""
        try:
            sys.stderr.write(f"   📖 Reading file: {file_path}\n")
            sys.stderr.flush()
            self.logger.info(f"📖 Reading file: {file_path}")
            with open(file_path, 'r') as f:
                content = f.read()
            sys.stderr.write(f"   ✅ File content ({len(content)} chars): {content[:200]}{'...' if len(content) > 200 else ''}\n")
            sys.stderr.flush()
            return {"content": content, "success": True}
        except Exception as e:
            self.logger.error(f"Error reading file: {e}")
            return {"error": str(e), "success": False}
    
    def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Write content to a file."""
        try:
            sys.stderr.write(f"   ✍️  Writing to file: {file_path}\n")
            sys.stderr.write(f"   📝 Content ({len(content)} chars): {content[:200]}{'...' if len(content) > 200 else ''}\n")
            sys.stderr.flush()
            self.logger.info(f"✍️ Writing to file: {file_path}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)
            sys.stderr.write(f"   ✅ File written successfully\n")
            sys.stderr.flush()
            return {"success": True}
        except Exception as e:
            self.logger.error(f"Error writing file: {e}")
            return {"error": str(e), "success": False}
    
    def solve_problem(self, problem_description: str) -> Tuple[str, Dict[str, Any]]:
        """Solve a terminal bench problem using OpenAI with tool calling.
        
        Returns:
            Tuple of (solution_text, usage_info) where usage_info contains:
            - total_tokens: Total tokens used
            - prompt_tokens: Prompt tokens used
            - completion_tokens: Completion tokens used
            - request_count: Number of API requests made
        """
        system_prompt = """You are an expert in terminal operations, system administration, and programming.
        You have access to tools to execute bash commands, read files, and write files.
        Use these tools to solve terminal bench problems step by step.
        
        Always verify your work by checking the results of commands before proceeding.
        Be precise and practical in your solutions."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_description}
        ]
        
        tool_call_log = []
        max_iterations = 10
        
        # Track token usage across all API calls
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        request_count = 0
        
        try:
            for iteration in range(max_iterations):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    max_tokens=2000,
                    temperature=0.3
                )
                
                request_count += 1
                
                # Extract token usage from response
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    if hasattr(usage, 'prompt_tokens'):
                        total_prompt_tokens += usage.prompt_tokens
                    if hasattr(usage, 'completion_tokens'):
                        total_completion_tokens += usage.completion_tokens
                    if hasattr(usage, 'total_tokens'):
                        total_tokens += usage.total_tokens
                
                # Print token usage for this request to stderr
                if hasattr(response, 'usage') and response.usage:
                    request_usage = response.usage
                    sys.stderr.write(f"\n{'='*60}\n")
                    sys.stderr.write(f"📊 API Request #{request_count} - Token Usage:\n")
                    sys.stderr.write(f"   Prompt Tokens: {request_usage.prompt_tokens if hasattr(request_usage, 'prompt_tokens') else 'N/A'}\n")
                    sys.stderr.write(f"   Completion Tokens: {request_usage.completion_tokens if hasattr(request_usage, 'completion_tokens') else 'N/A'}\n")
                    sys.stderr.write(f"   Total Tokens: {request_usage.total_tokens if hasattr(request_usage, 'total_tokens') else 'N/A'}\n")
                    sys.stderr.write(f"   Cumulative Total: {total_tokens}\n")
                    sys.stderr.flush()
                
                assistant_message = response.choices[0].message
                messages.append(assistant_message)
                
                # Print assistant's response to stderr
                if assistant_message.content:
                    sys.stderr.write(f"\n🤖 Assistant Response:\n")
                    sys.stderr.write(f"   {assistant_message.content[:300]}{'...' if len(assistant_message.content) > 300 else ''}\n")
                    sys.stderr.flush()
                
                # Check if the model wants to call a function
                if assistant_message.tool_calls:
                    for tool_call in assistant_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        sys.stderr.write(f"\n🔧 Tool Call #{len(tool_call_log) + 1}: {function_name}\n")
                        sys.stderr.write(f"   Arguments: {json.dumps(function_args, indent=2)}\n")
                        sys.stderr.flush()
                        
                        self.logger.info(f"🔧 Tool call: {function_name}({function_args})")
                        tool_call_log.append(f"Tool: {function_name} | Args: {function_args}")
                        
                        # Execute the function
                        if function_name == "execute_bash_command":
                            function_response = self.execute_bash_command(**function_args)
                        elif function_name == "read_file":
                            function_response = self.read_file(**function_args)
                        elif function_name == "write_file":
                            function_response = self.write_file(**function_args)
                        else:
                            function_response = {"error": f"Unknown function: {function_name}"}
                        
                        # Print tool response to stderr
                        sys.stderr.write(f"   ✅ Tool Response:\n")
                        response_str = json.dumps(function_response, indent=2)
                        if len(response_str) > 200:
                            sys.stderr.write(f"   {response_str[:200]}...\n")
                        else:
                            sys.stderr.write(f"   {response_str}\n")
                        sys.stderr.flush()
                        
                        # Add function response to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps(function_response)
                        })
                else:
                    # No more tool calls, return the final response
                    final_response = assistant_message.content or "Task completed"
                    
                    sys.stderr.write(f"\n✅ Final Response:\n")
                    sys.stderr.write(f"   {final_response[:500]}{'...' if len(final_response) > 500 else ''}\n")
                    sys.stderr.flush()
                    
                    # Add tool call log
                    if tool_call_log:
                        final_response += "\n\n=== Tool Execution Log ===\n"
                        final_response += "\n".join(tool_call_log)
                    
                    # Prepare usage information
                    usage_info = {
                        "total_tokens": total_tokens,
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "request_count": request_count
                    }
                    
                    sys.stderr.write(f"\n📊 Total Token Usage Summary:\n")
                    sys.stderr.write(f"   Total Tokens: {total_tokens}\n")
                    sys.stderr.write(f"   Prompt Tokens: {total_prompt_tokens}\n")
                    sys.stderr.write(f"   Completion Tokens: {total_completion_tokens}\n")
                    sys.stderr.write(f"   Requests: {request_count}\n")
                    sys.stderr.write(f"   Tokens Per Request: {total_tokens / request_count if request_count > 0 else 0:.2f}\n")
                    sys.stderr.write(f"{'='*60}\n\n")
                    sys.stderr.flush()
                    
                    self.logger.info(f"Token usage: {usage_info}")
                    return final_response, usage_info
            
            # Maximum iterations reached
            usage_info = {
                "total_tokens": total_tokens,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "request_count": request_count
            }
            return "Maximum iterations reached. Task may not be complete.", usage_info
            
        except Exception as e:
            self.logger.error(f"Error solving problem: {e}")
            usage_info = {
                "total_tokens": total_tokens,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "request_count": request_count
            }
            return f"Error: {str(e)}", usage_info
    
    def handle_message(self, message: Message) -> Task:
        """
        Handle an incoming A2A message and return a Task.
        This implements the A2A protocol's message.send method.
        """
        try:
            # Extract text from message parts
            problem_description = ""
            for part in message.parts:
                if isinstance(part, TextPart):
                    problem_description += part.text + "\n"
            
            if not problem_description.strip():
                problem_description = "No problem description provided"
            
            # Get or create context
            context_id = message.contextId or f"ctx-{os.urandom(6).hex()}"
            if context_id not in self.contexts:
                self.contexts[context_id] = []
            
            # Store the incoming message
            self.contexts[context_id].append(message)
            
            # Print incoming task to stderr so it shows in terminal even when stdout is redirected
            sys.stderr.write(f"\n{'='*60}\n")
            sys.stderr.write(f"📨 Received Task Request:\n")
            sys.stderr.write(f"   Task ID: {context_id}\n")
            sys.stderr.write(f"   Description: {problem_description[:200]}{'...' if len(problem_description) > 200 else ''}\n")
            sys.stderr.write(f"{'='*60}\n")
            sys.stderr.flush()
            
            # Solve the problem and get token usage
            solution, usage_info = self.solve_problem(problem_description)
            
            # Create artifact with the solution
            artifact = create_text_artifact(
                text=solution,
                name="terminal_bench_solution",
                description=f"Solution for: {problem_description[:100]}..."
            )
            
            # Create agent response message
            agent_message = create_text_message(
                text=f"I've analyzed the problem and provided a solution.",
                role="agent",
                context_id=context_id
            )
            
            # Create completed task with token usage in metadata
            task = create_completed_task(
                message=agent_message,
                artifacts=[artifact],
                context_id=context_id
            )
            
            # Add token usage to task metadata
            task.metadata = {
                "usage": {
                    "total_tokens": usage_info.get("total_tokens", 0),
                    "prompt_tokens": usage_info.get("prompt_tokens", 0),
                    "completion_tokens": usage_info.get("completion_tokens", 0)
                },
                "request_count": usage_info.get("request_count", 1)
            }
            
            self.logger.info(f"Task {task.id} completed with token usage: {task.metadata['usage']}")
            
            # Store task
            self.tasks[task.id] = task
            self.contexts[context_id].append(agent_message)
            
            return task
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            
            # Create error task
            error_artifact = create_text_artifact(
                text=f"Error processing request: {str(e)}",
                name="error",
                description="Error during task processing"
            )
            
            error_task = Task(
                contextId=message.contextId or f"ctx-{os.urandom(6).hex()}",
                status=TaskStatus(state="failed", message=str(e)),
                artifacts=[error_artifact],
                history=[message]
            )
            
            self.tasks[error_task.id] = error_task
            return error_task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str, reason: Optional[str] = None) -> Optional[Task]:
        """Cancel a task."""
        task = self.tasks.get(task_id)
        if task and task.status.state in ["submitted", "working"]:
            task.status = TaskStatus(state="canceled", message=reason)
            return task
        return None
    
    def get_agent_card(self) -> AgentCard:
        """Get the agent card for A2A discovery."""
        return self.agent_card

# Create a simple Agent class for backward compatibility
class Agent:
    """Simple agent wrapper for backward compatibility."""
    
    def __init__(self, model="gpt-4o-mini", name="white_agent", description="A versatile agent capable of handling various tasks and problem-solving.", instruction="You are a helpful AI assistant. Respond to user queries clearly and concisely."):
        self.model = model
        self.name = name
        self.description = description
        self.instruction = instruction
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=api_key)
    
    def run(self, user_input):
        """Run the agent with user input and return response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.instruction},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

# A2A FastAPI Server (JSON-RPC 2.0 + A2A Protocol)
class A2ATerminalBenchServer:
    """A2A-compliant FastAPI server implementing the official A2A protocol."""
    
    def __init__(self, port=8001, host="0.0.0.0"):
        self.port = port
        self.host = host
        self.agent = TerminalBenchAgent(base_url=f"http://{host}:{port}")
        self.logger = logging.getLogger(__name__)
        self.app = FastAPI(
            title="Terminal Bench A2A Agent",
            description="A2A-compliant agent for terminal bench problems (JSON-RPC 2.0)",
            version="1.0.0"
        )
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes for A2A protocol."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint - agent status"""
            return {
                "message": "Terminal Bench A2A Agent",
                "status": "running",
                "protocol": "A2A 1.0 (JSON-RPC 2.0)",
                "agent_card_url": f"{self.agent.base_url}/.well-known/agent-card"
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "agent": self.agent.name}
        
        @self.app.get("/.well-known/agent-card")
        async def get_agent_card_wellknown():
            """Agent discovery via well-known URI (A2A standard)"""
            return self.agent.get_agent_card()
        
        @self.app.get("/agent-card")
        async def get_agent_card():
            """Agent card endpoint (alternative)"""
            return self.agent.get_agent_card()
        
        @self.app.post("/")
        async def handle_jsonrpc(request: Request):
            """
            Main JSON-RPC 2.0 endpoint for A2A protocol.
            Handles: message.send, task.get, task.cancel
            """
            try:
                body = await request.json()
                rpc_request = JsonRpcRequest(**body)
                
                # Handle different RPC methods (A2A v0.3.0 format)
                if rpc_request.method == "message/send":
                    return await self._handle_message_send(rpc_request)
                elif rpc_request.method == "tasks/get":
                    return await self._handle_task_get(rpc_request)
                elif rpc_request.method == "tasks/cancel":
                    return await self._handle_task_cancel(rpc_request)
                else:
                    # Method not found
                    return JsonRpcResponse(
                        id=rpc_request.id,
                        error=JsonRpcError(
                            code=-32601,
                            message="Method not found",
                            data={"method": rpc_request.method}
                        ).dict()
                    )
                    
            except Exception as e:
                self.logger.error(f"Error handling JSON-RPC request: {e}")
                return JsonRpcResponse(
                    id="unknown",
                    error=JsonRpcError(
                        code=-32603,
                        message="Internal error",
                        data={"error": str(e)}
                    ).dict()
                )
        
        async def _handle_message_send(self, rpc_request: JsonRpcRequest) -> JsonRpcResponse:
            """Handle message/send RPC method"""
            try:
                params = rpc_request.params or {}
                message_data = params.get("message", {})
                
                # Ensure message has proper structure
                if not message_data:
                    return JsonRpcResponse(
                        id=rpc_request.id,
                        error=JsonRpcError(
                            code=-32602,
                            message="Invalid params: message required"
                        ).dict()
                    )
                
                # Create Message object
                message = Message(**message_data)
                
                # Process message and get task
                task = self.agent.handle_message(message)
                
                if not task:
                    return JsonRpcResponse(
                        id=rpc_request.id,
                        error=JsonRpcError(
                            code=-32603,
                            message="Failed to create task"
                        ).dict()
                    )
                
                return JsonRpcResponse(
                    id=rpc_request.id,
                    result=task.dict()
                )
                
            except Exception as e:
                self.logger.error(f"Error in message/send: {e}")
                import traceback
                traceback.print_exc()
                return JsonRpcResponse(
                    id=rpc_request.id,
                    error=JsonRpcError(
                        code=-32603,
                        message="Failed to process message",
                        data={"error": str(e)}
                    ).dict()
                )
        
        async def _handle_task_get(self, rpc_request: JsonRpcRequest) -> JsonRpcResponse:
            """Handle tasks/get RPC method"""
            try:
                params = rpc_request.params or {}
                task_id = params.get("id")  # A2A v0.3.0 uses 'id' not 'taskId'
                
                if not task_id:
                    return JsonRpcResponse(
                        id=rpc_request.id,
                        error=JsonRpcError(
                            code=-32602,
                            message="Invalid params: id required"
                        ).dict()
                    )
                
                task = self.agent.get_task(task_id)
                
                if task:
                    return JsonRpcResponse(
                        id=rpc_request.id,
                        result=task.dict()
                    )
                else:
                    return JsonRpcResponse(
                        id=rpc_request.id,
                        error=JsonRpcError(
                            code=-32001,  # TaskNotFoundError per A2A spec
                            message="Task not found",
                            data={"id": task_id}
                        ).dict()
                    )
                    
            except Exception as e:
                self.logger.error(f"Error in task.get: {e}")
                return JsonRpcResponse(
                    id=rpc_request.id,
                    error=JsonRpcError(
                        code=-32603,
                        message="Failed to get task",
                        data={"error": str(e)}
                    ).dict()
                )
        
        async def _handle_task_cancel(self, rpc_request: JsonRpcRequest) -> JsonRpcResponse:
            """Handle tasks/cancel RPC method"""
            try:
                params = rpc_request.params or {}
                task_id = params.get("id")  # A2A v0.3.0 uses 'id'
                reason = params.get("reason")
                
                if not task_id:
                    return JsonRpcResponse(
                        id=rpc_request.id,
                        error=JsonRpcError(
                            code=-32602,
                            message="Invalid params: id required"
                        ).dict()
                    )
                
                task = self.agent.cancel_task(task_id, reason)
                
                if task:
                    return JsonRpcResponse(
                        id=rpc_request.id,
                        result=task.dict()
                    )
                else:
                    return JsonRpcResponse(
                        id=rpc_request.id,
                        error=JsonRpcError(
                            code=-32002,  # TaskNotCancelableError per A2A spec
                            message="Task cannot be canceled",
                            data={"id": task_id}
                        ).dict()
                    )
                    
            except Exception as e:
                self.logger.error(f"Error in task.cancel: {e}")
                return JsonRpcResponse(
                    id=rpc_request.id,
                    error=JsonRpcError(
                        code=-32603,
                        message="Failed to cancel task",
                        data={"error": str(e)}
                    ).dict()
                )
        
        # Store methods as instance methods
        self._handle_message_send = _handle_message_send.__get__(self, A2ATerminalBenchServer)
        self._handle_task_get = _handle_task_get.__get__(self, A2ATerminalBenchServer)
        self._handle_task_cancel = _handle_task_cancel.__get__(self, A2ATerminalBenchServer)
    
    def run(self):
        """Run the A2A server."""
        self.logger.info(f"Starting A2A Terminal Bench Agent on {self.host}:{self.port}")
        self.logger.info(f"Agent Card URL: http://{self.host}:{self.port}/.well-known/agent-card")
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )

# Create the agent instance for backward compatibility
agent = Agent(
    model="gpt-4o-mini",
    name="white_agent",
    description="A versatile agent capable of handling various tasks and problem-solving.",
    instruction="You are a helpful AI assistant. Respond to user queries clearly and concisely."
)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Terminal Bench A2A Agent")
    parser.add_argument("--server", action="store_true", help="Run as A2A server")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    
    args = parser.parse_args()
    
    if args.server:
        # Run as A2A server
        print("🚀 Starting A2A Terminal Bench Agent Server...")
        server = A2ATerminalBenchServer(port=args.port, host=args.host)
        server.run()
    elif args.test:
        # Test mode
        print("🧪 Testing Terminal Bench Agent...")
        print(f"Agent '{agent.name}' initialized with model '{agent.model}'")
        
        # Test the A2A agent
        test_agent = TerminalBenchAgent()
        print(f"A2A-compatible agent: {test_agent.name}")
        print(f"Skills: {[skill.name for skill in test_agent.skills]}")
        
        # Test a simple problem
        sample_problem = "Write a bash script to find all files larger than 100MB in the current directory"
        print(f"\nTesting with sample problem: {sample_problem}")
        # Note: Uncomment the next line to test with OpenAI (requires API key)
        # print(f"Response: {test_agent.solve_problem(sample_problem)}")
        
        # Test A2A message handling
        print("\n🔧 Testing A2A message handling...")
        test_message = create_text_message(
            text=sample_problem,
            role="user"
        )
        
        try:
            task = test_agent.handle_message(test_message)
            print(f"✅ A2A message handling successful")
            print(f"Task ID: {task.id}")
            print(f"Status: {task.status.state}")
            print(f"Context ID: {task.contextId}")
            print(f"Artifacts: {len(task.artifacts)}")
            if task.artifacts:
                print(f"Response length: {len(task.artifacts[0].parts[0].text)} characters")
        except Exception as e:
            print(f"❌ A2A message handling failed: {e}")
