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
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
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
    """A2A-compliant agent for handling terminal bench problems with tool calling."""
    
    def __init__(self, model="gpt-5", name="terminal_bench_agent", base_url="http://localhost:8001", log_dir=None):
        self.model = model
        self.name = name
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=api_key)
        
        # Set up chat completion logging
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), "logs", "openai_chat")
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger.info(f"OpenAI chat logs will be saved to: {self.log_dir}")
        
        # Store for active tasks and contexts
        self.tasks: Dict[str, Task] = {}
        self.contexts: Dict[str, List[Message]] = {}
        
        # Simple React agent: maintain conversation history
        self.history: List[Dict[str, Any]] = []
        
        # System prompt for tool-based execution
        system_prompt = """You are an expert in terminal operations, system administration, and programming.
You have access to tools for executing bash commands and stopping the task loop.

Use the execute_bash_command tool to run commands in the container.
Use the stop tool when the task is fully complete and verified.

After you execute each tool, you will receive a tool message indicating whether the tool call was successful or not, as well as any outputs from the tool call. Use this information to decide your next step.

Be precise, follow instructions literally, and verify your work before stopping."""
        
        # Add system prompt to history
        self.history.append({"role": "system", "content": system_prompt})
        
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
    
    
    def call_llm_with_tools(self, tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Call OpenAI with current history and tools.
        
        Args:
            tools: Tool definitions in OpenAI format
            
        Returns:
            Dict with assistant message and tool calls
        """
        try:
            # Prepare API call
            api_params = {
                "model": self.model,
                "messages": self.history
            }
            
            # Add tools if provided
            if tools:
                # Convert green agent tool format to OpenAI format
                openai_tools = []
                for tool in tools:
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool["description"],
                            "parameters": tool["inputSchema"]
                        }
                    })
                api_params["tools"] = openai_tools
            
            # Call OpenAI
            response = self.client.chat.completions.create(**api_params)
            
            assistant_message = response.choices[0].message
            
            # Build response dict
            result = {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": []
            }
            
            # Extract tool calls if present
            if assistant_message.tool_calls:
                for tc in assistant_message.tool_calls:
                    result["tool_calls"].append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
            
            self.logger.info(f"LLM returned {len(result['tool_calls'])} tool calls")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            raise
    
    def handle_message(self, message: Message, tools: Optional[List[Dict[str, Any]]] = None) -> Task:
        """
        Handle an incoming A2A message and return a Task.
        This implements the A2A protocol's message.send method.
        """
        try:
            # Extract text from message parts
            user_message = ""
            for part in message.parts:
                if isinstance(part, TextPart):
                    user_message += part.text + "\n"
            
            if not user_message.strip():
                user_message = "No message provided"
            
            # Get or create context
            context_id = message.contextId or f"ctx-{os.urandom(6).hex()}"
            if context_id not in self.contexts:
                self.contexts[context_id] = []
            
            # Store the incoming message
            self.contexts[context_id].append(message)
            
            self.logger.info(f"Received message: {user_message}...")
            
            # Add user message to history
            self.history.append({"role": message.role, "content": user_message.strip()})
            
            # Call LLM with tools
            llm_response = self.call_llm_with_tools(tools=tools)
            
            # Add assistant response to history (important for tool call context!)
            self.history.append(llm_response)
            
            # Create artifact with tool calls
            artifact_data = {
                "tool_calls": llm_response["tool_calls"]
            }
            
            artifact = create_text_artifact(
                text=json.dumps(artifact_data, indent=2),
                name="tool_calls",
                description="Tool calls to execute"
            )
            
            # Create agent response message
            agent_message = create_text_message(
                text=llm_response["content"] or "Making tool calls",
                role="agent",
                context_id=context_id
            )
            
            # Create completed task
            task = create_completed_task(
                message=agent_message,
                artifacts=[artifact],
                context_id=context_id
            )
            
            # Store task
            self.tasks[task.id] = task
            self.contexts[context_id].append(agent_message)
            
            return task
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            import traceback
            traceback.print_exc()
            
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



# A2A FastAPI Server (JSON-RPC 2.0 + A2A Protocol)
class A2ATerminalBenchServer:
    """A2A-compliant FastAPI server implementing the official A2A protocol."""
    
    def __init__(self, port=8001, host="0.0.0.0"):
        self.port = port
        self.host = host
        self.agent = TerminalBenchAgent(model="gpt-5-nano", base_url=f"http://{host}:{port}")
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
                tools = params.get("tools")  # Extract tools from params
                tool_messages = params.get("tool_messages")  # Extract tool messages from params
                
                # Handle tool results
                if tool_messages:
                    # Add tool messages directly to history
                    for tm in tool_messages:
                        self.agent.history.append(tm)
                        self.logger.info(f"Added tool result: {tm.get('tool_call_id')}")
                    
                    # Call LLM again with updated history
                    llm_response = self.agent.call_llm_with_tools(tools=tools)
                    
                    # Add assistant response to history
                    self.agent.history.append(llm_response)
                    
                    # Create task with new tool calls
                    artifact_data = {"tool_calls": llm_response["tool_calls"]}
                    artifact = create_text_artifact(
                        text=json.dumps(artifact_data, indent=2),
                        name="tool_calls",
                        description="Tool calls to execute"
                    )
                    
                    agent_message = create_text_message(
                        text=llm_response["content"] or "Making tool calls",
                        role="agent"
                    )
                    
                    task = create_completed_task(
                        message=agent_message,
                        artifacts=[artifact]
                    )
                    
                    self.agent.tasks[task.id] = task
                    
                    return JsonRpcResponse(
                        id=rpc_request.id,
                        result=task.dict()
                    )
                
                # Regular message handling
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
                
                # Process message and get task (pass tools)
                task = self.agent.handle_message(message, tools=tools)
                
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
