"""
A2A Compatible Agent for Terminal Bench Problems

This module implements a fully compliant A2A (Agent-to-Agent) protocol agent
following the official specification at https://google.github.io/a2a-protocol-spec/
"""
import os
import logging
import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# Import A2A protocol models
from a2a_protocol import (
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
    
    def solve_problem(self, problem_description: str) -> str:
        """Solve a terminal bench problem using OpenAI."""
        system_prompt = """You are an expert in terminal operations, system administration, and programming. 
        Solve the given problem step by step, providing clear instructions and explanations.
        
        For terminal problems, provide:
        1. The exact commands to run
        2. Explanation of what each command does
        3. Expected output
        4. Troubleshooting tips if applicable
        
        Be precise and practical in your solutions."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": problem_description}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error solving problem: {e}")
            return f"Error: {str(e)}"
    
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
            
            # Solve the problem
            solution = self.solve_problem(problem_description)
            
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
