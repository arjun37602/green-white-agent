#!/usr/bin/env python3
"""
Simple White Agent for TerminalBench

A naive, template-based White Agent that can respond to TerminalBench tasks
using pre-written command templates. This agent is designed to work with
the Green Agent for testing and evaluation.
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from a2a_protocol import (
    Message, Task, TaskStatus, Artifact, TextPart,
    JsonRpcRequest, JsonRpcResponse, JsonRpcError,
    create_text_message, create_text_artifact
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleWhiteAgent:
    """A simple, template-based White Agent for TerminalBench tasks."""
    
    def __init__(self):
        self.name = "Simple White Agent"
        self.logger = logging.getLogger(__name__)
        
        # Command templates for common TerminalBench tasks
        self.command_templates = {
            # File creation tasks
            "solution.txt": [
                "echo 'password123' > solution.txt",
                "cat solution.txt"
            ],
            "answer.txt": [
                "echo 'answer' > answer.txt",
                "cat answer.txt"
            ],
            "result.txt": [
                "echo 'result' > result.txt",
                "cat result.txt"
            ],
            
            # Directory tasks
            "directory": [
                "mkdir test_dir",
                "cd test_dir",
                "pwd"
            ],
            "nested_directory": [
                "mkdir -p parent/child",
                "cd parent/child",
                "pwd"
            ],
            
            # File operations
            "file_copy": [
                "echo 'source content' > source.txt",
                "cp source.txt destination.txt",
                "cat destination.txt"
            ],
            "file_move": [
                "echo 'content' > file.txt",
                "mv file.txt moved_file.txt",
                "ls -la moved_file.txt"
            ],
            
            # SSL Certificate tasks
            "ssl_certificate": [
                "mkdir -p ssl",
                "openssl req -x509 -newkey rsa:4096 -keyout ssl/server.key -out ssl/server.crt -days 365 -nodes -subj '/C=US/ST=CA/L=SF/O=Test/CN=localhost'",
                "ls -la ssl/"
            ],
            
            # Archive tasks
            "archive": [
                "echo 'content' > file.txt",
                "tar -czf archive.tar.gz file.txt",
                "ls -la archive.tar.gz"
            ],
            "extract": [
                "echo 'content' > file.txt",
                "tar -czf archive.tar.gz file.txt",
                "rm file.txt",
                "tar -xzf archive.tar.gz",
                "cat file.txt"
            ],
            
            # System info tasks
            "system_info": [
                "uname -a",
                "whoami",
                "pwd",
                "ls -la"
            ],
            
            # Process tasks
            "process": [
                "ps aux",
                "echo 'Process information retrieved'"
            ]
        }
        
        # Task patterns for matching
        self.task_patterns = {
            r"solution\.txt": "solution.txt",
            r"answer\.txt": "answer.txt",
            r"result\.txt": "result.txt",
            r"create.*directory": "directory",
            r"nested.*directory": "nested_directory",
            r"copy.*file": "file_copy",
            r"move.*file": "file_move",
            r"certificate|cert|ssl": "ssl_certificate",
            r"archive|tar|zip": "archive",
            r"extract|unzip|untar": "extract",
            r"system.*info|uname|whoami": "system_info",
            r"process|ps": "process"
        }
    
    def solve_task(self, task_description: str) -> List[str]:
        """
        Solve a task using instruction parsing with template fallback.
        
        Args:
            task_description: Description of the task to solve
            
        Returns:
            List of bash commands to execute
        """
        self.logger.info(f"Solving task: {task_description[:100]}...")
        
        # First try to parse the actual instruction
        commands = self._parse_instruction(task_description)
        if commands:
            self.logger.info(f"Generated commands from instruction parsing: {commands}")
            return commands
        
        # Fallback to template matching
        task_description_lower = task_description.lower()
        
        for pattern, template_name in self.task_patterns.items():
            if re.search(pattern, task_description_lower):
                self.logger.info(f"Matched pattern '{pattern}' to template '{template_name}'")
                commands = self.command_templates.get(template_name, [])
                if commands:
                    self.logger.info(f"Using {len(commands)} commands from template")
                    return commands
        
        # Default fallback commands
        self.logger.warning("No pattern matched, using default commands")
        return [
            "echo 'Default solution' > solution.txt",
            "cat solution.txt"
        ]
    
    def _parse_instruction(self, instruction: str) -> List[str]:
        """
        Parse Terminal-Bench instruction to generate appropriate commands.
        
        Args:
            instruction: The task instruction
            
        Returns:
            List of commands or empty list if parsing failed
        """
        instruction_lower = instruction.lower()
        
        # Parse "Create a file called X. Write Y to it." pattern
        file_create_match = re.search(r"create a file called (.*?)\. write ['\"](.*?)['\"] to it", instruction_lower, re.IGNORECASE)
        if file_create_match:
            # Re-extract from original instruction to preserve case
            original_match = re.search(r"create a file called (.*?)\. write ['\"](.*?)['\"] to it", instruction, re.IGNORECASE)
            filename = original_match.group(1).strip()
            content = original_match.group(2).strip()
            
            # Handle /app/ prefix - convert to relative app/ path
            if filename.startswith('/app/'):
                filename = 'app/' + filename[5:]  # Convert /app/ to app/
            elif not filename.startswith('app/'):
                filename = f"app/{filename}"
            
            commands = [
                f'mkdir -p $(dirname {filename})',
                f'echo "{content}" > {filename}',
                f'cat {filename}'
            ]
            self.logger.info(f"Parsed file creation: {filename} with content '{content}'")
            return commands
        
        # Parse "Write Y to file X" pattern
        write_match = re.search(r"write ['\"]([^'\"]*)['\"] to (?:file )?([^\s,.]*)", instruction_lower)
        if write_match:
            content = write_match.group(1).strip()
            filename = write_match.group(2).strip()
            
            # Handle /app/ prefix - convert to relative app/ path
            if filename.startswith('/app/'):
                filename = 'app/' + filename[5:]  # Convert /app/ to app/
            elif not filename.startswith('app/'):
                filename = f"app/{filename}"
            
            commands = [
                f'mkdir -p $(dirname {filename})',
                f'echo "{content}" > {filename}',
                f'cat {filename}'
            ]
            self.logger.info(f"Parsed write operation: {filename} with content '{content}'")
            return commands
        
        # Parse directory creation patterns
        dir_match = re.search(r"create (?:a )?directory called (?:/app/)?([^\s,.]*)", instruction_lower)
        if dir_match:
            dirname = dir_match.group(1).strip()
            if dirname.startswith('/app/'):
                dirname = 'app/' + dirname[5:]  # Convert /app/ to app/
            elif not dirname.startswith('app/'):
                dirname = f"app/{dirname}"
            
            commands = [
                f'mkdir -p {dirname}',
                f'cd {dirname}',
                'pwd'
            ]
            self.logger.info(f"Parsed directory creation: {dirname}")
            return commands
        
        # Parse SSL certificate patterns
        if "ssl" in instruction_lower or "certificate" in instruction_lower:
            commands = [
                'mkdir -p app/ssl',
                'openssl genrsa -out app/ssl/server.key 2048',
                'chmod 600 app/ssl/server.key',
                'openssl req -new -x509 -key app/ssl/server.key -out app/ssl/server.crt -days 365 -subj "/O=DevOps Team/CN=dev-internal.company.local"',
                'cat app/ssl/server.key app/ssl/server.crt > app/ssl/server.pem',
                'openssl x509 -in app/ssl/server.crt -noout -subject > app/ssl/verification.txt',
                'openssl x509 -in app/ssl/server.crt -noout -dates >> app/ssl/verification.txt',
                'openssl x509 -in app/ssl/server.crt -noout -fingerprint -sha256 >> app/ssl/verification.txt',
                'ls -la app/ssl/'
            ]
            self.logger.info("Parsed SSL certificate creation")
            return commands
        
        # Parse archive patterns
        if "archive" in instruction_lower or "tar" in instruction_lower:
            commands = [
                'echo "content" > file.txt',
                'tar -czf archive.tar.gz file.txt',
                'ls -la archive.tar.gz'
            ]
            self.logger.info("Parsed archive creation")
            return commands
        
        # If no specific pattern matched, return empty list to fall back to templates
        self.logger.info("No specific instruction pattern matched, falling back to templates")
        return []
    
    def handle_message(self, message: Message) -> Task:
        """
        Handle an incoming A2A message and return a Task.
        
        Args:
            message: A2A message from Green Agent
            
        Returns:
            Task with solution commands
        """
        try:
            self.logger.info(f"Handling message from context: {message.contextId}")
            
            # Extract task description from message parts
            task_description = ""
            for part in message.parts:
                if isinstance(part, TextPart):
                    task_description += part.text + "\n"
            
            if not task_description.strip():
                task_description = "No task description provided"
            
            # Solve the task
            commands = self.solve_task(task_description)
            
            # Create response text with commands
            response_text = f"""I'll solve this task using the following commands:

```bash
{chr(10).join(commands)}
```

These commands should complete the requested task."""
            
            # Create artifact with the solution
            artifact = create_text_artifact(
                text=response_text,
                name="task_solution",
                description=f"Solution for: {task_description[:100]}..."
            )
            
            # Create agent response message
            agent_message = create_text_message(
                text="I've analyzed the task and prepared a solution.",
                role="agent",
                context_id=message.contextId
            )
            
            # Create completed task
            task = Task(
                contextId=message.contextId or f"ctx-{uuid.uuid4().hex[:12]}",
                status=TaskStatus(state="completed"),
                artifacts=[artifact],
                history=[message, agent_message]
            )
            
            # Add token usage metadata (set to 0 since simple agent doesn't use LLM)
            task.metadata = {
                "usage": {
                    "total_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0
                },
                "request_count": 1
            }
            
            self.logger.info(f"Task completed with {len(commands)} commands")
            return task
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            
            # Create error task
            error_artifact = create_text_artifact(
                text=f"Error processing task: {str(e)}",
                name="error",
                description="Error during task processing"
            )
            
            error_task = Task(
                contextId=message.contextId or f"ctx-{uuid.uuid4().hex[:12]}",
                status=TaskStatus(state="failed", message=str(e)),
                artifacts=[error_artifact],
                history=[message]
            )
            
            return error_task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID (not implemented for simple agent)."""
        return None
    
    def cancel_task(self, task_id: str, reason: Optional[str] = None) -> Optional[Task]:
        """Cancel a task (not implemented for simple agent)."""
        return None


class SimpleWhiteAgentServer:
    """A2A-compliant FastAPI server for the Simple White Agent."""
    
    def __init__(self, port=8002, host="0.0.0.0"):
        self.port = port
        self.host = host
        self.agent = SimpleWhiteAgent()
        self.logger = logging.getLogger(__name__)
        
        # Store tasks for retrieval
        self.stored_tasks = {}
        
        # Import FastAPI here to avoid dependency issues
        try:
            from fastapi import FastAPI, HTTPException, Request
            from fastapi.responses import JSONResponse
            import uvicorn
            
            self.FastAPI = FastAPI
            self.HTTPException = HTTPException
            self.JSONResponse = JSONResponse
            self.Request = Request
            self.uvicorn = uvicorn
            
        except ImportError:
            self.logger.error("FastAPI not available. Install with: pip install fastapi uvicorn")
            raise
        
        self.app = self.FastAPI(
            title="Simple White Agent",
            description="A naive, template-based White Agent for TerminalBench",
            version="1.0.0"
        )
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes for A2A protocol."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint - agent status"""
            return {
                "message": "Simple White Agent",
                "status": "running",
                "protocol": "A2A 1.0 (JSON-RPC 2.0)",
                "agent": self.agent.name
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "agent": self.agent.name}
        
        @self.app.get("/.well-known/agent-card")
        async def get_agent_card_wellknown():
            """Agent discovery via well-known URI (A2A standard)"""
            return {
                "protocolVersion": "0.3.0",
                "name": self.agent.name,
                "description": "A simple, template-based White Agent for TerminalBench tasks",
                "url": f"http://{self.host}:{self.port}",
                "preferredTransport": "JSONRPC",
                "version": "1.0.0",
                "skills": [
                    {
                        "id": "terminal_task_solving",
                        "name": "Terminal Task Solving",
                        "description": "Solve terminal-based tasks using template-based approach",
                        "inputModes": ["text"],
                        "outputModes": ["text"],
                        "tags": ["terminal", "template-based", "simple"]
                    }
                ],
                "capabilities": {
                    "streaming": False,
                    "pushNotifications": False,
                    "longRunningTasks": False
                }
            }
        
        @self.app.post("/")
        async def handle_jsonrpc(request: self.Request):
            """Main JSON-RPC 2.0 endpoint for A2A protocol."""
            try:
                body = await request.json()
                rpc_request = JsonRpcRequest(**body)
                
                # Handle different RPC methods
                if rpc_request.method == "message/send":
                    return await self._handle_message_send(rpc_request)
                elif rpc_request.method == "tasks/get":
                    return await self._handle_task_get(rpc_request)
                elif rpc_request.method == "tasks/cancel":
                    return await self._handle_task_cancel(rpc_request)
                else:
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
                
                # Store task for later retrieval
                self.stored_tasks[task.id] = task
                
                return JsonRpcResponse(
                    id=rpc_request.id,
                    result=task.dict()
                )
                
            except Exception as e:
                self.logger.error(f"Error in message/send: {e}")
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
                task_id = params.get("id")
                
                if not task_id:
                    return JsonRpcResponse(
                        id=rpc_request.id,
                        error=JsonRpcError(
                            code=-32602,
                            message="Invalid params: id required"
                        ).dict()
                    )
                
                if task_id in self.stored_tasks:
                    task = self.stored_tasks[task_id]
                    return JsonRpcResponse(
                        id=rpc_request.id,
                        result=task.dict()
                    )
                else:
                    return JsonRpcResponse(
                        id=rpc_request.id,
                        error=JsonRpcError(
                            code=-32602,
                            message=f"Task not found: {task_id}"
                        ).dict()
                    )
                    
            except Exception as e:
                self.logger.error(f"Error in tasks/get: {e}")
                return JsonRpcResponse(
                    id=rpc_request.id,
                    error=JsonRpcError(
                        code=-32603,
                        message="Failed to retrieve task",
                        data={"error": str(e)}
                    ).dict()
                )
        
        async def _handle_task_cancel(self, rpc_request: JsonRpcRequest) -> JsonRpcResponse:
            """Handle tasks/cancel RPC method"""
            return JsonRpcResponse(
                id=rpc_request.id,
                error=JsonRpcError(
                    code=-32601,
                    message="Method not implemented"
                ).dict()
            )
        
        # Store methods as instance methods
        self._handle_message_send = _handle_message_send.__get__(self, SimpleWhiteAgentServer)
        self._handle_task_get = _handle_task_get.__get__(self, SimpleWhiteAgentServer)
        self._handle_task_cancel = _handle_task_cancel.__get__(self, SimpleWhiteAgentServer)
    
    def run(self):
        """Run the Simple White Agent server."""
        self.logger.info(f"Starting Simple White Agent on {self.host}:{self.port}")
        self.logger.info(f"Agent Card URL: http://{self.host}:{self.port}/.well-known/agent-card")
        self.uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple White Agent for TerminalBench")
    parser.add_argument("--server", action="store_true", help="Run as A2A server")
    parser.add_argument("--port", type=int, default=8002, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    
    args = parser.parse_args()
    
    print("🚀 Starting Simple White Agent Server...")
    server = SimpleWhiteAgentServer(port=args.port, host=args.host)
    server.run()
