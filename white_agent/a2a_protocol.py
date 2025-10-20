"""
A2A Protocol Data Models

This module implements the official A2A (Agent-to-Agent) protocol specification.
Reference: https://google.github.io/a2a-protocol-spec/
"""

from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


# ============================================================================
# Core A2A Protocol Models (JSON-RPC 2.0 + A2A Extensions)
# ============================================================================

class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 Request"""
    jsonrpc: str = "2.0"
    id: str
    method: str
    params: Optional[Dict[str, Any]] = None


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 Response"""
    class Config:
        # Exclude None values from JSON output
        exclude_none = True
    
    jsonrpc: str = "2.0"
    id: str
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 Error"""
    code: int
    message: str
    data: Optional[Any] = None


# ============================================================================
# Part Types (Content Containers)
# ============================================================================

class TextPart(BaseModel):
    """Text content part"""
    kind: Literal["text"] = "text"
    text: str


class FileInline(BaseModel):
    """Inline file data (Base64 encoded)"""
    name: str
    mimeType: str
    bytes: str  # Base64 encoded


class FileUri(BaseModel):
    """File reference via URI"""
    name: str
    mimeType: str
    uri: str


class FilePart(BaseModel):
    """File content part"""
    kind: Literal["file"] = "file"
    file: Union[FileInline, FileUri]


class DataPart(BaseModel):
    """Structured JSON data part"""
    kind: Literal["data"] = "data"
    data: Dict[str, Any]


# Union of all part types
Part = Union[TextPart, FilePart, DataPart]


# ============================================================================
# Message Types
# ============================================================================

class Message(BaseModel):
    """A message in A2A protocol"""
    messageId: str = Field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:12]}")
    kind: Literal["message"] = "message"
    role: Literal["user", "agent"]
    parts: List[Part]
    contextId: Optional[str] = None
    taskId: Optional[str] = None  # For continuing existing tasks
    referenceTaskIds: Optional[List[str]] = None
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Optional[Dict[str, Any]] = {}


# ============================================================================
# Task Types
# ============================================================================

class TaskStatus(BaseModel):
    """Task status information"""
    state: Literal[
        "submitted",      # Task received, queued for processing
        "working",        # Task in progress
        "input-required", # Waiting for user input
        "auth-required",  # Waiting for authentication
        "completed",      # Task finished successfully
        "canceled",       # Task canceled by user
        "rejected",       # Task rejected by agent
        "failed"          # Task failed with error
    ]
    message: Optional[str] = None
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat())


class Artifact(BaseModel):
    """An artifact produced by a task"""
    artifactId: str = Field(default_factory=lambda: f"artifact-{uuid.uuid4().hex[:12]}")
    name: str
    description: Optional[str] = None
    parts: List[Part]
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat())


class Task(BaseModel):
    """A stateful unit of work in A2A"""
    id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex[:12]}")
    kind: Literal["task"] = "task"
    contextId: str = Field(default_factory=lambda: f"ctx-{uuid.uuid4().hex[:12]}")
    status: TaskStatus
    artifacts: Optional[List[Artifact]] = []
    history: Optional[List[Message]] = []  # Changed from messages to history per spec
    createdAt: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updatedAt: Optional[str] = Field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Optional[Dict[str, Any]] = {}


# ============================================================================
# Agent Discovery (Agent Card)
# ============================================================================

class AgentSkill(BaseModel):
    """A skill/capability advertised by an agent"""
    id: str
    name: str
    description: str
    inputModes: Optional[List[str]] = ["text"]
    outputModes: Optional[List[str]] = ["text"]
    examples: Optional[List[str]] = []
    tags: Optional[List[str]] = []


class SecurityScheme(BaseModel):
    """Authentication scheme"""
    type: Literal["bearer", "oauth2", "apiKey", "openIdConnect"]
    scheme: Optional[str] = None
    bearerFormat: Optional[str] = None
    authorizationUrl: Optional[str] = None
    tokenUrl: Optional[str] = None


class AgentCapabilities(BaseModel):
    """Agent capabilities"""
    streaming: Optional[bool] = False
    pushNotifications: Optional[bool] = False
    longRunningTasks: Optional[bool] = True


class AgentProvider(BaseModel):
    """Information about the agent provider"""
    organization: str
    url: Optional[str] = None


class TransportProtocol(BaseModel):
    """Transport protocol declaration"""
    url: str
    transport: str  # "JSONRPC", "GRPC", or "HTTP+JSON"


class AgentCard(BaseModel):
    """Agent Card for A2A discovery"""
    protocolVersion: str = "0.3.0"  # A2A protocol version
    name: str
    description: str
    url: str
    preferredTransport: str = "JSONRPC"  # Main transport protocol
    version: str
    provider: Optional[AgentProvider] = None
    iconUrl: Optional[str] = None
    documentationUrl: Optional[str] = None
    skills: List[AgentSkill]
    capabilities: Optional[AgentCapabilities] = Field(default_factory=AgentCapabilities)
    securitySchemes: Optional[Dict[str, SecurityScheme]] = None
    security: Optional[List[Dict[str, List[str]]]] = None
    defaultInputModes: Optional[List[str]] = ["text/plain", "application/json"]
    defaultOutputModes: Optional[List[str]] = ["text/plain", "application/json"]
    additionalInterfaces: Optional[List[TransportProtocol]] = None
    supportsAuthenticatedExtendedCard: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = {}


# ============================================================================
# A2A API Request/Response Models
# ============================================================================

class SendMessageRequest(BaseModel):
    """Request to send a message to an agent"""
    message: Message


class SendMessageResponse(BaseModel):
    """Response from sending a message"""
    result: Union[Task, Message]  # Can be a Task or immediate Message response


class GetTaskRequest(BaseModel):
    """Request to get task status"""
    taskId: str


class GetTaskResponse(BaseModel):
    """Response with task information"""
    result: Task


class CancelTaskRequest(BaseModel):
    """Request to cancel a task"""
    taskId: str
    reason: Optional[str] = None


class CancelTaskResponse(BaseModel):
    """Response from canceling a task"""
    result: Task


# ============================================================================
# Streaming Events (SSE)
# ============================================================================

class TaskStatusUpdateEvent(BaseModel):
    """Event for task status updates (streaming)"""
    eventType: Literal["task-status-update"] = "task-status-update"
    task: Task


class TaskArtifactUpdateEvent(BaseModel):
    """Event for artifact updates (streaming)"""
    eventType: Literal["task-artifact-update"] = "task-artifact-update"
    taskId: str
    artifact: Artifact


class TaskMessageEvent(BaseModel):
    """Event for new messages (streaming)"""
    eventType: Literal["task-message"] = "task-message"
    taskId: str
    message: Message


StreamingEvent = Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent, TaskMessageEvent]


# ============================================================================
# Helper Functions
# ============================================================================

def create_text_message(text: str, role: Literal["user", "agent"] = "user", 
                       context_id: Optional[str] = None) -> Message:
    """Helper to create a simple text message"""
    return Message(
        role=role,
        parts=[TextPart(text=text)],
        contextId=context_id
    )


def create_completed_task(message: Message, artifacts: List[Artifact], 
                         context_id: Optional[str] = None) -> Task:
    """Helper to create a completed task"""
    return Task(
        contextId=context_id or f"ctx-{uuid.uuid4().hex[:12]}",
        status=TaskStatus(state="completed"),
        artifacts=artifacts,
        messages=[message]
    )


def create_text_artifact(text: str, name: str = "response", 
                        description: Optional[str] = None) -> Artifact:
    """Helper to create a text artifact"""
    return Artifact(
        name=name,
        description=description,
        parts=[TextPart(text=text)]
    )

