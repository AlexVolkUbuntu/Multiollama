"""
server/api_models.py — request/response models for the control API.

The field sets mirror the proven PyQt dialogs one-to-one:

  * AgentCreate / AgentUpdate  = EditAgentDialog.get_data()
  * InjectRequest              = PromptInjectDialog
  * QueueSessionRequest        = the "session.name + message before it"
                                 control requirement
  * QueueInjectRequest         = the "session.name + agent name" injection
                                 control requirement
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ======================================================================
# Chat / user controls
# ======================================================================
class MessageRequest(BaseModel):
    """Send a chat message.

    mode:
      * "queue"     — the desktop 'Send to queue' button: the text waits
                      in pending_user_messages until the engines drain it.
      * "send_now"  — the desktop 'Send' button: queue + unblock every
                      engine's user-turn wait so they drain immediately.
    """
    text: str
    sender: str = "user"
    mode: str = Field(default="queue", pattern="^(queue|send_now|now)$")


class SetUserNameRequest(BaseModel):
    name: str


class PauseUserInputRequest(BaseModel):
    paused: bool


class AttachFileRequest(BaseModel):
    """The desktop 'Add file' button: the raw path is pushed verbatim
    onto the pending queue (the agents see it as a plain string)."""
    path: str


# ======================================================================
# Agent controls (EditAgentDialog equivalents)
# ======================================================================
class AgentCreateRequest(BaseModel):
    name: str
    system_prompt: str = ""
    model: str = "qwen3.5:9b"
    num_ctx: int = 32000
    temperature: float = 0.5
    tools: List[str] = Field(default_factory=list)
    max_context_messages: int = 30
    max_tool_iterations: int = 1000
    endpoint: str = "local"
    active: bool = True
    silent: bool = False
    description: str = ""
    use_tools: bool = True
    image: bool = False
    audio: bool = False
    mcp: bool = False
    mcp_servers: Optional[List[str]] = None
    tool_files: Optional[List[str]] = None
    prompts: Optional[List[str]] = None


class AgentUpdateRequest(BaseModel):
    """Every field optional — only provided fields are updated
    (AgentManager.update_agent semantics)."""
    name: Optional[str] = None              # rename target
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    num_ctx: Optional[int] = None
    temperature: Optional[float] = None
    tools: Optional[List[str]] = None
    max_context_messages: Optional[int] = None
    max_tool_iterations: Optional[int] = None
    endpoint: Optional[str] = None
    active: Optional[bool] = None
    silent: Optional[bool] = None
    description: Optional[str] = None
    use_tools: Optional[bool] = None
    image: Optional[bool] = None
    audio: Optional[bool] = None
    mcp: Optional[bool] = None
    mcp_servers: Optional[List[str]] = None
    tool_files: Optional[List[str]] = None
    prompts: Optional[List[str]] = None


class AgentRenameRequest(BaseModel):
    new_name: str


class AgentActiveRequest(BaseModel):
    active: bool


class InjectRequest(BaseModel):
    """Inject by agent name in the BODY (queue-style / tooling use)."""
    agent_name: str
    text: str


class InjectTextRequest(BaseModel):
    """Inject with the agent name taken from the URL path
    (POST /control/agents/{name}/inject — the desktop 'Inject' button)."""
    text: str


class SaveAgentRequest(BaseModel):
    name: str
    path: Optional[str] = None


class LoadAgentRequest(BaseModel):
    path: str
    new_name: Optional[str] = None


# ======================================================================
# Session controls
# ======================================================================
class NewSessionRequest(BaseModel):
    name: Optional[str] = None
    user_name: str = "User"
    # Initial state of the 'Pause User Input' (wait-user-input) checkbox.
    # None -> server default = True: the session WAITS for your first
    # message instead of agents chatting immediately (PyQt parity).
    wait_user_input: Optional[bool] = None


class SaveSessionRequest(BaseModel):
    path: Optional[str] = None


class LoadSessionRequest(BaseModel):
    path: str
    # Same semantics as in NewSessionRequest (None -> True).
    wait_user_input: Optional[bool] = None


class SetSessionNameRequest(BaseModel):
    name: str


# ======================================================================
# MCP controls
# ======================================================================
class MCPServerRequest(BaseModel):
    """Register / update one MCP server definition.

    transport: "stdio" (command + args) or "sse"/"http" (url).
    Matches the proven MCPServerEditDialog fields."""
    name: str
    transport: str = "stdio"
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None
    env: Optional[dict] = None
    description: Optional[str] = None


class MCPConnectRequest(BaseModel):
    wait: float = 60.0


class MCPSubscribeRequest(BaseModel):
    agent_name: str
    server: str


# ======================================================================
# Endpoints / models
# ======================================================================
class ScanLanRequest(BaseModel):
    port: int = 11434


# ======================================================================
# Queue — the additional control endpoints (design brief items 4.1/4.2)
# ======================================================================
class QueueSessionRequest(BaseModel):
    """Requirement 4.1: session name to run + message to send before it.

    Enqueues the session. When it is dequeued the runner loads the
    session's newest auto-save (or a file from sessions/), optionally
    re-activates agents, queues *message* as a user message BEFORE the
    engines start, and runs the session until it quiesces (all agents
    inactive) — then auto-saves it and moves to the next entry.
    """
    session_name: str
    message: Optional[str] = None
    sender: str = "user"
    activate_all: bool = False
    activate: Optional[List[str]] = None
    idle_timeout: Optional[float] = None
    auto_finish: bool = True


class QueueMoveRequest(BaseModel):
    position: int


class QueueInjectRequest(BaseModel):
    """Requirement 4.2: session name + agent name + message.

    Injects the message so it is prepended to that agent's NEXT TOOL
    RESULT — without adding the session to the run queue. When the
    named session is currently running, the injection goes live
    immediately; otherwise it is staged and delivered automatically the
    next time that session runs.
    """
    session_name: str
    agent_name: str
    text: str


# ======================================================================
# System
# ======================================================================
class ShutdownRequest(BaseModel):
    confirm: bool = False
