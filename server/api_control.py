"""
server/api_control.py — the FULL-CONTROL endpoint set.

Every control the PyQt GUI exposed is drivable from here, so a future
frontend (web browser, mobile app, other backends) can fully replace
the desktop window:

  PyQt control                      REST equivalent
  ---------------------------------------------------------------
  input box / Send                  POST /control/message {mode:send_now}
  Send to queue                     POST /control/message {mode:queue}
  pending queue display             GET  /control/queue
  Clear queue                       DELETE /control/queue
  User name box                     GET/POST /control/user
  'Pause User Input' checkbox       GET/POST /control/pause
  Send Now                          POST /control/send-now
  Add file                          POST /control/attach
  agent dropdown                    GET  /control/agents
  Create/Edit agent dialog          POST /control/agents, PATCH .../{name}
  Delete agent                      DELETE /control/agents/{name}
  Rename agent                      PUT  /control/agents/{name}
  sidebar Stop (per agent)          POST /control/agents/{name}/stop
  sidebar Inject                    POST /control/agents/{name}/inject
  core_agent_active                 POST /control/agents/{name}/active
  Save/Load agent to/from file      POST /control/agents/save|load
  Session name box                  GET/POST /control/session/name
  Save Session                      POST /control/session/save
  Load Session                      POST /control/session/load
  New Session                       POST /control/session/new
  (new) finish / abort              POST /control/session/finish|abort
  Refresh tools / prompts           POST /control/tools/reload, ...
  MCP servers dialog                /control/mcp/*
  endpoint & model dropdowns        /control/endpoints/*
  model memory badges               GET /control/endpoints/usage

Session-scoped operations require an ACTIVE session (queue-launched or
interactive) — otherwise they answer 409 with an explanatory message.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

import tool_injection  # top-level module (backend root is on sys.path)
from .api_models import (
    AgentActiveRequest, AgentCreateRequest, AgentRenameRequest,
    AgentUpdateRequest, AttachFileRequest, InjectRequest,
    InjectTextRequest, LoadAgentRequest, LoadSessionRequest,
    MCPConnectRequest, MCPServerRequest, MCPSubscribeRequest,
    MessageRequest, NewSessionRequest, PauseUserInputRequest,
    SaveAgentRequest, SaveSessionRequest, ScanLanRequest,
    SetSessionNameRequest, SetUserNameRequest, ShutdownRequest,
)
from .state import _auto_rescan, _auto_rescan_registry

router = APIRouter(prefix="/control", tags=["control"])


# ======================================================================
# Helpers
# ======================================================================
def get_app(request: Request):
    return request.app.state.app


def require_api_key(request: Request, app=Depends(get_app)):
    """Optional shared-secret gate (set ML_API_KEY to enable)."""
    key = app.config.api_key
    if key:
        provided = request.headers.get("X-API-Key") \
            or request.query_params.get("token")
        if provided != key:
            raise HTTPException(status_code=401, detail="invalid or missing API key")
    return app


def active_runner(app):
    runner = app.queue_manager.active_runner
    if runner is None or runner.done:
        raise HTTPException(
            status_code=409,
            detail=("no active session — start one with "
                    "POST /control/session/new, POST /control/session/load "
                    "or POST /queue/sessions"))
    return runner


def _err(exc: Exception, code: int = 400) -> HTTPException:
    return HTTPException(status_code=code, detail=str(exc))


# ======================================================================
# Chat / user controls
# ======================================================================
@router.get("/overview")
def control_overview(app=Depends(require_api_key)):
    """Machine-readable map of every control endpoint (frontend helper)."""
    return {
        "message": "POST /control/message {text, mode: queue|send_now}",
        "queue": "GET/DELETE /control/queue",
        "user": "GET/POST /control/user {name}",
        "pause": "GET/POST /control/pause {paused}",
        "send_now": "POST /control/send-now",
        "attach": "POST /control/attach {path}",
        "agents": ("GET/POST /control/agents; GET/PATCH/DELETE "
                   "/control/agents/{name}; PUT rename; POST "
                   "/control/agents/{name}/active|stop|inject"),
        "agent_files": "POST /control/agents/save | /control/agents/load",
        "session": ("GET /control/session; POST /control/session/new|save|"
                    "load|finish|abort; GET/POST /control/session/name"),
        "tools": ("GET /control/tools (auto-reloads from tools/ on every "
                  "call); POST /control/tools/reload (force)"),
        "prompts": ("GET /control/prompts (auto-reloads from prompts/ on "
                    "every call); POST /control/prompts/reload (force); "
                    "GET /control/prompts/{name}"),
        "mcp": ("GET/POST /control/mcp/servers (auto-reloads "
                "mcp_servers.json); DELETE /control/mcp/servers/{name}; "
                "POST /control/mcp/servers/{name}/connect|disconnect; GET "
                "/control/mcp/servers/{name}/tools; GET /control/mcp/status; "
                "POST /control/mcp/reload; POST /control/mcp/subscribe|"
                "unsubscribe"),
        "catalog": ("POST /control/catalog/reload — force-reload tools + "
                    "prompts + MCP registry in one call"),
        "endpoints": ("GET /control/endpoints; GET "
                      "/control/endpoints/models?endpoint=; POST "
                      "/control/endpoints/scan; GET /control/endpoints/usage"),
        "system": "POST /system/shutdown",
        "names": "GET /control/names (names.txt pool for agent auto-naming)",
    }


@router.post("/message")
def send_message(body: MessageRequest, app=Depends(require_api_key)):
    """Send a chat message — the input box + Send/Send-to-queue buttons."""
    runner = active_runner(app)
    text = (body.text or "").strip()
    if not text:
        raise _err(ValueError("text must not be empty"))
    with runner.conversation._lock:
        runner.conversation.pending_user_messages.append(text)
    app.broadcast.publish("queue_updated", {
        "pending_user_messages": runner.pending_messages(),
    })
    if body.mode in ("send_now", "now"):
        runner.unblock_user()
    return {"ok": True, "mode": body.mode,
            "pending": runner.pending_messages()}


@router.get("/queue")
def get_queue(app=Depends(require_api_key)):
    """The pending-user-message queue of the active session."""
    runner = active_runner(app)
    return {"pending_user_messages": runner.pending_messages()}


@router.delete("/queue")
def clear_queue(app=Depends(require_api_key)):
    runner = active_runner(app)
    with runner.conversation._lock:
        runner.conversation.pending_user_messages.clear()
    app.broadcast.publish("queue_updated", {"pending_user_messages": []})
    return {"ok": True, "pending": []}


@router.get("/user")
def get_user(app=Depends(require_api_key)):
    runner = active_runner(app)
    return {"user_name": runner.user_name}


@router.post("/user")
def set_user(body: SetUserNameRequest, app=Depends(require_api_key)):
    runner = active_runner(app)
    name = (body.name or "").strip() or "User"
    runner.set_user_name(name)
    return {"ok": True, "user_name": runner.user_name}


@router.get("/pause")
def get_pause(app=Depends(require_api_key)):
    runner = active_runner(app)
    return {"paused_user_input": runner.user_paused}


@router.post("/pause")
def set_pause(body: PauseUserInputRequest, app=Depends(require_api_key)):
    """The 'Pause User Input' checkbox."""
    runner = active_runner(app)
    runner.set_user_paused(bool(body.paused))
    return {"ok": True, "paused_user_input": runner.user_paused}


@router.post("/send-now")
def send_now(app=Depends(require_api_key)):
    """The 'Send' button: unblock every engine's user-turn wait."""
    runner = active_runner(app)
    runner.unblock_user()
    return {"ok": True}


@router.post("/attach")
def attach_file(body: AttachFileRequest, app=Depends(require_api_key)):
    """The 'Add file' button — the raw path is queued verbatim."""
    runner = active_runner(app)
    path = (body.path or "").strip()
    if not path:
        raise _err(ValueError("path must not be empty"))
    if not os.path.exists(path):
        raise _err(ValueError(f"path does not exist: {path}"))
    with runner.conversation._lock:
        runner.conversation.pending_user_messages.append(path)
    app.broadcast.publish("queue_updated", {
        "pending_user_messages": runner.pending_messages(),
    })
    # UI confirmation only — never enters the history (proven behaviour).
    app.broadcast.publish("chat", {
        "sender": "system",
        "text": f"Added file to queue: {path}",
        "role": "system",
        "display_only": True,
    })
    return {"ok": True, "pending": runner.pending_messages()}


# ======================================================================
# Agent controls
# ======================================================================
@router.get("/agents")
def list_agents(app=Depends(require_api_key)):
    runner = active_runner(app)
    from .state import _agent_public
    agents = []
    for name in runner.manager.list_agents():
        agent = runner.manager.get_agent(name)
        if agent is not None:
            agents.append(_agent_public(agent, name in runner.running_agents))
    return {"agents": agents}


@router.post("/agents")
def create_agent(body: AgentCreateRequest, app=Depends(require_api_key)):
    runner = active_runner(app)
    try:
        agent = runner.manager.create_agent(**body.model_dump())
    except ValueError as exc:
        raise _err(exc)
    except TypeError as exc:
        raise _err(exc)
    return {"ok": True, "agent": {"name": agent.name,
                                  "endpoint": agent.endpoint}}


@router.get("/agents/{name}")
def get_agent(name: str, app=Depends(require_api_key)):
    runner = active_runner(app)
    agent = runner.manager.get_agent(name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"no agent '{name}'")
    from .state import _agent_public
    return {"agent": _agent_public(agent, name in runner.running_agents)}


@router.patch("/agents/{name}")
def update_agent(name: str, body: AgentUpdateRequest,
                 app=Depends(require_api_key)):
    """Edit Agent dialog: only provided fields change."""
    runner = active_runner(app)
    data = {k: v for k, v in body.model_dump().items() if v is not None}
    data.pop("name", None)  # renames go through PUT
    if not data:
        raise _err(ValueError("no fields to update"))
    try:
        runner.manager.update_agent(name, **data)
    except ValueError as exc:
        raise _err(exc)
    agent = runner.manager.get_agent(name)
    from .state import _agent_public
    return {"ok": True, "agent": _agent_public(
        agent, name in runner.running_agents)}


@router.put("/agents/{name}")
def rename_agent(name: str, body: AgentRenameRequest,
                 app=Depends(require_api_key)):
    runner = active_runner(app)
    try:
        runner.manager.rename_agent(name, body.new_name)
        runner.conversation.rename_sender(name, body.new_name)
    except ValueError as exc:
        raise _err(exc)
    return {"ok": True, "old_name": name, "new_name": body.new_name}


@router.delete("/agents/{name}")
def delete_agent(name: str, app=Depends(require_api_key)):
    runner = active_runner(app)
    if runner.manager.get_agent(name) is None:
        raise HTTPException(status_code=404, detail=f"no agent '{name}'")
    runner.manager.delete_agent(name)
    return {"ok": True, "deleted": name}


@router.post("/agents/{name}/active")
def set_agent_active(name: str, body: AgentActiveRequest,
                     app=Depends(require_api_key)):
    """core_agent_active equivalent — in/out of the round-robin."""
    runner = active_runner(app)
    agent = runner.manager.get_agent(name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"no agent '{name}'")
    agent.active = bool(body.active)
    runner.manager.agents_changed.emit()
    return {"ok": True, "name": name, "active": agent.active}


@router.post("/agents/{name}/stop")
def stop_agent(name: str, app=Depends(require_api_key)):
    """Sidebar Stop — stop the agent's currently running stream."""
    runner = active_runner(app)
    if runner.manager.get_agent(name) is None:
        raise HTTPException(status_code=404, detail=f"no agent '{name}'")
    ok = runner.stop_agent(name)
    app.broadcast.publish("chat", {
        "sender": "system",
        "text": f"[Stop requested for {name}]",
        "role": "system",
        "display_only": True,
    })
    return {"ok": ok}


@router.post("/agents/{name}/inject")
def inject_to_agent(name: str, body: InjectTextRequest,
                    app=Depends(require_api_key)):
    """Sidebar Inject — queue an operator prompt-injection for the agent.

    Delivered prepended to the agent's NEXT tool result (proven
    tool_injection semantics); text is stripped from history afterwards.
    """
    runner = active_runner(app)
    if runner.manager.get_agent(name) is None:
        raise HTTPException(status_code=404, detail=f"no agent '{name}'")
    text = (body.text or "").strip()
    if not text:
        raise _err(ValueError("text must not be empty"))
    count = tool_injection.add_injection(name, text)
    app.broadcast.publish("injection", {
        "agent_name": name, "count": count, "live": True, "staged": False,
    })
    return {"ok": True, "agent_name": name, "pending": count}


@router.get("/agents/{name}/injections")
def agent_injections(name: str, app=Depends(require_api_key)):
    active_runner(app)  # 409 when idle
    return {"agent_name": name,
            "pending": tool_injection.pending_injection_count(name),
            "queue": [i.get("text") for i in
                      tool_injection.get_pending_injections(name)]}


@router.post("/agents/save")
def save_agent_to_file(body: SaveAgentRequest, app=Depends(require_api_key)):
    """Save Agent to File — portable JSON (no history)."""
    runner = active_runner(app)
    try:
        data = runner.manager.export_agent(body.name)
    except ValueError as exc:
        raise _err(exc)
    path = body.path or f"{body.name}.json"
    if not path.lower().endswith(".json"):
        path += ".json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise _err(exc, code=500)
    return {"ok": True, "path": os.path.abspath(path)}


@router.post("/agents/load")
def load_agent_from_file(body: LoadAgentRequest, app=Depends(require_api_key)):
    runner = active_runner(app)
    try:
        with open(body.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        agent = runner.manager.import_agent(data, new_name=body.new_name)
    except OSError as exc:
        raise _err(exc, code=500)
    except ValueError as exc:
        raise _err(exc)
    return {"ok": True, "name": agent.name}


# ======================================================================
# Session controls
# ======================================================================
@router.get("/session")
def get_session(app=Depends(require_api_key)):
    runner = app.queue_manager.active_runner
    if runner is None:
        return {"active": False,
                "hint": "start one via /control/session/new, "
                        "/control/session/load or POST /queue/sessions"}
    return {"active": True, **runner.summary()}


@router.post("/session/new")
def new_session(body: NewSessionRequest, app=Depends(require_api_key)):
    """New Session — wipes the current interactive session after a final
    autosave and starts a fresh empty one.

    wait_user_input (default True): the fresh session starts with the
    'Pause User Input' checkbox ON — agents WAIT for your first message
    (or Send Now) instead of chatting immediately, exactly like the
    proven PyQt default.
    """
    name = (body.name or "").strip() or app.session_store.next_default_session_name()
    wait_user_input = True if body.wait_user_input is None \
        else bool(body.wait_user_input)
    data = {
        "app_dir": app.app_dir,
        "session_name": name,
        "user_name": body.user_name or "User",
        "agents": {},
        "conversation": {"history": []},
        "mcp_registry": None,
    }
    try:
        runner = app.queue_manager.start_interactive(
            data, name, wait_user_input=wait_user_input)
    except RuntimeError as exc:
        raise _err(exc, code=409)
    return {"ok": True, "session_name": runner.session_name,
            "wait_user_input": runner.user_paused}


@router.post("/session/load")
def load_session(body: LoadSessionRequest, app=Depends(require_api_key)):
    """Load Session (from a file path) as the active interactive session.

    wait_user_input (default True): the loaded session starts WAITING
    for your next message instead of resuming the chat immediately
    (same default as POST /control/session/new).
    """
    try:
        data = app.session_store.load_from_path(body.path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"file not found: {body.path}")
    except (ValueError, json.JSONDecodeError) as exc:
        raise _err(exc)
    name = (str(data.get("session_name") or "").strip()
            or app.session_store.next_default_session_name())
    wait_user_input = True if body.wait_user_input is None \
        else bool(body.wait_user_input)
    try:
        runner = app.queue_manager.start_interactive(
            data, name, wait_user_input=wait_user_input)
    except RuntimeError as exc:
        raise _err(exc, code=409)
    return {"ok": True, "session_name": runner.session_name,
            "path": body.path, "wait_user_input": runner.user_paused}


@router.post("/session/save")
def save_session(body: SaveSessionRequest, app=Depends(require_api_key)):
    runner = active_runner(app)
    path = body.path
    if not path:
        base = (runner.session_name or "session").strip() or "session"
        path = f"{base}.json"
    if not path.lower().endswith(".json"):
        path += ".json"
    try:
        written = app.session_store.save_to_path(
            runner.session_name, runner.user_name,
            runner.manager, runner.conversation, path)
    except OSError as exc:
        raise _err(exc, code=500)
    return {"ok": True, "path": os.path.abspath(written)}


@router.get("/session/name")
def get_session_name(app=Depends(require_api_key)):
    runner = active_runner(app)
    return {"session_name": runner.session_name}


@router.post("/session/name")
def set_session_name(body: SetSessionNameRequest, app=Depends(require_api_key)):
    """The session Name box — applies from the next auto-save on."""
    runner = active_runner(app)
    name = (body.name or "").strip()
    if not name:
        raise _err(ValueError("name must not be empty"))
    runner.session_name = name
    return {"ok": True, "session_name": runner.session_name}


@router.post("/session/finish")
def finish_session(app=Depends(require_api_key)):
    """Graceful finish: running turns complete, then final auto-save."""
    runner = active_runner(app)
    runner.request_finish()
    return {"ok": True, "finishing": runner.session_name}


@router.post("/session/abort")
def abort_session(app=Depends(require_api_key)):
    """Hard stop: cancel streams, then final auto-save."""
    runner = active_runner(app)
    runner.abort()
    return {"ok": True, "aborting": runner.session_name}


# ======================================================================
# Tools / prompts / names
#
# AUTO-RELOAD (the proven PyQt behaviour, kept in the fullstack version):
# every list call re-reads its sources from disk first — tool files
# dropped into tools/ (including ones written by agents via
# core_create_tool), prompt files in prompts/, servers hand-edited into
# mcp_servers.json and names.txt all appear on the very next call, with
# no app restart and no explicit POST .../reload. The check is a couple
# of stat() calls; a real reload only happens when something changed.
# ======================================================================
@router.get("/names")
def list_names(app=Depends(require_api_key)):
    """The agent-name pool (names.txt — working dir first, then app dir).

    One candidate name per non-empty line. Used by frontends for the
    name picker and for the random auto-suggested name when a new agent
    is created (the proven PyQt behaviour). The file is re-read FRESH
    on every call. Also embedded in every /state snapshot as the
    "names" block.
    """
    from .state import names_block
    return names_block(app)


@router.get("/tools")
def list_tools(app=Depends(require_api_key)):
    loader = app.loader
    _auto_rescan(loader)                 # fresh from tools/ on every call
    return {
        "tool_files": [
            {
                "file": fname,
                "doc": loader.get_tool_file_docstring(fname),
                "tools": loader.file_tools.get(fname, []),
            }
            for fname in loader.get_tool_file_names()
        ],
        "tools": [
            {"name": t, "doc": loader.get_tool_docstring(t),
             "file": loader.get_tool_file_for_tool(t)}
            for t in loader.get_tool_names()
        ],
    }


@router.post("/tools/reload")
def reload_tools(app=Depends(require_api_key)):
    loader = app.loader
    loader.reload()
    return {"ok": True, "tools": len(loader.get_tool_names())}


@router.get("/prompts")
def list_prompts(app=Depends(require_api_key)):
    pl = app.prompt_loader
    _auto_rescan(pl)                     # fresh from prompts/ on every call
    return {"prompts": [
        {"name": n, "description": pl.get_prompt_description(n)}
        for n in pl.get_prompt_names()
    ]}


@router.post("/prompts/reload")
def reload_prompts(app=Depends(require_api_key)):
    app.prompt_loader.reload()
    return {"ok": True, "prompts": len(app.prompt_loader.get_prompt_names())}


@router.get("/prompts/{name}")
def get_prompt(name: str, app=Depends(require_api_key)):
    text = app.prompt_loader.get_prompt_text(name)
    if text is None:
        # The prompts folder is re-scanned before answering, so a prompt
        # dropped in moments ago is found without any reload call.
        _auto_rescan(app.prompt_loader)
        text = app.prompt_loader.get_prompt_text(name)
    if text is None:
        raise HTTPException(status_code=404, detail=f"no prompt '{name}'")
    return {"name": name, "text": text}


# ======================================================================
# MCP
# ======================================================================
@router.get("/mcp/servers")
def mcp_servers(app=Depends(require_api_key)):
    mgr = app.mcp_manager
    if mgr is None:
        return {"available": False, "servers": {}}
    _auto_rescan_registry(mgr)          # fresh from mcp_servers.json
    return {"available": True, "servers": mgr.list_registry_servers()}


@router.post("/mcp/servers")
def mcp_register(body: MCPServerRequest, app=Depends(require_api_key)):
    mgr = app.mcp_manager
    if mgr is None:
        raise HTTPException(status_code=503, detail="MCP engine unavailable")
    cfg = {"transport": body.transport}
    for key in ("command", "args", "url", "env", "description"):
        value = getattr(body, key)
        if value is not None:
            cfg[key] = value
    try:
        mgr.register_server(body.name, **cfg)
    except Exception as exc:
        raise _err(exc)
    return {"ok": True, "name": body.name, "config": cfg}


@router.delete("/mcp/servers/{name}")
def mcp_remove(name: str, app=Depends(require_api_key)):
    mgr = app.mcp_manager
    if mgr is None:
        raise HTTPException(status_code=503, detail="MCP engine unavailable")
    removed = mgr.remove_server(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"no MCP server '{name}'")
    return {"ok": True, "removed": name}


@router.post("/mcp/servers/{name}/connect")
def mcp_connect(name: str, body: MCPConnectRequest,
                app=Depends(require_api_key)):
    mgr = app.mcp_manager
    if mgr is None:
        raise HTTPException(status_code=503, detail="MCP engine unavailable")
    ok, message = mgr.connect_server(name, wait=float(body.wait))
    return {"ok": bool(ok), "message": message}


@router.post("/mcp/servers/{name}/disconnect")
def mcp_disconnect(name: str, app=Depends(require_api_key)):
    mgr = app.mcp_manager
    if mgr is None:
        raise HTTPException(status_code=503, detail="MCP engine unavailable")
    return {"ok": True, "message": mgr.disconnect_server(name)}


@router.get("/mcp/servers/{name}/tools")
def mcp_tools(name: str, app=Depends(require_api_key)):
    mgr = app.mcp_manager
    if mgr is None:
        raise HTTPException(status_code=503, detail="MCP engine unavailable")
    _auto_rescan_registry(mgr)          # definitions fresh before listing
    return {"server": name, "tools": mgr.list_tools_info(name)}


@router.get("/mcp/status")
def mcp_status(app=Depends(require_api_key)):
    mgr = app.mcp_manager
    if mgr is None:
        return {"available": False, "status": []}
    _auto_rescan_registry(mgr)          # definitions fresh before reporting
    return {"available": True, "status": mgr.status_info()}


@router.post("/mcp/reload")
def mcp_reload(app=Depends(require_api_key)):
    """Force a re-read of mcp_servers.json (registry definitions only —
    live connections are untouched). Rarely needed: every listing
    already re-reads the file automatically."""
    mgr = app.mcp_manager
    if mgr is None:
        raise HTTPException(status_code=503, detail="MCP engine unavailable")
    mgr.reload_registry()
    return {"ok": True, "servers": len(mgr.list_registry_servers())}


@router.post("/catalog/reload")
def reload_catalog(app=Depends(require_api_key)):
    """Force-reload EVERYTHING the agent-creation lists show — tools
    (full re-import), prompts and the MCP server registry. The REST
    equivalent of pressing Tools Refresh + Prompts Refresh + MCP
    Refresh in the original PyQt GUI; normally unnecessary because all
    list endpoints auto-reload on every call.
    """
    loader = app.loader
    loader.reload()
    app.prompt_loader.reload()
    mcp_reloaded = False
    mgr = app.mcp_manager
    if mgr is not None:
        try:
            mgr.reload_registry()
            mcp_reloaded = True
        except Exception:
            pass
    return {
        "ok": True,
        "tool_files": len(loader.get_tool_file_names()),
        "tools": len(loader.get_tool_names()),
        "prompts": len(app.prompt_loader.get_prompt_names()),
        "mcp_registry_reloaded": mcp_reloaded,
    }


@router.post("/mcp/subscribe")
def mcp_subscribe(body: MCPSubscribeRequest, app=Depends(require_api_key)):
    """Subscribe an agent to an MCP server (native tools surface)."""
    runner = active_runner(app)
    mgr = app.mcp_manager
    if mgr is None:
        raise HTTPException(status_code=503, detail="MCP engine unavailable")
    agent = runner.manager.get_agent(body.agent_name)
    if agent is None:
        raise HTTPException(status_code=404,
                            detail=f"no agent '{body.agent_name}'")
    message = mgr.subscribe_agent(body.agent_name, body.server)
    agent.mcp = True
    if body.server not in agent.mcp_servers:
        agent.mcp_servers.append(body.server)
    runner.manager.agents_changed.emit()
    return {"ok": True, "message": message}


@router.post("/mcp/unsubscribe")
def mcp_unsubscribe(body: MCPSubscribeRequest, app=Depends(require_api_key)):
    runner = active_runner(app)
    mgr = app.mcp_manager
    if mgr is None:
        raise HTTPException(status_code=503, detail="MCP engine unavailable")
    agent = runner.manager.get_agent(body.agent_name)
    if agent is None:
        raise HTTPException(status_code=404,
                            detail=f"no agent '{body.agent_name}'")
    message = mgr.unsubscribe_agent(body.agent_name, body.server)
    if body.server in agent.mcp_servers:
        agent.mcp_servers.remove(body.server)
    runner.manager.agents_changed.emit()
    return {"ok": True, "message": message}


# ======================================================================
# Endpoints / models / usage
# ======================================================================
@router.get("/endpoints")
def list_endpoints(app=Depends(require_api_key)):
    runner = app.queue_manager.active_runner
    if runner is None:
        return {"endpoints": []}
    result = []
    for ep in runner.manager.get_endpoints():
        text, color = runner.endpoint_status.get(ep, ("Ready", "green"))
        result.append({
            "endpoint": ep,
            "agents": runner.manager.get_agents_on_endpoint(ep),
            "status_text": text,
            "status_color": color,
        })
    return {"endpoints": result}


@router.get("/endpoints/models")
def endpoint_models(endpoint: str = "local", app=Depends(require_api_key)):
    """Model dropdown content for an endpoint (blocks a few seconds)."""
    from ollama_scan import fetch_models_for_endpoint
    try:
        models = fetch_models_for_endpoint(endpoint)
    except Exception as exc:
        raise _err(exc, code=502)
    return {"endpoint": endpoint, "models": models}


@router.post("/endpoints/scan")
def scan_lan(body: ScanLanRequest, app=Depends(require_api_key)):
    """LAN scan for Ollama servers (psutil must be installed)."""
    from ollama_scan import scan_lan_ollama, HAVE_PSUTIL
    if not HAVE_PSUTIL:
        raise HTTPException(status_code=503,
                            detail="psutil is not installed — scan unavailable")
    try:
        hosts = scan_lan_ollama(port=body.port)
    except Exception as exc:
        raise _err(exc, code=502)
    return {"hosts": hosts, "port": body.port}


@router.get("/endpoints/usage")
def endpoint_usage(app=Depends(require_api_key)):
    """Model memory usage per endpoint (Ollama /api/ps)."""
    endpoints = set(app.model_usage.keys())
    runner = app.queue_manager.active_runner
    if runner is not None:
        endpoints.update(runner.manager.get_endpoints())
    usage = {}
    for ep in sorted(endpoints):
        try:
            client = app.client_manager.get_client(ep)
            usage[ep] = client.get_model_usage()
        except Exception:
            usage[ep] = app.model_usage.get(ep, {})
    app.model_usage.update(usage)
    return {"usage": usage}


# ======================================================================
# System
# ======================================================================
system_router = APIRouter(prefix="/system", tags=["system"])


@system_router.get("/info")
def system_info(app=Depends(require_api_key)):
    from .state import build_full_state
    return build_full_state(app)


@system_router.post("/shutdown")
def system_shutdown(body: ShutdownRequest, request: Request,
                    app=Depends(require_api_key)):
    """Graceful server shutdown (final autosave of the active session).

    The API answers first, then the process exits a moment later."""
    if not body.confirm:
        raise _err(ValueError("pass {\"confirm\": true} to shut down"))
    import threading

    def _stop():
        import time as _t
        _t.sleep(0.5)
        app.broadcast.publish("log", {"text": "server shutting down"})
        try:
            app.shutdown()
        finally:
            os._exit(0)

    threading.Thread(target=_stop, daemon=True).start()
    return {"ok": True, "detail": "shutting down"}
