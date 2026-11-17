"""
server/state.py — full-state snapshots for the output endpoints.

GET /state and the WebSocket/SSE "snapshot" event both render through
build_full_state(): the COMPLETE observable state of the backend —

    server              identity, uptime, config, folder layout
    active_session      name, source, options, user, per-agent config +
                        running flags, per-endpoint status, pending
                        user messages, history length
    session_queue       pending entries (+ running entry)
    recent_queue        last finished entries (results + autosave paths)
    staged_injections   operator injections waiting for their session
    live_injections     pending per-agent injection counts
    tools               tool files, tool names, docstrings
    prompts             prompt files + descriptions
    mcp                 registry servers + live status
    names               names.txt agent-name pool (file + list)
    model_usage         /api/ps memory usage per endpoint
    sessions_on_disk    every loadable session file (autosave/ + sessions/)

The chat HISTORY itself is intentionally NOT embedded (it can be huge);
pull it separately from GET /output/history or replay it from
GET /output/recent when reconnecting.
"""

from __future__ import annotations

import os
from typing import Optional


# Agent auto-naming pool file (same convention as the proven PyQt app):
# one candidate name per non-empty line, looked up in the process working
# directory first, then next to the app. A missing file simply means the
# names list is empty.
NAMES_FILE_NAME = "names.txt"


# ----------------------------------------------------------------------
# Auto-reload helpers (the proven PyQt behaviour: a list is re-read from
# disk whenever it is called). Every catalog listing below funnels through
# these, so files dropped into tools/ or prompts/, or a hand-edited
# mcp_servers.json, are reflected on the very next snapshot / list call —
# no restart, no explicit Refresh. Each check is a handful of stat()
# calls; a real reload only happens when something actually changed.
# ----------------------------------------------------------------------
def _auto_rescan(loader_like) -> bool:
    """Best-effort maybe_reload() on a ToolLoader / PromptLoader.
    Returns True when a reload happened; never raises."""
    fn = getattr(loader_like, "maybe_reload", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            pass
    return False


def _auto_rescan_registry(mgr) -> bool:
    """Best-effort maybe_reload_registry() on an MCPManager.
    Returns True when the registry was re-read; never raises."""
    fn = getattr(mgr, "maybe_reload_registry", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            pass
    return False


def _agent_public(agent, running: bool) -> dict:
    return {
        "name": agent.name,
        "description": getattr(agent, "description", "") or "",
        "system_prompt": agent.system_prompt,
        "model": agent.model,
        "num_ctx": agent.num_ctx,
        "temperature": agent.temperature,
        "tools": list(agent.tools or []),
        "max_context_messages": agent.max_context_messages,
        "max_tool_iterations": agent.max_tool_iterations,
        "endpoint": agent.endpoint or "local",
        "active": bool(getattr(agent, "active", True)),
        "silent": bool(getattr(agent, "silent", False)),
        "use_tools": bool(getattr(agent, "use_tools", True)),
        "image": bool(getattr(agent, "image", False)),
        "audio": bool(getattr(agent, "audio", False)),
        "mcp": bool(getattr(agent, "mcp", False)),
        "mcp_servers": list(getattr(agent, "mcp_servers", []) or []),
        "tool_files": getattr(agent, "tool_files", None),
        "prompts": list(getattr(agent, "prompts", []) or []),
        "running": running,
    }


def _active_session_block(app) -> Optional[dict]:
    runner = app.queue_manager.active_runner
    if runner is None:
        return None

    agents = {}
    for name in runner.manager.list_agents():
        agent = runner.manager.get_agent(name)
        if agent is not None:
            agents[name] = _agent_public(agent, name in runner.running_agents)

    endpoints = {}
    for ep in runner.manager.get_endpoints():
        text, color = runner.endpoint_status.get(ep, ("Ready", "green"))
        engine = runner.engines.get(ep)
        endpoints[ep] = {
            "agents": runner.manager.get_agents_on_endpoint(ep),
            "status_text": text,
            "status_color": color,
            "running_agent": engine.get_running_agent() if engine else None,
            "endpoint_down": bool(engine.endpoint_down) if engine else False,
        }

    return {
        "session_name": runner.session_name,
        "source": runner.options.source,           # queue | interactive
        "auto_finish": runner.options.auto_finish,
        "user_name": runner.user_name,
        "paused_user_input": runner.user_paused,
        "started_at": runner.started_at,
        "finished_at": runner.finished_at,
        "finish_reason": runner.finish_reason,
        "autosave_path": runner.autosave_path,
        "last_activity": runner.last_activity,
        "agents": agents,
        "endpoints": endpoints,
        "pending_user_messages": runner.pending_messages(),
        "history_messages": len(runner.conversation.history),
        "options": {
            "message": runner.options.message,
            "activate_all": runner.options.activate_all,
            "activate": runner.options.activate,
            "idle_timeout": runner.options.idle_timeout,
        },
    }


def _tools_block(app) -> dict:
    loader = app.loader
    _auto_rescan(loader)          # fresh from tools/ on every snapshot
    files = []
    for file_name in loader.get_tool_file_names():
        files.append({
            "file": file_name,
            "doc": loader.get_tool_file_docstring(file_name),
            "tools": [
                {
                    "name": t,
                    "doc": loader.get_tool_docstring(t),
                }
                for t in loader.file_tools.get(file_name, [])
            ],
        })
    return {
        "files": files,
        "tool_count": len(loader.get_tool_names()),
    }


def _prompts_block(app) -> list:
    _auto_rescan(app.prompt_loader)   # fresh from prompts/ on every snapshot
    out = []
    for name in app.prompt_loader.get_prompt_names():
        out.append({
            "name": name,
            "description": app.prompt_loader.get_prompt_description(name),
        })
    return out


def _mcp_block(app) -> dict:
    mgr = app.mcp_manager
    if mgr is None:
        return {"available": False, "servers": {}, "status": []}
    _auto_rescan_registry(mgr)        # fresh from mcp_servers.json
    try:
        servers = mgr.list_registry_servers()
    except Exception:
        servers = {}
    try:
        status = mgr.status_info()
    except Exception:
        status = []
    return {"available": True, "servers": servers, "status": status}


def names_block(app) -> dict:
    """The agent-name pool from names.txt (PyQt-compatible lookup).

    Search order (first hit wins): working directory, then the app
    folder. Each non-empty line is one candidate name — the file is read
    fresh on every call so edits made while the server runs are picked
    up by the next state snapshot.

    Returns {"file": <path or None>, "names": [...]}. Frontends use it
    for the single-selection name picker and for the auto-suggested
    random name when creating a new agent (the proven behaviour).
    """
    for base in (os.getcwd(), app.app_dir):
        if not base:
            continue
        path = os.path.join(base, NAMES_FILE_NAME)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except Exception:
            continue
        names = [line.strip() for line in lines if line.strip()]
        return {"file": path, "names": names}
    return {"file": None, "names": []}


def build_full_state(app) -> dict:
    """The complete state snapshot (see module docstring)."""
    queue_mgr = app.queue_manager
    active = _active_session_block(app)

    server_block = {
        "name": "multillama-backend",
        "version": "1.0.0",
        "uptime": round(app.uptime(), 1),
        "app_dir": app.app_dir,
        "pid": __import__("os").getpid(),
        "config": app.config.as_public_dict(),
        "folders": {
            "tools": app.tools_dir,
            "prompts": app.prompts_dir,
            "autosave": app.session_store.autosave_dir,
            "sessions": app.session_store.sessions_dir,
            "dialogues": app.dialogue_store.base_dir,
        },
        "output_subscribers": app.broadcast.subscriber_count(),
    }

    return {
        "server": server_block,
        "active_session": active,
        "session_queue": queue_mgr.queue_snapshot(),
        "recent_queue": queue_mgr.recent_snapshot(limit=20),
        "staged_injections": queue_mgr.staged_snapshot(),
        "live_injections": queue_mgr.live_injection_counts(),
        "tools": _tools_block(app),
        "prompts": _prompts_block(app),
        "mcp": _mcp_block(app),
        "names": names_block(app),
        "model_usage": getattr(app, "model_usage", {}) or {},
        "sessions_on_disk": app.session_store.list_sessions(),
    }
