#!/usr/bin/env python3
"""
ml_mcp_server.py — a STANDALONE MCP (Model Context Protocol) server that
exposes EVERY feature of a running MultiOllama headless backend
(`MultiOllama_backend/` — see its README.md and API_GUIDE.md) as
first-class MCP tools, so ANY MCP-capable agentic app can drive it:

    Claude Desktop, Cursor, Cline, Windsurf, VS Code (MCP), Zed, your own
    agents — and even MultiOllama agents themselves (the backend's own MCP
    engine can register this very server; see README.md §7).

 Architecture
 ════════════

    ┌──────────────────────┐  JSON-RPC over stdio (default)   ┌───────────────────────────┐
    │  MCP client app       │ ────────────────────────────►  │  ml_mcp_server.py          │
    │  (any agentic app)    │ ◄────────────────────────────   │  ~63 tools, thin HTTP      │
    └──────────────────────┘                                  │  client, no state of its  │
                                                              │  own                      │
                                                              └─────────────┬─────────────┘
                                                                plain HTTP + │ X-API-Key
                                                                             ▼
                                                   ┌────────────────────────────────────────┐
                                                   │  MultiOllama backend                    │
                                                   │  python run_server.py --port 8200      │
                                                   │  FastAPI: /state /control /queue       │
                                                   │  /output /files /health /docs          │
                                                   └────────────────────────────────────────┘

 The server is a pure ADAPTER: it holds no sessions, no agents, no files
 — everything stays in the one authoritative backend. One MCP server
 instance per backend URL (run a second copy with --backend-url for a
 second backend).

 Quick start
 ═══════════
     # 1) backend up (any machine)
     cd MultiOllama_backend && python run_server.py --port 8200

     # 2) verify THIS server can reach it
     python ml_mcp_server.py --check --backend-url http://127.0.0.1:8200

     # 3) inspect the tool surface
     python ml_mcp_server.py --list-tools

     # 4) wire it into your MCP client (README.md §5 has copy-paste
     #    configs for Claude Desktop, Cursor, Cline, Windsurf, generic
     #    stdio clients and remote streamable-http setups)

 Configuration (CLI flag > environment variable > default)
 ════════════════════════════════════════════════════════
     --backend-url   ML_BACKEND_URL    http://127.0.0.1:8200
     --api-key      ML_MCP_API_KEY     none
                    (fallback: ML_API_KEY — same value the backend
                     checks, so one env var configures both sides)
     --timeout      ML_MCP_TIMEOUT     30 (seconds per backend call)
     --transport    ML_MCP_TRANSPORT   stdio  (stdio | sse | streamable-http)
     --host/--port  ML_MCP_HOST/PORT    127.0.0.1:8090 (HTTP transports only)

 Transports
 ══════════
     stdio (default)  — for Claude Desktop, Cursor, Cline, ... the client
                        spawns this file and speaks JSON-RPC over stdin/
                        stdout. NOTHING else may print to stdout here.
     streamable-http — run it as a small service for REMOTE agents:
                        python ml_mcp_server.py --transport streamable-http \
                               --host 0.0.0.0 --port 8090 --backend-url http://10.0.0.4:8200
                        MCP endpoint: http://<host>:8090/mcp
     sse              — legacy variant of the same (endpoint /sse).

 The ONE backend rule agents must know
 ═════════════════════════════════════
     AGENTS LIVE INSIDE A SESSION. The backend boots with no session, so
     session-scoped calls answer 409 until you start one:
         ml_session_new (interactive, waits for your first message),
         ml_session_load (from a file — paths via ml_sessions_on_disk), or
         ml_queue_enqueue (unattended queued run).
     Typical flow: ml_backend_health → ml_session_new → ml_agent_create
     → ml_ask / ml_chat_send → ml_chat_history (with 'files' download
     links) → ml_files_url / ml_files_fetch → ml_session_save /
     ml_session_finish.

 Error contract: every tool returns the backend's JSON verbatim on
 success; failures become {"ok": false, "error_kind": ..., "detail": ...,
 "hint": ...} objects (409 → the session rule, 401 → the API key, 502 →
 Ollama endpoint down, unreachable → the backend is not running).

 Requirements: Python >= 3.10, `mcp` >= 1.2, `httpx` >= 0.27
 (see requirements.txt). Independent from the backend's own process and
 dependencies — the ONLY thing they share is the HTTP API.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import inspect
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote as _quote

import httpx

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings
except ImportError as _exc:  # pragma: no cover - clear early failure
    print("ml_mcp_server: the 'mcp' package is required "
          "(pip install -r requirements.txt)", file=sys.stderr)
    raise _exc

try:
    from mcp.types import ToolAnnotations
except ImportError:  # pragma: no cover - annotation hints are optional
    ToolAnnotations = None  # type: ignore



VERSION = "1.0.0"
SERVER_NAME = "MultiOllama-backend"
DEFAULT_BACKEND_URL = "http://127.0.0.1:8200"


# ======================================================================
# Configuration (env first; CLI flags override in main())
# ======================================================================
class _Config:
    """Runtime configuration, resolved once per process."""

    def __init__(self) -> None:
        self.backend_url = (os.environ.get("ML_BACKEND_URL")
                             or DEFAULT_BACKEND_URL).rstrip("/")
        # ML_MCP_API_KEY wins over ML_API_KEY so this server can use a
        # different key than the backend-side env var if ever needed.
        self.api_key = (os.environ.get("ML_MCP_API_KEY")
                        or os.environ.get("ML_API_KEY") or "")
        try:
            self.timeout = float(os.environ.get("ML_MCP_TIMEOUT", "30") or 30)
        except ValueError:
            self.timeout = 30.0


CONFIG = _Config()


INSTRUCTIONS = f"""MultiOllama backend control — every tool drives ONE running headless
backend (currently {DEFAULT_BACKEND_URL if not os.environ.get("ML_BACKEND_URL") else os.environ.get("ML_BACKEND_URL")}) over HTTP.

THE ONE RULE: AGENTS LIVE INSIDE A SESSION. The backend starts with no
session loaded, so session-scoped calls fail with 409 until you start one.

Standard flow:
  1. ml_backend_health — confirm the backend is reachable.
  2. ml_session_new — create the ACTIVE session (starts waiting for input).
  3. ml_agent_create (repeat) — populate the session (models via
     ml_endpoints_models, tools via ml_tools_list).
  4. ml_chat_send or ml_ask — talk to the team; ml_ask waits for the turn
     to finish and returns the appended messages.
  5. ml_chat_history — read the chat; message entries may carry a "files"
     array (real agent-generated files) with ready download URLs — feed
     those to ml_files_url / ml_files_fetch.
  6. ml_session_save / ml_session_finish — persist and close.

Alternatives: ml_queue_enqueue runs whole sessions unattended;
ml_agent_inject / ml_queue_inject steer an agent mid-turn; ml_mcp_* manage
the backend's OWN MCP client registry; ml_endpoints_* inspect Ollama
endpoints; ml_system_shutdown stops the backend itself.

Every tool returns the backend JSON verbatim on success and a
{{"ok": false, "error_kind", "detail", "hint"}} object on failure — read
the hint before retrying. Full catalog with workflows: README.md next to
ml_mcp_server.py."""


# ======================================================================
# HTTP plumbing — one shared client, one error contract
# ======================================================================
_CLIENT: Optional[httpx.AsyncClient] = None
_CLIENT_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _get_client() -> httpx.AsyncClient:
    """The shared httpx client, re-created when the event loop changes
    (stdio runs a single loop; tests may run several)."""
    global _CLIENT, _CLIENT_LOOP
    loop = asyncio.get_running_loop()
    if _CLIENT is None or _CLIENT_LOOP is not loop:
        headers = {"Accept": "application/json"}
        if CONFIG.api_key:
            headers["X-API-Key"] = CONFIG.api_key
        _CLIENT = httpx.AsyncClient(
            base_url=CONFIG.backend_url,
            headers=headers,
            timeout=CONFIG.timeout,
            follow_redirects=True,
        )
        _CLIENT_LOOP = loop
    return _CLIENT


_ERROR_HINTS: Dict[int, str] = {
    400: "read 'detail' — usually a bad field value or an unknown name "
         "(list valid ones with the matching ml_*_list tool)",
    401: "the backend requires an API key — pass --api-key / "
         "ML_MCP_API_KEY / ML_API_KEY with the backend's ML_API_KEY value",
    404: "the name/path does not exist — list valid ones with the "
         "matching ml_*_list tool",
    405: "wrong method for that path — this is a server-side mapping bug, "
         "report it with the tool name",
    409: "no ACTIVE session: agents live inside a session — call "
         "ml_session_new (or ml_session_load / ml_queue_enqueue) first",
    502: "upstream Ollama call failed — check the endpoint and model "
         "(ml_endpoints_list / ml_endpoints_models)",
    503: "optional backend feature missing — pip install psutil (LAN scan) "
         "or mcp (backend MCP engine) on the BACKEND machine",
}


def _error(kind: str, detail: str, hint: str = "", **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": False, "error_kind": kind, "detail": detail}
    if hint:
        out["hint"] = hint
    out.update(extra)
    return out


def _http_detail(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict) and data.get("detail") is not None:
            return str(data["detail"])[:500]
        return json.dumps(data)[:500]
    except Exception:
        return (resp.text or "<no body>")[:500]


async def _call(method: str, path: str, *, body: Any = None,
                params: Any = None, timeout: Optional[float] = None) -> Any:
    """One backend HTTP call -> parsed JSON, or the error contract.

    Success returns the backend payload VERBATIM (POST/DELETE endpoints
    already include an "ok" flag; GET payloads are self-describing).
    Everything that can go wrong becomes {"ok": false, ...} so the MCP
    client always receives a structured, hint-carrying answer.
    """
    client = _get_client()
    try:
        resp = await client.request(method, path, json=body,
                                    params=params, timeout=timeout)
    except httpx.ConnectError as exc:
        return _error(
            "unreachable", str(exc),
            f"no backend answering at {CONFIG.backend_url} — start it first: "
            f"cd MultiOllama_backend && python run_server.py --port 8200 "
            f"(then point --backend-url / ML_BACKEND_URL at it)")
    except httpx.TimeoutException as exc:
        return _error("timeout", str(exc),
                      "the backend did not answer in time — raise the "
                      "timeout (ML_MCP_TIMEOUT / --timeout) or check "
                      "whether the backend is stuck")
    except httpx.HTTPError as exc:
        return _error("transport", str(exc),
                      "HTTP transport failure — check the backend URL "
                      f"({CONFIG.backend_url})")
    if resp.status_code >= 400:
        return _error("http", _http_detail(resp),
                      _ERROR_HINTS.get(resp.status_code, ""),
                      status=resp.status_code)
    ctype = resp.headers.get("content-type", "")
    if "json" in ctype:
        try:
            data = resp.json()
        except Exception:
            return {"ok": True, "data": resp.text}
        return data if isinstance(data, dict) else {"ok": True, "data": data}
    return {"ok": True, "content_type": ctype, "data": resp.text}


def _q(segment: str) -> str:
    """URL-quote one path segment (agent / session / server names)."""
    return _quote(str(segment), safe="")


# ======================================================================
# State compaction (ml_backend_state brief mode)
# ======================================================================
_AGENT_BRIEF_FIELDS = ("name", "description", "model", "endpoint",
                      "active", "running", "silent", "use_tools",
                      "mcp", "image", "audio")


def _brief_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compact a full /state snapshot: keep the decision-relevant fields,
    drop the heavy ones (per-agent full prompts, endpoint details, tool
    docstrings, the whole names pool, every session file on disk)."""
    try:
        server = dict(state.get("server") or {})
        server.pop("folders", None)
        out: Dict[str, Any] = {"server": server}

        active = state.get("active_session")
        if isinstance(active, dict) and active:
            agents = {}
            for name, ag in (active.get("agents") or {}).items():
                agents[name] = ({k: ag.get(k) for k in _AGENT_BRIEF_FIELDS}
                                if isinstance(ag, dict) else ag)
            a = dict(active)
            a["agents"] = agents
            a.pop("endpoints", None)
            out["active_session"] = a
        else:
            out["active_session"] = None

        out["session_queue"] = state.get("session_queue")
        out["queue_counts"] = {
            "recent_queue": len(state.get("recent_queue") or []),
            "staged_injections": len(state.get("staged_injections") or []),
            "live_injections": state.get("live_injections") or {},
        }

        tools = state.get("tools") or {}
        out["tools"] = {
            "tool_count": tools.get("tool_count"),
            "files": [
                {"file": f.get("file"),
                 "tools": [t.get("name") for t in (f.get("tools") or [])]}
                for f in (tools.get("files") or [])
            ],
        }

        out["prompts"] = [p.get("name") for p in state.get("prompts") or []]

        mcp_raw = state.get("mcp") or {}
        mcp = dict(mcp_raw)
        mcp.pop("servers", None)
        mcp["registered"] = list((mcp_raw.get("servers") or {}).keys())
        out["mcp"] = mcp

        names = state.get("names") or {}
        pool = names.get("names") or []
        out["names"] = {"file": names.get("file"), "count": len(pool),
                        "sample": pool[:12]}

        on_disk = state.get("sessions_on_disk") or []
        if isinstance(on_disk, list):
            out["sessions_on_disk"] = {"count": len(on_disk),
                                       "first": on_disk[:15]}
        else:
            out["sessions_on_disk"] = on_disk

        out["model_usage"] = state.get("model_usage")
        return out
    except Exception:
        return state          # compaction must never corrupt the answer


# ======================================================================
# The MCP server instance + the tool surface
# ======================================================================
mcp = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS)

# report THIS server's version in serverInfo (FastMCP would otherwise
# show the SDK's own version there)
try:
    mcp._mcp_server.version = VERSION
except Exception:
    pass


# ----------------------------------------------------------------------
# Backend / system
# ----------------------------------------------------------------------
@mcp.tool()
async def ml_backend_health() -> dict:
    """Cheap connectivity probe (GET /health, no auth): status, uptime,
    the active session name and the run-queue length. Call this FIRST to
    confirm the backend is reachable before anything else."""
    return await _call("GET", "/health")


@mcp.tool()
async def ml_backend_state(brief: bool = True) -> dict:
    """Complete backend snapshot (GET /state): server, active session
    with every agent's config + running flags, run queue, tools, prompts,
    MCP registry status, names pool, sessions on disk.
    brief=true (default) compacts the payload to the decision-relevant
    fields; brief=false returns everything (large — use sparingly)."""
    data = await _call("GET", "/state")
    if isinstance(data, dict) and data.get("ok") is False:
        return data
    return _brief_state(data) if brief else data


@mcp.tool()
async def ml_backend_overview() -> dict:
    """Machine-readable map of every backend REST endpoint
    (GET /control/overview) — the complete surface this MCP server wraps,
    for when an agent wants to go beyond the tools."""
    return await _call("GET", "/control/overview")


@mcp.tool(annotations=(ToolAnnotations(readOnlyHint=False,
                                        destructiveHint=True)
                       if ToolAnnotations else None))
async def ml_system_shutdown(confirm: bool = False) -> dict:
    """Gracefully shut the BACKEND process down (running turns finish,
    final autosave, server exits). DANGEROUS and requires confirm=true.
    This MCP server keeps running — only the backend stops."""
    if not confirm:
        return _error(
            "usage", "refusing to shut down without confirmation",
            "call again with confirm=true only when you really want to "
            "stop the backend process")
    return await _call("POST", "/system/shutdown", body={"confirm": True})


# ----------------------------------------------------------------------
# Sessions — the lifecycle everything else hangs on
# ----------------------------------------------------------------------
@mcp.tool()
async def ml_session_new(name: Optional[str] = None,
                         user_name: str = "User",
                         wait_user_input: bool = True) -> dict:
    """Create a new interactive session — it becomes the ACTIVE session
    (agents live inside a session; until one exists every
    session-scoped tool answers 409). A previous interactive session is
    autosaved first, never lost.
    wait_user_input=true (default): the session starts WAITING for your
    first message instead of the agents chatting immediately."""
    return await _call("POST", "/control/session/new",
                       body={"name": name, "user_name": user_name,
                             "wait_user_input": wait_user_input})


@mcp.tool()
async def ml_session_load(path: str, wait_user_input: bool = True) -> dict:
    """Load a session file as the ACTIVE interactive session.
    Find loadable files with ml_sessions_on_disk (autosave/ + sessions/
    folders, newest first per name)."""
    return await _call("POST", "/control/session/load",
                       body={"path": path,
                             "wait_user_input": wait_user_input})


@mcp.tool()
async def ml_session_get() -> dict:
    """Active-session summary (GET /control/session) — never 409s:
    returns {"active": false, "hint": ...} when no session is running,
    so this is the safe first call after ml_backend_health."""
    return await _call("GET", "/control/session")


@mcp.tool()
async def ml_session_save(path: Optional[str] = None) -> dict:
    """Save the active session to a file. path optional — default
    '<session name>.json' next to the backend. The autosave folder keeps
    its own rolling history independent of this manual save."""
    return await _call("POST", "/control/session/save", body={"path": path})


@mcp.tool()
async def ml_session_rename(name: str) -> dict:
    """Rename the active session (applies from the next autosave on)."""
    return await _call("POST", "/control/session/name", body={"name": name})


@mcp.tool()
async def ml_session_finish() -> dict:
    """Graceful finish: running turns complete, a final autosave
    happens, then the session stops being active (session-scoped tools
    go back to 409)."""
    return await _call("POST", "/control/session/finish")


@mcp.tool()
async def ml_session_abort() -> dict:
    """Hard stop of the active session — running streams are cancelled,
    the final autosave still happens."""
    return await _call("POST", "/control/session/abort")


@mcp.tool()
async def ml_sessions_on_disk() -> dict:
    """Every loadable session file on the backend (autosave/ + sessions/
    folders) — feed one of these paths into ml_session_load. Cheaper
    than ml_backend_state(brief=false) when you only need this block."""
    data = await _call("GET", "/state")
    if isinstance(data, dict) and data.get("ok") is False:
        return data
    return {"ok": True, "active_session": data.get("active_session", {}).get("session_name") if data.get("active_session") else None,
            "sessions_on_disk": data.get("sessions_on_disk")}


# ----------------------------------------------------------------------
# Agents (active session only — see the 409 rule)
# ----------------------------------------------------------------------
@mcp.tool()
async def ml_agents_list() -> dict:
    """All agents of the ACTIVE session with their running flags
    (GET /control/agents)."""
    return await _call("GET", "/control/agents")


@mcp.tool()
async def ml_agent_get(name: str) -> dict:
    """One agent's full configuration: system prompt, model, endpoint,
    tools, limits, flags, MCP subscriptions."""
    return await _call("GET", f"/control/agents/{_q(name)}")


@mcp.tool()
async def ml_agent_create(
        name: str,
        system_prompt: str = "",
        description: str = "",
        model: str = "qwen3.5:9b",
        endpoint: str = "local",
        temperature: float = 0.5,
        num_ctx: int = 32000,
        tools: Optional[List[str]] = None,
        max_context_messages: int = 30,
        max_tool_iterations: int = 1000,
        active: bool = True,
        silent: bool = False,
        use_tools: bool = True,
        image: bool = False,
        audio: bool = False,
        mcp: bool = False,
        mcp_servers: Optional[List[str]] = None,
        prompts: Optional[List[str]] = None) -> dict:
    """Create an agent in the ACTIVE session (the desktop Edit-Agent
    dialog equivalent). All fields optional except name.
    tools: backend tool names (ml_tools_list); model: an installed model
    (ml_endpoints_models); endpoint: 'local' or 'ip:port' of another
    Ollama server; mcp/mcp_servers: subscribe to backend-registered MCP
    servers (ml_mcp_servers_list); prompts: snippet names
    (ml_prompts_list)."""
    body: Dict[str, Any] = {
        "name": name,
        "system_prompt": system_prompt,
        "description": description,
        "model": model,
        "endpoint": endpoint,
        "temperature": temperature,
        "num_ctx": num_ctx,
        "max_context_messages": max_context_messages,
        "max_tool_iterations": max_tool_iterations,
        "active": active,
        "silent": silent,
        "use_tools": use_tools,
        "image": image,
        "audio": audio,
        "mcp": mcp,
    }
    if tools is not None:
        body["tools"] = tools
    if mcp_servers is not None:
        body["mcp_servers"] = mcp_servers
    if prompts is not None:
        body["prompts"] = prompts
    return await _call("POST", "/control/agents", body=body)


@mcp.tool()
async def ml_agent_edit(
        name: str,
        system_prompt: Optional[str] = None,
        description: Optional[str] = None,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        tools: Optional[List[str]] = None,
        max_context_messages: Optional[int] = None,
        max_tool_iterations: Optional[int] = None,
        active: Optional[bool] = None,
        silent: Optional[bool] = None,
        use_tools: Optional[bool] = None,
        image: Optional[bool] = None,
        audio: Optional[bool] = None,
        mcp: Optional[bool] = None,
        mcp_servers: Optional[List[str]] = None,
        prompts: Optional[List[str]] = None) -> dict:
    """Edit an agent — ONLY the provided fields change (PATCH
    semantics). Renames are NOT done here (use ml_agent_rename)."""
    body: Dict[str, Any] = {}
    for key, value in (("system_prompt", system_prompt),
                       ("description", description),
                       ("model", model),
                       ("endpoint", endpoint),
                       ("temperature", temperature),
                       ("num_ctx", num_ctx),
                       ("tools", tools),
                       ("max_context_messages", max_context_messages),
                       ("max_tool_iterations", max_tool_iterations),
                       ("active", active),
                       ("silent", silent),
                       ("use_tools", use_tools),
                       ("image", image),
                       ("audio", audio),
                       ("mcp", mcp),
                       ("mcp_servers", mcp_servers),
                       ("prompts", prompts)):
        if value is not None:
            body[key] = value
    if not body:
        return _error("usage", "no fields to update",
                      "pass at least one field to change")
    return await _call("PATCH", f"/control/agents/{_q(name)}", body=body)


@mcp.tool()
async def ml_agent_rename(name: str, new_name: str) -> dict:
    """Rename an agent (history sender names are rewritten too)."""
    return await _call("PUT", f"/control/agents/{_q(name)}",
                       body={"new_name": new_name})


@mcp.tool()
async def ml_agent_delete(name: str) -> dict:
    """Delete an agent from the ACTIVE session."""
    return await _call("DELETE", f"/control/agents/{_q(name)}")


@mcp.tool()
async def ml_agent_set_active(name: str, active: bool) -> dict:
    """Put an agent in/out of the round-robin (active=false = the agent
    is skipped by the engines but stays in the session)."""
    return await _call("POST", f"/control/agents/{_q(name)}/active",
                       body={"active": active})


@mcp.tool()
async def ml_agent_stop(name: str) -> dict:
    """Stop the agent's currently running generation stream (the desktop
    sidebar Stop button)."""
    return await _call("POST", f"/control/agents/{_q(name)}/stop")


@mcp.tool()
async def ml_agent_inject(name: str, text: str) -> dict:
    """Inject operator text into a RUNNING agent: the text is prepended
    to that agent's NEXT tool result (proven tool-injection semantics —
    it never persists into the chat history)."""
    return await _call("POST", f"/control/agents/{_q(name)}/inject",
                       body={"text": text})


@mcp.tool()
async def ml_agent_injections(name: str) -> dict:
    """The agent's pending injection count + texts."""
    return await _call("GET", f"/control/agents/{_q(name)}/injections")


@mcp.tool()
async def ml_agent_save(name: str, path: Optional[str] = None) -> dict:
    """Save one agent to a portable JSON file (config only, no history)."""
    return await _call("POST", "/control/agents/save",
                       body={"name": name, "path": path})


@mcp.tool()
async def ml_agent_load(path: str, new_name: Optional[str] = None) -> dict:
    """Load an agent from a saved agent JSON into the ACTIVE session."""
    return await _call("POST", "/control/agents/load",
                       body={"path": path, "new_name": new_name})


# ----------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------
@mcp.tool()
async def ml_chat_send(text: str, mode: str = "send_now") -> dict:
    """Send a chat message into the ACTIVE session.
    mode='send_now' = the Send button: the message is queued AND every
    engine's user-turn wait is unblocked, so the team answers now.
    mode='queue' = the Send-to-queue button: the text waits in
    pending_user_messages until the engines naturally come around."""
    return await _call("POST", "/control/message",
                       body={"text": text, "mode": mode})


@mcp.tool()
async def ml_chat_send_now() -> dict:
    """The Send-Now button (no text): unblock every engine's user-turn
    wait so the pending queue drains immediately."""
    return await _call("POST", "/control/send-now")


@mcp.tool()
async def ml_chat_history(limit: int = 0, offset: int = 0) -> dict:
    """Chat history of the ACTIVE session (oldest→newest).
    limit=0 (default) = everything after offset.
    Message entries may carry a 'files' array — the real local files
    mentioned in that message, each with {name, kind, size, path, url,
    url_download}: the backend's chat download-links feature. Feed those
    paths/urls to ml_files_url / ml_files_fetch to download or read them."""
    return await _call("GET", "/output/history",
                       params={"limit": limit, "offset": offset})


@mcp.tool()
async def ml_chat_pending() -> dict:
    """The pending user messages of the ACTIVE session (queued but not
    yet drained by the engines)."""
    return await _call("GET", "/control/queue")


@mcp.tool()
async def ml_chat_pending_clear() -> dict:
    """Clear the pending user-message queue of the ACTIVE session."""
    return await _call("DELETE", "/control/queue")


@mcp.tool()
async def ml_chat_user(name: Optional[str] = None) -> dict:
    """Get (no argument) or set the user name shown in the chat."""
    if name is None:
        return await _call("GET", "/control/user")
    return await _call("POST", "/control/user", body={"name": name})


@mcp.tool()
async def ml_chat_wait_user(paused: Optional[bool] = None) -> dict:
    """Get (no argument) or set 'Wait user input' — ON = the round-robin
    waits for your message (or ml_chat_send_now) before each agent turn;
    OFF = agents keep talking freely."""
    if paused is None:
        return await _call("GET", "/control/pause")
    return await _call("POST", "/control/pause", body={"paused": paused})


@mcp.tool()
async def ml_chat_attach(path: str) -> dict:
    """The desktop 'Add file' button: queue a file path verbatim as the
    next user message (agents see it as a plain string — they then open
    it with their own file tools)."""
    return await _call("POST", "/control/attach", body={"path": path})


# ----------------------------------------------------------------------
# Waiting / one-shot turns (composite convenience tools)
# ----------------------------------------------------------------------
async def _wait_idle(timeout_seconds: float,
                     poll_seconds: float) -> Dict[str, Any]:
    """Poll /state until the ACTIVE session is idle: no agent running,
    no pending user messages, and the round-robin is held (wait-user ON,
    session finished, or every agent inactive). The condition must hold
    for two consecutive polls to avoid turn-start races."""
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    poll = max(0.2, float(poll_seconds))
    stable = 0
    summary: Dict[str, Any] = {}
    while True:
        st = await _call("GET", "/state")
        if isinstance(st, dict) and st.get("ok") is False:
            return {"idle": False, "timed_out": False, "state_error": st}
        active = (st or {}).get("active_session")
        if not isinstance(active, dict) or not active:
            return {"idle": True, "timed_out": False,
                    "active_session": None,
                    "reason": "no active session"}
        agents = [a for a in (active.get("agents") or {}).values()
                  if isinstance(a, dict)]
        running = [a.get("name") for a in agents if a.get("running")]
        pending = list(active.get("pending_user_messages") or [])
        finished = active.get("finished_at") is not None
        wait_user = bool(active.get("paused_user_input"))
        any_active = any(a.get("active") for a in agents)
        summary = {
            "session_name": active.get("session_name"),
            "running_agents": running,
            "pending_user_messages": len(pending),
            "wait_user_input": wait_user,
            "finished": finished,
            "active_agents": sum(1 for a in agents if a.get("active")),
        }
        idle_now = ((not running) and (not pending)
                    and (finished or wait_user or not any_active))
        if idle_now:
            stable += 1
            if stable >= 2:
                return {"idle": True, "timed_out": False, **summary}
        else:
            stable = 0
        if time.monotonic() >= deadline:
            return {"idle": False, "timed_out": True, **summary}
        await asyncio.sleep(poll)


@mcp.tool()
async def ml_wait_idle(timeout_seconds: float = 120,
                       poll_seconds: float = 2) -> dict:
    """Block until the ACTIVE session goes idle (no agent running, no
    pending user messages, round-robin held — e.g. back to waiting for
    user input). Returns the state summary; on timeout
    {idle: false, timed_out: true} with what is still running. Useful
    after ml_chat_send(mode='queue') or a queued session run."""
    return await _wait_idle(timeout_seconds, poll_seconds)


@mcp.tool()
async def ml_ask(text: str,
                 timeout_seconds: float = 180,
                 poll_seconds: float = 2,
                 mode: str = "send_now") -> dict:
    """One-shot conversation turn — the convenient default way to talk
    to the team: sends text (mode like ml_chat_send), waits until the
    session goes idle, then returns ONLY the messages appended by that
    turn, including their 'files' download arrays. Requires an active
    session (ml_session_new first — 409 otherwise). If the team keeps
    chatting (wait_user_input off) you get timed_out=true plus the
    messages so far — fetch the rest with ml_chat_history."""
    if not (text or "").strip():
        return _error("usage", "text must not be empty")
    before = await _call("GET", "/output/history", params={"limit": 1})
    if isinstance(before, dict) and before.get("ok") is False:
        return before
    old_total = int((before or {}).get("total") or 0)
    sent = await _call("POST", "/control/message",
                       body={"text": text, "mode": mode})
    if isinstance(sent, dict) and sent.get("ok") is False:
        return sent
    wait = await _wait_idle(timeout_seconds, poll_seconds)
    after = await _call("GET", "/output/history",
                        params={"offset": old_total, "limit": 0})
    if isinstance(after, dict) and after.get("ok") is False:
        return after
    msgs = (after or {}).get("messages") or []
    return {
        "ok": True,
        "session_name": (after or {}).get("session_name"),
        "appended": len(msgs),
        "idle": wait.get("idle"),
        "timed_out": wait.get("timed_out"),
        "wait": wait,
        "messages": msgs,
    }


# ----------------------------------------------------------------------
# Output events
# ----------------------------------------------------------------------
@mcp.tool()
async def ml_output_recent(limit: int = 200) -> dict:
    """The last N live events from the backend's in-memory ring buffer
    (chat, thinking/tool streams, status, session lifecycle) — the
    reconnect gap-filler. 1..5000. Thinking streams are separate events
    and never scanned for file links (only final chat messages are)."""
    return await _call("GET", "/output/recent",
                       params={"limit": max(1, min(int(limit), 5000))})


# ----------------------------------------------------------------------
# Files — the chat download-links feature, from the MCP side
# ----------------------------------------------------------------------
@mcp.tool()
async def ml_files_url(path: str, download: bool = True,
                       include_token: bool = True) -> dict:
    """Build absolute URL(s) for one local file the backend can serve
    (GET /files): inline view + forced-download variants. path may be
    absolute, ./-relative, ~/… or a bare name (resolved
    working-directory-first, then the backend folder).
    include_token=true (default) appends ?token= when an API key is
    configured — <a href> links cannot send headers, so this keeps the
    URL clickable in browsers; the URL then EMBODIES the secret, treat
    it accordingly. Verify existence first via a message's 'files'
    array (ml_chat_history) — this tool never checks the file."""
    if not (path or "").strip():
        return _error("usage", "path must not be empty")
    quoted = _quote(path, safe="")
    url = f"{CONFIG.backend_url}/files?path={quoted}"
    url_download = url + "&download=1"
    token_included = False
    if CONFIG.api_key and include_token:
        tok = f"&token={_quote(CONFIG.api_key, safe='')}"
        url += tok
        url_download += tok
        token_included = True
    return {"ok": True, "path": path, "url": url,
            "url_download": url_download, "token_included": token_included}


@mcp.tool()
async def ml_files_fetch(path: str, max_bytes: int = 2097152) -> dict:
    """Fetch one local file THROUGH the backend (GET /files) and return
    it base64-encoded with MIME type and size — how an MCP client
    'reads' agent-generated images/audio/video/documents without
    filesystem access to the backend machine.
    Default cap 2 MiB (raise max_bytes for bigger files; hard cap
    64 MiB). 404 when the reference resolves to no existing file — take
    the exact 'path' from a message's 'files' array (ml_chat_history)
    when that happens."""
    if not (path or "").strip():
        return _error("usage", "path must not be empty")
    cap = max(1, min(int(max_bytes), 64 * 1024 * 1024))
    client = _get_client()
    try:
        resp = await client.get("/files", params={"path": path},
                                timeout=max(CONFIG.timeout, 60.0))
    except httpx.ConnectError as exc:
        return _error("unreachable", str(exc),
                      "no backend answering at "
                      f"{CONFIG.backend_url} — start it first")
    except httpx.TimeoutException as exc:
        return _error("timeout", str(exc), "file download timed out")
    except httpx.HTTPError as exc:
        return _error("transport", str(exc), "HTTP failure fetching the file")
    if resp.status_code == 404:
        return _error("http", _http_detail(resp),
                      "the reference resolves to no existing file — take "
                      "the exact path from a message's 'files' array "
                      "(ml_chat_history)", status=404)
    if resp.status_code >= 400:
        return _error("http", _http_detail(resp),
                      _ERROR_HINTS.get(resp.status_code, ""),
                      status=resp.status_code)
    data = resp.content
    if len(data) > cap:
        return _error("too_large",
                      f"{len(data)} bytes exceeds max_bytes={cap}",
                      f"re-call with max_bytes >= {len(data)} "
                      "(hard cap 64 MiB)", size=len(data))
    return {"ok": True, "path": path, "size": len(data),
            "mime": resp.headers.get("content-type",
                                     "application/octet-stream"),
            "base64": base64.b64encode(data).decode("ascii")}


# ----------------------------------------------------------------------
# The session run queue (unattended / batch operation)
# ----------------------------------------------------------------------
@mcp.tool()
async def ml_queue_enqueue(session_name: str,
                          message: Optional[str] = None,
                          sender: str = "user",
                          activate: Optional[List[str]] = None,
                          activate_all: bool = False,
                          idle_timeout: Optional[float] = None,
                          auto_finish: bool = True) -> dict:
    """Queue an UNATTENDED session run. When its turn comes the backend
    loads that session's newest autosave (or a sessions/ file), applies
    activate/activate_all, queues `message` as a user message BEFORE the
    engines start, runs the round-robin until the session quiesces, then
    autosaves and moves on. Only session_name is required.
    May preempt an interactive session (autosaved first, never lost)."""
    body: Dict[str, Any] = {"session_name": session_name, "sender": sender,
                            "activate_all": activate_all,
                            "auto_finish": auto_finish}
    if message is not None:
        body["message"] = message
    if activate is not None:
        body["activate"] = activate
    if idle_timeout is not None:
        body["idle_timeout"] = idle_timeout
    return await _call("POST", "/queue/sessions", body=body)


@mcp.tool()
async def ml_queue_list() -> dict:
    """The run queue: pending entries, the running entry, recent
    finished entries (with results + autosave paths) and staged
    injections."""
    return await _call("GET", "/queue/sessions")


@mcp.tool()
async def ml_queue_move(entry_id: str, position: int) -> dict:
    """Reorder the run queue — position=0 = runs next."""
    return await _call("PATCH", f"/queue/sessions/{_q(entry_id)}/move",
                       body={"position": position})


@mcp.tool()
async def ml_queue_cancel(entry_id: Optional[str] = None) -> dict:
    """Cancel ONE queue entry by id — or ALL pending entries when
    entry_id is omitted."""
    if entry_id:
        return await _call("DELETE", f"/queue/sessions/{_q(entry_id)}")
    return await _call("DELETE", "/queue/sessions")


@mcp.tool()
async def ml_queue_inject(session_name: str, agent_name: str,
                          text: str) -> dict:
    """Inject into an agent's NEXT TOOL RESULT — unlike
    ml_queue_enqueue this NEVER enqueues the session for a run.
    Running session → delivered_live=true immediately; otherwise staged
    (delivered automatically the next time that session runs)."""
    return await _call("POST", "/queue/inject",
                       body={"session_name": session_name,
                             "agent_name": agent_name, "text": text})


@mcp.tool()
async def ml_queue_inject_list() -> dict:
    """Staged + live injection overview (ids for
    ml_queue_inject_cancel)."""
    return await _call("GET", "/queue/inject")


@mcp.tool()
async def ml_queue_inject_cancel(injection_id: str) -> dict:
    """Remove one STAGED injection by id (see ml_queue_inject_list)."""
    return await _call("DELETE", f"/queue/inject/{_q(injection_id)}")


# ----------------------------------------------------------------------
# Backend agent tools / prompts / names pool
# ----------------------------------------------------------------------
@mcp.tool()
async def ml_tools_list() -> dict:
    """Every backend agent TOOL (per file, with docstrings) — the toolbox
    you can hand to agents via ml_agent_create's tools field. The backend
    ships core file/exec/agent tools; drop more .py files into its tools/
    folder and ml_tools_reload picks them up."""
    return await _call("GET", "/control/tools")


@mcp.tool()
async def ml_tools_reload() -> dict:
    """Re-scan the backend's tools/ folder (pick up hand-edited or newly
    dropped tool files without restarting the backend)."""
    return await _call("POST", "/control/tools/reload")


@mcp.tool()
async def ml_prompts_list() -> dict:
    """Prompt snippets in the backend's prompts/ folder (name +
    description) — reusable prompt fragments for ml_agent_create's
    prompts field."""
    return await _call("GET", "/control/prompts")


@mcp.tool()
async def ml_prompts_reload() -> dict:
    """Re-scan the backend's prompts/ folder."""
    return await _call("POST", "/control/prompts/reload")


@mcp.tool()
async def ml_prompt_get(name: str) -> dict:
    """The full text of one prompt snippet (ml_prompts_list for names)."""
    return await _call("GET", f"/control/prompts/{_q(name)}")


@mcp.tool()
async def ml_names_list() -> dict:
    """The names.txt agent-name pool (one name per line; working dir
    first, then the backend folder) — the desktop app's agent
    auto-naming source. Empty when no file exists."""
    return await _call("GET", "/control/names")


# ----------------------------------------------------------------------
# The backend's OWN MCP engine (it is an MCP *client* — registry,
# connections, per-agent subscriptions)
# ----------------------------------------------------------------------
@mcp.tool()
async def ml_mcp_servers_list() -> dict:
    """The backend's own MCP-server REGISTRY (its MCP client engine):
    every server the backend's agents can subscribe to, with transport
    definitions. Requires the optional `mcp` package on the BACKEND."""
    return await _call("GET", "/control/mcp/servers")


@mcp.tool()
async def ml_mcp_server_add(
        name: str,
        transport: str = "stdio",
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        url: Optional[str] = None,
        env: Optional[dict] = None,
        description: Optional[str] = None) -> dict:
    """Register / update an MCP server in the BACKEND's registry (its
    agents subscribe to it via ml_mcp_subscribe). transport='stdio' →
    command + args; 'sse'/'http' → url.
    Dogfood tip: THIS very server can be registered — command='python',
    args=['<absolute path to ml_mcp_server.py>'] — so the backend's own
    agents get the same toolset your MCP clients have."""
    body: Dict[str, Any] = {"name": name, "transport": transport}
    for key, value in (("command", command), ("args", args), ("url", url),
                       ("env", env), ("description", description)):
        if value is not None:
            body[key] = value
    return await _call("POST", "/control/mcp/servers", body=body)


@mcp.tool()
async def ml_mcp_server_connect(name: str, wait: float = 60.0) -> dict:
    """Connect the backend to a registered MCP server (launch / dial +
    handshake + tool listing). wait = seconds to wait for the
    handshake."""
    return await _call("POST", f"/control/mcp/servers/{_q(name)}/connect",
                       body={"wait": wait},
                       timeout=max(CONFIG.timeout, float(wait) + 15.0))


@mcp.tool()
async def ml_mcp_server_disconnect(name: str) -> dict:
    """Disconnect the backend from a registered MCP server (stays in the
    registry; agents' subscriptions survive)."""
    return await _call("POST", f"/control/mcp/servers/{_q(name)}/disconnect")


@mcp.tool()
async def ml_mcp_server_tools(name: str) -> dict:
    """The tools a CONNECTED MCP server exposes to the backend's agents
    (they surface as mcp__<server>__<tool>)."""
    return await _call("GET", f"/control/mcp/servers/{_q(name)}/tools")


@mcp.tool()
async def ml_mcp_server_delete(name: str) -> dict:
    """Remove a server from the backend's MCP registry (disconnects it
    first; agents' subscriptions to it are dropped)."""
    return await _call("DELETE", f"/control/mcp/servers/{_q(name)}")


@mcp.tool()
async def ml_mcp_status() -> dict:
    """Live connection status per registered MCP server (connected?
    tool count? last error?)."""
    return await _call("GET", "/control/mcp/status")


@mcp.tool()
async def ml_mcp_subscribe(agent_name: str, server: str) -> dict:
    """Subscribe an agent to an MCP server: ALL of that server's tools
    become first-class tools in the agent's Ollama tool list (named
    mcp__<server>__<tool>) — the backend's native-MCP behaviour."""
    return await _call("POST", "/control/mcp/subscribe",
                       body={"agent_name": agent_name, "server": server})


@mcp.tool()
async def ml_mcp_unsubscribe(agent_name: str, server: str) -> dict:
    """Remove an agent's subscription to an MCP server."""
    return await _call("POST", "/control/mcp/unsubscribe",
                       body={"agent_name": agent_name, "server": server})


# ----------------------------------------------------------------------
# Ollama endpoints / models / usage
# ----------------------------------------------------------------------
@mcp.tool()
async def ml_endpoints_list() -> dict:
    """Every Ollama endpoint the backend knows (local + ip:port ones)
    with status, the agents sitting on each and model memory usage."""
    return await _call("GET", "/control/endpoints")


@mcp.tool()
async def ml_endpoints_models(endpoint: str = "local") -> dict:
    """The models installed on one endpoint (Ollama /api/tags) — feeds
    ml_agent_create's / ml_agent_edit's model field. Can block a few
    seconds; a 502 means the endpoint is down."""
    return await _call("GET", "/control/endpoints/models",
                       params={"endpoint": endpoint}, timeout=120.0)


@mcp.tool()
async def ml_endpoints_scan(port: int = 11434) -> dict:
    """Scan the LAN for Ollama servers (psutil on the BACKEND; the found
    servers become usable endpoints)."""
    return await _call("POST", "/control/endpoints/scan",
                       body={"port": port}, timeout=60.0)


@mcp.tool()
async def ml_endpoints_usage() -> dict:
    """Model memory usage per endpoint (Ollama /api/ps) — what is
    actually loaded right now."""
    return await _call("GET", "/control/endpoints/usage")


# ======================================================================
# CLI entry point
# ======================================================================
_EPILOG = """examples:
  python ml_mcp_server.py --check --backend-url http://127.0.0.1:8200
  python ml_mcp_server.py --list-tools
  python ml_mcp_server.py                                     # stdio (for MCP clients)
  python ml_mcp_server.py --transport streamable-http --host 0.0.0.0 --port 8090
  ML_BACKEND_URL=http://10.0.0.4:8200 python ml_mcp_server.py

environment: ML_BACKEND_URL, ML_MCP_API_KEY (fallback ML_API_KEY),
ML_MCP_TIMEOUT, ML_MCP_TRANSPORT, ML_MCP_HOST, ML_MCP_PORT — CLI flags win."""


async def _run_check() -> int:
    """--check: verify the backend is reachable + summarise it."""
    client = _get_client()
    print(f"backend : {CONFIG.backend_url}")
    print(f"api key: {'configured' if CONFIG.api_key else 'none'}")
    try:
        resp = await client.get("/health", timeout=10.0)
    except httpx.HTTPError as exc:
        print(f"FAILED  — backend unreachable: {exc}")
        print("start it first:  cd MultiOllama_backend && "
              "python run_server.py --port 8200")
        return 1
    if resp.status_code != 200:
        print(f"FAILED  — HTTP {resp.status_code}: {_http_detail(resp)}")
        return 1
    print("health  :", json.dumps(resp.json(), indent=2))
    state = await _call("GET", "/state")
    if isinstance(state, dict) and state.get("ok") is False:
        print("state   :", json.dumps(state, indent=2))
        return 1
    brief = _brief_state(state)
    active = brief.get("active_session")
    summary = {
        "server_version": (brief.get("server") or {}).get("version"),
        "uptime": (brief.get("server") or {}).get("uptime"),
        "active_session": (active or {}).get("session_name"),
        "agents": sorted((active or {}).get("agents") or {}),
        "queue_entries": len(brief.get("session_queue") or []),
        "tools_available": (brief.get("tools") or {}).get("tool_count"),
        "mcp_engine_available": (brief.get("mcp") or {}).get("available"),
    }
    print("state   :", json.dumps(summary, indent=2))
    print("OK — the backend is reachable; this MCP server can talk to it.")
    return 0


async def _list_tools() -> int:
    """--list-tools: print the whole tool surface."""
    res = mcp.list_tools()
    tools = await res if inspect.isawaitable(res) else res
    print(f"{len(tools)} tools exposed by {SERVER_NAME} v{VERSION} "
          f"(backend: {CONFIG.backend_url}):")
    for t in sorted(tools, key=lambda t: t.name):
        first = ((t.description or "").strip().splitlines() or [""])[0]
        print(f"  {t.name:30s} {first}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml_mcp_server",
        description="Standalone MCP server exposing the MultiOllama "
                    "headless backend as MCP tools.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend-url", default=None, metavar="URL",
        help=f"backend base URL (default {DEFAULT_BACKEND_URL} or "
             f"ML_BACKEND_URL)")
    parser.add_argument(
        "--api-key", default=None, metavar="KEY",
        help="backend API key (default ML_MCP_API_KEY / ML_API_KEY env)")
    parser.add_argument(
        "--timeout", type=float, default=None, metavar="SEC",
        help="seconds per backend call (default 30 or ML_MCP_TIMEOUT)")
    parser.add_argument(
        "--transport", default=None,
        choices=("stdio", "sse", "streamable-http"),
        help="MCP transport (default stdio or ML_MCP_TRANSPORT)")
    parser.add_argument(
        "--host", default=None, metavar="HOST",
        help="bind host for the HTTP transports (default 127.0.0.1)")
    parser.add_argument(
        "--port", type=int, default=None, metavar="PORT",
        help="bind port for the HTTP transports (default 8090)")
    parser.add_argument(
        "--check", action="store_true",
        help="verify the backend is reachable, print a summary, exit")
    parser.add_argument(
        "--list-tools", action="store_true",
        help="print every exposed tool, exit")
    parser.add_argument(
        "--version", action="store_true", help="print the version, exit")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.version:
        import mcp as _mcp_pkg
        print(f"ml_mcp_server {VERSION} "
              f"(mcp SDK {getattr(_mcp_pkg, '__version__', '?')})")
        return 0

    if args.backend_url:
        CONFIG.backend_url = args.backend_url.rstrip("/")
    if args.api_key:
        CONFIG.api_key = args.api_key
    if args.timeout:
        CONFIG.timeout = args.timeout

    if args.list_tools:
        return asyncio.run(_list_tools())
    if args.check:
        return asyncio.run(_run_check())

    transport = args.transport or os.environ.get("ML_MCP_TRANSPORT", "stdio")
    host = args.host or os.environ.get("ML_MCP_HOST", "127.0.0.1")
    try:
        port = args.port or int(os.environ.get("ML_MCP_PORT", "8090"))
    except ValueError:
        port = 8090

    if transport != "stdio":
        # the HTTP transports need the bind address; stdio ignores it
        try:
            mcp.settings.host = host
            mcp.settings.port = port
        except Exception:
            pass

    # The MCP SDK auto-arms DNS-rebinding protection with a localhost-only
    # Host allowlist (FastMCP was built with host=127.0.0.1). Closed LAN:
    # accept any Host/Origin.
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False
    )

    mcp.run(transport=transport)
    return 0

    # IMPORTANT: from here on NOTHING may write to stdout — the stdio
    # transport speaks JSON-RPC there. (Logging goes to stderr.)
    mcp.run(transport=transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
