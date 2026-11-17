"""
mcp_manager.py — NATIVE MCP (Model Context Protocol) engine for the agents.

WHY THIS MODULE EXISTS
======================
Previously, agents could only reach MCP servers through hand-written facade
tools (like tools/godot_client.py): the model saw 3-4 meta-tools such as
``godot_call_mcp(tool_name, arguments)`` and had to GUESS the real MCP tool
names / argument shapes behind them. That produced long retry loops while
the model hunted for the right syntax — burning time and tokens.

This module implements NATIVE MCP usage instead:

  * Every agent can SUBSCRIBE to any number of MCP servers (per-agent
    subscription list saved on the Agent object → saved with the agent,
    with the session, and in saved-agent files).
  * All tools of every subscribed server are exposed to the model DIRECTLY
    as first-class tool definitions in the Ollama ``tools`` parameter —
    exactly like built-in tools. No wrapper syntax to learn.
  * Tool calls to those names are routed straight to the MCP server and
    the result is fed back through the normal tool-result pipeline
    (thinking tab, tool-call counter, operator injections, history).

Tool naming
-----------
MCP tools are namespaced to avoid collisions between servers (and with
built-in tools):

    mcp__<server>__<tool>

e.g. the Godot server's ``godot_get_version`` becomes
``mcp__godot__godot_get_version``. The mapping is maintained by this
manager, so the model never needs to know anything about MCP itself.

Server registry (mcp_servers.json)
----------------------------------
Server DEFINITIONS live in a JSON registry file next to the app
(auto-created). Example::

    {
      "mcp_registry_version": 1,
      "servers": {
        "godot": {
          "transport": "stdio",
          "command": "/home/alex/.local/bin/godot-mcp",
          "args": [],
          "env": {"GODOT_PATH": "/home/alex/godot/Godot_v4.7.2-stable_linux.x86_64"},
          "description": "Godot engine control"
        },
        "filesystem": {
          "transport": "stdio",
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
          "env": {},
          "description": "Filesystem access"
        },
        "remote-thing": {
          "transport": "sse",
          "url": "http://192.168.3.5:3000/sse",
          "headers": {},
          "description": "Remote SSE server"
        }
      }
    }

The registry can be edited by hand, via the GUI ("MCP Servers..." dialog),
via the ``mcp_subscribe`` core tool (agents register + subscribe from info
pasted in chat), and definitions travel inside saved-session /
saved-agent files so they survive moves between machines.

Transports: stdio (local commands), sse (legacy remote), http
(streamable-http remote). Persistent connections: each server process is
started ONCE and kept alive (one shared connection no matter how many
agents subscribe); it auto-reconnects with backoff on crash and is shut
down cleanly on app exit.

Threading model
---------------
The ``mcp`` python package is asyncio-based while the whole app is Qt +
blocking worker threads. The manager therefore owns ONE daemon thread that
runs a dedicated asyncio event loop for the entire MCP engine. All public
methods are SYNCHRONOUS and thread-safe: they submit coroutines to that
loop with ``asyncio.run_coroutine_threadsafe(...)`` and wait on the
future. This lets BackendThreads, GUI code and tool code call the manager
without knowing anything about asyncio.

The ``mcp`` package is imported lazily, so the app still boots (and the
registry still works) on machines without it — connecting simply reports
"mcp package not installed".
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import app_paths

# ----------------------------------------------------------------------
# Optional imports (never required at import time)
# ----------------------------------------------------------------------
# `mcp` is imported lazily inside the connection coroutines so this module
# (and the whole app) stays importable on machines without the package.
try:
    import context as _app_context
except Exception:  # pragma: no cover - defensive, context.py is stdlib-only
    _app_context = None

try:
    from anyio import ClosedResourceError as _ClosedResourceError
except Exception:  # pragma: no cover
    class _ClosedResourceError(Exception):
        pass

try:
    from anyio.streams.memory import BrokenResourceError as _BrokenResourceError
except Exception:  # pragma: no cover
    class _BrokenResourceError(Exception):
        pass

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# Registry file (server DEFINITIONS) — lives next to the app.
REGISTRY_FILE_NAME = "mcp_servers.json"
REGISTRY_VERSION = 1

# Per-server stderr logs directory (next to the app).
LOG_DIR_NAME = "mcp_logs"

# Tool-name prefix + separator for namespaced MCP tools.
MCP_TOOL_PREFIX = "mcp__"
MCP_TOOL_SEPARATOR = "__"

# Management core tools (defined in core_tools.py → tools/mcp_tools.py).
# Auto-available to any agent whose "MCP" capability checkbox is ON.
MANAGEMENT_TOOL_NAMES: Tuple[str, ...] = (
    "mcp_list_servers",
    "mcp_subscribe",
    "mcp_unsubscribe",
    "mcp_list_tools",
    "mcp_call_tool",
    "mcp_status",
)

# Timeouts (seconds).
INIT_TIMEOUT = 60.0          # session.initialize()
LIST_TOOLS_TIMEOUT = 60.0    # tools/list round-trip
CONNECT_WAIT_TIMEOUT = 60.0  # how long subscribe/connect waits for ready
CALL_TOOL_TIMEOUT = 300.0    # default per tool call (mcp_call_tool may override)
RECONNECT_WAIT = 30.0        # extra wait for auto-reconnect during a call
TOOL_CACHE_TTL = 60.0        # seconds before the cached tool list is refreshed

# Reconnect backoff: first retry after 1s, doubling up to 30s; after this
# many CONSECUTIVE failed connect attempts the connection is parked in
# "failed" state until an explicit reconnect / re-subscribe happens.
RECONNECT_BACKOFF_START = 1.0
RECONNECT_BACKOFF_MAX = 30.0
MAX_CONSECUTIVE_FAILURES = 5

VALID_TRANSPORTS = ("stdio", "sse", "http")

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_SERVER_SLUG = 24
_MAX_TOOL_NAME_LEN = 64


def _slug(name: str, max_len: int = _MAX_SERVER_SLUG) -> str:
    """Sanitize a server name for use inside tool names / file names."""
    s = _SAFE_NAME_RE.sub("_", str(name or "").strip())
    s = s.strip("_") or "server"
    return s[:max_len]


def _safe_tool_name(name: str) -> str:
    """Sanitize a raw MCP tool name for the Ollama API tool-name rules."""
    s = _SAFE_NAME_RE.sub("_", str(name or "").strip())
    return s[:_MAX_TOOL_NAME_LEN]


def make_mcp_tool_name(server_name: str, tool_name: str) -> str:
    """Build the namespaced tool name exposed to the model."""
    return f"{MCP_TOOL_PREFIX}{_slug(server_name)}{MCP_TOOL_SEPARATOR}{_safe_tool_name(tool_name)}"


def parse_mcp_tool_name(tool_name: str) -> Optional[Tuple[str, str]]:
    """Split 'mcp__<server>__<tool>' back into (server_slug, tool_name).

    Returns None when the name is not a namespaced MCP tool. NOTE: the
    first component is the server SLUG (sanitized), which the manager
    resolves back to the real server name via its slug map.
    """
    if not tool_name or not tool_name.startswith(MCP_TOOL_PREFIX):
        return None
    rest = tool_name[len(MCP_TOOL_PREFIX):]
    if MCP_TOOL_SEPARATOR not in rest:
        return None
    server_slug, tool = rest.split(MCP_TOOL_SEPARATOR, 1)
    if not server_slug or not tool:
        return None
    return server_slug, tool


# ----------------------------------------------------------------------
# Child-watcher hardening for asyncio subprocess support
# ----------------------------------------------------------------------
# asyncio.create_subprocess_exec() -> _UnixSelectorEventLoop
# ._make_subprocess_transport() calls events.get_child_watcher(), which
# goes through the GLOBAL event-loop policy:
#
#     def get_child_watcher():
#         return get_event_loop_policy().get_child_watcher()
#
# uvicorn (versions that still use setup_event_loop, i.e. < 0.36) and
# any code that calls uvloop.install() REPLACE the default policy with
# uvloop.EventLoopPolicy.  uvloop.EventLoopPolicy inherits
# BaseDefaultEventLoopPolicy.get_child_watcher(), the BASE stub that
# unconditionally raises NotImplementedError.  After that, EVERY
# create_subprocess_exec on a non-uvloop loop dies — which is exactly
# what kills every stdio MCP server the moment stdio_client() runs.
#
# We cannot undo the policy globally: uvicorn's main loop is already
# running on a uvloop.Loop and replacing the policy would break loop
# creation elsewhere.  The policy isn't actually the thing we need —
# only events.get_child_watcher() is.  So we patch THAT function on the
# asyncio.events module (and the top-level asyncio alias) to return a
# process-wide PidfdChildWatcher.  _make_subprocess_transport looks the
# function up at call time, so the patch takes effect for every
# subprocess spawn on any loop, and survives any later policy changes.
#
# PidfdChildWatcher (the Python 3.12 default on Linux) uses pidfd_open
# and does NOT need attach_loop(); it is safe to retrieve from any
# thread or loop.  We fall back to ThreadedChildWatcher on older Python.
_CHILD_WATCHER_PATCH_LOCK = threading.Lock()
_CHILD_WATCHER_PATCHED = False
# One watcher PER RUNNING LOOP. PidfdChildWatcher.is_active() only returns
# True after attach_loop() has been called on it; the old code shared ONE
# never-attached instance across every loop, which made every stdio spawn
# raise: RuntimeError("Cannot create a subprocess because the child
# watcher is not active").
_CHILD_WATCHERS: Dict[Any, Any] = {}
_CHILD_WATCHERS_LOCK = threading.Lock()


def _make_child_watcher():
    """Create a fresh child watcher, or None if none is available."""
    if hasattr(os, "pidfd_open") and hasattr(asyncio, "PidfdChildWatcher"):
        try:
            return asyncio.PidfdChildWatcher()
        except (AttributeError, NotImplementedError):
            pass
    try:
        return asyncio.ThreadedChildWatcher()
    except AttributeError:
        return None


def _install_child_watcher_patch() -> None:
    """Ensure asyncio.events.get_child_watcher() returns a working watcher
    even when the global policy is one that doesn't implement it (uvloop's
    policy, the base BaseDefaultEventLoopPolicy, etc.).

    Idempotent and safe to call from any thread.  Calling it right before
    every stdio spawn is belt-and-suspenders: even if something later
    restores the broken policy, the module-level function we patch here
    stays patched, so subprocess spawning keeps working.
    """
    global _CHILD_WATCHER_PATCHED
    if _CHILD_WATCHER_PATCHED:
        return
    with _CHILD_WATCHER_PATCH_LOCK:
        if _CHILD_WATCHER_PATCHED:
            return

        import asyncio.events as _events

        def _get_child_watcher():
            # Must return a watcher whose is_active() is True for the
            # CURRENT running loop, otherwise _UnixSelectorEventLoop
            # ._make_subprocess_transport() raises:
            #   RuntimeError("Cannot create a subprocess because the
            #                 child watcher is not active")
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # Called from sync context (not expected from the MCP
                # engine, but be safe): return a fresh, unattached
                # watcher rather than a broken shared one.
                return _make_child_watcher()

            with _CHILD_WATCHERS_LOCK:
                watcher = _CHILD_WATCHERS.get(loop)
                if watcher is None:
                    watcher = _make_child_watcher()
                    if watcher is not None:
                        try:
                            watcher.attach_loop(loop)
                        except Exception:
                            pass
                    _CHILD_WATCHERS[loop] = watcher
                return watcher

        _events.get_child_watcher = _get_child_watcher
        asyncio.get_child_watcher = _get_child_watcher
        _CHILD_WATCHER_PATCHED = True
        print("[mcp] installed child-watcher patch (per-loop, "
              "attach_loop()-ed) — current policy "
              f"{type(asyncio.get_event_loop_policy()).__name__} left in place.")

# ======================================================================
# _ServerConnection — one live MCP server (state + session handle)
# ======================================================================
class _ServerConnection:
    """Bookkeeping for ONE server connection.

    The session itself only exists while the connection task runs on the
    engine's asyncio loop; everything here is plain, lock-free state that
    the manager guards with its own lock.
    """

    def __init__(self, name: str, config: dict, app_dir: Optional[str] = None):
        self.name = name
        self.config = dict(config)
        self.state = "disconnected"     # connecting|connected|disconnected|failed
        self.session = None             # mcp ClientSession (loop-owned)
        self.stop_event: Optional[asyncio.Event] = None
        self.want_running = False
        self.ready = threading.Event()  # set once initialize() + tools/list done
        self.tools_cache: List[dict] = []   # [{name, description, schema}]
        self.tools_cached_at = 0.0
        self.server_info: dict = {}
        self.last_error = ""
        self.consecutive_failures = 0
        self.task_future = None         # concurrent.futures.Future of the runner
        base_dir = app_dir or app_paths.get_app_dir()
        self.log_path = os.path.join(base_dir, LOG_DIR_NAME, f"{_slug(name, 60)}.log")

    # -- state helpers (called from manager threads; loop-safe fields) --
    @property
    def connected(self) -> bool:
        return self.state == "connected" and self.session is not None

    def describe(self) -> str:
        transport = self.config.get("transport", "stdio")
        if transport == "stdio":
            endpoint = self.config.get("command", "")
            args = self.config.get("args") or []
            if args:
                endpoint += " " + " ".join(str(a) for a in args[:4])
        else:
            endpoint = self.config.get("url", "")
        return f"{transport}: {endpoint}"


# ======================================================================
# MCPManager — the engine singleton
# ======================================================================
class MCPManager:
    """Owns the registry, all live server connections and the asyncio loop.

    Public API (all SYNCHRONOUS, safe from any thread):

      Registry:
        list_registry_servers()               -> {name: config_dict}
        register_server(name, **cfg)          -> config (creates/updates + saves)
        remove_server(name)                   -> bool
        registry_snapshot() / merge_registry_snapshot(snap)

      Subscriptions (persisted on the Agent object itself):
        subscribe_agent(agent_name, server)   -> str  (result message)
        unsubscribe_agent(agent_name, server) -> str
        get_agent_subscriptions(agent_name)   -> [server names]

      Connections:
        connect_server(name, wait=..)         -> (ok, message)
        disconnect_server(name)               -> str
        ensure_subscribed_connected(agent, t) -> None

      Native tool surface (used by Agent.call):
        native_tool_defs_for_agent(agent_name)   -> [OpenAI-style defs]
        extend_allowed_tools(agent_name, base)   -> [names]
        is_tool_allowed(agent_name, tool_name)   -> bool
        handles_tool(agent_name, tool_name)      -> bool
        call_tool(agent_name, tool_name, args, timeout) -> str

      Info / maintenance:
        list_tools_info(server_name)          -> [lines]
        status_info()                         -> [lines]
        refresh_tools(server_name)            -> (ok, message)
        shutdown_all()
    """

    def __init__(self, app_dir: Optional[str] = None):
        self._app_dir = app_dir or app_paths.get_app_dir()
        self._registry_path = os.path.join(self._app_dir, REGISTRY_FILE_NAME)
        self._lock = threading.RLock()
        self._registry: Dict[str, dict] = {}
        # (mtime_ns, size) of mcp_servers.json at the time of the last
        # registry read — the auto-reload baseline used by
        # maybe_reload_registry().
        self._registry_sig: Optional[tuple] = None
        self._connections: Dict[str, _ServerConnection] = {}
        self._server_slug_to_name: Dict[str, str] = {}

        # ---- the single asyncio engine loop, on its own daemon thread ----
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_started = threading.Event()
        self._shutting_down = False
        self._start_loop_thread()

        self._load_registry()

    # ------------------------------------------------------------------
    # Module-level singleton plumbing (mcp_manager.get_manager())
    # ------------------------------------------------------------------
    # main.py creates the manager at startup and publishes it BOTH here
    # (module singleton, used by agent.py / tools) and as context.mcp_manager
    # (for user tools that already reach singletons through context).

    # ------------------------------------------------------------------
    # Engine loop thread
    # ------------------------------------------------------------------
    def _start_loop_thread(self):
        def _run():
            # Make sure asyncio can spawn subprocesses from this loop even
            # if the web framework later replaces the global event-loop
            # policy with one that doesn't implement get_child_watcher()
            # (uvloop.EventLoopPolicy, the base BaseDefaultEventLoopPolicy,
            # ...). The patch is on asyncio.events.get_child_watcher itself,
            # so it survives later policy changes — but we also re-apply it
            # in _run_connection before every stdio spawn as a safety net.
            _install_child_watcher_patch()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._loop_started.set()
            try:
                loop.run_forever()
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass

        try:
            self._loop_thread = threading.Thread(
                target=_run, name="mcp-engine", daemon=True
            )
            self._loop_thread.start()
            self._loop_started.wait(timeout=10.0)
        except Exception as e:
            print(f"[mcp] failed to start engine loop: {e}")
            self._loop = None

    def _submit(self, coro, timeout: Optional[float] = None):
        """Run a coroutine on the engine loop; wait for the result."""
        if self._loop is None or self._shutting_down:
            raise RuntimeError("MCP engine loop is not running.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout)

    # ------------------------------------------------------------------
    # Registry (mcp_servers.json)
    # ------------------------------------------------------------------
    def _registry_signature(self) -> Optional[tuple]:
        """(mtime_ns, size) of the registry file, or None when absent."""
        try:
            st = os.stat(self._registry_path)
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def _load_registry(self):
        with self._lock:
            self._registry = {}
            if os.path.isfile(self._registry_path):
                try:
                    with open(self._registry_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    servers = data.get("servers") if isinstance(data, dict) else None
                    if isinstance(servers, dict):
                        for name, cfg in servers.items():
                            if isinstance(cfg, dict):
                                self._registry[str(name)] = self._normalize_config(cfg)
                except Exception as e:
                    print(f"[mcp] failed to load registry {self._registry_path}: {e}")
            self._rebuild_slug_map()
            # Fingerprint AFTER the read: a write that lands between the
            # read and this stat simply triggers one harmless re-read on
            # the next maybe_reload_registry().
            self._registry_sig = self._registry_signature()

    def reload_registry(self) -> None:
        """Force a re-read of mcp_servers.json (definitions only).

        Live connections are NOT touched — connect/disconnect and
        register_server still manage those explicitly."""
        self._load_registry()

    def maybe_reload_registry(self) -> bool:
        """Auto re-read of mcp_servers.json before any REGISTRY LISTING
        (the proven PyQt behaviour: whenever the server list is called —
        e.g. the agent-creation MCP picker — the registry file is
        re-read, so servers hand-edited into mcp_servers.json or written
        by ANOTHER process appear immediately, without restarting the
        app and without pressing Refresh).

        A single stat() when nothing changed. Returns True when the
        registry was re-read. Never raises."""
        try:
            with self._lock:
                sig = self._registry_signature()
                if sig == self._registry_sig:
                    return False
            self._load_registry()
            return True
        except Exception:
            return False

    def _save_registry_locked(self):
        try:
            os.makedirs(self._app_dir, exist_ok=True)
            payload = {
                "mcp_registry_version": REGISTRY_VERSION,
                "servers": self._registry,
            }
            with open(self._registry_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            # Keep the auto-reload baseline in sync with what we just
            # wrote, so our own save does not look like an external edit.
            self._registry_sig = self._registry_signature()
        except Exception as e:
            print(f"[mcp] failed to save registry: {e}")

    @staticmethod
    def _normalize_config(cfg: dict) -> dict:
        """Validate/normalize one server definition."""
        transport = str(cfg.get("transport") or "stdio").strip().lower()
        if transport not in VALID_TRANSPORTS:
            transport = "stdio"
        out = {
            "transport": transport,
            "command": str(cfg.get("command") or "").strip(),
            "args": [str(a) for a in (cfg.get("args") or [])],
            "env": {str(k): str(v) for k, v in (cfg.get("env") or {}).items()},
            "url": str(cfg.get("url") or "").strip(),
            "headers": {str(k): str(v) for k, v in (cfg.get("headers") or {}).items()},
            "cwd": str(cfg.get("cwd") or "").strip() or None,
            "description": str(cfg.get("description") or "").strip(),
        }
        return out

    def _rebuild_slug_map(self):
        # called with self._lock held
        self._server_slug_to_name = {
            _slug(name): name for name in self._registry
        }

    # -- public registry API -------------------------------------------

    def list_registry_servers(self) -> Dict[str, dict]:
        with self._lock:
            return {name: dict(cfg) for name, cfg in self._registry.items()}

    def register_server(self, name: str, transport: str = "stdio",
                        command: str = "", args: Optional[list] = None,
                        env: Optional[dict] = None, url: str = "",
                        headers: Optional[dict] = None, cwd: str = "",
                        description: str = "") -> dict:
        """Create or update a server definition and persist the registry.

        If an existing connection for this name is alive and the definition
        meaningfully changed, the connection is restarted automatically.
        """
        name = str(name or "").strip()
        if not name:
            raise ValueError("Server name cannot be empty.")

        cfg = self._normalize_config({
            "transport": transport, "command": command, "args": args or [],
            "env": env or {}, "url": url, "headers": headers or {},
            "cwd": cwd, "description": description,
        })

        if cfg["transport"] == "stdio" and not cfg["command"]:
            raise ValueError("stdio servers need a 'command' (e.g. 'npx' or a path).")
        if cfg["transport"] in ("sse", "http") and not cfg["url"]:
            raise ValueError(f"{cfg['transport']} servers need a 'url'.")

        with self._lock:
            old = self._registry.get(name)
            self._registry[name] = cfg
            self._rebuild_slug_map()
            self._save_registry_locked()

        # Restart a live connection when its definition changed.
        conn = self._connections.get(name)
        if conn is not None and conn.want_running and old != cfg:
            print(f"[mcp] definition of '{name}' changed — restarting connection.")
            self.disconnect_server(name)
            self.connect_server(name, wait=CONNECT_WAIT_TIMEOUT)
        return dict(cfg)

    def remove_server(self, name: str) -> bool:
        """Delete a server definition, disconnect it, and unsubscribe every
        agent that was subscribed to it."""
        name = str(name or "").strip()
        with self._lock:
            existed = name in self._registry
            self._registry.pop(name, None)
            self._rebuild_slug_map()
            self._save_registry_locked()
        self.disconnect_server(name)
        # Drop the subscription from every agent that references it.
        manager = getattr(_app_context, "manager", None) if _app_context else None
        if manager is not None:
            for agent_name in list(manager.list_agents()):
                agent = manager.get_agent(agent_name)
                if agent is None:
                    continue
                subs = list(getattr(agent, "mcp_servers", []) or [])
                if name in subs:
                    subs.remove(name)
                    agent.mcp_servers = subs
        return existed

    def registry_snapshot(self) -> dict:
        """Snapshot for session files (definitions travel with the session)."""
        with self._lock:
            return {
                "mcp_registry_version": REGISTRY_VERSION,
                "servers": {name: dict(cfg) for name, cfg in self._registry.items()},
            }

    def merge_registry_snapshot(self, snapshot: Optional[dict]) -> int:
        """Merge definitions from a loaded session / saved-agent file.

        Existing local definitions WIN (the user may have fixed paths for
        THIS machine); only definitions the local registry is missing are
        added. Returns how many servers were added.
        """
        if not isinstance(snapshot, dict):
            return 0
        servers = snapshot.get("servers")
        if not isinstance(servers, dict):
            return 0
        added = 0
        with self._lock:
            changed = False
            for name, cfg in servers.items():
                name = str(name)
                if name in self._registry or not isinstance(cfg, dict):
                    continue
                try:
                    self._registry[name] = self._normalize_config(cfg)
                    changed = True
                    added += 1
                except Exception:
                    continue
            if changed:
                self._rebuild_slug_map()
                self._save_registry_locked()
        return added

    # ------------------------------------------------------------------
    # Agent subscription bookkeeping (subscriptions live on the Agent)
    # ------------------------------------------------------------------

    def _get_agent(self, agent_name: str):
        manager = getattr(_app_context, "manager", None) if _app_context else None
        if manager is None:
            return None
        return manager.get_agent(agent_name)

    def get_agent_subscriptions(self, agent_name: str) -> List[str]:
        agent = self._get_agent(agent_name)
        if agent is None:
            return []
        return list(getattr(agent, "mcp_servers", []) or [])

    def subscribe_agent(self, agent_name: str, server_name: str) -> str:
        """Subscribe an agent to a registry server and connect it.

        Returns a human/model-readable result message (includes the tool
        names the agent just gained — so the model immediately knows what
        it can call, no extra discovery round-trip needed).
        """
        agent = self._get_agent(agent_name)
        if agent is None:
            return f"Error: no agent named '{agent_name}'."

        name = str(server_name or "").strip()
        with self._lock:
            if name not in self._registry:
                return (f"Error: no MCP server named '{name}' in the registry. "
                        f"Use mcp_list_servers() to see the available servers, or "
                        f"register one first via mcp_subscribe(command=..., args=...).")

        # Connect (or reuse an existing connection).
        ok, info = self.connect_server(name, wait=CONNECT_WAIT_TIMEOUT)
        if not ok:
            return f"Error subscribing to '{name}': {info}"

        # Record the subscription on the agent object (persisted with it).
        subs = list(getattr(agent, "mcp_servers", []) or [])
        if name not in subs:
            subs.append(name)
            agent.mcp_servers = subs

        # Subscribe implies intent: if the agent's MCP capability flag is
        # off (e.g. the management tools were selected manually in
        # Available Tools), switch it on — otherwise the subscription
        # would be recorded but no native tools would ever flow.
        flipped = False
        if not getattr(agent, "mcp", False):
            agent.mcp = True
            flipped = True
            print(f"[mcp] agent '{agent_name}' mcp capability auto-enabled "
                  f"by subscribe.")

        tools = self._cached_tool_names(name)
        tool_list = ", ".join(tools[:40]) + ("..." if len(tools) > 40 else "")
        return (f"Subscribed agent '{agent_name}' to MCP server '{name}' "
                f"({len(tools)} native tools now available: {tool_list}). "
                + ("MCP capability was OFF — it has been switched ON for "
                   "this agent. " if flipped else "")
                + f"They appear as regular tools named '{MCP_TOOL_PREFIX}{_slug(name)}{MCP_TOOL_SEPARATOR}*<tool>' "
                f"and can be called directly.")

    def unsubscribe_agent(self, agent_name: str, server_name: str) -> str:
        agent = self._get_agent(agent_name)
        if agent is None:
            return f"Error: no agent named '{agent_name}'."
        name = str(server_name or "").strip()
        subs = list(getattr(agent, "mcp_servers", []) or [])
        if name not in subs:
            return (f"Agent '{agent_name}' is not subscribed to '{name}'. "
                    f"Current subscriptions: {', '.join(subs) or '(none)'}")
        subs.remove(name)
        agent.mcp_servers = subs

        # Disconnect the server when NO agent uses it anymore.
        still_used = False
        manager = getattr(_app_context, "manager", None) if _app_context else None
        if manager is not None:
            for other in manager.list_agents():
                other_agent = manager.get_agent(other)
                if other_agent is not None and name in (getattr(other_agent, "mcp_servers", []) or []):
                    still_used = True
                    break
        if not still_used:
            self.disconnect_server(name)

        return (f"Unsubscribed agent '{agent_name}' from MCP server '{name}'."
                + ("" if still_used else " (server disconnected — no other agent uses it)"))

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect_server(self, name: str, wait: float = CONNECT_WAIT_TIMEOUT) -> Tuple[bool, str]:
        """Start (or reuse) the persistent connection to a registry server."""
        name = str(name or "").strip()
        with self._lock:
            cfg = self._registry.get(name)
            if cfg is None:
                return False, f"no MCP server named '{name}' in the registry"
            conn = self._connections.get(name)

        if conn is not None and conn.ready.is_set() and conn.want_running:
            return True, f"already connected ({len(conn.tools_cache)} tools cached)"

        with self._lock:
            # Fresh connection object (drops any stale failed one).
            conn = _ServerConnection(name, cfg, app_dir=self._app_dir)
            conn.want_running = True
            conn.state = "connecting"
            self._connections[name] = conn

        if self._loop is None:
            conn.state = "failed"
            conn.last_error = "MCP engine loop is not running"
            return False, conn.last_error

        conn.task_future = asyncio.run_coroutine_threadsafe(
            self._run_connection(conn), self._loop
        )

        if wait and not conn.ready.wait(timeout=wait):
            state = conn.state
            err = conn.last_error or "timeout"
            if state == "connected":
                return True, "connected (tool listing still in progress)"
            return False, f"connection not ready (state: {state}; {err})"
        if conn.state == "failed":
            return False, conn.last_error or "connection failed"
        if conn.ready.is_set():
            return True, f"connected — {len(conn.tools_cache)} tools available"
        return True, "connecting"

    def disconnect_server(self, name: str) -> str:
        """Stop a server connection (process exits for stdio servers)."""
        name = str(name or "").strip()
        with self._lock:
            conn = self._connections.get(name)
        if conn is None:
            return f"No live connection for '{name}'."
        conn.want_running = False
        self._signal_stop(conn)
        # Give the loop a moment to wind the task down (best effort).
        deadline = time.time() + 5.0
        while time.time() < deadline and conn.state in ("connected", "connecting"):
            time.sleep(0.05)
        with self._lock:
            if self._connections.get(name) is conn:
                self._connections.pop(name, None)
        return f"Disconnected MCP server '{name}'."

    def _signal_stop(self, conn: _ServerConnection):
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(conn.stop_event.set)
        except Exception:
            # Loop already closed — nothing to signal.
            pass

    def ensure_subscribed_connected(self, agent_name: str, timeout: float = 30.0):
        """Make sure every server the agent subscribes to is connected.

        Called at the start of an MCP-enabled agent's turn: session restores
        (or app restarts) leave subscriptions without live connections, and
        the model should see its full native tool surface on turn one.
        Best-effort: failures are reported to the console and the failing
        server's tools are simply absent until it recovers.
        """
        subs = self.get_agent_subscriptions(agent_name)
        for name in subs:
            with self._lock:
                cfg = self._registry.get(name)
                conn = self._connections.get(name)
            if cfg is None:
                print(f"[mcp] agent '{agent_name}' subscribes to '{name}' "
                      f"but it is not in the registry — skipping.")
                continue
            if conn is not None and (conn.ready.is_set() or conn.state == "connecting"):
                continue
            try:
                ok, info = self.connect_server(name, wait=max(1.0, timeout / max(1, len(subs))))
                if not ok:
                    print(f"[mcp] could not connect '{name}' for agent "
                          f"'{agent_name}': {info}")
            except Exception as e:
                print(f"[mcp] connect '{name}' failed: {e}")

    # ------------------------------------------------------------------
    # The connection runner (asyncio, on the engine loop)
    # ------------------------------------------------------------------

    async def _run_connection(self, conn: _ServerConnection):
        """Persistent connection with auto-reconnect and backoff.

        Runs on the engine loop until want_running goes False. Each pass
        opens the transport, initializes the session, caches the tool list
        and then parks until stop is signalled. Crashes retry with backoff;
        after MAX_CONSECUTIVE_FAILURES the connection is parked 'failed'
        (an explicit reconnect / re-subscribe revives it).
        """
        conn.stop_event = asyncio.Event()
        backoff = RECONNECT_BACKOFF_START
        try:
            while conn.want_running:
                errlog = None
                try:
                    conn.state = "connecting"
                    conn.last_error = ""

                    if conn.config.get("transport") == "stdio":
                        from mcp import ClientSession, StdioServerParameters
                        from mcp.client.stdio import stdio_client

                        # Re-apply the child-watcher patch right before every
                        # spawn.  It is idempotent — if it was already applied
                        # at engine startup this is a no-op; if something
                        # between then and now restored the broken policy AND
                        # somehow undid our patch, this brings it back.
                        _install_child_watcher_patch()

                        env = dict(os.environ)
                        env.update(conn.config.get("env") or {})
                        params = StdioServerParameters(
                            command=conn.config.get("command") or "",
                            args=list(conn.config.get("args") or []),
                            env=env,
                        )
                        os.makedirs(os.path.dirname(conn.log_path), exist_ok=True)
                        errlog = open(conn.log_path, "a", encoding="utf-8", errors="replace")
                        errlog.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} connecting =====\n")
                        errlog.flush()

                        async with stdio_client(params, errlog=errlog) as (read, write):
                            async with ClientSession(read, write) as session:
                                await self._session_ready(conn, session)
                                await conn.stop_event.wait()
                    elif conn.config.get("transport") == "sse":
                        from mcp import ClientSession
                        from mcp.client.sse import sse_client

                        async with sse_client(
                            conn.config.get("url") or "",
                            headers=conn.config.get("headers") or {},
                        ) as (read, write):
                            async with ClientSession(read, write) as session:
                                await self._session_ready(conn, session)
                                await conn.stop_event.wait()
                    else:  # http — streamable http
                        from mcp import ClientSession
                        from mcp.client.streamable_http import streamablehttp_client

                        async with streamablehttp_client(
                            conn.config.get("url") or "",
                            headers=conn.config.get("headers") or {},
                        ) as (read, write, _get_session_id):
                            async with ClientSession(read, write) as session:
                                await self._session_ready(conn, session)
                                await conn.stop_event.wait()

                    # Clean stop (want_running False / stop_event set).
                    conn.state = "disconnected"
                    conn.session = None
                    conn.ready.clear()
                    return

                except Exception as e:
                    conn.session = None
                    conn.ready.clear()
                    conn.state = "disconnected"
                    conn.last_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    conn.consecutive_failures += 1
                    print(f"[mcp] connection '{conn.name}' failed "
                          f"({conn.consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): "
                          f"{conn.last_error}")
                    if errlog is not None:
                        try:
                            errlog.write(f"===== error: {conn.last_error}\n")
                        except Exception:
                            pass
                    if not conn.want_running:
                        return
                    if conn.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        conn.state = "failed"
                        print(f"[mcp] giving up on '{conn.name}' after "
                              f"{conn.consecutive_failures} consecutive failures "
                              f"(reconnect manually or re-subscribe to retry).")
                        return
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX)
                finally:
                    if errlog is not None:
                        try:
                            errlog.close()
                        except Exception:
                            pass
        except asyncio.CancelledError:
            conn.state = "disconnected"
            conn.session = None
            conn.ready.clear()

    async def _session_ready(self, conn: _ServerConnection, session):
        """initialize() + first tools/list; marks the connection ready."""
        await asyncio.wait_for(session.initialize(), INIT_TIMEOUT)
        conn.session = session
        conn.state = "connected"
        conn.consecutive_failures = 0
        try:
            tools = await self._fetch_tools(session)
            conn.tools_cache = tools
            conn.tools_cached_at = time.time()
            print(f"[mcp] '{conn.name}' connected — {len(tools)} tools")
        except Exception as e:
            # Connected but listing failed: still usable via mcp_call_tool.
            conn.last_error = f"tools/list failed: {e}"
            print(f"[mcp] '{conn.name}' connected but tools/list failed: {e}")
        conn.ready.set()

    async def _fetch_tools(self, session) -> List[dict]:
        """Fetch ALL tools (cursor-paginated), normalized."""
        from mcp import ClientSession  # noqa: F401  (type doc only)

        tools: List[dict] = []
        cursor = None
        for _ in range(50):  # hard page cap
            res = await asyncio.wait_for(
                session.list_tools(cursor=cursor), LIST_TOOLS_TIMEOUT
            )
            for t in (getattr(res, "tools", None) or []):
                schema = getattr(t, "inputSchema", None)
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}}
                schema = dict(schema)
                schema.setdefault("type", "object")
                if not isinstance(schema.get("properties"), dict):
                    schema["properties"] = {}
                if not isinstance(schema.get("required"), list):
                    schema.pop("required", None)
                tools.append({
                    "name": _safe_tool_name(getattr(t, "name", "") or ""),
                    "description": str(getattr(t, "description", "") or ""),
                    "schema": schema,
                })
            cursor = getattr(res, "nextCursor", None)
            if not cursor:
                break
        # De-duplicate by name (defensive: weird servers).
        seen = set()
        unique = []
        for t in tools:
            if t["name"] and t["name"] not in seen:
                seen.add(t["name"])
                unique.append(t)
        return unique

    # ------------------------------------------------------------------
    # Tool surface for Agent.call
    # ------------------------------------------------------------------

    def _get_connection(self, server_name: str) -> Optional[_ServerConnection]:
        with self._lock:
            return self._connections.get(server_name)

    def _resolve_server_by_slug(self, slug: str) -> Optional[str]:
        with self._lock:
            return self._server_slug_to_name.get(slug)

    def _cached_tool_names(self, server_name: str) -> List[str]:
        conn = self._get_connection(server_name)
        if conn is None:
            return []
        return [make_mcp_tool_name(server_name, t["name"]) for t in conn.tools_cache]

    def native_tool_defs_for_agent(self, agent_name: str) -> List[dict]:
        """OpenAI-style tool definitions for every subscribed server's tools.

        Refreshes a server's cached tool list when older than TOOL_CACHE_TTL
        (best-effort, non-blocking on failure). Agents whose 'mcp' flag is
        off never reach this method (Agent.call gates it).
        """
        defs: List[dict] = []
        subs = self.get_agent_subscriptions(agent_name)
        for name in subs:
            conn = self._get_connection(name)
            if conn is None or not conn.want_running:
                continue
            if conn.ready.is_set() and (time.time() - conn.tools_cached_at) > TOOL_CACHE_TTL:
                try:
                    self.refresh_tools(name)
                except Exception as e:
                    print(f"[mcp] tool refresh for '{name}' failed: {e}")
            conn = self._get_connection(name)
            if conn is None:
                continue
            for t in conn.tools_cache:
                defs.append({
                    "type": "function",
                    "function": {
                        "name": make_mcp_tool_name(name, t["name"]),
                        "description": (t["description"] or "").strip()
                                       or f"MCP tool '{t['name']}' from server '{name}'.",
                        "parameters": t["schema"],
                    },
                })
        return defs

    def extend_allowed_tools(self, agent_name: str, base: List[str]) -> List[str]:
        """Base whitelist + MCP management tools + native MCP tool names.

        Used for build_context's tool-leak filter: past MCP tool_calls must
        survive the rebuild, so the allowed list has to cover them.
        """
        allowed = list(base)
        for n in MANAGEMENT_TOOL_NAMES:
            if n not in allowed:
                allowed.append(n)
        for ns in self.get_agent_subscriptions(agent_name):
            for n in self._cached_tool_names(ns):
                if n not in allowed:
                    allowed.append(n)
        return allowed

    def handles_tool(self, agent_name: str, tool_name: str) -> bool:
        """True when tool_name is a native MCP tool of a SUBSCRIBED server."""
        parsed = parse_mcp_tool_name(tool_name)
        if parsed is None:
            return False
        slug, _tool = parsed
        server = self._resolve_server_by_slug(slug)
        if server is None:
            return False
        return server in self.get_agent_subscriptions(agent_name)

    def is_tool_allowed(self, agent_name: str, tool_name: str) -> bool:
        """Whitelist check for MCP-enabled agents (management + native tools)."""
        if tool_name in MANAGEMENT_TOOL_NAMES:
            return True
        return self.handles_tool(agent_name, tool_name)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def call_tool(self, agent_name: str, tool_name: str,
                  arguments: Optional[dict] = None,
                  timeout: float = CALL_TOOL_TIMEOUT) -> str:
        """Route one namespaced (or server-qualified) MCP tool call.

        Returns the rendered result as a string (never raises — errors come
        back as readable text so the model can react). Automatically waits
        for an auto-reconnecting server once, then retries the call.
        """
        arguments = dict(arguments or {})
        server, raw_tool = self._resolve_tool_target(agent_name, tool_name)
        if server is None:
            return (f"Error: '{tool_name}' is not an MCP tool available to agent "
                    f"'{agent_name}'. Use mcp_list_tools() to see your MCP tools.")

        conn = self._get_connection(server)
        if conn is None or not conn.want_running:
            # Lazily revive (e.g. session restore, server parked as failed).
            try:
                ok, info = self.connect_server(server, wait=CONNECT_WAIT_TIMEOUT)
                if not ok:
                    return f"Error connecting MCP server '{server}': {info}"
            except Exception as e:
                return f"Error connecting MCP server '{server}': {e}"
            conn = self._get_connection(server)

        if conn is None or conn.session is None:
            return (f"Error: MCP server '{server}' is not connected "
                    f"(state: {conn.state if conn else 'unknown'}). Try again shortly "
                    f"or call mcp_status() to inspect.")

        try:
            result = self._submit(
                self._do_call(conn, raw_tool, arguments, timeout),
                timeout + 30.0,
            )
            return self._render_result(result, tool_name)
        except Exception as first_error:
            # Dead session mid-call.  NOTE: we cannot just wait on
            # conn.ready here — the connection runner is parked on
            # stop_event.wait() and does NOT notice that the underlying
            # session died, so conn.ready stays set forever and
            # conn.session still points at the dead session.  Waiting on
            # ready therefore returns instantly and the "retry" retries
            # the same dead session.  Force a real teardown + reconnect.
            if self._looks_like_dead_session(first_error):
                print(f"[mcp] session for '{server}' looks dead "
                      f"({type(first_error).__name__}: {first_error}); "
                      f"forcing reconnect.")
                try:
                    self.disconnect_server(server)
                    ok, info = self.connect_server(
                        server, wait=RECONNECT_WAIT
                    )
                    if not ok:
                        return (f"Error calling MCP tool '{tool_name}' after "
                                f"reconnect: {info}")
                    conn = self._get_connection(server)
                    if conn is None or conn.session is None:
                        return (f"Error calling MCP tool '{tool_name}' after "
                                f"reconnect: no live session after reconnect")
                    result = self._submit(
                        self._do_call(conn, raw_tool, arguments, timeout),
                        timeout + 30.0,
                    )
                    return self._render_result(result, tool_name)
                except Exception as second_error:
                    return (f"Error calling MCP tool '{tool_name}' after "
                            f"reconnect: {second_error}")
            return f"Error calling MCP tool '{tool_name}': {first_error}"

    def _resolve_tool_target(self, agent_name: str, tool_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Accept 'mcp__server__tool', or 'server:tool', or a bare tool name
        found on exactly one subscribed server.

        WHITELIST: every spelling resolves ONLY against servers the agent
        is SUBSCRIBED to — an unsubscribed server's tools are never
        callable, mirroring the strict per-agent tool whitelist the app
        enforces for built-in tools.
        """
        my_subs = self.get_agent_subscriptions(agent_name)
        parsed = parse_mcp_tool_name(tool_name)
        if parsed is not None:
            slug, tool = parsed
            server = self._resolve_server_by_slug(slug)
            if server is not None and server in my_subs:
                return server, tool
            return None, None
        if ":" in tool_name:
            server, tool = tool_name.split(":", 1)
            server = server.strip()
            if server in my_subs:
                return server, tool.strip()
            return None, None
        # Bare name: search the agent's subscribed servers.
        candidates = []
        for ns in my_subs:
            conn = self._get_connection(ns)
            if conn is None:
                continue
            for t in conn.tools_cache:
                if t["name"] == tool_name:
                    candidates.append((ns, t["name"]))
                    break
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            # Ambiguous — the model must use the namespaced name.
            return None, None
        return None, None

    @staticmethod
    def _looks_like_dead_session(exc: Exception) -> bool:
        if isinstance(exc, (_ClosedResourceError, _BrokenResourceError)):
            return True
        text = str(exc).lower()
        return any(k in text for k in (
            "closed", "broken", "session is closed", "disconnected", "cancelled",
        ))

    async def _do_call(self, conn: _ServerConnection, tool: str,
                       arguments: dict, timeout: float):
        if conn.session is None:
            raise RuntimeError(f"MCP server '{conn.name}' session is not open")
        return await asyncio.wait_for(
            conn.session.call_tool(tool, arguments or {}), timeout
        )

    @staticmethod
    def _render_result(result, tool_name: str) -> str:
        """Flatten an MCP CallToolResult into model-friendly text."""
        try:
            is_error = bool(getattr(result, "isError", False))
        except Exception:
            is_error = False

        text_parts: List[str] = []
        has_text = False
        for c in (getattr(result, "content", None) or []):
            text = getattr(c, "text", None)
            if text:
                has_text = True
                text_parts.append(str(text))
                continue
            ctype = type(c).__name__
            if ctype == "ImageContent" or getattr(c, "data", None) is not None:
                try:
                    mime = getattr(c, "mimeType", "unknown")
                    size = len(getattr(c, "data", "") or "")
                    text_parts.append(f"[MCP {ctype}: {mime}, {size} chars of data — "
                                      f"not displayable as text]")
                except Exception:
                    text_parts.append(f"[MCP {ctype}]")
            else:
                text_parts.append(f"[MCP {ctype}]")

        parts = text_parts

        # structuredContent: only used when the server sent NO text
        # content. FastMCP-style servers send BOTH (the text part is the
        # human-readable form of the same payload) — appending both would
        # duplicate every result, so text wins when present.
        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict) and structured and not has_text:
            try:
                parts.append(json.dumps(structured, indent=2, ensure_ascii=False))
            except Exception:
                parts.append(str(structured))

        text = "\n".join(p for p in parts if p is not None).strip()
        # Pretty-print embedded JSON when the whole payload is JSON.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (dict, list)):
                text = json.dumps(parsed, indent=2, ensure_ascii=False)
        except (ValueError, TypeError):
            pass

        if not text:
            text = f"(MCP tool '{tool_name}' returned no content)"
        if is_error:
            return f"MCP tool error (server reported isError): {text}"
        return text

    # ------------------------------------------------------------------
    # Info / maintenance
    # ------------------------------------------------------------------

    def refresh_tools(self, server_name: str) -> Tuple[bool, str]:
        """Force a fresh tools/list on a live connection."""
        name = str(server_name or "").strip()
        conn = self._get_connection(name)
        if conn is None or conn.session is None:
            return False, f"server '{name}' is not connected"
        try:
            tools = self._submit(self._fetch_tools(conn.session), LIST_TOOLS_TIMEOUT + 15.0)
            conn.tools_cache = tools
            conn.tools_cached_at = time.time()
            return True, f"refreshed — {len(tools)} tools"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def list_tools_info(self, server_name: Optional[str] = None) -> List[str]:
        """Formatted tool listing for one server (or all subscribed/current)."""
        out: List[str] = []
        with self._lock:
            names = ([server_name] if server_name else list(self._connections.keys()))
            if server_name and server_name not in self._connections:
                if server_name in self._registry:
                    out.append(f"MCP server '{server_name}' is registered but not "
                               f"connected — subscribe/connect first to list tools.")
                    return out
                out.append(f"No MCP server named '{server_name}'. "
                           f"Use mcp_list_servers() to see the registry.")
                return out

        for name in names:
            conn = self._get_connection(name)
            if conn is None:
                continue
            out.append(f"== MCP server '{name}' ({conn.describe()}) "
                       f"[{conn.state}] — {len(conn.tools_cache)} tools ==")
            for t in conn.tools_cache:
                ns = make_mcp_tool_name(name, t["name"])
                desc = (t["description"] or "").strip().splitlines()
                desc_line = desc[0][:160] if desc else ""
                required = ", ".join(t["schema"].get("required") or []) or "-"
                props = ", ".join(list(t["schema"].get("properties", {}).keys())[:12])
                out.append(f"  {ns}")
                if desc_line:
                    out.append(f"    {desc_line}")
                out.append(f"    args: {props or '(none)'} | required: {required}")
        if not out:
            out.append("No connected MCP servers. Use mcp_list_servers() to see "
                       "the registry and mcp_subscribe() to connect one.")
        return out

    def list_tools_info_multi(self, server_names: List[str]) -> List[str]:
        """Formatted tool listing for several servers at once.

        Used by the mcp_list_tools core tool for the calling agent's whole
        subscription list. Servers that are subscribed but not (yet)
        connected produce a hint line instead of failing.
        """
        out: List[str] = []
        any_listed = False
        for name in (server_names or []):
            conn = self._get_connection(name)
            if conn is None or not conn.tools_cache:
                out.append(f"== MCP server '{name}' — connected: "
                           f"{'no' if conn is None else conn.state} — no tools "
                           f"listed yet (tools appear once it is connected) ==")
                continue
            any_listed = True
            out.append(f"== MCP server '{name}' ({conn.describe()}) "
                       f"[{conn.state}] — {len(conn.tools_cache)} tools ==")
            for t in conn.tools_cache:
                ns = make_mcp_tool_name(name, t["name"])
                desc = (t["description"] or "").strip().splitlines()
                desc_line = desc[0][:160] if desc else ""
                required = ", ".join(t["schema"].get("required") or []) or "-"
                props = ", ".join(list(t["schema"].get("properties", {}).keys())[:12])
                out.append(f"  {ns}")
                if desc_line:
                    out.append(f"    {desc_line}")
                out.append(f"    args: {props or '(none)'} | required: {required}")
        if not any_listed and not out:
            out.append("No MCP servers to list.")
        return out

    def status_info(self) -> List[str]:
        """One status line per registry server (+ engine line)."""
        out = [f"MCP engine: running (native tools, prefix "
               f"'{MCP_TOOL_PREFIX}<server>{MCP_TOOL_SEPARATOR}<tool>')"]
        with self._lock:
            names = sorted(self._registry.keys())
        if not names:
            out.append("Registry is empty — add servers with mcp_subscribe(...), "
                       "the 'MCP Servers...' dialog, or by editing mcp_servers.json.")
            return out
        for name in names:
            out.append(f"  {name}: {self.status_info_for(name)}")
        return out

    def status_info_for(self, server_name: str) -> str:
        """One compact status line for a single registry server."""
        name = str(server_name or "").strip()
        with self._lock:
            cfg = dict(self._registry.get(name) or {})
        if not cfg:
            return f"{name}: not in registry"
        conn = self._get_connection(name)
        transport = cfg.get("transport", "stdio")
        if transport == "stdio":
            endpoint = cfg.get("command", "")
            args = cfg.get("args") or []
            if args:
                endpoint += " " + " ".join(str(a) for a in args[:4])
        else:
            endpoint = cfg.get("url", "")
        if conn is None:
            state = "disconnected (not started)"
            tools = 0
        else:
            state = conn.state
            tools = len(conn.tools_cache)
            if conn.last_error:
                state += f" | {conn.last_error[:80]}"
        return f"{transport}: {endpoint} — {state} — {tools} tools"

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown_all(self):
        """Stop every connection and the engine loop (app exit). Idempotent."""
        if self._shutting_down:
            return
        self._shutting_down = True
        with self._lock:
            conns = list(self._connections.values())
        for conn in conns:
            conn.want_running = False
            self._signal_stop(conn)
        # Give the loop a moment to close transports / terminate processes.
        deadline = time.time() + 5.0
        while time.time() < deadline and any(
            c.state in ("connected", "connecting") for c in conns
        ):
            time.sleep(0.05)
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
        with self._lock:
            self._connections.clear()
        print("[mcp] engine shut down.")


# ----------------------------------------------------------------------
# Module-level singleton
# ----------------------------------------------------------------------
_MANAGER: Optional[MCPManager] = None
_MANAGER_LOCK = threading.Lock()


def set_manager(manager: Optional[MCPManager]):
    """Publish the app-wide MCPManager (called once by main.py)."""
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = manager


def get_manager() -> Optional[MCPManager]:
    """Return the app-wide MCPManager, or None when MCP is not initialized.

    Callers (Agent.call, core tools, GUI) must treat None as "feature
    unavailable" and degrade gracefully — the app must keep working on
    setups where the engine could not start.
    """
    with _MANAGER_LOCK:
        return _MANAGER

