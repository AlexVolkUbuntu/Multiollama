"""
server/app_context.py — process-wide application context + bootstrap.

This is the headless counterpart of the PyQt app's main(): it loads ALL
modules the proven core needs, publishes the singletons into `context`,
starts the MCP engine, and prepares the folders:

    tools/     — regenerated from core_tools.CORE_TOOLS_CODE at startup
                 (user-added tool files are left untouched)
    prompts/   — reusable prompt snippets (.txt), auto-created
    autosave/  — session auto-saves (PyQt-compatible naming + pruning)
    sessions/  — manually placed session JSONs (import path for the queue)
    agent_dialogues/ — created lazily by DialogueStore

Per the design brief it loads NO sessions at startup: the backend boots
with an empty idle state, the queue manager parks in "wait" mode, and
sessions are loaded only when they are dequeued from the queue (or an
interactive session is created via the control API).

The AppContext object is the single dependency every API route and every
server component receives; it holds:

    config          server/config.py Config
    broadcast       server/broadcast.py BroadcastBus (full output stream)
    session_store   server/session_store.py SessionStore
    queue_manager   server/queue_manager.py SessionQueueManager
    loader          agent.ToolLoader                (tools/)
    prompt_loader   agent.PromptLoader              (prompts/)
    client_manager  ollama_client.ClientManager     (per-endpoint clients)
    dialogue_store  dialogue_store.DialogueStore
    mcp_manager     mcp_manager.MCPManager | None   (native MCP engine)
    model_usage     {endpoint: {model: {size, size_vram}}} polled cache
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import context

import core_tools
from agent import ToolLoader, PromptLoader
from agent_manager import AgentManager, Conversation
from dialogue_store import DialogueStore
from ollama_client import ClientManager

from .broadcast import BroadcastBus
from .config import Config, get_config
from .queue_manager import SessionQueueManager
from .session_store import SessionStore

try:
    import mcp_manager as mcp_engine
except Exception:
    mcp_engine = None


class AppContext:
    """Everything the server needs, in one injectable object."""

    def __init__(self, config: Optional[Config] = None,
                 app_dir: Optional[str] = None):
        self.config = config or get_config()
        self.started_at = time.time()
        self.app_dir = app_dir or str(Path(__file__).resolve().parent.parent)
        self.config.app_dir = self.app_dir

        self.broadcast = BroadcastBus(ring_size=self.config.recent_events_buffer)
        self.session_store = SessionStore(keep=self.config.autosave_keep)

        # ---- module bootstrap (main.py equivalent, minus the GUI) -----
        tools_dir = Path(self.app_dir) / "tools"
        self._write_core_tools(tools_dir)
        self.tools_dir = str(tools_dir)

        prompts_dir = Path(self.app_dir) / "prompts"
        self.prompt_loader = PromptLoader(str(prompts_dir))
        self.prompts_dir = str(prompts_dir)

        self.loader = ToolLoader(str(tools_dir))
        self.dialogue_store = DialogueStore()
        self.client_manager = ClientManager(
            request_timeout=180, max_retries=2)

        # NATIVE MCP engine — one shared engine, failures never take the
        # backend down (proven main.py behaviour).
        self.mcp_manager = None
        try:
            if mcp_engine is not None:
                mgr = mcp_engine.MCPManager()
                mcp_engine.set_manager(mgr)
                self.mcp_manager = mgr
        except Exception as exc:
            print(f"[mcp] engine failed to start — MCP disabled: {exc}")
            self.mcp_manager = None

        # ---- publish singletons for the proven tools -------------------
        context.loader = self.loader
        context.prompt_loader = self.prompt_loader
        context.dialogue_store = self.dialogue_store
        context.client_manager = self.client_manager
        context.mcp_manager = self.mcp_manager
        # An empty manager/conversation so nothing explodes before the
        # first session binds its own (runner.start rebinds these).
        context.manager = AgentManager(self.loader)
        context.conversation = Conversation(context.manager)
        context.thinking_router = None
        context.agent_status_router = None
        context.message_router = None

        # ---- the session queue -------------------------------------------
        self.queue_manager = SessionQueueManager(self)

        # ---- display-only model memory usage cache ----------------------
        self.model_usage: Dict[str, dict] = {}
        self._usage_thread = None
        self._usage_stop = False

    # ------------------------------------------------------------------
    # Bootstrap helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _write_core_tools(tools_dir: Path) -> None:
        """(Re-)write the built-in tool files; user tools are untouched.

        Identical to main.write_core_tools."""
        tools_dir.mkdir(exist_ok=True)
        for filename, code in core_tools.CORE_TOOLS_CODE.items():
            filepath = tools_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"Written: {filepath}")

    # ------------------------------------------------------------------
    # Model usage polling (the GUI's 3s QTimer equivalent)
    # ------------------------------------------------------------------
    def start_usage_polling(self) -> None:
        if self._usage_thread is not None:
            return

        def _poll():
            interval = max(1.0, float(self.config.usage_poll_interval))
            while not self._usage_stop:
                endpoints = set()
                runner = self.queue_manager.active_runner
                if runner is not None:
                    endpoints.update(runner.manager.get_endpoints())
                if not endpoints:
                    # fall back to endpoints we already know about
                    endpoints.update(self.client_manager._clients.keys())
                usage: Dict[str, dict] = {}
                for ep in endpoints:
                    try:
                        client = self.client_manager.get_client(ep)
                        usage[ep] = client.get_model_usage()
                    except Exception:
                        continue
                if usage != self.model_usage:
                    self.model_usage = usage
                time.sleep(interval)

        self._usage_thread = __import__("threading").Thread(
            target=_poll, name="model-usage-poll", daemon=True)
        self._usage_thread.start()

    def stop_usage_polling(self) -> None:
        self._usage_stop = True

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Orderly shutdown: finish the active session (final autosave),
        stop the queue, stop MCP servers and Ollama clients."""
        try:
            self.queue_manager.shutdown(abort_active=False)
        except Exception as exc:
            print(f"[shutdown] queue manager error: {exc}")
        self.stop_usage_polling()
        if self.mcp_manager is not None:
            try:
                self.mcp_manager.shutdown_all()
            except Exception as exc:
                print(f"[shutdown] mcp shutdown error: {exc}")
        try:
            self.client_manager.stop_all()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def uptime(self) -> float:
        return time.time() - self.started_at
