"""
server/runner.py — SessionRunner: the lifecycle of ONE active session.

The runner is the headless replacement for "one running instance of the
PyQt MainWindow": it owns the per-session AgentManager + Conversation,
spawns one EngineThread per endpoint, publishes every observable event
onto the BroadcastBus, keeps the throttled auto-save going, and decides
when the session is DONE.

Session sources
===============
  * "queue"       — started by the SessionQueueManager from a QueueEntry.
                    The session data is loaded from the newest auto-save
                    (or sessions/) file for that session name; the entry
                    may carry a message that is queued as a user message
                    BEFORE the engines start ("message to send before
                    it"). auto_finish is ON: the session finishes when it
                    quiesces.
  * "interactive" — started by POST /control/session/new or
                    /control/session/load. auto_finish is OFF: it runs
                    until the operator finishes it explicitly (or a
                    queued session preempts it).

When is a queue session DONE ("quiesced")?
==========================================
    * no engine currently runs an agent turn, AND
    * the pending-user-message queue is empty (the initial message was
      drained into history), AND
    * every agent is inactive — the proven `core_agent_active(name,
      False)` "I'm done, leaving the rotation" mechanism is exactly the
      completion signal, AND
    * a small startup grace period has passed (so the initial message
      can never be mistaken for an already-finished session).

Additional finish triggers:
    * explicit request  — request_finish() (graceful: the running turn
      completes, no new turn starts) or abort() (streams are stopped).
    * idle_timeout      — per-entry ceiling on inactivity.
    * all endpoints down longer than config.endpoint_down_timeout — an
      unattended queue must never hang forever on a dead Ollama box.

On completion the runner ALWAYS writes a final auto-save (requirement:
"once one session is done - autosaves it") before the queue manager
loads the next one.

Context rebinding
=================
The proven tools (generated into tools/*.py) reach the app through the
module-level singletons in `context`. The runner rebinds manager /
conversation / routers for the session it runs, exactly like MainWindow
did at startup — core_agent_execute, core_prune_agent_context, the
dialogue tools and every user tool keep working unmodified.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import context
import tool_injection
from agent import ToolLoader
from agent_manager import AgentManager, Conversation

from .engine import EngineHooks, EngineThread, endpoint_label
from .broadcast import chat_event_role
from .media_links import enrich_chat_event
from .session_store import SessionStore


class RunnerOptions:
    """How a session should be run (queue entries carry these)."""

    def __init__(self,
                 source: str = "queue",            # "queue" | "interactive"
                 auto_finish: bool = True,
                 message: Optional[str] = None,
                 sender: str = "user",
                 activate_all: bool = False,
                 activate: Optional[List[str]] = None,
                 idle_timeout: Optional[float] = None,
                 wait_user_input: bool = False):
        self.source = source
        self.auto_finish = auto_finish
        self.message = message
        self.sender = sender or "user"
        self.activate_all = bool(activate_all)
        self.activate = list(activate or [])
        self.idle_timeout = idle_timeout
        # Initial state of the 'Pause User Input' / 'wait user input'
        # checkbox. The proven PyQt app defaults it to ON (the "ISSUE #1
        # FIX": the round-robin only advances after the user sends a
        # message or presses Send Now). Interactive sessions therefore
        # start WAITING; queue entries keep it OFF because they run
        # unattended (their entry message is the trigger).
        self.wait_user_input = bool(wait_user_input)


class SessionRunner:
    """Runs one session from start to finish and reports completion."""

    def __init__(self, app, session_data: dict, session_name: str,
                 options: RunnerOptions,
                 staged_injections: Optional[List[dict]] = None):
        """
        *app* is the AppContext (server/app_context.py): it supplies the
        process-wide loader / prompt loader / client manager / dialogue
        store / MCP engine / broadcast bus / session store / config.
        """
        self.app = app
        self.options = options
        self.session_name = session_name
        self.user_name = str(session_data.get("user_name") or "User")

        # ---- per-session state -----------------------------------------
        self.manager = AgentManager(app.loader)
        self.conversation = Conversation(self.manager)
        self.engines: Dict[str, EngineThread] = {}
        self._engines_lock = threading.RLock()
        # 'Send Now' that arrived while no engine existed yet — applied
        # to engines the moment sync_engines() creates them.
        self._pending_send_now = False

        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.finish_reason: Optional[str] = None
        self.autosave_path: Optional[str] = None
        self.last_autosave = 0.0
        self.last_activity = time.time()
        # 'wait user input' initial state (see RunnerOptions.wait_user_input)
        self._user_paused = bool(getattr(options, "wait_user_input", False))

        self._finish_requested = False       # graceful finish
        self._abort_requested = False        # hard stop
        self._done_event = threading.Event()
        self._agents_dirty = threading.Event()
        self._finalized = False

        # Status snapshot for the state API (endpoint -> (text, color)).
        self.endpoint_status: Dict[str, tuple] = {}
        # Agents currently running (endpoint, name) -> model.
        self.running_agents: Dict[str, str] = {}

        self._staged_injections = list(staged_injections or [])

        # ---- hooks: forward engine events onto the broadcast bus -------
        self.hooks = self._build_hooks()

        # ---- load the session data --------------------------------------
        self._apply_session_data(session_data)

    # ==================================================================
    # Session construction
    # ==================================================================
    def _apply_session_data(self, data: dict) -> None:
        """Populate manager + conversation from a session dict.

        Mirrors gui_main._apply_session_data minus all Qt UI work:
        MCP registry merge, manager/conversation rebuild, media-path
        re-anchoring (when the app folder moved) and the optional
        activation overrides from the runner options.
        """
        # NATIVE MCP: merge the session's embedded registry BEFORE the
        # agents are rebuilt (local definitions always win).
        mcp_mgr = getattr(self.app, "mcp_manager", None)
        if mcp_mgr is not None and data.get("mcp_registry"):
            try:
                added = mcp_mgr.merge_registry_snapshot(data.get("mcp_registry"))
                if added:
                    print(f"[runner] merged {added} MCP server definition(s) "
                          f"from session '{self.session_name}'.")
            except Exception as exc:
                print(f"[runner] MCP registry merge failed: {exc}")

        self.manager.from_dict(data.get("agents", {}) or {})
        self.conversation.from_dict(data.get("conversation", {}) or {})

        # Re-anchor absolute media paths recorded under a previous app
        # location (proven gui_main._reanchor_loaded_history behaviour).
        try:
            from markdown_renderer import reanchor_text_paths
            old_base = data.get("app_dir") or ""
            new_base = self.app.app_dir
            if isinstance(old_base, str) and old_base:
                import os as _os
                if _os.path.normpath(_os.path.expanduser(old_base)) != \
                        _os.path.normpath(new_base):
                    count = 0
                    with self.conversation._lock:
                        for msg in self.conversation.history:
                            if not isinstance(msg, dict):
                                continue
                            content = msg.get("content")
                            if not isinstance(content, str) or not content:
                                continue
                            try:
                                new_content, n = reanchor_text_paths(
                                    content, old_base, new_base)
                            except Exception:
                                continue
                            if n:
                                msg["content"] = new_content
                                count += n
                    if count:
                        print(f"[runner] re-anchored {count} media path(s) "
                              f"in session '{self.session_name}'.")
        except Exception as exc:
            print(f"[runner] media re-anchoring skipped: {exc}")

        # Activation overrides (queue entries may re-activate agents that
        # were saved inactive because they had finished their last job).
        if self.options.activate_all:
            for agent in self.manager.agents.values():
                agent.active = True
        for name in self.options.activate:
            agent = self.manager.get_agent(name)
            if agent is not None:
                agent.active = True

    # ==================================================================
    # Hooks — engine events -> broadcast bus + bookkeeping
    # ==================================================================
    def _build_hooks(self) -> EngineHooks:
        bus = self.app.broadcast
        runner = self

        class _Hooks(EngineHooks):
            def thinking(self, endpoint, agent_name, event_type, detail, extra):
                bus.publish("thinking", {
                    "endpoint": endpoint,
                    "agent": agent_name,
                    "event_type": event_type,
                    "detail": detail,
                    "extra": extra,
                })

            def message(self, sender, text):
                runner.last_activity = time.time()
                # media_links scan: attach download links for every real
                # file the message mentions (thinking streams never come
                # through here — they use the thinking() hook above).
                # `role` lets frontends render proper chat bubbles
                # (user / assistant / system / error) — without it every
                # live message degraded to the system-note style while a
                # history reload showed the same text as a real bubble.
                bus.publish("chat", enrich_chat_event({
                    "sender": sender,
                    "text": text,
                    "role": chat_event_role(sender, runner.user_name),
                    "session": runner.session_name,
                }))

            def status(self, endpoint, text, color):
                runner.endpoint_status[endpoint] = (text, color)
                bus.publish("status", {
                    "endpoint": endpoint, "text": text, "color": color,
                })

            def agent_status(self, endpoint, agent_name, running, model):
                if running:
                    runner.running_agents[agent_name] = model
                else:
                    runner.running_agents.pop(agent_name, None)
                bus.publish("agent_status", {
                    "endpoint": endpoint, "agent": agent_name,
                    "running": running, "model": model,
                })

            def queue_updated(self):
                bus.publish("queue_updated", {
                    "pending_user_messages":
                        list(runner.conversation.pending_user_messages),
                })

            def autosave(self, agent_name):
                runner._throttled_autosave()

            def turn_started(self, endpoint, agent_name):
                runner.last_activity = time.time()

            def turn_finished(self, endpoint, agent_name):
                runner.last_activity = time.time()

            def endpoint_unavailable(self, endpoint):
                bus.publish("status", {
                    "endpoint": endpoint,
                    "text": "Ollama unavailable",
                    "color": "red",
                })

            def endpoint_ready_again(self, endpoint):
                pass

        return _Hooks()

    # ==================================================================
    # Lifecycle: start / finish / abort / monitor
    # ==================================================================
    def start(self) -> None:
        """Bind globals, deliver staged injections + the entry message,
        spawn the engines and the monitor thread."""
        # --- bind the global context for the proven tools ---------------
        context.manager = self.manager
        context.loader = self.app.loader
        context.prompt_loader = self.app.prompt_loader
        context.conversation = self.conversation
        context.client_manager = self.app.client_manager
        context.dialogue_store = self.app.dialogue_store
        context.mcp_manager = self.app.mcp_manager
        # Routers used by core_agent_execute to forward a called agent's
        # live events — plain callables, thread-safe via the bus.
        context.thinking_router = (
            lambda ep, agent, etype, detail, extra:
            self.app.broadcast.publish("thinking", {
                "endpoint": ep, "agent": agent, "event_type": etype,
                "detail": detail, "extra": extra})
        )
        context.agent_status_router = (
            lambda ep, agent, running, model:
            self.app.broadcast.publish("agent_status", {
                "endpoint": ep, "agent": agent,
                "running": running, "model": model})
        )
        context.message_router = (
            lambda sender, text:
            self.app.broadcast.publish("chat", enrich_chat_event({
                "sender": sender, "text": text,
                "role": chat_event_role(sender, self.user_name),
                "session": self.session_name}))
        )

        # Agent CRUD (also from core tools) -> engines must resync.
        self.manager.agents_changed.connect(self._on_agents_changed)

        # --- staged injections: delivered to the named agents' next tool
        # result (they sit in tool_injection's per-agent buffer until the
        # agent actually calls a tool — proven delivery semantics).
        for inj in self._staged_injections:
            count = tool_injection.add_injection(inj["agent_name"],
                                                 inj["text"])
            self.app.broadcast.publish("injection", {
                "session_name": self.session_name,
                "agent_name": inj["agent_name"],
                "count": count,
                "staged": True,
                "delivered_to_buffer": True,
            })

        # --- the queue entry's "message to send before it" --------------
        if self.options.message and str(self.options.message).strip():
            with self.conversation._lock:
                self.conversation.pending_user_messages.append(
                    str(self.options.message).strip())
            self.app.broadcast.publish("queue_updated", {
                "pending_user_messages":
                    list(self.conversation.pending_user_messages),
            })

        # --- spawn engines ------------------------------------------------
        self.sync_engines()

        # --- monitor thread ------------------------------------------------
        self._monitor = threading.Thread(
            target=self._monitor_loop,
            name=f"monitor-{self.session_name}",
            daemon=True)
        self._monitor.start()

        self.app.broadcast.publish("session", {
            "event": "started",
            "session_name": self.session_name,
            "source": self.options.source,
            "auto_finish": self.options.auto_finish,
            "wait_user_input": self._user_paused,
            "agents": self.manager.list_agents(),
        })
        print(f"[runner] session '{self.session_name}' started "
              f"({self.options.source}, agents: "
              f"{self.manager.list_agents()}, "
              f"wait_user_input={self._user_paused})")

    def _on_agents_changed(self) -> None:
        self._agents_dirty.set()
        names = self.manager.list_agents()
        self.app.broadcast.publish("agents_changed", {"agents": names})

    # ------------------------------------------------------------------
    # Engine management
    # ------------------------------------------------------------------
    def sync_engines(self) -> None:
        """Reconcile the engine-per-endpoint set with the manager.

        * endpoints that lost all their agents -> engine shut down
        * new endpoints                        -> engine created
        * every engine gets its endpoint's current agent list
        """
        with self._engines_lock:
            if self._finish_requested or self._abort_requested:
                return
            pending_send_now = self._pending_send_now
            endpoints = self.manager.get_endpoints()
            for ep in list(self.engines.keys()):
                if ep not in endpoints:
                    engine = self.engines.pop(ep)
                    try:
                        engine.shutdown()
                    except Exception:
                        pass
            for ep in endpoints:
                engine = self.engines.get(ep)
                agents_on_ep = self.manager.get_agents_on_endpoint(ep)
                if engine is None:
                    client = self.app.client_manager.get_client(ep)
                    engine = EngineThread(
                        self.manager, self.app.loader, self.conversation,
                        client, ep, agents_on_ep, self.hooks,
                        user_name=self.user_name,
                        user_paused=self.user_paused,
                    )
                    self.engines[ep] = engine
                    engine.start()
                else:
                    engine.update_agents(agents_on_ep)

            # A 'Send Now' that arrived BEFORE any engine existed (agent
            # created a moment ago; the monitor had not reconciled yet)
            # is applied to the freshly created engines so it is never
            # silently lost.
            if pending_send_now:
                for engine in list(self.engines.values()):
                    engine.unblock_user()
                self._pending_send_now = False

    @property
    def user_paused(self) -> bool:
        return self._user_paused

    def set_user_paused(self, paused: bool) -> None:
        self._user_paused = bool(paused)
        with self._engines_lock:
            engines = list(self.engines.values())
        for engine in engines:
            engine.set_user_paused(self._user_paused)

    def set_user_name(self, name: str) -> None:
        self.user_name = str(name or "User")
        with self._engines_lock:
            engines = list(self.engines.values())
        for engine in engines:
            engine.set_user_name(self.user_name)

    def unblock_user(self) -> None:
        """'Send Now' — unblock any user-turn wait in progress.

        When NO engine exists yet (an agent was created a moment ago and
        the monitor has not reconciled the engine set), the request is
        remembered and applied by sync_engines() the moment the engine
        is created — a Send Now that races engine creation is never
        silently lost."""
        with self._engines_lock:
            engines = list(self.engines.values())
            if not engines:
                self._pending_send_now = True
        for engine in engines:
            engine.unblock_user()

    def stop_agent(self, agent_name: str) -> bool:
        """Stop the current stream of one agent (sidebar Stop button)."""
        agent = self.manager.get_agent(agent_name)
        if agent is None:
            return False
        ep = agent.endpoint or "local"
        with self._engines_lock:
            engine = self.engines.get(ep)
        if engine is None:
            return False
        engine.stop_current()
        return True

    # ------------------------------------------------------------------
    # Completion detection
    # ------------------------------------------------------------------
    def _quiesced(self) -> bool:
        """True when a queue session has nothing left to do."""
        now = time.time()
        if now - self.started_at < self.app.config.done_grace:
            return False
        with self._engines_lock:
            engines = list(self.engines.values())
        if any(engine.get_running_agent() for engine in engines):
            return False
        with self.conversation._lock:
            if self.conversation.pending_user_messages:
                return False
        for agent in self.manager.agents.values():
            if bool(getattr(agent, "active", True)):
                return False
        return True

    def _all_endpoints_down_too_long(self) -> bool:
        with self._engines_lock:
            engines = list(self.engines.values())
        if not engines:
            return False
        limit = self.app.config.endpoint_down_timeout
        now = time.time()
        for engine in engines:
            down_since = engine._down_since
            if down_since is None or (now - down_since) < limit:
                return False
        return True

    def _monitor_loop(self) -> None:
        """Watch the session; finalize it when it is done."""
        while True:
            if self._abort_requested:
                self._finalize("aborted")
                return
            if self._finish_requested:
                # Graceful: wait for running turns to end, then finalize.
                with self._engines_lock:
                    engines = list(self.engines.values())
                if not any(engine.get_running_agent() for engine in engines):
                    self._finalize("finished")
                    return
            elif self.options.auto_finish and self._quiesced():
                self._finalize("completed")
                return
            elif (self.options.idle_timeout
                  and time.time() - self.last_activity > self.options.idle_timeout):
                self._finalize("idle_timeout")
                return
            elif self._all_endpoints_down_too_long():
                self._finalize("endpoints_unreachable")
                return

            # Periodic engine reconciliation (covers endpoints added by
            # core_create_agent / update_agent mid-session).
            if self._agents_dirty.is_set():
                self._agents_dirty.clear()
                self.sync_engines()
            time.sleep(0.5)

    # ------------------------------------------------------------------
    # Finishing
    # ------------------------------------------------------------------
    def request_finish(self) -> None:
        """Graceful finish: running turns complete, no new turns start."""
        if self.finished_at is not None:
            return
        self._finish_requested = True
        with self._engines_lock:
            engines = list(self.engines.values())
        for engine in engines:
            engine.request_finish()
        self.app.broadcast.publish("session", {
            "event": "finishing",
            "session_name": self.session_name,
        })

    def abort(self) -> None:
        """Hard stop: stop streams, finish immediately."""
        if self.finished_at is not None:
            return
        self._abort_requested = True
        with self._engines_lock:
            engines = list(self.engines.values())
        for engine in engines:
            engine.stop_current()
            engine.shutdown()

    def _throttled_autosave(self) -> None:
        """Mid-run auto-save (at most one per AUTOSAVE_MIN_INTERVAL)."""
        now = time.time()
        if now - self.last_autosave < self.app.config.autosave_min_interval:
            return
        self.last_autosave = now
        path = self.app.session_store.autosave(
            self.session_name, self.user_name,
            self.manager, self.conversation)
        if path:
            self.autosave_path = path
            self.app.broadcast.publish("session", {
                "event": "autosaved",
                "session_name": self.session_name,
                "path": path,
            })

    def _finalize(self, reason: str) -> None:
        """Stop engines, write the FINAL auto-save, mark done."""
        if self._finalized:
            return
        self._finalized = True
        self.finish_reason = reason
        try:
            self.manager.agents_changed.disconnect(self._on_agents_changed)
        except Exception:
            pass

        # Stop every engine.
        with self._engines_lock:
            engines = list(self.engines.values())
            self.engines.clear()
        for engine in engines:
            try:
                engine.shutdown()
            except Exception:
                pass
        for engine in engines:
            engine.join(timeout=5.0)

        # Final auto-save — "once one session is done - autosaves it".
        # Bypasses the throttle: this is the definitive end-of-session write.
        self.last_autosave = 0.0
        path = self.app.session_store.autosave(
            self.session_name, self.user_name,
            self.manager, self.conversation)
        if path:
            self.autosave_path = path

        self.finished_at = time.time()
        self.app.broadcast.publish("session", {
            "event": "finished",
            "session_name": self.session_name,
            "reason": reason,
            "path": path,
            "history_messages": len(self.conversation.history),
        })
        print(f"[runner] session '{self.session_name}' finished "
              f"({reason}); autosave: {path}")
        self._done_event.set()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def done(self) -> bool:
        return self._done_event.is_set()

    def wait_done(self, timeout: Optional[float] = None) -> bool:
        return self._done_event.wait(timeout)

    def summary(self) -> dict:
        return {
            "session_name": self.session_name,
            "source": self.options.source,
            "auto_finish": self.options.auto_finish,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "finish_reason": self.finish_reason,
            "autosave_path": self.autosave_path,
            "user_name": self.user_name,
            "history_messages": len(self.conversation.history),
            "agents": self.manager.list_agents(),
        }

    # ------------------------------------------------------------------
    # Chat history access (state API)
    # ------------------------------------------------------------------
    def history_snapshot(self) -> List[dict]:
        return self.conversation.get_history_snapshot()

    def pending_messages(self) -> List[str]:
        with self.conversation._lock:
            return list(self.conversation.pending_user_messages)
