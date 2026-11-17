"""
server/engine.py — the headless equivalent of the proven BackendThread.

One EngineThread runs per distinct Ollama endpoint and executes the
EXACT same round-robin turn loop the PyQt app used:

    [user] + [this endpoint's agents]  ->  round-robin  ->  one turn

The turn logic itself (_handle_agent_turn) is a line-for-line port of
gui_main.BackendThread._handle_agent_turn with exactly two changes:

  1. Qt signals (thinking_update / message_ready / status_changed /
     agent_status_changed / queue_updated / autosave_requested /
     agent_done) are replaced by calls into an EngineHooks object —
     plain callables that are safe to invoke from this thread.
  2. There is no initial "wait for the user to press Send Now" gate:
     in the headless backend a session is started precisely when it
     should run (the queue manager or the /control/session/new call
     decided that already).

Everything else is preserved on purpose — the endpoint pause-gate
coordination for remote core_agent_execute runs, the inactive/busy
skipping, ensure_ready() health checks, the silent-agent rules, the
autosave-before-turn safety net and the error reporting into the chat
channel.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

import context


def endpoint_label(endpoint: str) -> str:
    """Human-readable label for an endpoint (same as gui_main)."""
    if not endpoint or endpoint == "local":
        return "local"
    return endpoint


class EngineHooks:
    """Callback bundle the runner installs on every engine.

    Every hook is optional at construction time (missing attributes fall
    back to no-ops) and every hook MUST be safe to call from the engine
    thread — the runner's hooks only touch thread-safe structures
    (BroadcastBus.publish, throttled autosave under a lock).
    """

    def thinking(self, endpoint: str, agent_name: str, event_type: str,
                 detail: str, extra: str) -> None:
        """Live per-agent stream event (thinking/stream/prompt/tool_call/
        tool_result/error/status/done/stream_end — same vocabulary the
        PyQt Thinking tab consumed)."""

    def message(self, sender: str, text: str) -> None:
        """A chat message should be displayed (user turn, agent answer,
        system note)."""

    def status(self, endpoint: str, text: str, color: str) -> None:
        """Endpoint status line changed (Running: X / Ready / ...)."""

    def agent_status(self, endpoint: str, agent_name: str,
                     running: bool, model: str) -> None:
        """An agent flipped between idle and running."""

    def queue_updated(self) -> None:
        """The pending-user-message queue changed."""

    def autosave(self, agent_name: str) -> None:
        """An agent turn is about to start — write a throttled auto-save."""

    def turn_started(self, endpoint: str, agent_name: str) -> None:
        """An agent turn started (activity marker)."""

    def turn_finished(self, endpoint: str, agent_name: str) -> None:
        """An agent turn finished (activity marker)."""

    def endpoint_unavailable(self, endpoint: str) -> None:
        """ensure_ready() failed on this endpoint (health marker)."""

    def endpoint_ready_again(self, endpoint: str) -> None:
        """ensure_ready() succeeded again after a failure streak."""


class EngineThread(threading.Thread):
    """Round-robin inference scheduler for ONE Ollama endpoint."""

    def __init__(self, manager, loader, conversation, client,
                 endpoint: str, agents: List[str],
                 hooks: EngineHooks,
                 user_name: str = "User",
                 user_paused: bool = False):
        super().__init__(name=f"engine-{endpoint_label(endpoint)}", daemon=True)
        self.manager = manager
        self.loader = loader
        self.conversation = conversation
        self.client = client              # OllamaClient bound to this endpoint
        self.endpoint = endpoint
        self.hooks = hooks
        self._agents: List[str] = list(agents)
        self._agents_lock = threading.Lock()
        self.user_name = user_name
        self._shutdown = False
        self._finish_requested = False      # graceful: end after current turn
        self._user_send_event = threading.Event()
        self._user_paused = user_paused
        self._current_agent_name: Optional[str] = None
        self._turn_index = 0
        # Health tracking: when did this endpoint last succeed / fail?
        # The session runner uses these to detect "all endpoints down".
        self._last_ready_ok: float = time.time()
        self._down_since: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API (used by the session runner / control API)
    # ------------------------------------------------------------------
    @property
    def agents(self) -> List[str]:
        with self._agents_lock:
            return list(self._agents)

    def update_agents(self, agents: List[str]) -> None:
        """Atomically swap this endpoint's agent list; effective from the
        next loop iteration (same contract as BackendThread)."""
        with self._agents_lock:
            self._agents = list(agents)

    def get_running_agent(self) -> Optional[str]:
        return self._current_agent_name

    @property
    def endpoint_down(self) -> bool:
        """True while ensure_ready() keeps failing (health marker)."""
        return self._down_since is not None

    def set_user_name(self, name: str) -> None:
        self.user_name = name

    def set_user_paused(self, paused: bool) -> None:
        """Toggle the pause mode (skip-user / wait-for-user). Turning
        pause OFF also unblocks a thread waiting on a user turn."""
        self._user_paused = paused
        if not paused:
            self._user_send_event.set()

    def unblock_user(self) -> None:
        """'Send Now' — unblock any user-turn wait in progress."""
        self._user_send_event.set()

    def add_to_queue(self, text: str) -> None:
        with self.conversation._lock:
            self.conversation.pending_user_messages.append(text)
        self.hooks.queue_updated()

    def clear_queue(self) -> None:
        with self.conversation._lock:
            self.conversation.pending_user_messages.clear()
        self.hooks.queue_updated()

    def stop_current(self) -> None:
        """Stop the currently running agent's stream on this endpoint."""
        self.client.stop()

    def request_finish(self) -> None:
        """Graceful end-of-session: stop STARTING new turns only.

        Unlike shutdown() this NEVER interrupts the turn that is
        currently streaming — the loop simply exits after it finishes
        (or immediately when idle). User-pause waits are unblocked so a
        parked engine can observe the flag and exit.
        """
        self._finish_requested = True
        self._user_send_event.set()

    def shutdown(self) -> None:
        """Hard stop: also signal the in-flight stream to cancel."""
        self._shutdown = True
        self._finish_requested = True
        self._user_send_event.set()
        self.client.stop()

    # ------------------------------------------------------------------
    # Main loop — the proven BackendThread.run() port
    # ------------------------------------------------------------------
    def run(self) -> None:
        while not self._shutdown and not self._finish_requested:
            try:
                # Drain pending user messages into shared history.
                # The first engine to grab the lock wins; others see an
                # empty queue (the history is shared) — proven behaviour.
                combined = self.conversation.drain_pending(self.user_name)
                if combined:
                    self.hooks.message(self.user_name, combined)
                    self.hooks.queue_updated()

                agents = self.agents
                if not agents:
                    time.sleep(0.1)
                    continue

                # Round-robin: user + this machine's agents.
                participants = ["user"] + agents
                participant = participants[self._turn_index % len(participants)]
                self._turn_index += 1

                # Regular pause: only block user turns.
                if self._user_paused and participant == "user":
                    self.hooks.status(self.endpoint, "Waiting for user...", "orange")
                    self._user_send_event.wait()
                    self._user_send_event.clear()
                    if self._shutdown or self._finish_requested:
                        break
                    time.sleep(0.1)
                    continue

                if participant == "user":
                    self._handle_user_turn()
                else:
                    self._handle_agent_turn(participant)
            except Exception as exc:
                print(f"Turn loop error on {self.endpoint}: {exc}")
                self.hooks.message(
                    "system",
                    f"[{endpoint_label(self.endpoint)} loop error: {exc}]")
                time.sleep(1)

    def _handle_user_turn(self) -> None:
        time.sleep(0.1)

    def _handle_agent_turn(self, agent_name: str) -> None:
        agent = self.manager.get_agent(agent_name)
        if agent is None:
            time.sleep(0.2)
            return

        # --- active flag: inactive agents are skipped by the scheduler --
        if not bool(getattr(agent, "active", True)):
            time.sleep(0.05)
            return

        # --- busy flag: agent is run inside core_agent_execute now ------
        if getattr(agent, "_busy", False):
            time.sleep(0.1)
            return

        if not self.client.ensure_ready():
            now = time.time()
            if self._down_since is None:
                self._down_since = now
                self.hooks.endpoint_unavailable(self.endpoint)
            self.hooks.message(
                "system",
                f"[Ollama at {endpoint_label(self.endpoint)} not ready, "
                f"skipping turn for {agent_name}]"
            )
            self.hooks.status(self.endpoint, "Ollama unavailable", "red")
            time.sleep(1)
            return
        if self._down_since is not None:
            self._down_since = None
            self.hooks.endpoint_ready_again(self.endpoint)
        self._last_ready_ok = time.time()

        # --- remote-execution pause gate (proven EndpointGate logic) -----
        gate = context.get_endpoint_gate(self.endpoint)
        if gate.is_paused():
            self.hooks.status(
                self.endpoint,
                f"Waiting (remote agent on {endpoint_label(self.endpoint)})",
                "orange"
            )
            # Not running anything — mark quiesced, then wait.
            gate.mark_quiesced()
            while gate.is_paused() and not self._shutdown \
                    and not self._finish_requested:
                time.sleep(0.05)
            if self._shutdown or self._finish_requested:
                return
        gate.begin_turn()

        self.hooks.status(self.endpoint, f"Running: {agent_name}", "blue")
        self._current_agent_name = agent_name
        self.hooks.agent_status(self.endpoint, agent_name, True, agent.model)
        self.hooks.turn_started(self.endpoint, agent_name)

        # Auto-save BEFORE the agent starts (crash safety net) — the
        # runner throttles this exactly like the GUI did.
        self.hooks.autosave(agent_name)

        def thinking_callback(event_type, agent_name_cb, detail, extra):
            self.hooks.thinking(self.endpoint, agent_name_cb,
                                event_type, detail, extra)

        # Capture the silent flag BEFORE the call (proven behaviour).
        agent_silent = bool(getattr(agent, "silent", False))

        try:
            history = self.conversation.get_history_snapshot()
            new_msgs = agent.call(history, self.loader, self.client,
                                  thinking_callback)

            with self.conversation._lock:
                if not agent_silent:
                    self.conversation.add_messages(new_msgs)

            # Chat-output behaviour — non-silent agents' answers are
            # emitted to the chat channel (the broadcast bus).
            if new_msgs and not agent_silent:
                for msg in new_msgs:
                    if msg["role"] == "assistant" and msg["content"].strip():
                        self.hooks.message(msg["sender"], msg["content"])
            elif not new_msgs and not agent_silent:
                self.hooks.message("system",
                                   f"[{agent_name} returned no response]")
        except Exception as exc:
            error_msg = f"Error during {agent_name}: {str(exc)[:200]}"
            print(error_msg)
            # Errors are always shown — even for silent agents.
            self.hooks.message("Error", error_msg)
        finally:
            gate.end_turn()
            self._current_agent_name = None
            self.hooks.agent_status(self.endpoint, agent_name, False, "")
            self.hooks.status(self.endpoint, "Ready", "green")
            self.hooks.turn_finished(self.endpoint, agent_name)
            time.sleep(0.1)
