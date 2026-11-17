"""
server/queue_manager.py — the session queue (the core new behaviour).

Required flow (verbatim from the design brief):

    "must handle different sessions in queue — once one session is done —
     autosaves it — next session (from queue) loaded from autosave and
     ran — if no new — just wait"

This module implements exactly that loop on a dedicated daemon thread:

    while running:
        entry = next pending queue entry        (or WAIT when empty)
        if an interactive session is active:    (optional preemption)
            finish it gracefully, wait for its final autosave
        find the newest saved file for entry.session_name
            (autosave/ first, then sessions/)
        load it, apply entry options (activation overrides, initial
            message, idle timeout)
        run it with SessionRunner (auto-finish on quiesce)
        ── the runner auto-saves during the run AND a final autosave
           the moment the session is done ──
        record the result, broadcast it, pick the next entry (or wait)

Waiting is a real wait: an Event blocks the thread with zero CPU until
either a new entry is enqueued or shutdown is signalled.

The queue manager also owns the STAGED INJECTIONS registry — operator
messages destined for a specific (session_name, agent_name) that must
NOT enqueue that session (requirement 4.2). When the named session
eventually runs, the runner delivers them through the proven
tool_injection buffer (prepended to the agent's next tool result).
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Dict, List, Optional

from .runner import RunnerOptions, SessionRunner


def _now() -> float:
    return time.time()


class QueueEntry:
    """One requested session run."""

    __slots__ = ("id", "session_name", "message", "sender", "activate_all",
                 "activate", "idle_timeout", "auto_finish", "created_at",
                 "started_at", "finished_at", "status", "result",
                 "enqueued_by", "error")

    def __init__(self, session_name: str,
                 message: Optional[str] = None,
                 sender: str = "user",
                 activate_all: bool = False,
                 activate: Optional[List[str]] = None,
                 idle_timeout: Optional[float] = None,
                 auto_finish: bool = True,
                 enqueued_by: str = "api"):
        self.id = uuid.uuid4().hex[:12]
        self.session_name = str(session_name or "").strip()
        self.message = message
        self.sender = sender or "user"
        self.activate_all = bool(activate_all)
        self.activate = list(activate or [])
        self.idle_timeout = idle_timeout
        self.auto_finish = bool(auto_finish)
        self.created_at = _now()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.status = "pending"     # pending | running | completed | failed | cancelled
        self.result: Optional[str] = None
        self.enqueued_by = enqueued_by
        self.error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_name": self.session_name,
            "message": self.message,
            "sender": self.sender,
            "activate_all": self.activate_all,
            "activate": self.activate,
            "idle_timeout": self.idle_timeout,
            "auto_finish": self.auto_finish,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "result": self.result,
            "enqueued_by": self.enqueued_by,
            "error": self.error,
        }


class StagedInjection:
    """An operator message for (session_name, agent_name).

    Delivered through tool_injection's per-agent buffer when that
    session starts running. Enqueuing an injection NEVER adds the
    session to the run queue."""

    __slots__ = ("id", "session_name", "agent_name", "text",
                 "created_at", "delivered")

    def __init__(self, session_name: str, agent_name: str, text: str):
        self.id = uuid.uuid4().hex[:12]
        self.session_name = str(session_name or "").strip()
        self.agent_name = str(agent_name or "").strip()
        self.text = str(text or "").strip()
        self.created_at = _now()
        self.delivered = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_name": self.session_name,
            "agent_name": self.agent_name,
            "text": self.text,
            "created_at": self.created_at,
            "delivered": self.delivered,
        }


class SessionQueueManager:
    """Owns the session queue + the single active session runner."""

    def __init__(self, app):
        self.app = app
        self._lock = threading.RLock()
        self._queue: List[QueueEntry] = []
        self._recent: List[QueueEntry] = []          # last 50 finished
        self._staged: List[StagedInjection] = []
        self._wake = threading.Event()
        self._shutdown = False

        self.active_runner: Optional[SessionRunner] = None
        self._thread: Optional[threading.Thread] = None
        self._start_lock = threading.Lock()   # guards session starts
        self._idle_announced = False

        # Live (delivered-to-buffer) injection counts, for the state API.
        # Keys are agent names — same keys tool_injection uses.
        self._live_injections: Dict[str, int] = {}

    # ==================================================================
    # Queue manipulation (API side)
    # ==================================================================
    def enqueue(self, session_name: str, message: Optional[str] = None,
                sender: str = "user", activate_all: bool = False,
                activate: Optional[List[str]] = None,
                idle_timeout: Optional[float] = None,
                auto_finish: bool = True,
                enqueued_by: str = "api") -> QueueEntry:
        if not str(session_name or "").strip():
            raise ValueError("session_name is required")
        entry = QueueEntry(
            session_name=session_name, message=message, sender=sender,
            activate_all=activate_all, activate=activate,
            idle_timeout=idle_timeout, auto_finish=auto_finish,
            enqueued_by=enqueued_by)
        with self._lock:
            self._queue.append(entry)
        self._wake.set()
        self._broadcast_queue()
        return entry

    def queue_snapshot(self) -> List[dict]:
        with self._lock:
            return [e.to_dict() for e in self._queue]

    def recent_snapshot(self, limit: int = 50) -> List[dict]:
        with self._lock:
            return [e.to_dict() for e in self._recent[:limit]]

    def remove_entry(self, entry_id: str) -> Optional[QueueEntry]:
        with self._lock:
            for i, e in enumerate(self._queue):
                if e.id == entry_id:
                    removed = self._queue.pop(i)
                    removed.status = "cancelled"
                    removed.finished_at = _now()
                    self._recent.insert(0, removed)
                    self._trim_recent()
                    self._broadcast_queue()
                    return removed
        return None

    def clear_pending(self) -> int:
        with self._lock:
            count = len(self._queue)
            for e in self._queue:
                e.status = "cancelled"
                e.finished_at = _now()
                self._recent.insert(0, e)
            self._queue.clear()
            self._trim_recent()
        if count:
            self._broadcast_queue()
        return count

    def move_entry(self, entry_id: str, position: int) -> Optional[QueueEntry]:
        """Move one pending entry to *position* (0 = next to run)."""
        with self._lock:
            idx = next((i for i, e in enumerate(self._queue)
                        if e.id == entry_id), None)
            if idx is None:
                return None
            entry = self._queue.pop(idx)
            position = max(0, min(int(position), len(self._queue)))
            self._queue.insert(position, entry)
        self._broadcast_queue()
        return entry

    # ==================================================================
    # Staged injections (requirement 4.2 — never enqueues a session)
    # ==================================================================
    def stage_injection(self, session_name: str, agent_name: str,
                        text: str) -> StagedInjection:
        session_name = str(session_name or "").strip()
        agent_name = str(agent_name or "").strip()
        text = str(text or "").strip()
        if not session_name:
            raise ValueError("session_name is required")
        if not agent_name:
            raise ValueError("agent_name is required")
        if not text:
            raise ValueError("text is required")

        runner = self.active_runner
        if runner is not None and runner.session_name == session_name:
            # The session is running RIGHT NOW: deliver live through the
            # proven buffer (delivered prepended to the agent's next
            # tool result).
            import tool_injection
            count = tool_injection.add_injection(agent_name, text)
            with self._lock:
                self._live_injections[agent_name] = count
            staged = StagedInjection(session_name, agent_name, text)
            staged.delivered = True
            self.app.broadcast.publish("injection", {
                "session_name": session_name,
                "agent_name": agent_name,
                "count": count,
                "staged": False,
                "live": True,
                "text": text,
            })
            return staged

        staged = StagedInjection(session_name, agent_name, text)
        with self._lock:
            self._staged.append(staged)
        self.app.broadcast.publish("injection", {
            "session_name": session_name,
            "agent_name": agent_name,
            "staged": True,
            "live": False,
            "text": text,
        })
        return staged

    def staged_snapshot(self) -> List[dict]:
        with self._lock:
            return [s.to_dict() for s in self._staged]

    def remove_staged(self, injection_id: str) -> Optional[StagedInjection]:
        with self._lock:
            for i, s in enumerate(self._staged):
                if s.id == injection_id:
                    return self._staged.pop(i)
        return None

    def take_staged_for_session(self, session_name: str) -> List[dict]:
        """Drain (remove and return) staged injections for a session."""
        with self._lock:
            remaining = []
            taken = []
            for s in self._staged:
                if s.session_name == session_name:
                    s.delivered = True
                    taken.append({
                        "agent_name": s.agent_name,
                        "text": s.text,
                        "id": s.id,
                    })
                else:
                    remaining.append(s)
            self._staged = remaining
        return taken

    def live_injection_counts(self) -> Dict[str, int]:
        """Pending injection counts per agent (live buffer + mirrors)."""
        import tool_injection
        counts = {}
        # The tool_injection buffer is GLOBAL and keyed by agent name —
        # report every agent that has anything pending (the proven
        # semantics: the buffer survives until that agent's next tool
        # result, whichever session it runs in).
        try:
            counts.update(tool_injection.all_pending_counts())
        except Exception:
            pass
        return counts

    def clear_live_injection(self, agent_name: str) -> int:
        import tool_injection
        dropped = tool_injection.clear_injections(agent_name)
        with self._lock:
            self._live_injections.pop(agent_name, None)
        return dropped

    # ==================================================================
    # Interactive sessions (created via /control/session/new or load)
    # ==================================================================
    def start_interactive(self, session_data: dict, session_name: str,
                          wait_user_input: bool = True) -> SessionRunner:
        """Start (or replace) an INTERACTIVE session immediately.

        Refused when a QUEUED session is currently running (the queue
        owns scheduling in that case — finish it first). If another
        interactive session is active it is finished gracefully first
        (final autosave included). There is always at most ONE active
        session.

        *wait_user_input* is the initial state of the 'Pause User Input'
        checkbox — True by default, mirroring the proven PyQt default
        ("wait user turn"): agents hold their turns until the user sends
        a message / presses Send Now, instead of chatting away the
        moment the session is created. Pass False (or use the API's
        ``wait_user_input: false`` body field) to start chatting
        immediately. Queue entries always run with it OFF.
        """
        with self._start_lock:
            current = self.active_runner
            if current is not None and not current.done:
                if current.options.source == "queue":
                    raise RuntimeError(
                        "a queued session is currently running "
                        f"('{current.session_name}'); finish or abort it "
                        "before starting an interactive session")
                self._retire_active("superseded",
                                    timeout=self.app.config.preempt_timeout)
            options = RunnerOptions(source="interactive", auto_finish=False,
                                    wait_user_input=bool(wait_user_input))
            runner = SessionRunner(
                self.app, session_data, session_name, options,
                staged_injections=self.take_staged_for_session(session_name))
            self.active_runner = runner
            self._idle_announced = False
            runner.start()
            # Wake the queue loop so it notices this session immediately
            # (it may be parked in the idle wait).
            self._wake.set()
            return runner

    # ==================================================================
    # The queue loop
    # ==================================================================
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop, name="session-queue", daemon=True)
        self._thread.start()

    def shutdown(self, abort_active: bool = True) -> None:
        self._shutdown = True
        self._wake.set()
        runner = self.active_runner
        if runner is not None and not runner.done:
            if abort_active:
                runner.abort()
            else:
                runner.request_finish()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def _run_loop(self) -> None:
        """Dequeues sessions one at a time; waits when the queue is empty."""
        while not self._shutdown:
            entry = self._next_entry()
            if entry is None:
                # A finished session (e.g. an interactive one the operator
                # ended via /control/session/finish) must not linger as
                # "active" — clear it before idling.
                if (self.active_runner is not None
                        and self.active_runner.done):
                    self.active_runner = None

                # Nothing pending: if an interactive session is running,
                # let it run; otherwise just WAIT for a new arrival.
                if self.active_runner is not None and not self.active_runner.done:
                    self._wake.wait(timeout=1.0)
                    self._wake.clear()
                    # An arrival while an interactive session runs:
                    # preempt it (it gets auto-saved, not lost).
                    if (not self._shutdown and self._has_pending()
                            and self.app.config.preempt_interactive):
                        self._retire_active(
                            "preempted",
                            timeout=self.app.config.preempt_timeout)
                    continue
                # Truly idle — wait for something to happen (short poll
                # so a session started by an API thread is picked up
                # promptly and finished ones get cleared).
                if not self._idle_announced:
                    self._idle_announced = True
                    self.app.broadcast.publish("session", {
                        "event": "idle",
                        "detail": "queue empty — waiting for sessions",
                    })
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue

            # A queue entry exists — retire any interactive session first.
            if (self.active_runner is not None
                    and not self.active_runner.done):
                self._retire_active("preempted",
                                    timeout=self.app.config.preempt_timeout)

            with self._start_lock:
                if self._shutdown:
                    break
                self._run_entry(entry)

        # Final cleanup on shutdown.
        runner = self.active_runner
        if runner is not None and not runner.done:
            runner.abort()
            runner.wait_done(timeout=15.0)

    def _next_entry(self) -> Optional[QueueEntry]:
        with self._lock:
            for entry in self._queue:
                if entry.status == "pending":
                    entry.status = "running"
                    entry.started_at = _now()
                    return entry
        return None

    def _has_pending(self) -> bool:
        with self._lock:
            return any(e.status == "pending" for e in self._queue)

    def _run_entry(self, entry: QueueEntry) -> None:
        """Load one session from disk and run it to completion."""
        store = self.app.session_store
        bus = self.app.broadcast

        # ---- locate the session's newest saved file --------------------
        path = store.find_latest_for_name(entry.session_name)
        if path is None:
            entry.status = "failed"
            entry.error = (f"no saved session named '{entry.session_name}' "
                           f"found in autosave/ or sessions/")
            entry.finished_at = _now()
            self._record_finished(entry)
            bus.publish("session", {
                "event": "failed",
                "session_name": entry.session_name,
                "error": entry.error,
                "entry": entry.to_dict(),
            })
            print(f"[queue] FAILED: {entry.error}")
            return

        # ---- load --------------------------------------------------------
        try:
            data = store.load_from_path(path)
        except Exception as exc:
            entry.status = "failed"
            entry.error = f"could not load {path}: {exc}"
            entry.finished_at = _now()
            self._record_finished(entry)
            bus.publish("session", {
                "event": "failed",
                "session_name": entry.session_name,
                "error": entry.error,
                "entry": entry.to_dict(),
            })
            return

        session_name = (str(data.get("session_name") or ""
                           ).strip() or entry.session_name)
        self._idle_announced = False

        # ---- run ---------------------------------------------------------
        options = RunnerOptions(
            source="queue",
            auto_finish=entry.auto_finish,
            message=entry.message,
            sender=entry.sender,
            activate_all=entry.activate_all,
            activate=entry.activate,
            idle_timeout=entry.idle_timeout,
        )
        runner = SessionRunner(
            self.app, data, session_name, options,
            staged_injections=self.take_staged_for_session(session_name))
        self.active_runner = runner
        runner.start()
        runner.wait_done()

        # ---- record ------------------------------------------------------
        entry.status = ("completed"
                        if runner.finish_reason in ("completed", "finished")
                        else "completed" if runner.finish_reason else "failed")
        if runner.finish_reason in ("aborted", "endpoints_unreachable"):
            entry.status = "failed"
            entry.error = f"finished with reason: {runner.finish_reason}"
        entry.result = runner.finish_reason
        entry.finished_at = _now()
        self.active_runner = None
        self._record_finished(entry)
        bus.publish("session", {
            "event": "queue_entry_done",
            "session_name": session_name,
            "entry": entry.to_dict(),
            "runner": runner.summary(),
        })
        print(f"[queue] entry '{session_name}' done "
              f"({runner.finish_reason}); next…")

    def _retire_active(self, reason: str, timeout: float) -> None:
        """Gracefully finish the active (interactive) session."""
        runner = self.active_runner
        if runner is None or runner.done:
            self.active_runner = None
            return
        runner.request_finish()
        if not runner.wait_done(timeout=timeout):
            print(f"[queue] active session '{runner.session_name}' did not "
                  f"finish within {timeout}s — aborting.")
            runner.abort()
            runner.wait_done(timeout=15.0)
        self.active_runner = None

    def _record_finished(self, entry: QueueEntry) -> None:
        with self._lock:
            # Keep only entries that are no longer pending/running.
            self._queue = [e for e in self._queue if e.id != entry.id]
            self._recent.insert(0, entry)
            self._trim_recent()
        self._broadcast_queue()

    def _trim_recent(self) -> None:
        self._recent = self._recent[:50]

    def _broadcast_queue(self) -> None:
        self.app.broadcast.publish("queue_updated", {
            "session_queue": self.queue_snapshot(),
            "pending_user_messages": (
                self.active_runner.pending_messages()
                if self.active_runner else []),
        })
