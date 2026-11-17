"""
server/broadcast.py — the full-output event bus.

Every observable thing that happens inside the backend is published here
as a JSON-serialisable event dict:

    {"type": "thinking",  "data": {...}}   — live per-agent stream events
    {"type": "chat",      "data": {...}}   — chat messages (incl. user turns)
    {"type": "status",    "data": {...}}   — per-endpoint status changes
    {"type": "agent_status", "data": {...}}— agent running/idle flips
    {"type": "queue_updated", "data": {...}}
    {"type": "session",   "data": {...}}   — session lifecycle
    {"type": "injection", "data": {...}}
    {"type": "agents_changed", "data": {...}}
    {"type": "log",       "data": {...}}

Consumers:

  * WebSocket  /ws/output      (api_output.py)
  * SSE        /output/stream  (api_output.py)
  * the ring buffer below (reconnect support: GET /output/recent)

Threading model
===============
Events are published from ENGINE THREADS and worker threads (the same
threads that used to emit Qt signals in the PyQt version). publish()
therefore must be callable from ANY thread: it hands the event to the
bound asyncio loop via loop.call_soon_threadsafe(...), and the loop then
fans the event out into every subscriber's asyncio.Queue.

If no loop is bound yet (startup race) or the loop is already closed
(shutdown race), the event is only appended to the ring buffer — publish
NEVER raises and never blocks the caller.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set


# ----------------------------------------------------------------------
# Chat-event shape helpers
# ----------------------------------------------------------------------
def chat_event_role(sender: str, user_name: str = "User") -> str:
    """Presentation role for a live ``chat`` event, derived from its sender.

    Chat events historically carried only ``sender`` + ``text``, so
    frontends could not tell a user turn from an agent answer and
    rendered EVERYTHING with the system-note style (dashed, centered,
    muted) — while a history reload showed the same messages with
    proper user/assistant bubbles. The engines know the difference but
    the hook signature is sender-only, so we derive the role here using
    the exact same discriminators the engine uses when it calls
    ``hooks.message(sender, text)``:

        sender == "Error"     -> "error"      (turn errors)
        sender == "system"    -> "system"     (status notes, display-only)
        sender == user_name   -> "user"       (drained user turn)
        anything else         -> "assistant"  (agent answer / agent tool)

    The value is display-only metadata: the broadcast payload and the
    stored history are otherwise unchanged.
    """
    s = str(sender or "").strip()
    if s == "Error":
        return "error"
    if s.lower() == "system":
        return "system"
    un = str(user_name or "User").strip().lower()
    if s and un and s.lower() == un:
        return "user"
    return "assistant"


class BroadcastBus:
    """Thread-safe one-to-many event bus feeding asyncio consumers."""

    def __init__(self, ring_size: int = 1000):
        self._lock = threading.RLock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribers: Set[asyncio.Queue] = set()
        self._ring: Deque[dict] = deque(maxlen=max(1, int(ring_size)))
        self._seq = 0

    # ------------------------------------------------------------------
    # Loop binding (called once from the FastAPI lifespan)
    # ------------------------------------------------------------------
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._loop = loop

    def unbind_loop(self) -> None:
        with self._lock:
            self._loop = None

    # ------------------------------------------------------------------
    # Publishing (any thread, never raises, never blocks)
    # ------------------------------------------------------------------
    def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast one event. Callable from any thread."""
        with self._lock:
            self._seq += 1
            event = {
                "type": str(event_type),
                "seq": self._seq,
                "ts": round(time.time(), 3),
                "data": data if isinstance(data, dict) else {"value": data},
            }
            self._ring.append(event)
            loop = self._loop
            subs = list(self._subscribers)

        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(self._dispatch, event, subs)
            except RuntimeError:
                # Loop closed between the check and the call — swallow.
                pass

    def _dispatch(self, event: dict, subs: List[asyncio.Queue]) -> None:
        """Fan the event out to subscriber queues (runs on the loop)."""
        for q in subs:
            try:
                q.put_nowait(event)
            except Exception:
                # Full queue / closed consumer — drop for that consumer
                # only; a slow client must never stall the others.
                pass

    # ------------------------------------------------------------------
    # Subscription (async side)
    # ------------------------------------------------------------------
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=4096)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    # ------------------------------------------------------------------
    # Ring buffer (reconnect support)
    # ------------------------------------------------------------------
    def recent(self, limit: int = 200) -> List[dict]:
        with self._lock:
            items = list(self._ring)
        if limit and limit > 0:
            items = items[-limit:]
        return items
