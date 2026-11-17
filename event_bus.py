"""
event_bus.py — Qt-free signal/event replacement for the headless backend.

WHY THIS MODULE EXISTS
======================

The proven PyQt application used Qt signals (pyqtSignal / .emit()) for
cross-thread event delivery:

  * AgentManager.agents_changed        — fired on every agent CRUD change
  * BackendThread.thinking_update etc. — fired from backend worker threads

The headless backend must run on machines WITHOUT PyQt5 installed
(Linux servers, Windows boxes, containers), so the same one-to-many
"connect / emit" contract is re-implemented here with pure Python.

The API is deliberately a subset of Qt's:

    sig = Signal()
    sig.connect(callable)      # subscribe (a plain function / method)
    sig.disconnect(callable)   # unsubscribe
    sig.emit(*args, **kwargs)  # synchronous, thread-safe fan-out

Differences from Qt signals (by design):

  * Delivery is SYNCHRONOUS on the emitting thread. In the backend,
    every listener used here is itself a thin, thread-safe forwarder
    (the BroadcastBus publisher), so this matches the Qt behaviour of
    "queued connection onto the server loop" closely enough.
  * Listener exceptions are swallowed (printed) — one broken subscriber
    must never break the emitter, exactly like Qt.
  * A listener may be connected multiple times; it will then be called
    once per registration (same as Qt).

The proven core modules that called ``manager.agents_changed.emit()``
(the generated agent_tools.py among them) continue to work VERBATIM,
because Signal exposes the same ``.emit()`` method name.
"""

import threading
from typing import Any, Callable, List


class Signal:
    """A minimal, thread-safe, synchronous pub/sub signal."""

    __slots__ = ("_listeners", "_lock")

    def __init__(self):
        self._listeners: List[Callable[..., Any]] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------
    def connect(self, listener: Callable[..., Any]) -> None:
        """Register *listener*; it will be called with the emit args."""
        if not callable(listener):
            raise TypeError("Signal.connect expects a callable")
        with self._lock:
            self._listeners.append(listener)

    def disconnect(self, listener: Callable[..., Any]) -> bool:
        """Remove one registration of *listener*. True when removed."""
        with self._lock:
            try:
                self._listeners.remove(listener)
                return True
            except ValueError:
                return False

    def disconnect_all(self) -> None:
        """Remove every listener."""
        with self._lock:
            self._listeners.clear()

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------
    def emit(self, *args: Any, **kwargs: Any) -> None:
        """Call every listener synchronously with the given arguments.

        Exceptions inside a listener are printed and swallowed — the
        emitter must never be taken down by a subscriber (this mirrors
        how Qt isolates slot errors from the emitting thread).
        """
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — deliberate
                print(f"[event_bus] listener {getattr(listener, '__name__', listener)!r} "
                      f"raised: {exc}")

    # Qt-ish alias — some call sites use the property-style accessor.
    @property
    def listeners(self) -> List[Callable[..., Any]]:
        with self._lock:
            return list(self._listeners)
