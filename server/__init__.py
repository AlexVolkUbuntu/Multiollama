"""
server — the headless backend package for MultiLlama.

Layout:

    config.py        env-driven runtime configuration
    broadcast.py     thread-safe event bus (WebSocket / SSE fan-out)
    app_context.py   module bootstrap (tools, prompts, MCP, clients)
    session_store.py PyQt-compatible session save/load/autosave
    engine.py        per-endpoint round-robin scheduler (no Qt)
    runner.py        one active session: lifecycle + done detection
    queue_manager.py the session queue loop + staged injections
    state.py         full-state snapshots
    tcp_events.py    legacy :8888 event server (send_event compat)
    api_models.py    request models
    api_control.py   FULL-CONTROL endpoints
    api_queue.py     additional control endpoints (queue + inject)
    api_output.py    FULL-OUTPUT endpoints (WS + SSE + REST)
    app.py           FastAPI assembly
"""

from .app import create_app
from .app_context import AppContext

__all__ = ["create_app", "AppContext"]
