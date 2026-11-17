"""
server/api_output.py — the FULL-OUTPUT endpoints.

One connection gives a frontend EVERYTHING:

  * on connect          -> a complete state snapshot
                          (all current state: server, session, agents,
                           endpoints, queue, injections, tools, prompts,
                           mcp, model usage, sessions on disk)
  * then, live          -> every event as it happens:
                          chat messages, per-agent streaming tokens and
                          thinking, full prompt dumps, tool calls and
                          (composed) tool results, endpoint status,
                          agent running/idle flips, queue updates,
                          session lifecycle, injections, agent changes

Transports
==========
  WebSocket   /ws/output        primary (browser, mobile, other backends)
  SSE         /output/stream    fallback for simple HTTP clients
  REST        /state            one-shot full state
  REST        /output/recent    ring-buffer replay (reconnect support)
  REST        /output/history   chat history of the active session
              (chat-view: user / agent answers / system notes; tool
               traffic filtered — include_tools=true for the raw
               context; tail=N keeps the newest N)

Protocol (WebSocket)
====================
Server -> client: JSON frames
    {"type": "snapshot", "seq": 0, "ts": ..., "data": {full state}}
    {"type": "thinking", "seq": N, "ts": ..., "data": {
        endpoint, agent, event_type, detail, extra}}
    ... (see server/broadcast.py for the event vocabulary)

Client -> server: JSON frames (all optional)
    {"action": "refresh"}                 -> re-send a full snapshot
    {"action": "ping"}                    -> {"type": "pong"}
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .api_control import require_api_key
from .media_links import enrich_history_messages
from .state import build_full_state

router = APIRouter(tags=["output"])


# ======================================================================
# Chat-view filter for /output/history
# ======================================================================
def chat_visible_messages(history: list) -> list:
    """The CHAT-DIALOGUE subset of the shared history.

    The shared history IS the LLM context, so it also stores the raw
    tool traffic: ``role:"tool"`` results and content-less assistant
    tool-call REQUESTS. The live chat broadcast never shows those
    (engine.py broadcasts only user turns, agent ANSWERS and system
    notes) — but /output/history used to return the raw list, so the
    UI's "Reload history" button painted tool outputs as chat
    bubbles while the live view looked correct. This filter mirrors
    the live rule EXACTLY, so reload and live show the same chat:

      * role "user"                                -> kept
      * role "assistant" with non-empty content    -> kept
        (the same gate the engine uses before broadcasting;
         a message that carries text AND tool_calls was
         broadcast live too, so it stays visible here)
      * role "system"                              -> kept
        (TCP events / legacy notes enter history as system)
      * role "tool", empty tool-call requests,
        anything else                              -> dropped

    The dicts are shared with the conversation (read-only contract)
    and are NEVER mutated here — only selected.
    """
    out = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").lower()
        if role == "user":
            out.append(msg)
        elif role == "assistant":
            if str(msg.get("content") or "").strip():
                out.append(msg)
        elif role == "system":
            out.append(msg)
    return out


def _authed(app, token: Optional[str]) -> bool:
    key = app.config.api_key
    if not key:
        return True
    return token == key


# ======================================================================
# REST: one-shot state + replay buffers + history
# ======================================================================
@router.get("/state")
def full_state(request: Request, app=Depends(require_api_key)):
    """The complete current state (same payload the WS sends on connect)."""
    return build_full_state(app)


@router.get("/output/recent")
def recent_events(limit: int = Query(default=200, ge=1, le=5000),
                  app=Depends(require_api_key)):
    """Recent events from the ring buffer — reconnect gap filler."""
    return {"events": app.broadcast.recent(limit=limit)}


@router.get("/output/history")
def chat_history(offset: int = Query(default=0, ge=0),
                 limit: int = Query(default=0, ge=0),
                 tail: int = Query(default=0, ge=0),
                 include_tools: bool = Query(default=False),
                 app=Depends(require_api_key)):
    """Chat history of the active session (newest last) — the CHAT view.

    Default output is the dialogue a human sees in the chat tab: user
    messages, agent ANSWERS and system notes — exactly the set the
    engines broadcast live. The raw LLM-context traffic (tool-call
    requests, tool results) is filtered OUT; pass include_tools=true
    for the unfiltered context (debugging / power tools).

    Slicing — applied to the (filtered) list, in this order:
      tail    > 0  -> only the NEWEST ``tail`` messages. This is what
                      a chat "Reload history" wants: with plain limit,
                      long tool-heavy sessions kept their OLDEST N
                      messages and silently hid the recent chat end.
      offset       -> skip the oldest ``offset`` messages (paging)
      limit   > 0  -> at most ``limit`` messages from there
    limit=0 & tail=0 returns everything after offset.

    ``total`` counts the FILTERED messages (consistent with the
    slicing above); ``total_raw`` counts everything stored, including
    tool traffic — a quick way to notice hidden context weight.

    Every message that mentions a real local file ALSO carries a
    "files" array with ready-made download links (media_links scan —
    copies of the stored dicts; the history / LLM context itself is
    never modified, and thinking was never stored here to begin
    with, so the scan covers exactly the final answers).
    """
    runner = app.queue_manager.active_runner
    if runner is None:
        return {"active": False, "messages": [], "total": 0, "total_raw": 0}
    history = runner.history_snapshot()
    total_raw = len(history)
    if not include_tools:
        history = chat_visible_messages(history)
    total = len(history)
    if tail:
        history = history[-tail:]
    if offset:
        history = history[offset:]
    if limit:
        history = history[:limit]
    messages = enrich_history_messages(history)
    return {"active": True, "session_name": runner.session_name,
            "total": total, "total_raw": total_raw,
            "offset": offset, "messages": messages}


# ======================================================================
# Server-Sent Events fallback
# ======================================================================
@router.get("/output/stream")
async def sse_stream(request: Request, token: Optional[str] = None):
    app = request.app.state.app
    if not _authed(app, token):
        raise HTTPException(status_code=401, detail="invalid or missing token")

    queue = app.broadcast.subscribe()

    async def event_source():
        try:
            # initial snapshot
            snapshot = {"type": "snapshot", "seq": 0,
                        "ts": __import__("time").time(),
                        "data": build_full_state(app)}
            yield f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    # keepalive comment — proxies love them
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            app.broadcast.unsubscribe(queue)

    return StreamingResponse(event_source(),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ======================================================================
# WebSocket — the primary full-output channel
# ======================================================================
@router.websocket("/ws/output")
async def ws_output(websocket: WebSocket, token: Optional[str] = None):
    app = websocket.app.state.app
    if not _authed(app, token):
        await websocket.close(code=4401, reason="invalid or missing token")
        return

    await websocket.accept()
    queue = app.broadcast.subscribe()

    async def _send_snapshot():
        await websocket.send_text(json.dumps({
            "type": "snapshot",
            "seq": 0,
            "ts": __import__("time").time(),
            "data": build_full_state(app),
        }, ensure_ascii=False))

    try:
        await _send_snapshot()

        receiver = asyncio.ensure_future(_receive_loop(websocket))
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    # periodic keepalive ping keeps intermediaries honest
                    await websocket.send_text(json.dumps({"type": "ping"}))
                    continue
                await websocket.send_text(
                    json.dumps(event, ensure_ascii=False))
        finally:
            receiver.cancel()
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as exc:
        print(f"[ws-output] error: {exc}")
    finally:
        app.broadcast.unsubscribe(queue)
        try:
            await websocket.close()
        except Exception:
            pass


async def _receive_loop(websocket: WebSocket):
    """Handle optional client->server control frames."""
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                frame = json.loads(raw)
            except Exception:
                continue
            action = str(frame.get("action", "")).lower()
            if action == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif action == "refresh":
                app = websocket.app.state.app
                await websocket.send_text(json.dumps({
                    "type": "snapshot",
                    "seq": 0,
                    "ts": __import__("time").time(),
                    "data": build_full_state(app),
                }, ensure_ascii=False))
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        pass
