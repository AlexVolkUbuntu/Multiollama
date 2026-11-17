"""
server/api_queue.py — the ADDITIONAL CONTROL endpoints (design brief):

  1) POST /queue/sessions
     "session.name to run and message to send before it"
        -> adds the session to the queue of sessions. The queue manager
           loads it from its newest auto-save when its turn comes,
           queues the message as a user message BEFORE the engines
           start, runs the session to completion (quiesce), auto-saves
           it and moves to the next entry.

  2) POST /queue/inject
     "session.name, agent name — to inject message to next tool output
      (not cause adding session to queue)"
        -> when the named session is RUNNING: goes live through the
           proven tool_injection buffer (prepended to that agent's next
           tool result). When it is NOT running: staged and delivered
           automatically the next time that session runs. Either way the
           session is NEVER added to the run queue by this endpoint.

Supporting endpoints for queue inspection / management:
  GET    /queue/sessions           queue + active + recent results
  DELETE /queue/sessions/{id}      cancel a pending entry
  PATCH  /queue/sessions/{id}/move reorder (0 = runs next)
  DELETE /queue/sessions           clear all pending entries
  GET    /queue/inject             list staged injections
  DELETE /queue/inject/{id}        remove a staged injection
  GET    /queue/state              queue-focused state block
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from .api_models import (
    QueueInjectRequest, QueueMoveRequest, QueueSessionRequest,
)
from .api_control import require_api_key
from .state import build_full_state

router = APIRouter(prefix="/queue", tags=["queue"])


def _queue_state(app) -> dict:
    state = build_full_state(app)
    return {
        "session_queue": state["session_queue"],
        "recent_queue": state["recent_queue"],
        "active_session": state["active_session"],
        "staged_injections": state["staged_injections"],
        "live_injections": state["live_injections"],
    }


@router.get("/state")
def queue_state(app=Depends(require_api_key)):
    return _queue_state(app)


# ======================================================================
# 1) Enqueue a session run
# ======================================================================
@router.post("/sessions")
def enqueue_session(body: QueueSessionRequest, app=Depends(require_api_key)):
    """Requirement 4.1 — session name + message to send before it."""
    try:
        entry = app.queue_manager.enqueue(
            session_name=body.session_name,
            message=body.message,
            sender=body.sender,
            activate_all=body.activate_all,
            activate=body.activate,
            idle_timeout=body.idle_timeout,
            auto_finish=body.auto_finish,
            enqueued_by="api",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "entry": entry.to_dict(),
            "queue": app.queue_manager.queue_snapshot()}


@router.get("/sessions")
def list_queue(app=Depends(require_api_key)):
    qm = app.queue_manager
    active = None
    if qm.active_runner is not None:
        active = qm.active_runner.summary()
    return {
        "queue": qm.queue_snapshot(),
        "active": active,
        "recent": qm.recent_snapshot(limit=50),
        "staged_injections": qm.staged_snapshot(),
        "live_injections": qm.live_injection_counts(),
    }


@router.delete("/sessions")
def clear_queue(app=Depends(require_api_key)):
    count = app.queue_manager.clear_pending()
    return {"ok": True, "cancelled": count}


@router.delete("/sessions/{entry_id}")
def remove_entry(entry_id: str, app=Depends(require_api_key)):
    removed = app.queue_manager.remove_entry(entry_id)
    if removed is None:
        raise HTTPException(status_code=404,
                            detail=f"no pending entry '{entry_id}'")
    return {"ok": True, "cancelled": removed.to_dict()}


@router.patch("/sessions/{entry_id}/move")
def move_entry(entry_id: str, body: QueueMoveRequest,
               app=Depends(require_api_key)):
    moved = app.queue_manager.move_entry(entry_id, body.position)
    if moved is None:
        raise HTTPException(status_code=404,
                            detail=f"no pending entry '{entry_id}'")
    return {"ok": True, "entry": moved.to_dict(),
            "queue": app.queue_manager.queue_snapshot()}


# ======================================================================
# 2) Inject to next tool output — never enqueues the session
# ======================================================================
@router.post("/inject")
def queue_inject(body: QueueInjectRequest, app=Depends(require_api_key)):
    """Requirement 4.2 — (session_name, agent_name, text)."""
    try:
        staged = app.queue_manager.stage_injection(
            session_name=body.session_name,
            agent_name=body.agent_name,
            text=body.text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "ok": True,
        "delivered_live": staged.delivered,
        "injection": staged.to_dict(),
        "staged": app.queue_manager.staged_snapshot(),
    }


@router.get("/inject")
def list_staged(app=Depends(require_api_key)):
    return {
        "staged": app.queue_manager.staged_snapshot(),
        "live": app.queue_manager.live_injection_counts(),
    }


@router.delete("/inject/{injection_id}")
def remove_staged(injection_id: str, app=Depends(require_api_key)):
    removed = app.queue_manager.remove_staged(injection_id)
    if removed is None:
        raise HTTPException(status_code=404,
                            detail=f"no staged injection '{injection_id}'")
    return {"ok": True, "removed": removed.to_dict()}
