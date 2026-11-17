"""
server/app.py — FastAPI application assembly.

Creates the AppContext (which loads ALL modules — tools, prompts, MCP,
Ollama client manager, dialogue store — and NO sessions), wires the
routers, starts the session queue manager, the model-usage poller and
the legacy TCP event server (keeps the proven send_event()
snippet working), and exposes:

    GET /        the built-in web console (webui.html, no build step)
    GET /info    the plain-HTML endpoint directory (the old root page)

CORS is enabled by default (ML_CORS_ORIGINS, "*" unless overridden) so
browser frontends served from OTHER origins (http://localhost:3000, a
mobile webview, a backend on another PC...) can call the API directly;
curl / Python / other backends are unaffected either way.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from .api_control import router as control_router, system_router
from .api_files import router as files_router
from .api_output import router as output_router
from .api_queue import router as queue_router
from .app_context import AppContext
from .media_links import enrich_chat_event
from .state import build_full_state
from .tcp_events import TcpEventServer

INFO_PAGE = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>multillama backend</title>
<style>
 body { font-family: system-ui, sans-serif; margin: 3rem; color: #222; }
 h1 { margin-bottom: .2rem; }
 code { background: #f0f2f5; padding: 2px 6px; border-radius: 4px; }
 table { border-collapse: collapse; margin-top: 1rem; }
 td { padding: 6px 14px 6px 0; vertical-align: top; }
</style></head>
<body>
<h1>MultiLlama backend</h1>
<p>Headless backend for the proven multi-agent / multi-endpoint engine.</p>
<table>
<tr><td>Web console</td><td><code>GET /</code></td></tr>
<tr><td>Full state</td><td><code>GET /state</code></td></tr>
<tr><td>Full output (stream)</td><td><code>WS /ws/output</code> &nbsp;|&nbsp; <code>GET /output/stream</code> (SSE)</td></tr>
<tr><td>Chat file downloads</td><td><code>GET /files?path=…&download=1</code></td></tr>
<tr><td>Full control</td><td><code>GET /control/overview</code></td></tr>
<tr><td>Session queue + inject</td><td><code>GET /queue/sessions</code> &nbsp;|&nbsp; <code>POST /queue/sessions</code> &nbsp;|&nbsp; <code>POST /queue/inject</code></td></tr>
<tr><td>API guide</td><td><code>API_GUIDE.md</code> in the backend folder</td></tr>
<tr><td>API docs</td><td><code>/docs</code> (OpenAPI)</td></tr>
<tr><td>Health</td><td><code>GET /health</code></td></tr>
</table>
</body></html>
"""

# A tiny inline SVG favicon (a llama-ish glyph) so browsers get an icon
# without a binary asset — Linux and Windows alike.
_FAVICON = (
    b"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    b"<rect width='32' height='32' rx='7' fill='#1b2330'/>"
    b"<text x='16' y='22' font-size='16' text-anchor='middle' "
    b"font-family='sans-serif' fill='#7dd3fc'>M</text></svg>"
)


def _handle_tcp_event(app: AppContext):
    """(sender, message) from agent-generated scripts -> chat + history.

    Mirrors the PyQt EventServer: the message enters the ACTIVE
    session's shared history as a system-role message with the
    "[Event] " prefix and is broadcast to every output client.
    """
    def handler(sender: str, message: str) -> None:
        text = f"[Event] {message}"
        runner = app.queue_manager.active_runner
        if runner is not None and not runner.done:
            runner.conversation.add_messages([{
                "role": "system",
                "sender": sender,
                "content": text,
            }])
        app.broadcast.publish("chat", enrich_chat_event({
            "sender": sender,
            "text": text,
            # same role this message is stored with in the shared
            # history — so the live bubble and a later history reload
            # render it identically.
            "role": "system",
            "session": (runner.session_name if runner else None),
        }))
        print(f"[tcp-events] {sender}: {message[:200]}")

    return handler


def create_app(app_context: AppContext = None) -> FastAPI:
    app = FastAPI(
        title="multillama-backend",
        description=(
            "Headless backend for the MultiLlama multi-agent, "
            "multi-endpoint engine: session queue, full control API, "
            "full output streaming."
        ),
        version="1.0.0",
        lifespan=None,   # replaced right below (needs the ctx)
    )

    ctx: AppContext = app_context or AppContext()

    # ---- CORS -----------------------------------------------------------
    # Browser frontends on OTHER origins need this (they send OPTIONS
    # preflight requests first — without the middleware those die with
    # 405 Method Not Allowed). curl / Python / WS-native clients are
    # unaffected. The browser cannot send the X-API-Key header on the
    # WebSocket itself, so the token query param stays the WS way in.
    origins = [o.strip() for o in (ctx.config.cors_origins or "").split(",")
               if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["*"],
            allow_headers=["*"],
            # No cookies / HTTP auth are used — the API key rides in a
            # header — so wildcard origins are safe here.
            allow_credentials=False,
        )

    tcp_server = TcpEventServer(_handle_tcp_event(ctx))

    @contextlib.asynccontextmanager
    async def lifespan(application: FastAPI):
        loop = asyncio.get_running_loop()
        ctx.broadcast.bind_loop(loop)
        await tcp_server.start(port=ctx.config.event_tcp_port)
        ctx.queue_manager.start()
        ctx.start_usage_polling()
        print(f"[backend] ready — HTTP/WS on port {ctx.config.port}, "
              f"TCP events on {ctx.config.event_tcp_port}")
        try:
            yield
        finally:
            await tcp_server.stop()
            ctx.broadcast.unbind_loop()
            ctx.shutdown()

    app.router.lifespan_context = lifespan

    # ---- routes ---------------------------------------------------------
    app.include_router(control_router)
    app.include_router(system_router)
    app.include_router(queue_router)
    app.include_router(output_router)
    app.include_router(files_router)

    webui_path = Path(ctx.app_dir) / "webui.html"

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """The built-in web console (single self-contained HTML file).

        Served from disk so the frontend can be tweaked without touching
        Python; if webui.html is missing the plain endpoint directory is
        shown instead.
        """
        if webui_path.is_file():
            return FileResponse(webui_path, media_type="text/html")
        return INFO_PAGE

    @app.get("/info", response_class=HTMLResponse)
    async def info():
        return INFO_PAGE

    @app.get("/favicon.ico")
    async def favicon():
        return Response(content=_FAVICON, media_type="image/svg+xml")

    @app.get("/health")
    async def health():
        runner = ctx.queue_manager.active_runner
        return {
            "status": "ok",
            "uptime": round(ctx.uptime(), 1),
            "active_session": (runner.session_name
                               if runner is not None else None),
            "queue_length": len(ctx.queue_manager.queue_snapshot()),
            "output_subscribers": ctx.broadcast.subscriber_count(),
        }

    # ---- error handling -------------------------------------------------
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    app.state.app = ctx
    return app
