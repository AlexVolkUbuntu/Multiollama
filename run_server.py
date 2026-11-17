#!/usr/bin/env python3
"""
run_server.py — entry point for the MultiOllama headless backend.

    python run_server.py                       # 0.0.0.0:8000, events :8888
    python run_server.py --host 127.0.0.1 --port 9000
    python run_server.py --tcp-port 8889

Everything is configurable through environment variables too (see
server/config.py): ML_BACKEND_HOST, ML_BACKEND_PORT, ML_EVENT_TCP_PORT,
ML_API_KEY, ...

Startup behaviour (per the design brief): the terminal process loads
ALL modules — core tools are (re)written into tools/, the prompts
folder is prepared, the MCP engine and Ollama client manager start —
but NO session is loaded. The session queue parks in "wait" mode until
POST /queue/sessions (or /control/session/new) starts one.
"""

import argparse
import sys
from pathlib import Path


def _ensure_backend_root_on_sys_path() -> None:
    """Make the backend folder importable no matter where we launch from.

    The proven core modules (context, agent, agent_manager, ...) and the
    generated tools/ modules do top-level imports of each other, so the
    backend root itself must sit on sys.path — exactly like running the
    original app from its own folder.
    """
    root = str(Path(__file__).resolve().parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    # tools/*.py are imported as the "tools" package (ToolLoader does
    # `import tools.file_tools` internally) — the parent of tools/ is the
    # backend root, which is now first on sys.path.
    sys.dont_write_bytecode = True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MultiOllama headless backend server")
    parser.add_argument("--host", default=None,
                        help="bind address (default 0.0.0.0 or ML_BACKEND_HOST)")
    parser.add_argument("--port", type=int, default=None,
                        help="HTTP/WS port (default 8000 or ML_BACKEND_PORT)")
    parser.add_argument("--tcp-port", type=int, default=None,
                        help="legacy TCP event port (default 8888)")
    parser.add_argument("--reload", action="store_true",
                        help="dev auto-reload (code changes restart the app)")
    args = parser.parse_args()

    _ensure_backend_root_on_sys_path()

    # Config reads env vars; CLI flags win over env.
    from server.config import get_config
    config = get_config()
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.tcp_port:
        config.event_tcp_port = args.tcp_port

    import uvicorn
    from server.app import create_app

    app = create_app()
    uvicorn.run(app, host=config.host, port=config.port,
                log_level=config.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
