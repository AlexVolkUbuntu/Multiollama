"""
server/config.py — runtime configuration for the headless backend.

Every value can be overridden with an environment variable so the SAME
backend folder deploys unmodified on Linux, Windows and containers:

  ML_BACKEND_HOST            bind address              (default 0.0.0.0)
  ML_BACKEND_PORT            HTTP/WS port              (default 8000)
  ML_EVENT_TCP_PORT          legacy TCP event port     (default 8888)
                              — keeps the proven `send_event()` snippet
                                produced by the core_get_send_message_code
                                tool working against this backend too.
  ML_CORS_ORIGINS            comma-separated list of browser origins allowed
                              to call the API (default "*" = every origin).
                              Example:
                              ML_CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
                              Set it to an empty string to disable CORS
                              entirely (same-origin / non-browser clients).
  ML_BACKEND_AUTOSAVE_KEEP   newest autosave files kept (default 20)
  ML_BACKEND_AUTOSAVE_MIN_INTERVAL
                              seconds between mid-run autosaves (default 1.0)
  ML_SESSIONS_DIR            folder with manually placed session JSONs
                              (default "sessions" next to the backend)
  ML_API_KEY                 when set, every /control and /queue call must
                              carry it in the X-API-Key header (and WS/SSE
                              clients in a ?token= query param). LAN-only
                              deployments can leave it unset.
  ML_PREEMPT_INTERACTIVE     "0" disables preemption of an interactive
                              session by queued sessions (default enabled)
  ML_PREEMPT_TIMEOUT         seconds to wait for an interactive session to
                              finish gracefully before aborting it when a
                              queued session is waiting (default 600)
  ML_ENDPOINT_DOWN_TIMEOUT   seconds ALL endpoints may stay unreachable
                              before a queued session is finished as failed
                              (default 120)
  ML_DONE_GRACE              seconds after session start before the
                              quiesce detector may fire (default 3)
  ML_USAGE_POLL_INTERVAL     model-usage (/api/ps) poll period (default 3)
  ML_BACKEND_LOG_LEVEL       uvicorn log level (default "info")

The Config object is a plain module-level singleton created lazily by
get_config(); nothing here imports the app core, so importing this
module is side-effect free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    # --- network -------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    event_tcp_port: int = 8188
    # Browser origins allowed to call the HTTP/WS API ("*" = all).
    # Only cross-origin BROWSER frontends need this; curl / Python /
    # other backends never send CORS preflights.
    cors_origins: str = "*"

    # --- persistence ----------------------------------------------------
    autosave_keep: int = 20
    autosave_min_interval: float = 1.0
    sessions_dir: str = "sessions"

    # --- security -------------------------------------------------------
    api_key: Optional[str] = None

    # --- queue behaviour -------------------------------------------------
    preempt_interactive: bool = True
    preempt_timeout: float = 600.0
    endpoint_down_timeout: float = 120.0
    done_grace: float = 3.0

    # --- misc -------------------------------------------------------------
    usage_poll_interval: float = 3.0
    log_level: str = "info"
    recent_events_buffer: int = 1000

    # --- filled at bootstrap (not from env) -------------------------------
    app_dir: str = ""

    def as_public_dict(self) -> dict:
        """Config subset that is safe to expose through the state API."""
        return {
            "host": self.host,
            "port": self.port,
            "event_tcp_port": self.event_tcp_port,
            "cors_origins": self.cors_origins,
            "autosave_keep": self.autosave_keep,
            "autosave_min_interval": self.autosave_min_interval,
            "sessions_dir": self.sessions_dir,
            "preempt_interactive": self.preempt_interactive,
            "preempt_timeout": self.preempt_timeout,
            "endpoint_down_timeout": self.endpoint_down_timeout,
            "done_grace": self.done_grace,
            "usage_poll_interval": self.usage_poll_interval,
            "auth_required": bool(self.api_key),
        }


_CONFIG: Optional[Config] = None


def get_config() -> Config:
    """Return the process-wide Config (created from env on first call)."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config(
            host=_env_str("ML_BACKEND_HOST", "0.0.0.0"),
            port=_env_int("ML_BACKEND_PORT", 8000),
            event_tcp_port=_env_int("ML_EVENT_TCP_PORT", 8188),
            cors_origins=_env_str("ML_CORS_ORIGINS", "*"),
            autosave_keep=_env_int("ML_BACKEND_AUTOSAVE_KEEP", 20),
            autosave_min_interval=_env_float("ML_BACKEND_AUTOSAVE_MIN_INTERVAL", 1.0),
            sessions_dir=_env_str("ML_SESSIONS_DIR", "sessions"),
            api_key=(os.environ.get("ML_API_KEY") or None),
            preempt_interactive=_env_bool("ML_PREEMPT_INTERACTIVE", True),
            preempt_timeout=_env_float("ML_PREEMPT_TIMEOUT", 600.0),
            endpoint_down_timeout=_env_float("ML_ENDPOINT_DOWN_TIMEOUT", 120.0),
            done_grace=_env_float("ML_DONE_GRACE", 3.0),
            usage_poll_interval=_env_float("ML_USAGE_POLL_INTERVAL", 3.0),
            log_level=_env_str("ML_BACKEND_LOG_LEVEL", "info"),
            recent_events_buffer=_env_int("ML_BACKEND_RECENT_EVENTS", 1000),
        )
    return _CONFIG
