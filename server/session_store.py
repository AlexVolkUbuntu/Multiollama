"""
server/session_store.py — session persistence for the headless backend.

The session FILE FORMAT is byte-compatible with the proven PyQt app
(gui_main._build_session_data), so sessions auto-saved by the desktop
app can be queued here and vice versa:

    {
      "app_dir": "<dir the session was saved from>",
      "session_name": "team-alpha",
      "user_name": "User",
      "agents": { "<name>": { ...AgentManager.to_dict()... } },
      "conversation": { "history": [ ... ] },
      "mcp_registry": { "servers": { ... } } | null
    }

Auto-save behaviour mirrors the desktop app exactly:

  * files live in            <app_dir>/autosave/
  * file names are           <sanitized session name>_<YYYYMMDD_HHMMSS>.json
    (with a _N suffix on same-second collisions)
  * only the newest AUTOSAVE_KEEP files are kept (default 20)
  * the newest loadable file for a given session NAME is what the queue
    loads when that session is dequeued ("loaded from autosave")

In addition to the autosave folder, a "sessions" folder next to the
backend is scanned for manually placed session JSON files (exported
from the desktop app's Save Session button). Lookup order when running
a session by name: newest match in autosave/, then newest match in
sessions/.

All methods are thread-safe and never raise on I/O problems (they
return None / False / empty results and print the reason, exactly like
the proven dialogue_store / gui_main code they were modelled on).
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional

import app_paths

# Exactly the constants the PyQt gui_main used, so file naming (and the
# session<N> default-name counter) behave identically in both apps.
SESSION_NAME_FALLBACK = "session"
SESSION_NAME_MAX_LEN = 60
DEFAULT_SESSION_NAME_RE = re.compile(r'^session(\d+)(?:[_\s\-.]|$)', re.IGNORECASE)
_UNSAFE_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def sanitize_session_name(name: str) -> str:
    """Make *name* safe as a file-name prefix (same rules as gui_main)."""
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", str(name or "").strip())
    cleaned = cleaned.strip(" .")
    return cleaned[:SESSION_NAME_MAX_LEN]


def _get_mcp_registry_snapshot() -> Optional[dict]:
    """Snapshot of the MCP server registry for session files (or None)."""
    try:
        import mcp_manager as _mm
        mgr = _mm.get_manager()
        if mgr is None:
            return None
        snap = mgr.registry_snapshot()
        return snap if snap else None
    except Exception as exc:
        print(f"[session_store] mcp registry snapshot failed: {exc}")
        return None


class SessionStore:
    """Load / save / autosave session JSONs (PyQt-compatible format)."""

    def __init__(self,
                 autosave_dir: Optional[str] = None,
                 sessions_dir: Optional[str] = None,
                 keep: int = 20):
        self.autosave_dir = autosave_dir or app_paths.get_autosave_dir()
        self.sessions_dir = sessions_dir or os.path.join(
            app_paths.get_app_dir(), "sessions")
        self.keep = max(1, int(keep))
        for folder in (self.autosave_dir, self.sessions_dir):
            try:
                os.makedirs(folder, exist_ok=True)
            except Exception as exc:
                print(f"[session_store] could not create folder {folder}: {exc}")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def build_session_data(self, session_name: str, user_name: str,
                           manager, conversation) -> dict:
        """Serialise the live session into the portable session dict.

        Identical layout to gui_main._build_session_data (including the
        mcp_registry snapshot) so both apps read each other's files.
        """
        return {
            "app_dir": app_paths.get_app_dir(),
            "session_name": (session_name or "").strip(),
            "user_name": user_name or "User",
            "agents": manager.to_dict(),
            "conversation": conversation.to_dict(),
            "mcp_registry": _get_mcp_registry_snapshot(),
        }

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def autosave_filename(self, folder: str, session_name: str) -> str:
        """<sanitized name>_<stamp>.json with collision counter."""
        base = sanitize_session_name(session_name) or SESSION_NAME_FALLBACK
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(folder, f"{base}_{stamp}.json")
        n = 1
        while os.path.exists(path):
            n += 1
            path = os.path.join(folder, f"{base}_{stamp}_{n}.json")
        return path

    def autosave(self, session_name: str, user_name: str,
                 manager, conversation) -> Optional[str]:
        """Write one auto-save of the live session. Never raises.

        Returns the file path on success, None on failure. Serialisation
        is retried a few times, exactly like the proven GUI code, in case
        the agents dict mutates mid-walk (agent_manager's lock makes this
        mostly unnecessary today, but the retry costs nothing).
        """
        try:
            payload = None
            last_exc = None
            for _ in range(3):
                try:
                    payload = json.dumps(
                        self.build_session_data(session_name, user_name,
                                                manager, conversation),
                        indent=2, ensure_ascii=False)
                    break
                except RuntimeError as exc:
                    last_exc = exc
                    time.sleep(0.05)
            if payload is None:
                print(f"[session_store] failed to serialise session: {last_exc}")
                return None
            path = self.autosave_filename(self.autosave_dir, session_name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(payload)
            self.prune_autosaves()
            return path
        except Exception as exc:
            print(f"[session_store] autosave failed: {exc}")
            return None

    def save_to_path(self, session_name: str, user_name: str,
                     manager, conversation, path: str) -> str:
        """Manual save (the desktop 'Save Session' equivalent).

        Raises IOError-ish exceptions on failure — the API layer turns
        them into a 500 response. Returns the path written.
        """
        data = self.build_session_data(session_name, user_name,
                                       manager, conversation)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def prune_autosaves(self) -> None:
        """Keep only the newest AUTOSAVE_KEEP files in the autosave dir."""
        try:
            files = [
                os.path.join(self.autosave_dir, name)
                for name in os.listdir(self.autosave_dir)
                if name.lower().endswith(".json")
            ]
            if len(files) <= self.keep:
                return
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for stale in files[self.keep:]:
                try:
                    os.remove(stale)
                except Exception:
                    pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    @staticmethod
    def validate(data) -> bool:
        """True when *data* looks like a loadable session dict."""
        return (isinstance(data, dict)
                and isinstance(data.get("agents"), dict)
                and isinstance(data.get("conversation"), dict))

    def load_from_path(self, path: str) -> dict:
        """Read + validate a session file. Raises on any problem."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not self.validate(data):
            raise ValueError(f"Not a valid session file: {path}")
        return data

    def _candidates_for_name(self, session_name: str) -> List[str]:
        """All session files in both folders that belong to *session_name*.

        A file belongs to the session when its base name (without .json
        and without the _<timestamp>[_N] auto-save suffix) equals the
        sanitized session name, or when the file's stored session_name
        matches — the stored name is the authoritative identity, the
        file-name match is the fast path.
        """
        base = sanitize_session_name(session_name)
        results: List[str] = []
        for folder in (self.autosave_dir, self.sessions_dir):
            try:
                names = os.listdir(folder)
            except Exception:
                continue
            for fn in names:
                if not fn.lower().endswith(".json"):
                    continue
                stem = fn[:-5]
                # fast path: file-name prefix produced by our own naming
                if stem == base or stem.startswith(base + "_"):
                    results.append(os.path.join(folder, fn))
                    continue
                # slow path: open the file and read session_name
                path = os.path.join(folder, fn)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        head = json.load(f)
                    if (isinstance(head, dict)
                            and str(head.get("session_name", "")).strip()
                            == str(session_name).strip()):
                        results.append(path)
                except Exception:
                    continue
        return results

    def find_latest_for_name(self, session_name: str) -> Optional[str]:
        """Newest loadable session file for *session_name* (or None).

        Scans autosave/ first, then sessions/ — exactly the "next session
        loaded from autosave" requirement, with the sessions folder as a
        manual import path. Files that fail validation are skipped in
        favour of the next-newest (a crash mid-write can leave truncated
        JSON behind), mirroring the proven _autoload_latest_session.
        """
        candidates = self._candidates_for_name(session_name)
        if not candidates:
            return None
        try:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        except Exception:
            return None
        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if self.validate(data):
                    return path
            except Exception:
                continue
        return None

    def list_sessions(self) -> List[dict]:
        """Summarise every session file found in autosave/ + sessions/.

        Returns newest-first:
            {file, name, folder, messages, mtime, agents, last_activity}
        """
        results: List[dict] = []
        for folder, origin in ((self.autosave_dir, "autosave"),
                               (self.sessions_dir, "sessions")):
            try:
                names = os.listdir(folder)
            except Exception:
                continue
            for fn in names:
                if not fn.lower().endswith(".json"):
                    continue
                path = os.path.join(folder, fn)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if not self.validate(data):
                        continue
                    history = data.get("conversation", {}).get("history", [])
                    ts_list = [m.get("timestamp") for m in history
                               if isinstance(m, dict)
                               and isinstance(m.get("timestamp"), (int, float))]
                    results.append({
                        "file": path,
                        "name": (str(data.get("session_name") or "").strip()
                                 or os.path.splitext(fn)[0]),
                        "folder": origin,
                        "messages": len(history),
                        "agents": sorted((data.get("agents") or {}).keys()),
                        "mtime": os.path.getmtime(path),
                        "last_activity": max(ts_list) if ts_list else None,
                    })
                except Exception:
                    continue
        results.sort(key=lambda item: item["mtime"], reverse=True)
        return results

    def next_default_session_name(self) -> str:
        """Return the next 'session<N>' default name.

        Same logic as the proven gui_main._next_default_session_name:
        one past the highest N found among auto-save file names following
        the default pattern; renamed sessions never disturb the counter.
        """
        max_num = 0
        try:
            listing = os.listdir(self.autosave_dir)
        except Exception:
            listing = []
        for name in listing:
            m = DEFAULT_SESSION_NAME_RE.match(name)
            if not m:
                continue
            try:
                num = int(m.group(1))
            except ValueError:
                continue
            if num > max_num:
                max_num = num
        return f"session{max_num + 1}"
