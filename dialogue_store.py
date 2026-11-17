# dialogue_store.py
#
# Persistent side chats between pairs of agents (agent-to-agent dialogues).
#
# A dialogue is a PRIVATE conversation between exactly two agents:
# the CALLER (who used the core_agent_execute_dialogue tool) and the
# CALLEE (the agent that was invoked). The caller's messages are stored
# with role "user"; the callee's answers (and their tool traffic) with
# role "assistant" / "tool". Nothing from the main shared chat ever
# enters a dialogue, and nothing from a dialogue ever enters the main
# shared chat.
#
# For every ORDERED pair of agents there is exactly ONE dialogue:
# A->B and B->A are two separate chats (in A->B, A is the "user" side
# and B is the "assistant" side; in B->A the roles swap). Each dialogue
# is kept in memory while the app runs and mirrored to disk as its own
# little session file under  <app_dir>/agent_dialogues/ :
#
#     dialogue_<caller>__<callee>.json
#
# ...so side chats survive restarts and app-folder moves (they only
# reference agent NAMES, never absolute paths). The JSON layout:
#
# {
#   "dialogue_file_format": "agent-dialogue",
#   "dialogue_file_version": 1,
#   "caller": "agent_a",
#   "callee": "agent_b",
#   "history": [
#     {"role": "user",      "sender": "agent_a", "content": "...",
#      "timestamp": 1760000000.123},
#     {"role": "assistant", "sender": "agent_b", "content": "...",
#      "timestamp": 1760000005.456},
#     ...
#   ]
# }
#
# This module has NO dependencies on the GUI (no PyQt5) and no imports
# from agent.py / agent_manager.py, so it can be used from anywhere —
# including the tool code written to the tools/ folder.

import hashlib
import json
import os
import re
import threading
import time

try:
    import app_paths
except Exception:  # pragma: no cover - app_paths always exists in the app
    app_paths = None

DIALOGUE_FORMAT = "agent-dialogue"
DIALOGUE_VERSION = 1
DIALOGUE_FOLDER_NAME = "agent_dialogues"


def _safe_name(name: str) -> str:
    """Filesystem-safe version of an agent name (used in file names only)."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    return s or "_"


class DialogueSession:
    """One side chat: `caller` (user side) talks to `callee` (assistant side).

    Thread-safe: all mutations happen under an internal lock, mirroring
    the Conversation class in agent_manager.py.
    """

    def __init__(self, caller: str, callee: str, history=None):
        self.caller = str(caller)
        self.callee = str(callee)
        self.history: list = list(history or [])
        self._lock = threading.RLock()

    def add_partner_message(self, text: str):
        """The CALLER speaks. Stored as a user-role message.

        Note there is deliberately no "sender: " prefix anywhere — the
        prefix decision happens at context-build time
        (agent.build_dialogue_context shows partner messages verbatim,
        so the callee believes it talks with the human user).
        """
        with self._lock:
            if str(text).strip():
                self.history.append({
                    "role": "user",
                    "sender": self.caller,
                    "content": str(text).strip(),
                    "timestamp": time.time(),
                })

    def add_messages(self, messages):
        """Append the CALLEE's produced messages (one turn's output).

        Same defense-in-depth rules as Conversation.add_messages:
        - the `thinking` field is stripped (never persisted);
        - messages without a timestamp get one stamped (shallow copy,
          the caller's dicts are not mutated).
        """
        with self._lock:
            now = time.time()
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if "thinking" in msg or "timestamp" not in msg:
                    msg = {k: v for k, v in msg.items() if k != "thinking"}
                    if "timestamp" not in msg:
                        msg["timestamp"] = now
                self.history.append(msg)

    def get_history_snapshot(self) -> list:
        """Copy of the dialogue history under lock (dicts are shared,
        read-only use is safe — same contract as Conversation)."""
        with self._lock:
            return list(self.history)

    def message_count(self) -> int:
        with self._lock:
            return len(self.history)

    def to_dict(self) -> dict:
        """Serialise for disk (thinking-stripped, like session files)."""
        with self._lock:
            clean = []
            for msg in self.history:
                if isinstance(msg, dict) and "thinking" in msg:
                    clean.append({k: v for k, v in msg.items() if k != "thinking"})
                else:
                    clean.append(msg)
            return {
                "dialogue_file_format": DIALOGUE_FORMAT,
                "dialogue_file_version": DIALOGUE_VERSION,
                "caller": self.caller,
                "callee": self.callee,
                "history": clean,
            }

    @classmethod
    def from_dict(cls, data: dict) -> "DialogueSession":
        """Rebuild a session from a saved dialogue file."""
        history = data.get("history", [])
        if not isinstance(history, list):
            history = []
        return cls(
            str(data.get("caller", "")),
            str(data.get("callee", "")),
            history,
        )


class DialogueStore:
    """Registry + disk persistence for agent-to-agent dialogues.

    One instance lives for the whole app run (published as
    context.dialogue_store by main.py). Sessions are cached in memory
    by (caller, callee) key and mirrored to disk after every change,
    so a crash loses at most the message that was being produced.
    """

    def __init__(self, base_dir=None):
        if base_dir is None:
            if app_paths is not None:
                base_dir = os.path.join(app_paths.get_app_dir(),
                                        DIALOGUE_FOLDER_NAME)
            else:
                base_dir = DIALOGUE_FOLDER_NAME
        self.base_dir = str(base_dir)
        self._sessions = {}  # (caller, callee) -> DialogueSession
        self._lock = threading.RLock()
        try:
            os.makedirs(self.base_dir, exist_ok=True)
        except Exception as exc:
            print(f"[dialogue_store] could not create folder "
                  f"{self.base_dir}: {exc}")

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------
    @staticmethod
    def pair_filename(caller: str, callee: str) -> str:
        """Deterministic file name for an ordered agent pair.

        Names are sanitised for the filesystem; when sanitisation changed
        anything, a short hash of the RAW pair is appended so two
        different weird names can never collapse onto the same file.
        """
        sc = _safe_name(caller)
        se = _safe_name(callee)
        base = f"dialogue_{sc}__{se}"
        if sc != str(caller) or se != str(callee):
            digest = hashlib.sha1(
                f"{caller}\x00{callee}".encode("utf-8")
            ).hexdigest()[:8]
            base += f"_{digest}"
        return base + ".json"

    def path_for(self, caller: str, callee: str) -> str:
        return os.path.join(self.base_dir, self.pair_filename(caller, callee))

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------
    def _load_from_disk(self, caller: str, callee: str):
        path = self.path_for(caller, callee)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        if data.get("dialogue_file_format") != DIALOGUE_FORMAT:
            return None
        # The file must belong to exactly this ordered pair — never mix
        # chats, even if a file was renamed by hand.
        if str(data.get("caller")) != str(caller) or \
                str(data.get("callee")) != str(callee):
            print(f"[dialogue_store] WARNING: {path} holds a different "
                  f"pair ({data.get('caller')} -> {data.get('callee')}); "
                  f"ignoring it.")
            return None
        return DialogueSession.from_dict(data)

    def get(self, caller: str, callee: str, create: bool = False):
        """Return the session for an ordered pair.

        Lookup order: in-memory cache, then disk. When nothing exists:
        create=True returns a brand-new EMPTY session (and caches it —
        nothing is written to disk until save()); create=False returns
        None (read-only semantics for the listing/reading tools).
        """
        key = (str(caller), str(callee))
        with self._lock:
            sess = self._sessions.get(key)
            if sess is not None:
                return sess
            sess = self._load_from_disk(*key)
            if sess is None and create:
                sess = DialogueSession(key[0], key[1], [])
            if sess is not None:
                self._sessions[key] = sess
            return sess

    def save(self, session: DialogueSession) -> bool:
        """Mirror one session to disk (atomic replace). Never raises."""
        try:
            payload = json.dumps(session.to_dict(), indent=2)
            path = self.path_for(session.caller, session.callee)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, path)
            return True
        except Exception as exc:
            print(f"[dialogue_store] failed to save dialogue "
                  f"{session.caller} -> {session.callee}: {exc}")
            return False

    def delete(self, caller: str, callee: str) -> bool:
        """Forget (and remove from disk) the dialogue for an ordered pair.

        Returns True when a session or file actually existed.
        """
        key = (str(caller), str(callee))
        with self._lock:
            existed = key in self._sessions
            self._sessions.pop(key, None)
        path = self.path_for(*key)
        removed = False
        try:
            if os.path.isfile(path):
                os.remove(path)
                removed = True
        except Exception as exc:
            print(f"[dialogue_store] failed to delete {path}: {exc}")
        return existed or removed

    def list_dialogues(self) -> list:
        """Summarise every dialogue file found on disk.

        Returns a list of dicts: file, caller, callee, messages,
        first_time, last_time (epoch seconds or None when the file holds
        no timestamps).
        """
        results = []
        try:
            names = sorted(os.listdir(self.base_dir))
        except Exception:
            return results
        for fn in names:
            if not fn.lower().endswith(".json"):
                continue
            path = os.path.join(self.base_dir, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            if data.get("dialogue_file_format") != DIALOGUE_FORMAT:
                continue
            history = data.get("history", [])
            if not isinstance(history, list):
                history = []
            ts_list = [m.get("timestamp") for m in history
                       if isinstance(m, dict)
                       and isinstance(m.get("timestamp"), (int, float))
                       and not isinstance(m.get("timestamp"), bool)]
            results.append({
                "file": fn,
                "caller": str(data.get("caller", "")),
                "callee": str(data.get("callee", "")),
                "messages": len(history),
                "first_time": min(ts_list) if ts_list else None,
                "last_time": max(ts_list) if ts_list else None,
            })
        return results
