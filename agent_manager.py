import threading
import time
import json
from typing import Dict, List, Optional

# HEADLESS BACKEND ADAPTATION (the ONLY change to this proven module):
# the PyQt5 QObject/pyqtSignal base was replaced by the pure-Python
# Signal from event_bus.py. The public surface is IDENTICAL —
# `agents_changed.connect(fn)` and `agents_changed.emit()` keep working
# exactly as before (the generated agent_tools.py calls
# `manager.agents_changed.emit()` verbatim). A re-entrant lock was also
# added around the agents dict so to_dict()/from_dict() are safe when
# core tools mutate agents from worker threads mid-serialisation.
from event_bus import Signal

from agent import Agent, ToolLoader


# ----------------------------------------------------------------------
# NATIVE MCP — small helpers that keep agent_manager.py decoupled from
# the MCP engine. agent_manager must never hard-import mcp_manager at
# module level (it is also used in stripped-down test setups), so the
# lookup is defensive in both directions.
# ----------------------------------------------------------------------
class MCPManagerSnapshotHelpers:
    """Export/import MCP server definitions for saved-agent files."""

    @staticmethod
    def _get_manager():
        try:
            import mcp_manager as _mm
            return _mm.get_manager()
        except Exception:
            return None

    @classmethod
    def export_server_defs(cls, agent) -> dict:
        """Snapshot of the DEFINITIONS of the agent's subscribed servers."""
        mgr = cls._get_manager()
        if mgr is None:
            return {}
        subs = list(getattr(agent, "mcp_servers", []) or [])
        registry = mgr.list_registry_servers()
        return {name: dict(registry[name]) for name in subs if name in registry}

    @classmethod
    def import_server_defs(cls, params: dict) -> int:
        """Merge a saved-agent file's embedded definitions into the registry."""
        mgr = cls._get_manager()
        if mgr is None:
            return 0
        defs = params.get("mcp_server_defs") if isinstance(params, dict) else None
        if not isinstance(defs, dict) or not defs:
            return 0
        return mgr.merge_registry_snapshot({"servers": defs})


# Saved-agent file format marker. Agent config files written by
# AgentManager.export_agent / core_agent_save / the GUI "Save Agent to File"
# button carry this marker so `core_saved_agents_list` can recognise them
# among all *.json files in the current folder.
SAVED_AGENT_FORMAT = "saved-agent"
SAVED_AGENT_VERSION = 1


class AgentManager:
    """Registry of agents (headless backend version — no Qt dependency).

    `agents_changed` is a plain Signal (event_bus.Signal): it is created
    PER INSTANCE (there is exactly one manager per active session in the
    backend) and carries the same connect()/emit() contract the PyQt
    version exposed.
    """

    def __init__(self, loader: ToolLoader):
        self.loader = loader
        self.agents: Dict[str, Agent] = {}
        self.agents_changed = Signal()
        # Guards the agents dict for atomic snapshot / replace operations
        # (to_dict / from_dict / listing) against concurrent mutation from
        # tool-execution worker threads (core_create_agent etc.).
        self._agents_lock = threading.RLock()

    def create_agent(self, name: str, system_prompt: str,
                     model: str = "qwen3.5:9b",
                     num_ctx: int = 32000,
                     temperature: float = 0.5,
                     tools: List[str] = None,
                     max_context_messages: int = 30,
                     max_tool_iterations: int = 1000,
                     endpoint: str = "local",
                     active: bool = True,
                     silent: bool = False,
                     description: str = "",
                     use_tools: bool = True,
                     image: bool = False,
                     audio: bool = False,
                     mcp: bool = False,
                     mcp_servers: Optional[List[str]] = None,
                     tool_files: Optional[List[str]] = None,
                     prompts: Optional[List[str]] = None) -> Agent:
        agent = Agent(
            name, system_prompt, model, num_ctx, temperature,
            tools, max_context_messages,
            max_tool_iterations=max_tool_iterations,
            endpoint=endpoint if endpoint else "local",
            active=active,
            silent=silent,
            description=description,
            use_tools=use_tools,
            image=image,
            audio=audio,
            mcp=mcp,
            mcp_servers=mcp_servers,
            tool_files=tool_files,
            prompts=prompts
        )
        with self._agents_lock:
            if name in self.agents:
                raise ValueError(f"Agent '{name}' already exists.")
            self.agents[name] = agent
        self.agents_changed.emit()
        return agent

    def get_agent(self, name: str) -> Optional[Agent]:
        return self.agents.get(name)

    def delete_agent(self, name: str):
        with self._agents_lock:
            if name in self.agents:
                del self.agents[name]
                self.agents_changed.emit()

    def update_agent(self, name: str, emit: bool = True, **kwargs):
        with self._agents_lock:
            agent = self.get_agent(name)
            if not agent:
                raise ValueError(f"Agent '{name}' not found.")
            for key, value in kwargs.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
            # Normalise empty endpoint
            if not agent.endpoint:
                agent.endpoint = "local"
        if emit:
            self.agents_changed.emit()

    def rename_agent(self, old_name: str, new_name: str, emit: bool = True):
        """Rename an existing agent while preserving its Agent object/config.

        The manager dictionary key and ``Agent.name`` are both changed.
        Conversation history is renamed separately by the owning MainWindow
        because AgentManager intentionally does not own the shared history.
        """
        old_name = (old_name or "").strip()
        new_name = (new_name or "").strip()
        if not old_name:
            raise ValueError("Old agent name cannot be empty.")
        if not new_name:
            raise ValueError("New agent name cannot be empty.")
        if old_name == new_name:
            with self._agents_lock:
                return self.agents.get(old_name)
        with self._agents_lock:
            if old_name not in self.agents:
                raise ValueError(f"Agent '{old_name}' not found.")
            if new_name in self.agents:
                raise ValueError(f"Agent '{new_name}' already exists.")

            agent = self.agents.pop(old_name)
            agent.name = new_name
            self.agents[new_name] = agent
        self.agents_changed.emit()
        return agent

    def list_agents(self) -> List[str]:
        with self._agents_lock:
            return list(self.agents.keys())

    # ------------------------------------------------------------------
    # Endpoint helpers – the GUI uses these to spawn one BackendThread
    # per distinct endpoint.
    # ------------------------------------------------------------------
    def get_endpoints(self) -> List[str]:
        """Return sorted list of distinct endpoints among all agents."""
        endpoints = set()
        with self._agents_lock:
            for agent in self.agents.values():
                endpoints.add(agent.endpoint or "local")
        return sorted(endpoints)

    def get_agents_on_endpoint(self, endpoint: str) -> List[str]:
        """Return sorted list of agent names whose endpoint matches."""
        ep = endpoint if endpoint else "local"
        with self._agents_lock:
            return sorted([n for n, a in self.agents.items()
                           if (a.endpoint or "local") == ep])

    def to_dict(self) -> dict:
        agents_data = {}
        with self._agents_lock:
            for name, agent in self.agents.items():
                agents_data[name] = {
                    "system_prompt": agent.system_prompt,
                    "model": agent.model,
                    "num_ctx": agent.num_ctx,
                    "temperature": agent.temperature,
                    "tools": agent.tools,
                    "max_context_messages": agent.max_context_messages,
                    "max_tool_iterations": agent.max_tool_iterations,
                    "endpoint": agent.endpoint or "local",
                    # Persist the scheduling / visibility flags so they
                    # survive save/load. Older session files (saved before
                    # these fields existed) will simply be missing —
                    # `from_dict` below defaults them to True / False.
                    "active": getattr(agent, "active", True),
                    "silent": getattr(agent, "silent", False),
                    # Persist the capability flags + description the same
                    # way (backward-compatible defaults in from_dict).
                    "description": getattr(agent, "description", ""),
                    "use_tools": getattr(agent, "use_tools", True),
                    "image": getattr(agent, "image", False),
                    "audio": getattr(agent, "audio", False),
                    # NATIVE MCP: capability flag + subscription list. The
                    # subscription names are resolved against the shared
                    # registry (mcp_servers.json); the registry itself is
                    # snapshotted into the session file separately (see
                    # gui_main._build_session_data) so a session restored on
                    # another machine re-registers any missing definitions.
                    "mcp": getattr(agent, "mcp", False),
                    "mcp_servers": list(getattr(agent, "mcp_servers", []) or []),
                    # GUI-only catalog filter. `None` means "all tool files";
                    # it never affects inference or the actual `tools` whitelist.
                    "tool_files": getattr(agent, "tool_files", None),
                    # Selected prompt files (names without .txt). Their texts
                    # are appended to the system prompt inside a <rules>...
                    # </rules> section at inference time (see
                    # agent.build_system_prompt).
                    "prompts": list(getattr(agent, "prompts", []) or []),
                }
        return agents_data

    def from_dict(self, data: dict):
        new_agents: Dict[str, Agent] = {}
        for name, params in data.items():
            new_agents[name] = Agent(
                name=name,
                system_prompt=params["system_prompt"],
                model=params["model"],
                num_ctx=params["num_ctx"],
                temperature=params["temperature"],
                tools=params.get("tools", []),
                max_context_messages=params.get("max_context_messages", 30),
                max_tool_iterations=params.get("max_tool_iterations", 1000),
                endpoint=params.get("endpoint", "local"),
                # Backward-compatible: old session files don't carry these
                # keys, so default to the same values Agent.__init__ uses.
                active=params.get("active", True),
                silent=params.get("silent", False),
                description=params.get("description", ""),
                use_tools=params.get("use_tools", True),
                image=params.get("image", False),
                audio=params.get("audio", False),
                mcp=params.get("mcp", False),
                mcp_servers=params.get("mcp_servers", []),
                tool_files=params.get("tool_files", None),
                prompts=params.get("prompts", []),
            )
        with self._agents_lock:
            self.agents.clear()
            self.agents.update(new_agents)
        self.agents_changed.emit()

    # ------------------------------------------------------------------
    # Single-agent export / import — the backing store for the agent
    # save/load feature. The exported dict contains ONLY the agent's
    # configuration (system prompt + settings + capability flags +
    # description). It deliberately contains NO chat history, so saved
    # agents can be exchanged between different setups / sessions.
    # ------------------------------------------------------------------
    def export_agent(self, name: str) -> dict:
        """Serialise ONE agent's config to a portable dict (no history).

        The dict carries a format marker (`agent_file_format`) so files
        written from it can be recognised by `core_saved_agents_list`
        even when they sit next to unrelated *.json files.
        """
        agent = self.get_agent(name)
        if agent is None:
            raise ValueError(f"Agent '{name}' not found.")
        return {
            "agent_file_format": SAVED_AGENT_FORMAT,
            "agent_file_version": SAVED_AGENT_VERSION,
            "agent": {
                "name": agent.name,
                "description": getattr(agent, "description", ""),
                "system_prompt": agent.system_prompt,
                "model": agent.model,
                "num_ctx": agent.num_ctx,
                "temperature": agent.temperature,
                "tools": list(agent.tools),
                "max_context_messages": agent.max_context_messages,
                "max_tool_iterations": agent.max_tool_iterations,
                "endpoint": agent.endpoint or "local",
                "active": getattr(agent, "active", True),
                "silent": getattr(agent, "silent", False),
                "use_tools": getattr(agent, "use_tools", True),
                "image": getattr(agent, "image", False),
                "audio": getattr(agent, "audio", False),
                # NATIVE MCP: the capability flag + the subscription list
                # (registry server NAMES). To keep saved-agent files fully
                # PORTABLE, a snapshot of the DEFINITIONS of exactly the
                # subscribed servers is embedded under 'mcp_server_defs';
                # import_agent re-registers any of them that the local
                # registry is missing (local definitions always win —
                # the user may have fixed paths/URLs for THIS machine).
                "mcp": getattr(agent, "mcp", False),
                "mcp_servers": list(getattr(agent, "mcp_servers", []) or []),
                "mcp_server_defs": MCPManagerSnapshotHelpers.export_server_defs(agent),
                "tool_files": getattr(agent, "tool_files", None),
                # Selected prompt files (names without .txt) — resolved
                # against the prompts/ folder of whatever machine loads
                # the agent, so a saved agent stays portable: only the
                # NAMES travel in the file, never the texts.
                "prompts": list(getattr(agent, "prompts", []) or []),
            },
        }

    def import_agent(self, data: dict, new_name: str = None) -> Agent:
        """Re-create an agent from a dict produced by export_agent().

        `new_name` lets the caller import under a different name when an
        agent with the stored name already exists (the GUI asks for a
        new name in that case; the core_agent_load tool takes it as an
        optional argument). Raises ValueError on invalid data or name
        conflicts.
        """
        if not isinstance(data, dict):
            raise ValueError("Not a saved-agent file: JSON root is not an object.")
        if data.get("agent_file_format") != SAVED_AGENT_FORMAT:
            raise ValueError(
                "Not a saved-agent file: missing 'agent_file_format' marker."
            )
        params = data.get("agent")
        if not isinstance(params, dict):
            raise ValueError("Not a saved-agent file: missing 'agent' object.")

        name = (new_name or params.get("name") or "").strip()
        if not name:
            raise ValueError("Saved agent has no name and no new_name was given.")
        if name in self.agents:
            raise ValueError(
                f"Agent '{name}' already exists. Import with a different "
                f"new_name (or delete/rename the existing one first)."
            )

        agent = Agent(
            name=name,
            system_prompt=params.get("system_prompt", ""),
            model=params.get("model", "qwen3.5:9b"),
            num_ctx=params.get("num_ctx", 32000),
            temperature=params.get("temperature", 0.5),
            tools=params.get("tools", []),
            max_context_messages=params.get("max_context_messages", 30),
            max_tool_iterations=params.get("max_tool_iterations", 1000),
            endpoint=params.get("endpoint", "local"),
            active=params.get("active", True),
            silent=params.get("silent", False),
            description=params.get("description", ""),
            use_tools=params.get("use_tools", True),
            image=params.get("image", False),
            audio=params.get("audio", False),
            mcp=params.get("mcp", False),
            mcp_servers=params.get("mcp_servers", []),
            tool_files=params.get("tool_files", None),
            prompts=params.get("prompts", []),
        )
        self.agents[name] = agent
        self.agents_changed.emit()

        # NATIVE MCP: merge the embedded server definitions (if any) into
        # the local registry so the agent's subscriptions resolve on this
        # machine too. Existing local definitions are never overwritten —
        # the local machine may have corrected paths/URLs.
        try:
            MCPManagerSnapshotHelpers.import_server_defs(params)
        except Exception as e:
            print(f"[agent-manager] MCP definition merge for '{name}' failed: {e}")

        return agent


class Conversation:
    """
    Shared conversation state across ALL BackendThreads (one per machine).

    Holds:
      - history: shared list of messages from all agents on all machines
      - pending_user_messages: shared queue of user inputs not yet drained

    MESSAGE TIMESTAMPS:
    Every message that enters `history` is stamped with a `timestamp`
    (unix epoch seconds, `time.time()`) at the moment it is added —
    `add_user_message`, `add_messages` and `drain_pending` all do it.
    The timestamp is pure metadata for the `core_chat_messages` tool
    (count / index / time-range queries over the chat); it is NEVER
    forwarded to the model — build_context / build_dialogue_context
    construct brand-new message dicts without it. Old session files
    loaded by `from_dict` may contain messages without timestamps;
    they still count for index queries but cannot match time queries.

    Scheduling of participants is NO LONGER done here – each BackendThread
    manages its own [user] + [its agents] rotation. This lets multiple
    machines run in parallel against the same single chat history.

    CRITICAL (thinking-leak defense-in-depth):
    `add_messages()` strips the `thinking` field from every assistant
    message before it enters `self.history`. The ollama Python client's
    `Message` model has a `thinking` field that is preserved end-to-end
    when present on a message dict, which means any `thinking` text that
    reaches `history` would be:
      * sent back to the SAME agent on the next tool-call iteration of
        the same turn (in-turn leak), and
      * available to `build_context` for any other agent whose context
        is rebuilt from this shared history (latent cross-agent leak).
    `agent.Agent.call()` already avoids attaching `thinking` to the
    assistant messages it produces, but this second guard cleans up any
    pre-existing thinking fields that may have been persisted by an
    older version of the app or loaded from a saved session JSON file.
    """
    def __init__(self, manager: AgentManager):
        self.manager = manager
        self.history: List[dict] = []
        self.pending_user_messages: List[str] = []
        self.paused = False
        self._lock = threading.RLock()

    def add_user_message(self, text: str, sender: str = "user"):
        with self._lock:
            if text.strip():
                self.history.append({
                    "role": "user",
                    "sender": sender,
                    "content": text.strip(),
                    "timestamp": time.time()
                })

    def add_messages(self, messages: List[dict]):
        """Append agent-produced messages to the shared history.

        Strips the `thinking` field from every message as a
        defense-in-depth measure (see class docstring). Only `role`,
        `content`, `sender`, `name`, `tool_name`, `tool_calls` and
        `timestamp` are preserved — these are the fields actually
        consumed by `build_context` and the GUI.

        Messages that don't carry a `timestamp` yet get one stamped
        here (unix epoch seconds) — see the class docstring on message
        timestamps. Messages that already have one (e.g. re-queued
        dialogue snapshots) keep their original time.
        """
        with self._lock:
            now = time.time()
            for msg in messages:
                # Defense-in-depth: drop `thinking` so it can never leak
                # back into any agent's next inference call, regardless of
                # whether the message came from a fresh agent.call() or
                # from a session JSON saved by an older app version.
                # While we're at it, stamp the entry time when missing
                # (shallow copy so the caller's dicts are not mutated).
                if isinstance(msg, dict):
                    if "thinking" in msg or "timestamp" not in msg:
                        msg = {k: v for k, v in msg.items() if k != "thinking"}
                        if "timestamp" not in msg:
                            msg["timestamp"] = now
                self.history.append(msg)

    def drain_pending(self, user_name: str = "user") -> Optional[str]:
        """
        Atomically drain pending user messages and append them to history.

        Returns the combined text if any messages were drained, else None.
        The first BackendThread to grab the lock wins; others will see an
        empty queue and return None. This is the desired behaviour because
        the history is shared.
        """
        with self._lock:
            if not self.pending_user_messages:
                return None
            combined = "\n---\n".join(self.pending_user_messages)
            self.pending_user_messages.clear()
            self.history.append({
                "role": "user",
                "sender": user_name,
                "content": combined.strip(),
                "timestamp": time.time()
            })
            return combined

    def get_history_snapshot(self) -> List[dict]:
        """Return a copy of the history under lock.

        The returned list is a shallow copy of the history list at the
        moment of the call; individual message dicts are shared with
        the internal history. Callers that need to mutate messages
        should copy them first. `build_context` only reads from these
        dicts and constructs brand-new dicts for the inference call, so
        sharing is safe.
        """
        with self._lock:
            return list(self.history)

    def rename_sender(self, old_name: str, new_name: str):
        """Rename an agent in shared history without touching tool names.

        Both `sender` and message-level `name` fields are updated when they
        contain the old agent name. Tool/function names are deliberately not
        changed.
        """
        old_name = (old_name or "").strip()
        new_name = (new_name or "").strip()
        if not old_name or not new_name or old_name == new_name:
            return
        with self._lock:
            for msg in self.history:
                if not isinstance(msg, dict):
                    continue
                if msg.get("sender") == old_name:
                    msg["sender"] = new_name
                if msg.get("name") == old_name:
                    msg["name"] = new_name

    def prune_history(self, keep_last_n: int) -> str:
        """Keep only the last ``keep_last_n`` messages in shared history.

        Used by the ``core_prune_agent_context`` tool so agents can
        manage their own context window without restarting the app.
        Thread-safe.
        """
        with self._lock:
            current = len(self.history)
            if keep_last_n < 0:
                return "keep_last_n must be >= 0."
            if keep_last_n >= current:
                return f"History has only {current} messages, nothing to prune."
            self.history = self.history[-keep_last_n:] if keep_last_n > 0 else []
            return f"Pruned shared history: kept last {keep_last_n} of {current} messages."

    def to_dict(self) -> dict:
        with self._lock:
            # Also strip `thinking` when serialising — saved sessions
            # should not carry raw reasoning text either, so that older
            # files loaded by newer code (and vice versa) stay clean.
            clean_history = []
            for msg in self.history:
                if isinstance(msg, dict) and "thinking" in msg:
                    clean_history.append(
                        {k: v for k, v in msg.items() if k != "thinking"}
                    )
                else:
                    clean_history.append(msg)
            return {"history": clean_history}

    def from_dict(self, data: dict):
        with self._lock:
            loaded = data.get("history", [])
            # Strip any `thinking` fields that may exist in a saved
            # session produced by an older app version.
            self.history = []
            for msg in loaded:
                if isinstance(msg, dict) and "thinking" in msg:
                    self.history.append(
                        {k: v for k, v in msg.items() if k != "thinking"}
                    )
                else:
                    self.history.append(msg)
            self.pending_user_messages.clear()
