# context.py
#
# Global singletons shared between the GUI / backend threads and the
# user-defined tools (loaded from tools/*.py).
#
# `prompt_loader` holds the PromptLoader that scans the prompts/ folder
# for reusable prompt snippets (.txt files). Agents select prompt files
# by name; at inference time Agent.call appends their texts to the
# system prompt inside a <rules>...</rules> section. Exposed here (like
# `loader`) so both Agent.call and the core tools (core_list_prompts)
# can reach it from anywhere.
#
# `dialogue_store` holds the DialogueStore (dialogue_store.py) that
# keeps the PRIVATE side chats between pairs of agents — the backing
# store of the core_agent_execute_dialogue tool. Each ordered pair of
# agents (caller -> callee) has exactly one dialogue, persisted as its
# own little session file under agent_dialogues/. The dialogue tools
# (core_agent_execute_dialogue / core_list_dialogues /
# core_get_dialogue / core_reset_dialogue) reach it through this
# attribute, exactly like the other shared singletons.
#
# `conversation` was added so that core_prune_agent_context (and any
# user tool that wants to) can prune shared history without reaching
# into the manager's internals. Every message that enters the shared
# history is now also stamped with a `timestamp`, which the
# core_chat_messages tool exposes (count / index / time queries).
#
# `client_manager` was added so that `core_agent_execute` can obtain
# the correct OllamaClient for any agent's endpoint — including when
# the calling agent and the called agent live on different machines.
#
# The three router callables (`thinking_router`, `agent_status_router`,
# `message_router`) are set by the MainWindow at startup. They let
# `core_agent_execute` forward an invoked agent's live-thinking stream,
# running/idle status, and (optional) chat output back into the GUI
# exactly as if the agent were running inside its own BackendThread.
# Each callable is a thin wrapper around a Qt signal's `.emit()`,
# which is thread-safe to call from any thread (including the
# ThreadPoolExecutor worker that runs tool code).
#
# NEW: the EndpointGate registry (bottom of this file) coordinates the
# "pause a remote endpoint while an agent is executed on it" behaviour
# requested for `core_agent_execute`. See the EndpointGate docstring
# for the full design.

import threading

manager = None        # AgentManager instance
loader = None         # ToolLoader instance
prompt_loader = None  # PromptLoader instance — reusable prompt snippets (prompts/*.txt)
conversation = None   # Conversation instance (shared history + queue)
client_manager = None  # ClientManager instance — for cross-endpoint agent execution
dialogue_store = None  # DialogueStore instance — agent-to-agent side chats (agent_dialogues/)

# NATIVE MCP engine (mcp_manager.MCPManager instance).
#
# Set once by main.py at startup. Gives every subscribed agent DIRECT
# access to the tools of any MCP server (mcp__<server>__<tool>) without
# wrapper tools. The same object is also published as the module-level
# singleton inside mcp_manager.py (mcp_manager.get_manager()) — user
# tools may reach it through EITHER path. None means the MCP feature is
# unavailable; every consumer must degrade gracefully in that case.
mcp_manager = None

# --- GUI-side routers (set by MainWindow.__init__) ---------------------
# Each of these is a callable; calling it from any thread is safe because
# the MainWindow wraps the actual GUI update behind a Qt signal.
#
#   thinking_router(endpoint: str, agent_name: str, event_type: str,
#                   detail: str, extra: str) -> None
#     Forwards a thinking-tab event for the given (endpoint, agent_name).
#     Same payload shape as BackendThread.thinking_update.
#
#   agent_status_router(endpoint: str, agent_name: str,
#                       running: bool, model: str) -> None
#     Flips the sidebar status of the named agent between
#     "Waiting" and "Running: <model>". Same payload shape as
#     BackendThread.agent_status_changed.
#
#   message_router(sender: str, text: str) -> None
#     Appends a message to the chat display. Used by
#     `core_agent_execute` to surface the called agent's final answer
#     in the chat UI when the agent is NOT silent.
thinking_router = None
agent_status_router = None
message_router = None


# ======================================================================
# EndpointGate — cooperative per-endpoint pause for remote agent execution
# ======================================================================
class EndpointGate:
    """
    Coordination gate for ONE Ollama endpoint ("local" or "ip:port").

    Purpose: when agent A (running on endpoint X) invokes agent B (on
    endpoint Y) via `core_agent_execute`, the tool wants endpoint Y to
    itself — Y's own scheduler thread may be mid-turn with another agent
    C, and two concurrent inference streams against the same Ollama
    server is exactly the situation we want to avoid.

    Mechanism (cooperative, never interrupts a running stream):

      1. `core_agent_execute` calls `request_pause()` on Y's gate and
         then waits for the gate to report "quiesced".

      2. "Quiesced" means: the endpoint's scheduler thread is NOT about
         to issue an inference request — either it is idle, it is
         blocked waiting for the gate to open, or its in-flight agent
         has paused itself at a safe point. It becomes true:
           - when an agent mid-turn reaches its NEXT tool-call iteration
             boundary (Agent.call checks the gate at the top of every
             iteration: "nearest tool call"), or
           - when the agent's current inference finishes and the turn
             ends (BackendThread's finally: "end of inference").

      3. Sub-agents executed via `core_agent_execute` themselves SKIP
         the gate check (they are the reason the endpoint is paused) —
         this is what `agent_call_stack` being non-empty identifies.

      4. When the remote execution finishes, `release_pause()` re-opens
         the gate and the paused scheduler / agent resumes exactly
         where it left off.

    Pause requests are ref-counted so nested remote executions on the
    same endpoint are handled correctly.

    All methods are thread-safe.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._lock = threading.Lock()
        self._pause_requests = 0                 # active remote executions wanting this endpoint paused
        self._open = threading.Event()           # set  -> scheduled turns may run
        self._open.set()
        self._quiesced = threading.Event()       # set  -> endpoint is at a safe pause point / idle
        self._quiesced.set()

    # --- used by core_agent_execute (runs on the CALLER's thread) ---

    def request_pause(self):
        """Ask this endpoint's scheduler to pause. Ref-counted."""
        with self._lock:
            self._pause_requests += 1
        self._open.clear()

    def release_pause(self):
        """Release one pause request. Re-opens the gate when the last
        remote execution on this endpoint finishes."""
        with self._lock:
            self._pause_requests = max(0, self._pause_requests - 1)
            remaining = self._pause_requests
        if remaining == 0:
            self._open.set()

    def wait_quiesced(self, timeout: float = None) -> bool:
        """Wait until the endpoint reaches a safe pause point (or is idle).

        Returns True when quiesced, False on timeout. Callers should
        treat a timeout as "proceed anyway with a warning" rather than
        an error — the endpoint's agent is simply taking very long to
        reach its next iteration boundary.
        """
        return self._quiesced.wait(timeout)

    # --- used by BackendThread / Agent.call (the endpoint's OWN thread) ---

    def is_paused(self) -> bool:
        return not self._open.is_set()

    def wait_open(self, timeout: float = None) -> bool:
        """Block while the endpoint is paused (used by Agent.call between
        tool-call iterations). Returns True when open, False on timeout."""
        return self._open.wait(timeout)

    def begin_turn(self):
        """Mark: the endpoint's scheduler is starting a turn that may issue
        inference requests (clears the quiesced flag)."""
        with self._lock:
            self._quiesced.clear()

    def mark_quiesced(self):
        """Mark: the endpoint's scheduler reached a safe pause point (it is
        blocked on the gate) or is idle. Remote executors waiting on
        wait_quiesced() may now proceed."""
        self._quiesced.set()

    def end_turn(self):
        """Mark: the endpoint's scheduler finished a turn (idle again)."""
        self._quiesced.set()

    # --- used on application shutdown ---

    def hard_release(self):
        """Force-open the gate regardless of pending pause requests.

        Called from MainWindow.closeEvent via
        context.release_all_endpoint_gates() so that agents blocked on a
        remote-execution pause can finish their turn and their threads
        can exit cleanly during shutdown.
        """
        with self._lock:
            self._pause_requests = 0
        self._open.set()
        self._quiesced.set()


# ----------------------------------------------------------------------
# EndpointGate registry — one gate per endpoint, created lazily.
# ----------------------------------------------------------------------
_endpoint_gates: dict = {}
_gates_lock = threading.Lock()


def get_endpoint_gate(endpoint: str) -> EndpointGate:
    """Return (creating if needed) the gate for an endpoint string.

    Accepts "local", "ip:port" or "http://..." — anything an Agent's
    `endpoint` field may contain. Normalised the same way AgentManager
    normalises endpoints ("" / None -> "local").
    """
    ep = (endpoint or "local").strip() or "local"
    with _gates_lock:
        if ep not in _endpoint_gates:
            _endpoint_gates[ep] = EndpointGate(ep)
        return _endpoint_gates[ep]


def release_all_endpoint_gates():
    """Force-open every gate (application shutdown).

    Lets agents that are blocked mid-turn on a remote-execution pause
    resume, finish their current turn, and let their BackendThreads
    observe the shutdown flag and exit.
    """
    with _gates_lock:
        gates = list(_endpoint_gates.values())
    for gate in gates:
        try:
            gate.hard_release()
        except Exception:
            pass
