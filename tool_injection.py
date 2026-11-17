"""
tool_injection.py

Prompt-injection / tool-result-prepend subsystem.

WHY THIS MODULE EXISTS
======================

Feature request (communicating with an agent DURING its inference):

    "Add an additional prepend (before tools result text) section
    (function) for each tool result returned to the agent — some kind
    of endpoint checking what the agent will get and what text it will
    receive (kind of communicating with the agent during his
    inference... prompt injection — but only when the agent uses a
    tool). It will accumulate all passed info; then after sending to
    the agent it will be cleared for the next tool-call gathering."

The design:

  * EVERY tool result that Agent.call returns to the model is
    prepended with a short SYSTEM section that states the tool-call
    usage count ("call #N, M left") — so the model always knows how
    much of its tool budget it has burned.

  * The human operator can queue free-form text ("prompt injection")
    for a specific agent from the GUI (the "Inject" button on the
    agent's sidebar widget). Queued texts ACCUMULATE in a per-agent
    buffer. The buffer is drained the next time the agent finishes a
    tool call: everything accumulated is delivered prepended to that
    tool result, and the buffer is CLEARED, ready for the next
    gathering cycle. Injection therefore only ever happens when the
    agent actually USES a tool — never mid-stream, never on
    text-only turns.

  * The injected tool-result message is composed only for the LOCAL
    message list of the running turn (what the model actually sees).
    The copy stored into shared history is CLEAN (no prepend), so
    injected text is delivered exactly once and does not pollute the
    agent's future contexts.

Thread safety: the injection buffer is guarded by a lock — the GUI
thread queues injections while a BackendThread worker drains them
mid-inference, without races.

PUBLIC API (all standalone, reusable in future projects)
=========================================================

    build_tool_result_prepend(...)        -> str
        THE standalone function. Builds the prepended SYSTEM section
        for a tool result (tool-call counter + operator injections).
        Pure: no module state needed if you pass pending_injections.

    format_messages_for_debug(...)        -> str
        Renders the EXACT message list a model receives (system +
        history, with tool_calls / images / name fields) as a
        readable dump — for the Thinking tab / debugging.

    tools_limit_exceeded_notice()       -> str
        STANDALONE: the exact text returned to the agent INSTEAD of
        tool usage when the per-turn tool budget is exhausted
        ("Tools limit exceeded. Generate final answer. ...").

    add_injection(agent_name, text)       -> int  (pending count)
        Queue an operator injection for an agent.

    take_pending_injections(agent_name)   -> list (atomic read+clear)
        Drain the agent's queue (used when composing a tool result).

    get_pending_injections(agent_name)    -> list (peek, no clear)
    clear_injections(agent_name)          -> None
    pending_injection_count(agent_name)   -> int
        Housekeeping / introspection helpers.
"""

import json
import threading
import time
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------------
# Per-agent injection buffer
# ----------------------------------------------------------------------
# agent_name -> list of {"text": str, "queued_at": float}
# Entries accumulate until drained by take_pending_injections() at the
# next tool-result composition for that agent.
_PENDING: Dict[str, List[Dict[str, Any]]] = {}
_LOCK = threading.RLock()


def add_injection(agent_name: str, text: str) -> int:
    """Queue an operator prompt-injection for *agent_name*.

    The text is delivered to the agent PREPENDED to its NEXT tool
    result (see build_tool_result_prepend). Multiple queued texts
    accumulate and are delivered together, then the buffer is cleared.

    Returns the number of pending injections for the agent after this
    call (useful for GUI confirmation messages).
    """
    agent_name = str(agent_name or "").strip()
    cleaned = str(text or "").strip()
    if not agent_name or not cleaned:
        with _LOCK:
            return len(_PENDING.get(agent_name, [])) if agent_name else 0
    with _LOCK:
        bucket = _PENDING.setdefault(agent_name, [])
        bucket.append({"text": cleaned, "queued_at": time.time()})
        return len(bucket)


def take_pending_injections(agent_name: str) -> List[Dict[str, Any]]:
    """Atomically return AND CLEAR the pending injections for an agent.

    Called by Agent.call when it composes a tool-result message: the
    drained entries are embedded in the prepend section of that tool
    result, and the buffer starts empty for the next gathering cycle.
    """
    with _LOCK:
        return _PENDING.pop(str(agent_name or "").strip(), [])


def get_pending_injections(agent_name: str) -> List[Dict[str, Any]]:
    """Peek at the pending injections WITHOUT clearing them."""
    with _LOCK:
        return [dict(item) for item in
                _PENDING.get(str(agent_name or "").strip(), [])]


def clear_injections(agent_name: str) -> int:
    """Discard any pending injections for an agent. Returns how many
    were dropped."""
    with _LOCK:
        return len(_PENDING.pop(str(agent_name or "").strip(), []))


def pending_injection_count(agent_name: str) -> int:
    """How many injections are currently queued for an agent."""
    with _LOCK:
        return len(_PENDING.get(str(agent_name or "").strip(), []))


def all_pending_counts() -> Dict[str, int]:
    """Pending injection counts for EVERY agent with a non-empty buffer.

    (Added for the headless backend's state API — pure introspection,
    no behaviour change: reads the same buffer the functions above use.)
    """
    with _LOCK:
        return {name: len(items) for name, items in _PENDING.items() if items}


# ----------------------------------------------------------------------
# The standalone prepend builder
# ----------------------------------------------------------------------
def build_tool_result_prepend(tool_name: str,
                              tool_call_number: int,
                              max_tool_calls: int = 0,
                              pending_injections: Optional[List[Any]] = None,
                              agent_name: Optional[str] = None,
                              extra_info: Optional[List[str]] = None) -> str:
    """Build the SYSTEM section prepended to a tool result.

    STANDALONE FUNCTION — deliberately usable outside this app: pass
    everything explicitly, get a string back, concatenate the raw tool
    result after it. It never touches module state unless you pass
    objects returned by take_pending_injections().

    Parameters
    ----------
    tool_name : str
        Name of the tool whose result follows.
    tool_call_number : int
        Which tool call this is within the current turn (1-based,
        cumulative — multiple parallel calls each get their own number).
    max_tool_calls : int
        The agent's tool-call budget for the turn (Agent.max_tool_iterations).
        0 (or negative) means UNLIMITED — then no "N left" is reported.
    pending_injections : list, optional
        Entries from take_pending_injections() (dicts with "text" /
        "queued_at") — or plain strings, both are accepted.
    agent_name : str, optional
        Included in the section header when given (helps the model know
        who the info applies to in multi-agent contexts).
    extra_info : list of str, optional
        Any extra lines to include in the section (future extensions).

    Returns
    -------
    str
        The section text INCLUDING the trailing blank-line separator;
        concatenate the raw tool result directly after it.
    """
    lines: List[str] = []
    lines.append("[SYSTEM INFO] (injected before tool result — read first)")

    if agent_name:
        lines.append(f"Agent: {agent_name}")
    lines.append(f"Tool: {tool_name}")

    # --- Tool usage count: "what number of tool was used + how many left"
    if max_tool_calls and int(max_tool_calls) > 0:
        remaining = max(0, int(max_tool_calls) - int(tool_call_number))
        lines.append(
            f"Tool call number {int(tool_call_number)} of {int(max_tool_calls)} "
            f"this turn — about {remaining} left."
        )
    else:
        lines.append(
            f"Tool call number {int(tool_call_number)} this turn "
            f"(no tool-call limit)."
        )

    # --- Accumulated operator injections (if any)
    for item in (pending_injections or []):
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            queued_at = item.get("queued_at")
            if queued_at:
                try:
                    stamp = time.strftime("%H:%M:%S",
                                           time.localtime(float(queued_at)))
                    lines.append(f"OPERATOR INJECTION (queued {stamp}):")
                except (TypeError, ValueError, OSError):
                    lines.append("OPERATOR INJECTION:")
            else:
                lines.append("OPERATOR INJECTION:")
            lines.append(text if text else "(empty)")
        else:
            lines.append("OPERATOR INJECTION:")
            lines.append(str(item).strip() or "(empty)")

    # --- Future extension lines
    for info in (extra_info or []):
        lines.append(str(info))

    lines.append("[END SYSTEM INFO]")

    return "\n".join(lines) + "\n\n"


# ----------------------------------------------------------------------
# Tools-limit-exceeded notice ("agent silently stops when it exceeds
# the tools limit and then calls a tool")
# ----------------------------------------------------------------------
# The agent loop returns this text INSTEAD of a tool result once the
# per-turn tool budget is exhausted, so the model can finish its job
# by generating a FINAL ANSWER instead of silently stopping.
#
# It is delivered in two situations (both mean "no more tools this
# turn — finalize now"):
#   1. as the tool result (role "tool") for a tool call that was
#      refused because the budget is already exhausted, and
#   2. as a final user-role guidance message when the iteration
#      budget runs out mid-turn (the model's last tool calls were
#      legitimate, their results were delivered, and no further
#      tool iterations are allowed).

TOOLS_LIMIT_EXCEEDED_MESSAGE = (
    "Tools limit exceeded. Generate final answer. "
    "Your tools limit will be renew on next turn"
)


def tools_limit_exceeded_notice() -> str:
    """STANDALONE FUNCTION — the exact text the agent receives INSTEAD
    of tool usage once its per-turn tool budget is exhausted.

    Deliberately a single fixed sentence, returned verbatim: the model
    must understand, without ambiguity, that it cannot use tools again
    this turn and must produce its final answer now. The notice
    replaces the tool result entirely (no prepend section, no counter,
    no operator injections) so nothing dilutes that instruction.

    Reusable anywhere — pure function, no module state.
    """
    return TOOLS_LIMIT_EXCEEDED_MESSAGE


# ----------------------------------------------------------------------
# Full-prompt debug dump (feature: "see exactly what the agent receives")
# ----------------------------------------------------------------------
def _tool_names(tools: Optional[List[Any]]) -> List[str]:
    """Normalise a tools argument (definition dicts, name strings, or
    None) into a list of tool name strings."""
    if not tools:
        return []
    names: List[str] = []
    for t in tools:
        if isinstance(t, dict):
            fn = t.get("function", {})
            name = fn.get("name", "") if isinstance(fn, dict) else ""
        else:
            name = getattr(t, "__name__", str(t))
        if name:
            names.append(str(name))
    return names


def format_messages_for_debug(messages: List[Dict[str, Any]],
                              agent_name: str = "",
                              model: str = "",
                              endpoint: str = "",
                              tools: Optional[List[Any]] = None,
                              num_ctx: Optional[int] = None,
                              temperature: Optional[float] = None,
                              iteration: Optional[int] = None,
                              as_json: bool = False) -> str:
    """Render the EXACT message list a model receives as a text dump.

    Every message is shown completely: role, full content, tool_calls
    (pretty JSON), images, and the name field. This is precisely the
    payload Agent.call sends to Ollama (the ollama client serialises
    role / content / tool_calls / images / name; internal metadata such
    as `sender` and `tool_name` is not transmitted).

    STANDALONE FUNCTION — usable anywhere; nothing module-global.

    Set ``as_json=True`` to emit the raw JSON serialisation of the
    messages list instead of the readable layout (still with a header).
    """
    bar = "=" * 78
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    out: List[str] = []

    out.append(bar)
    title = "FULL PROMPT — exactly what the model receives"
    if agent_name:
        title += f" — agent '{agent_name}'"
    out.append(title)
    meta = f"time: {stamp}"
    if endpoint:
        meta += f"   endpoint: {endpoint}"
    if model:
        meta += f"   model: {model}"
    out.append(meta)
    if iteration is not None:
        out.append(f"inference iteration: {iteration}")
    opts = []
    if num_ctx is not None:
        opts.append(f"num_ctx={num_ctx}")
    if temperature is not None:
        opts.append(f"temperature={temperature}")
    if opts:
        out.append("options: " + "  ".join(opts))
    names = _tool_names(tools)
    if names:
        out.append(f"tools offered ({len(names)}): {', '.join(names)}")
    else:
        out.append("tools offered: NONE (text-only inference)")
    out.append(f"messages: {len(messages)}")
    out.append(bar)

    if as_json:
        try:
            out.append(json.dumps(messages, ensure_ascii=False, indent=2))
        except (TypeError, ValueError):
            out.append(repr(messages))
        out.append(bar)
        out.append("END OF FULL PROMPT (raw JSON form)")
        out.append(bar)
        return "\n".join(out)

    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        header = f"--- messages[{i}] role={role}"
        if msg.get("name"):
            header += f" name={msg['name']}"
        if msg.get("images"):
            try:
                header += f" images=({len(msg['images'])} file(s))"
            except TypeError:
                pass
        out.append(header)
        out.append("-" * 60)

        content = msg.get("content", "")
        if not isinstance(content, str):
            content = repr(content)
        out.append(content if content else "(empty content)")

        if msg.get("tool_calls"):
            out.append("-" * 60)
            try:
                out.append("tool_calls: " + json.dumps(
                    msg["tool_calls"], ensure_ascii=False, indent=2))
            except (TypeError, ValueError):
                out.append(f"tool_calls: {msg['tool_calls']!r}")

        if msg.get("images"):
            try:
                out.append("images: " + ", ".join(
                    str(p) for p in msg["images"]))
            except TypeError:
                out.append(f"images: {msg['images']!r}")

        out.append("")  # blank line between messages

    out.append(bar)
    out.append("END OF FULL PROMPT")
    out.append(bar)
    return "\n".join(out)
