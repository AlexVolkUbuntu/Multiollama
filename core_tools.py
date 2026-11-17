# core_tools.py
#
# Built-in tools written to the tools/ directory at startup.
#
# FIX: core_prune_agent_context previously referenced `agent.messages`,
# which does not exist on the Agent class — the call always returned an
# error. It now prunes the SHARED conversation history via
# context.conversation.prune_history(...), which actually works.

CORE_TOOLS_CODE = {
    "file_tools.py": '''
import os

def core_read_file(path: str) -> str:
    """
    Read the entire contents of a file as a UTF-8 string.

    Args:
        path (str): Relative or absolute path to the file.

    Returns:
        str: The file content. If the file does not exist or cannot be read,
             an exception is raised (handled by the caller).
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def core_write_file(path: str, content: str) -> str:
    """
    Write (overwrite) a string to a file. Creates the file if it does not exist.

    Args:
        path (str): Relative or absolute path to the file.
        content (str): Text content to write.

    Returns:
        str: Confirmation message including the file path.
    """
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"Written to {path}"

def core_append_file(path: str, content: str) -> str:
    """
    Append a string to the end of an existing file. Creates the file if missing.

    Args:
        path (str): Relative or absolute path to the file.
        content (str): Text content to append.

    Returns:
        str: Confirmation message including the file path.
    """
    with open(path, 'a', encoding='utf-8') as f:
        f.write(content)
    return f"Appended to {path}"

def core_list_files(directory: str = ".") -> list:
    """
    Return a list of all entries (files and directories) in the given directory.

    Args:
        directory (str): Directory path. Defaults to the current directory ".".

    Returns:
        list: List of entry names as strings.
    """
    return os.listdir(directory)
''',

    "exec_tools.py": '''
import io
import sys
import contextlib

def core_run_python(code: str) -> str:
    """
    Execute arbitrary Python code and capture any printed output.

    This tool is intended for quick, self-contained code snippets. The code runs
    in a fresh local namespace and cannot directly affect the host process
    (though side-effects like file writes or network calls are possible).

    Args:
        code (str): A string containing valid Python code.

    Returns:
        str: The captured stdout from the execution, or a success message if no
             output was produced. If an exception occurs, the error traceback
             is returned as a string.
    """
    local_namespace = {}
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, globals(), local_namespace)
        output = stdout_capture.getvalue()
        return output if output else "Code executed successfully (no output)."
    except Exception as e:
        return f"Execution error: {e}"
''',

    "agent_tools.py": '''
import context
import sys
import importlib
import os
import threading
from pathlib import Path
from typing import Optional
import json
import ast
import time
from datetime import datetime

# Import the agent call-stack ContextVar from the agent module so that
# `core_agent_execute` can read / write it. This is the recursion-guard
# that prevents agent A from calling agent B which calls agent A again
# (each paused mid-turn waiting for the other to finish).
from agent import current_agent_name, agent_call_stack

# The persistent side-chat store (agent_dialogues folder) used by the
# dialogue tools below. Imported defensively so this tool file stays
# loadable even in stripped-down setups where the module is missing.
try:
    import dialogue_store as _dialogue_store
except Exception:
    _dialogue_store = None

_fallback_dialogue_store = None


def _get_dialogue_store():
    """Return the app's DialogueStore (agent-to-agent side chats).

    Normally main.py publishes it as context.dialogue_store at startup.
    The lazy fallback below keeps the dialogue tools working even when
    that attribute was never set (e.g. an older main.py was kept while
    the tool files were updated).
    """
    global _fallback_dialogue_store
    store = getattr(context, "dialogue_store", None)
    if store is not None:
        return store
    if _dialogue_store is None:
        return None
    if _fallback_dialogue_store is None:
        _fallback_dialogue_store = _dialogue_store.DialogueStore()
    context.dialogue_store = _fallback_dialogue_store
    return _fallback_dialogue_store

# ----- helper for converting string -> list -----
def _parse_string_to_list(value):
    """Convert a string representation of a list into a Python list.
    Supports JSON and Python literal syntax. Returns the original value
    if it is already a list, or None if value is None. Raises ValueError
    on failure.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        # Try JSON first (strict)
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        # Try Python literal (e.g., "['a','b']")
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        raise ValueError(f"Could not parse string as a list: {value}")
    raise ValueError(f"Expected list, string, or None, got {type(value).__name__}")

# ----- helper for normalising bool arguments -----
# Models often send booleans as 0/1, "true"/"false", "yes"/"no", or even
# the strings "True"/"False". This helper coerces all of them to a real
# Python bool so the rest of the code doesn't have to handle each case.
def _coerce_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "yes", "y", "on", "active", "enabled"):
            return True
        if s in ("0", "false", "no", "n", "off", "inactive", "disabled", ""):
            return False
    return bool(value)

def core_create_agent(name: str, prompt: str,
                 model: str = "qwen3.5:9b",
                 num_ctx: int = 32000,
                 temperature: float = 0.5,
                 tools: list = None,
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
                 mcp_servers: list = None,
                 prompts: list = None) -> str:
    """
    Create a new conversational agent with specified configuration.

    The agent will be added to the manager and can then be used for chat.

    Args:
        name (str): Unique identifier for the agent (used in future calls).
        prompt (str): System prompt defining the agent's personality and behavior.
        model (str): Name of the LLM model to use (e.g., 'qwen3.5:9b').
        num_ctx (int): Context window size in tokens (default 32000).
        temperature (float): Sampling temperature (0.0-1.0). Lower values = more deterministic.
        tools (list[str], optional): List of tool names the agent is allowed to call.
        max_context_messages (int): Maximum number of recent messages kept in context.
                                    Older messages are dropped to stay within token limits.
        max_tool_iterations (int): Max tool-call iterations per turn. 0 = unlimited.
                                    Default 1000.
        endpoint (str): Which Ollama machine this agent runs on. Use "local" for the
                        local Ollama server, or "ip:port" (e.g. "192.168.3.3:11434")
                        for a remote Ollama server on another PC in the LAN.
                        One parallel loop is spawned per distinct endpoint, so agents
                        on different machines run simultaneously.
        active (bool): Whether the agent is IN the round-robin rotation (True, default)
                       or SKIPPED by the scheduler (False). An inactive agent keeps
                       its config and history but is never inferenced until it is
                       re-activated (see `core_agent_active`). Useful for spinning
                       up an agent in a 'standby' state and letting another agent
                       switch it on at the right moment.
        silent (bool): Whether the agent's final answers are printed to the chat UI.
                      True  -> the answer is stored in shared history (so other
                               agents can read it) but NOT shown in the chat window;
                               the live thinking stream still appears in the
                               Thinking tab.
                      False (default) -> the answer is shown in chat as usual.
                      Useful for 'worker' agents whose output is meant for other
                      agents to consume rather than for the human user.
        description (str): Short human-readable note about what this agent is for.
                      Used for easy identification in agent management (GUI
                      tooltips, saved-agent listings). NEVER sent to the model —
                      it does not affect inference.
        use_tools (bool): Master switch for tool usage (default True). When False,
                      no tools are shown to the model on inference, no tool calls
                      are executed, and past tool calls are stripped from the
                      agent's context. The tools list is kept for re-enabling.
        image (bool): Enable image input (default False). When True, image files
                      referenced in the newest message are sent to the model in
                      the model call (including mid tool-usage). Only enable for
                      VISION models — non-vision models fail with images attached.
        audio (bool): Enable audio input (default False). When True, .wav files
                      referenced in the newest message are sent to the model the
                      same way images are (audio-capable models such as gemma
                      audio models accept base64 wav through the same channel).
                      Only .wav files are attached; can be combined with image.
        mcp (bool): Enable NATIVE MCP usage (default False). When True, the
                      agent may subscribe to any number of MCP servers and
                      every tool of every subscribed server is exposed to it
                      DIRECTLY as first-class tools named
                      mcp__<server>__<tool> (no wrapper syntax, no meta-tools).
                      The MCP management tools (mcp_list_servers,
                      mcp_subscribe, ...) become auto-available so the agent
                      can discover and subscribe to servers by itself.
        mcp_servers (list[str], optional): Registry names of MCP servers this
                      agent should be subscribed to from the start (see
                      `mcp_list_servers`). Unknown names are kept but produce
                      no tools until they are registered.
        prompts (list[str], optional): Names of prompt files from the prompts/
                      folder (WITHOUT the ".txt" extension) whose texts should
                      be appended to this agent's system prompt inside a
                      <rules>...</rules> section at inference time. Use
                      `core_list_prompts` to see the available names. Unknown
                      names are kept but silently skipped until the file exists.

    Returns:
        str: Success or error message.
    """
    manager = context.manager
    if manager is None:
        return "Error: Agent manager not initialized."

    # ----- convert string arguments to proper types -----
    try:
        if isinstance(num_ctx, str):
            num_ctx = int(num_ctx.strip())
        if isinstance(temperature, str):
            temperature = float(temperature.strip())
        if isinstance(max_context_messages, str):
            max_context_messages = int(max_context_messages.strip())
        if isinstance(max_tool_iterations, str):
            max_tool_iterations = int(max_tool_iterations.strip())
        if tools is not None:
            tools = _parse_string_to_list(tools)
            # Ensure all elements are strings
            tools = [str(t) for t in tools]
        if endpoint is None:
            endpoint = "local"
        else:
            endpoint = str(endpoint).strip() or "local"
        # Coerce the boolean flags — models often pass them as 0/1 or
        # the strings "true"/"false".
        active = _coerce_bool(active, default=True)
        silent = _coerce_bool(silent, default=False)
        use_tools = _coerce_bool(use_tools, default=True)
        image = _coerce_bool(image, default=False)
        audio = _coerce_bool(audio, default=False)
        mcp = _coerce_bool(mcp, default=False)
        if mcp_servers is not None:
            if isinstance(mcp_servers, str):
                try:
                    mcp_servers = _parse_string_to_list(mcp_servers)
                except ValueError:
                    mcp_servers = [p.strip() for p in mcp_servers.split(",")
                                   if p.strip()]
            mcp_servers = [str(s).strip() for s in mcp_servers if str(s).strip()]
        if description is None:
            description = ""
        else:
            description = str(description)
        # Prompt files: accept a list or a string like "['a','b']" / "a,b".
        if prompts is not None:
            if isinstance(prompts, str):
                try:
                    prompts = _parse_string_to_list(prompts)
                except ValueError:
                    # Fall back to comma-separated names — models often
                    # send plain comma-separated strings.
                    prompts = [p.strip() for p in prompts.split(",")
                               if p.strip()]
            # Normalise: strings, trimmed, ".txt" suffix tolerated and
            # stripped (the GUI stores names WITHOUT the extension).
            prompts = [str(p).strip() for p in prompts]
            prompts = [p[:-4] if p.lower().endswith(".txt") else p
                       for p in prompts]
            prompts = [p for p in prompts if p]
    except (ValueError, TypeError) as e:
        return f"Error in parameter conversion: {e}"

    try:
        manager.create_agent(
            name, prompt, model, num_ctx, temperature,
            tools or [], max_context_messages,
            max_tool_iterations=max_tool_iterations,
            endpoint=endpoint,
            active=active,
            silent=silent,
            description=description,
            use_tools=use_tools,
            image=image,
            audio=audio,
            mcp=mcp,
            mcp_servers=mcp_servers or [],
            prompts=prompts or []
        )
        rules_note = (f" Prompt files: {', '.join(prompts)}."
                      if prompts else "")
        mcp_note = (f" MCP enabled (servers: {', '.join(mcp_servers or []) or 'none yet'})."
                    if mcp else "")
        return f"Agent '{name}' created on {endpoint}.{rules_note}{mcp_note}"
    except Exception as e:
        return f"Error: {e}"

def core_delete_agent(name: str) -> str:
    """
    Permanently remove an agent and its conversation history.

    Args:
        name (str): Name of the agent to delete.

    Returns:
        str: Confirmation message or error if the agent does not exist.
    """
    manager = context.manager
    if manager is None:
        return "Error: Agent manager not initialized."
    manager.delete_agent(name)
    return f"Agent '{name}' deleted."

def core_list_agents() -> list:
    """
    Return a list of all currently existing agent names.

    Returns:
        list[str]: Agent names (empty list if none exist or manager unavailable).
    """
    manager = context.manager
    return manager.list_agents() if manager else []

def core_list_endpoints() -> list:
    """
    Return a list of all distinct endpoints (machines) currently in use
    across all agents.

    Returns:
        list[str]: Distinct endpoint strings (e.g. ["local", "192.168.3.3:11434"]).
    """
    manager = context.manager
    return manager.get_endpoints() if manager else []

def core_get_agent_info(name: str) -> dict:
    """
    Retrieve the full configuration of an agent.

    Args:
        name (str): Name of the agent.

    Returns:
        dict: Contains keys: name, description, system_prompt, model, num_ctx,
              temperature, tools (list), max_context_messages, max_tool_iterations,
              endpoint, active (bool), silent (bool), use_tools (bool),
              image (bool), audio (bool), mcp (bool), mcp_servers (list of
              registry server names), prompts (list of prompt-file names).
              If the agent is not found, returns {'error': 'Agent not found'}.
    """
    manager = context.manager
    if manager is None:
        return {"error": "Manager not initialized"}
    agent = manager.get_agent(name)
    if not agent:
        return {"error": "Agent not found"}
    return {
        "name": agent.name,
        "description": getattr(agent, "description", ""),
        "system_prompt": agent.system_prompt,
        "model": agent.model,
        "num_ctx": agent.num_ctx,
        "temperature": agent.temperature,
        "tools": agent.tools,
        "max_context_messages": agent.max_context_messages,
        "max_tool_iterations": agent.max_tool_iterations,
        "endpoint": agent.endpoint or "local",
        "active": getattr(agent, "active", True),
        "silent": getattr(agent, "silent", False),
        "use_tools": getattr(agent, "use_tools", True),
        "image": getattr(agent, "image", False),
        "audio": getattr(agent, "audio", False),
        "mcp": getattr(agent, "mcp", False),
        "mcp_servers": list(getattr(agent, "mcp_servers", []) or []),
        "prompts": list(getattr(agent, "prompts", []) or []),
    }

def core_update_agent(
    name: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    num_ctx: Optional[int] = None,
    temperature: Optional[float] = None,
    tools: Optional[list] = None,
    max_context_messages: Optional[int] = None,
    max_tool_iterations: Optional[int] = None,
    endpoint: Optional[str] = None,
    active: Optional[bool] = None,
    silent: Optional[bool] = None,
    description: Optional[str] = None,
    use_tools: Optional[bool] = None,
    image: Optional[bool] = None,
    audio: Optional[bool] = None,
    mcp: Optional[bool] = None,
    mcp_servers: Optional[list] = None,
    prompts: Optional[list] = None) -> str:
    """
    Change one or more parameters of an existing agent.

    If `endpoint` is changed, the agent moves to a different machine's loop
    on the next rebuild. Use "local" or "ip:port" (e.g. "192.168.3.3:11434").

    `max_tool_iterations` controls how many tool-call iterations the agent
    may make in a single turn. 0 = unlimited.

    `active` (bool, optional): when False, the agent is skipped by the
        scheduler — it keeps its config and history but is never
        inferenced. Pass True to put it back into the rotation. The same
        effect can be achieved with the dedicated `core_agent_active`
        tool, which is the lighter-weight call when you only want to
        flip this one flag.

    `silent` (bool, optional): when True, the agent's final answers are
        stored in shared history (visible to other agents and in the
        Thinking tab) but NOT printed to the chat UI. Pass False to
        restore normal chat output. NOTE: this is a config-level flag
        and is intentionally NOT exposed via `core_agent_active` — the
        design is that an agent should not be able to mute itself at
        runtime to hide its output from the human user.

    `description` (str, optional): short human-readable note about what
        this agent is for. Used for identification in agent management
        (GUI tooltips, saved-agent listings) — never sent to the model.

    `use_tools` (bool, optional): master switch for tool usage. Pass
        False to disable tools entirely for this agent: no tools are
        shown to the model on inference and no tool calls are executed.
        Pass True to re-enable (the agent's tools list is kept).

    `image` (bool, optional): enable/disable image input for this agent.
        Only vision models should have this enabled — non-vision models
        fail when images are attached to the call.

    `audio` (bool, optional): enable/disable .wav audio input for this
        agent (audio-capable models only; can be combined with image).

    `mcp` (bool, optional): enable/disable NATIVE MCP usage for this
        agent. When True, the agent may subscribe to any number of MCP
        servers and every tool of every subscribed server is exposed to
        it DIRECTLY as first-class tools named mcp__<server>__<tool>.

    `mcp_servers` (list[str], optional): REPLACES the agent's MCP server
        subscription list (registry names — see `mcp_list_servers`).
        Pass [] to clear all subscriptions.

    `prompts` (list[str], optional): REPLACES the agent's selected prompt
        files (names from the prompts/ folder, WITHOUT the ".txt"
        extension — a "name.txt" entry is tolerated and normalised).
        Their texts are appended to the agent's system prompt inside a
        <rules>...</rules> section on every inference. Pass [] to clear
        the selection. Use `core_list_prompts` to discover names.
    """
    manager = context.manager
    if manager is None:
        return "Error: Manager not initialized."
    try:
        updates = {}
        if system_prompt is not None:
            updates["system_prompt"] = str(system_prompt)
        if model is not None:
            updates["model"] = str(model)
        if num_ctx is not None:
            if isinstance(num_ctx, str):
                num_ctx = int(num_ctx.strip())
            updates["num_ctx"] = int(num_ctx)
        if temperature is not None:
            if isinstance(temperature, str):
                temperature = float(temperature.strip())
            updates["temperature"] = float(temperature)
        if tools is not None:
            tools = _parse_string_to_list(tools)
            updates["tools"] = [str(t) for t in tools]
        if max_context_messages is not None:
            if isinstance(max_context_messages, str):
                max_context_messages = int(max_context_messages.strip())
            updates["max_context_messages"] = int(max_context_messages)
        if max_tool_iterations is not None:
            if isinstance(max_tool_iterations, str):
                max_tool_iterations = int(max_tool_iterations.strip())
            updates["max_tool_iterations"] = int(max_tool_iterations)
        if endpoint is not None:
            ep = str(endpoint).strip()
            updates["endpoint"] = ep if ep else "local"
        if active is not None:
            updates["active"] = _coerce_bool(active, default=True)
        if silent is not None:
            updates["silent"] = _coerce_bool(silent, default=False)
        if description is not None:
            updates["description"] = str(description)
        if use_tools is not None:
            updates["use_tools"] = _coerce_bool(use_tools, default=True)
        if image is not None:
            updates["image"] = _coerce_bool(image, default=False)
        if audio is not None:
            updates["audio"] = _coerce_bool(audio, default=False)
        if mcp is not None:
            updates["mcp"] = _coerce_bool(mcp, default=False)
        if mcp_servers is not None:
            if isinstance(mcp_servers, str):
                try:
                    mcp_servers = _parse_string_to_list(mcp_servers)
                except ValueError:
                    mcp_servers = [p.strip() for p in mcp_servers.split(",")
                                   if p.strip()]
            updates["mcp_servers"] = [str(s).strip() for s in mcp_servers
                                      if str(s).strip()]
        if prompts is not None:
            if isinstance(prompts, str):
                try:
                    prompts = _parse_string_to_list(prompts)
                except ValueError:
                    # Fall back to comma-separated names.
                    prompts = [p.strip() for p in prompts.split(",")
                               if p.strip()]
            # Normalise: strings, trimmed, ".txt" suffix stripped.
            prompts = [str(p).strip() for p in prompts]
            prompts = [p[:-4] if p.lower().endswith(".txt") else p
                       for p in prompts]
            updates["prompts"] = [p for p in prompts if p]

        manager.update_agent(name, **updates)
        return f"Agent '{name}' updated."
    except (ValueError, TypeError) as e:
        return f"Error: Invalid value type - {e}"
    except Exception as e:
        return f"Error: {e}"

def core_list_tools() -> list:
    """
    Return the current tool catalog with per-tool availability for THIS agent.

    Important distinction:
        - EXISTING means the tool is loaded in the shared tool library and
          exists in the system, but it is NOT attached to the invoking agent.
        - AVAILABLE means the tool exists AND is attached to the invoking
          agent's own tool whitelist, so this agent may call it.

    This tool reports the complete loaded catalog, not just the invoking
    agent's attached subset. Seeing an EXISTING tool does NOT grant access
    to it. Only AVAILABLE tools may be called successfully by this agent.

    Returns:
        list[str]: One line per loaded tool, formatted as:
            "<tool_name> — AVAILABLE"
            "<tool_name> — EXISTING (not attached to this agent)"
    """
    loader = context.loader
    manager = context.manager
    if loader is None:
        return []

    # AUTO-RELOAD (PyQt behaviour): the tools directory is re-scanned
    # whenever the catalog is listed, so tool files dropped in by the
    # operator OR CREATED BY ANOTHER AGENT appear here immediately —
    # no restart, no core_reload_tools call needed.
    _maybe = getattr(loader, "maybe_reload", None)
    if callable(_maybe):
        try:
            _maybe()
        except Exception:
            pass

    caller = current_agent_name.get()
    agent = manager.get_agent(caller) if manager and caller else None
    attached = set(agent.tools) if agent is not None else set()

    result = []
    for tool_name in sorted(loader.get_tool_names(), key=str.lower):
        if tool_name in attached:
            result.append(f"{tool_name} — AVAILABLE")
        else:
            result.append(f"{tool_name} — EXISTING (not attached to this agent)")
    return result

def core_reload_tools() -> str:
    """
    Force a full reload of the tool catalog, the prompts folder AND the
    MCP server registry — everything the agent-creation lists show.

    Use this after adding or modifying files manually, or whenever a
    listing seems stale. New tools become callable immediately (no app
    restart): agents can create a tool with core_create_tool and then
    call it in the same conversation, and every other agent's next
    core_list_tools() already shows it.

    Returns:
        str: Summary of what was reloaded, or an error if the loader
             is not available.
    """
    loader = context.loader
    if loader is None:
        return "Error: Loader not initialized."

    parts = []
    try:
        loader.reload()
        parts.append(f"{len(loader.get_tool_names())} tools "
                     f"({len(loader.get_tool_file_names())} files)")
    except Exception as e:
        return f"Error reloading tools: {e}"

    prompt_loader = getattr(context, "prompt_loader", None)
    if prompt_loader is not None:
        try:
            prompt_loader.reload()
            parts.append(f"{len(prompt_loader.get_prompt_names())} prompts")
        except Exception:
            pass

    mgr = None
    try:
        import mcp_manager as _mm
        mgr = _mm.get_manager()
    except Exception:
        mgr = getattr(context, "mcp_manager", None)
    if mgr is not None:
        try:
            mgr.reload_registry()
            parts.append(f"{len(mgr.list_registry_servers())} MCP servers")
        except Exception:
            pass

    return "Reloaded: " + ", ".join(parts) + "."

def core_list_prompts() -> list:
    """
    List the reusable prompt files available in the prompts folder.

    Each prompts/<name>.txt file is a shared prompt snippet. Agents can
    select any number of them (via the `prompts` argument of
    `core_create_agent` / `core_update_agent`, or the Prompt Files
    selector in the GUI's Edit Agent dialog). At inference time the
    texts of the selected files are appended to the agent's system
    prompt inside a <rules>...</rules> section — shared rules written
    once and reused by many agents.

    This listing marks which prompt files are currently attached to the
    INVOKING agent (same AVAILABLE / EXISTING convention as
    `core_list_tools`): seeing an EXISTING prompt does not attach it —
    only `core_update_agent` / `core_create_agent` can change the
    selection.

    Returns:
        list[str]: One line per prompt file, formatted as:
            "<name> — AVAILABLE — <first non-empty line of the file>"
            "<name> — EXISTING (not attached to this agent) — <first line>"
        The list is empty when the prompts folder holds no .txt files.
    """
    prompt_loader = getattr(context, "prompt_loader", None)
    if prompt_loader is None:
        return ["Error: Prompt loader not initialized (no prompts folder)."]

    # AUTO-RELOAD (PyQt behaviour): the prompts folder is re-scanned
    # whenever the list is called, so prompt files dropped in or edited
    # while the app runs appear on this very listing.
    _maybe = getattr(prompt_loader, "maybe_reload", None)
    if callable(_maybe):
        try:
            _maybe()
        except Exception:
            pass

    manager = context.manager
    caller = current_agent_name.get()
    agent = manager.get_agent(caller) if manager and caller else None
    attached = set(getattr(agent, "prompts", []) or []) if agent else set()

    result = []
    for name in sorted(prompt_loader.get_prompt_names(), key=str.lower):
        desc = prompt_loader.get_prompt_description(name)
        desc_part = f" — {desc}" if desc else ""
        if name in attached:
            result.append(f"{name} — AVAILABLE{desc_part}")
        else:
            result.append(f"{name} — EXISTING (not attached to this agent){desc_part}")
    if not result:
        result.append("(no .txt prompt files found in the prompts folder)")
    return result

# --- New tools added below ---

def core_create_tool(tool_name: str, code: str) -> str:
    """
    Dynamically create a new tool by writing a Python file to the tools directory.

    The file is saved as '<tool_name>.py'. The tool is immediately loaded if the
    code is syntactically valid; otherwise the file is deleted and an error returned.

    Args:
        tool_name (str): Name of the tool. Must be a valid Python identifier.
        code (str): Complete Python source code for the tool. It should define
                    a function named 'tool_<tool_name>' that will be called by
                    the agent framework.

    Returns:
        str: Success message or error description.
    """
    loader = context.loader
    if loader is None:
        return "Error: Loader not initialized."
    if not tool_name.isidentifier():
        return f"Invalid tool name '{tool_name}': must be a valid Python identifier."

    tools_dir = loader.tools_dir  # assumes loader has a tools_dir attribute (Path)
    filepath = tools_dir / f"{tool_name}.py"
    if filepath.exists():
        return f"Tool '{tool_name}' already exists. Use a different name or delete first."

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)

        # Test import the new module to catch syntax/import errors early
        module_name = f"{tools_dir.name}.{tool_name}"
        if module_name in sys.modules:
            del sys.modules[module_name]
        importlib.import_module(module_name)

        # Full reload to make the tool available to the manager
        loader.reload()
        return f"Tool '{tool_name}' created and loaded."
    except Exception as e:
        # If anything fails, delete the file and report the error
        if filepath.exists():
            filepath.unlink()
        return f"Error creating tool: {e}"

def core_prune_agent_context(agent_name: str, keep_last_n: int) -> str:
    """
    Trim the SHARED conversation history to reduce token usage.

    The most recent `keep_last_n` messages are retained across all participants
    (user + all agents on all machines). Older messages are permanently removed.

    NOTE: history is shared — pruning affects every agent's view. The
    `agent_name` argument is kept for backward compatibility but is not
    used to filter; if the named agent does not exist a warning is
    returned alongside the prune result.

    Args:
        agent_name (str): Name of the agent requesting the prune (kept for
                          backward compatibility; pruning is global).
        keep_last_n (int): Number of recent messages to keep. Must be >= 0.
                           0 wipes the entire history.

    Returns:
        str: Confirmation message or error.
    """
    conversation = getattr(context, "conversation", None)
    if conversation is None:
        return "Error: Conversation not initialized (cannot prune history)."

    # ----- convert keep_last_n from string to int -----
    try:
        if isinstance(keep_last_n, str):
            keep_last_n = int(keep_last_n.strip())
    except ValueError:
        return f"keep_last_n must be an integer, got '{keep_last_n}'"

    if keep_last_n < 0:
        return "keep_last_n must be >= 0."

    # Verify the agent exists (optional, for UX)
    manager = context.manager
    if manager is not None and agent_name and agent_name not in manager.list_agents():
        # Don't fail hard — the prune is global, just warn.
        prefix = f"Warning: agent '{agent_name}' not found. "
    else:
        prefix = ""

    try:
        result = conversation.prune_history(keep_last_n)
        return prefix + result
    except Exception as e:
        return f"Error pruning history: {e}"

# ======================================================================
# Main-chat message counting & retrieval — let agents track old messages
# ======================================================================
#
# Every message that enters the shared conversation history is stamped
# with a "timestamp" (unix epoch seconds) at the moment it is added
# (Conversation.add_user_message / add_messages / drain_pending). The
# tools below expose that: count the messages, page through them by
# index, or query them by time — so an agent can find content that has
# long fallen out of its context window.

_CHAT_MESSAGES_PER_CALL = 200     # max messages returned by one call
_CHAT_MESSAGE_CONTENT_CAP = 1500  # max chars shown per single message


def _parse_time_arg(value):
    """Parse a time argument into unix epoch seconds.

    Accepts numbers (int/float — already epoch seconds), numeric strings
    ("1760000000", "1760000000.5") and ISO 8601 strings
    ("2026-09-16", "2026-09-16T15:30:00", optional "Z" or timezone
    offset; naive values are interpreted as LOCAL time).

    Raises ValueError with a helpful message when nothing matches.
    """
    if isinstance(value, bool):
        raise ValueError(f"Unrecognised time value: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        raise ValueError("Empty time value.")
    try:
        return float(s)
    except ValueError:
        pass
    iso = s
    if iso.endswith(("Z", "z")):
        iso = iso[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        raise ValueError(
            f"Unrecognised time value: '{value}' — use unix epoch seconds "
            f"(e.g. 1760000000) or ISO 8601 (e.g. 2026-09-16T15:30:00)."
        )
    if dt.tzinfo is None:
        dt = dt.astimezone()  # naive ISO strings are local time
    return dt.timestamp()


def _parse_index_arg(value):
    """Parse a 1-based message index. Raises ValueError on garbage."""
    if isinstance(value, bool):
        raise ValueError(f"Unrecognised index value: {value!r}")
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"Unrecognised index value: '{value}' (expected an integer)."
        )


def _format_ts(ts):
    """Human-readable local time for a timestamp ('no-time' when missing)."""
    if ts is None:
        return "no-time"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def _format_history_msg(idx, msg):
    """One history message -> '[idx] [time] who (role): content'."""
    role = str(msg.get("role", "?"))
    sender = msg.get("sender", "")
    content = str(msg.get("content", ""))
    if len(content) > _CHAT_MESSAGE_CONTENT_CAP:
        content = (content[:_CHAT_MESSAGE_CONTENT_CAP]
                   + f"... [truncated, {len(content)} chars total]")
    if role == "tool":
        tname = msg.get("tool_name", msg.get("name", "tool"))
        who = f"{tname} (tool result" + (f" for {sender}" if sender else "") + ")"
    else:
        who = f"{sender} ({role})" if sender else role
    return f"[{idx}] [{_format_ts(msg.get('timestamp'))}] {who}: {content}"


def core_chat_messages(x1: Optional[str] = None, x2: Optional[str] = None,
                       t1: Optional[str] = None, t2: Optional[str] = None) -> list:
    """
    Read the MAIN chat history: count messages or fetch them by index or time.

    Every message in the main chat gets a timestamp when it is stored.
    Use this tool to find out how big the chat is and when messages
    arrived, then page through old messages that fell out of your
    context window.

    Modes (index arguments win over time arguments):
      - No arguments: summary — total message count plus the timestamps
        of the oldest and the latest message.
      - x1 only: the single message number x1 (1 = oldest message).
      - x1 and x2: messages x1..x2 inclusive (1 = oldest).
      - t1 only: the first message at-or-after time t1.
      - t1 and t2: all messages from t1 to t2 inclusive.

    Times may be unix epoch seconds (as returned by the summary mode)
    or ISO 8601 strings like 2026-09-16T15:30:00.

    Args:
        x1 (str, optional): First message index (1-based, oldest = 1).
        x2 (str, optional): Last message index (inclusive). Requires x1.
        t1 (str, optional): Range start time (epoch seconds or ISO 8601).
        t2 (str, optional): Range end time (inclusive). Requires t1.

    Returns:
        list[str]: The summary (no-args mode) or the requested messages,
            one per line, formatted as  [index] [time] who (role): content.
            Long contents are truncated per line; a range returns at most
            200 messages per call — ask for the next chunk with a new x1.
    """
    conversation = getattr(context, "conversation", None)
    if conversation is None:
        return ["Error: Conversation not initialized."]
    history = conversation.get_history_snapshot()
    total = len(history)
    if total == 0:
        return ["(main chat history is empty)"]

    # ---------------- index mode ----------------
    if x1 is not None:
        try:
            i1 = _parse_index_arg(x1)
        except ValueError as e:
            return [f"Error: {e}"]
        if i1 < 1:
            return ["Error: message indices are 1-based (1 = the oldest message)."]
        if i1 > total:
            return [f"Error: x1={i1} is beyond the last message (history holds {total})."]
        if x2 is None:
            return [_format_history_msg(i1, history[i1 - 1])]
        try:
            i2 = _parse_index_arg(x2)
        except ValueError as e:
            return [f"Error: {e}"]
        if i2 < 1:
            return ["Error: message indices are 1-based (1 = the oldest message)."]
        if i2 < i1:
            return [f"Error: x2={i2} is smaller than x1={i1} (swap them)."]
        if i2 > total:
            i2 = total
        capped = False
        if i2 - i1 + 1 > _CHAT_MESSAGES_PER_CALL:
            i2 = i1 + _CHAT_MESSAGES_PER_CALL - 1
            capped = True
        out = [_format_history_msg(i, history[i - 1]) for i in range(i1, i2 + 1)]
        if capped:
            out.append(f"(capped at {_CHAT_MESSAGES_PER_CALL} messages per call — "
                       f"continue with x1={i2 + 1})")
        return out
    if x2 is not None:
        return ["Error: x2 was given without x1 — provide x1 too."]

    # ---------------- time mode ----------------
    if t1 is not None:
        try:
            tt1 = _parse_time_arg(t1)
        except ValueError as e:
            return [f"Error: {e}"]
        tt2 = None
        if t2 is not None:
            try:
                tt2 = _parse_time_arg(t2)
            except ValueError as e:
                return [f"Error: {e}"]
        matched = []
        no_ts = 0
        for i, msg in enumerate(history):
            ts = msg.get("timestamp") if isinstance(msg, dict) else None
            if not isinstance(ts, (int, float)) or isinstance(ts, bool):
                no_ts += 1
                continue
            if ts >= tt1 and (tt2 is None or ts <= tt2):
                matched.append((i + 1, msg))
        suffix = ""
        if no_ts:
            suffix = (f" ({no_ts} older message(s) have no timestamp and can "
                      f"only be reached by index)")
        if not matched:
            return [f"No messages found in that time range.{suffix}"]
        if tt2 is None:
            idx, msg = matched[0]
            return [_format_history_msg(idx, msg)
                    + f" (first message at-or-after the given time){suffix}"]
        capped = False
        if len(matched) > _CHAT_MESSAGES_PER_CALL:
            matched = matched[:_CHAT_MESSAGES_PER_CALL]
            capped = True
        out = [_format_history_msg(idx, msg) for idx, msg in matched]
        if capped:
            out.append(f"(capped at {_CHAT_MESSAGES_PER_CALL} messages per call — "
                       f"narrow the time range)")
        if suffix:
            out.append(suffix.strip())
        return out
    if t2 is not None:
        return ["Error: t2 was given without t1 — provide t1 too."]

    # ---------------- summary mode ----------------
    stamped = [(i + 1, m.get("timestamp")) for i, m in enumerate(history)
               if isinstance(m, dict)
               and isinstance(m.get("timestamp"), (int, float))
               and not isinstance(m.get("timestamp"), bool)]
    out = [f"Main chat: {total} message(s) (index 1 = oldest, {total} = latest)."]
    if stamped:
        out.append(f"Oldest: message {stamped[0][0]} at {stamped[0][1]:.0f} "
                   f"({_format_ts(stamped[0][1])}).")
        out.append(f"Latest: message {stamped[-1][0]} at {stamped[-1][1]:.0f} "
                   f"({_format_ts(stamped[-1][1])}).")
    else:
        out.append("No timestamps available (all messages predate timestamping).")
    missing = total - len(stamped)
    if missing:
        out.append(f"{missing} message(s) have no timestamp "
                   f"(they can only be fetched by index).")
    out.append("Usage: x1 [x2] = message indices; t1 [t2] = epoch seconds "
               "or ISO 8601 time range.")
    return out

def core_get_send_message_code() -> str:
    """
    Return a ready-to-use Python snippet for sending messages to the main chat UI.

    The code defines a function `send_event(message, sender)` that connects to
    a local TCP socket (default 127.0.0.1:8888) and sends a JSON payload. This
    allows scripts generated by an agent to display messages directly in the
    chat window.

    Example usage inside a generated script:

        send_event("Processing complete", "my_agent")

    Returns:
        str: The Python code as a string.
    """
    result = """
import socket
import json

def send_event(message, sender="external"):
    \'\'\'Send a message to the main chat via TCP socket.\'\'\'
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(('127.0.0.1', 8888))
        payload = json.dumps({"sender": sender, "message": message})
        sock.sendall(payload.encode())
"""
    return result


# ======================================================================
# Agentic-core tools — runtime control of agent scheduling & invocation
# ======================================================================
#
# These two tools are the user-facing surface of the "agentic core":
# they let one agent take itself in or out of the round-robin rotation
# at runtime (`core_agent_active`), and let one agent directly invoke
# another agent as a sub-routine (`core_agent_execute`).
#
# Together they enable patterns like:
#
#   - A "manager" agent that wakes up, decides which worker it needs,
#     activates that worker, invokes it via `core_agent_execute`, then
#     deactivates it again before answering the user.
#   - An agent that finishes its task and calls
#     `core_agent_active(my_own_name, False)` to step out of the
#     rotation, leaving the floor to the remaining agents.
#   - A pipeline of specialists (translator → reviewer → editor) where
#     each one invokes the next via `core_agent_execute` and the
#     final result is returned to the user-facing agent.
# ======================================================================


def core_agent_active(name: str, active: bool) -> str:
    """
    Toggle an agent's `active` flag at runtime — when True, the agent
    takes part in the round-robin scheduling loop as usual; when False,
    the scheduler SKIPS the agent entirely (it is not inferenced) until
    the flag is flipped back to True.

    This is the primary "I'm done / wake me up later" mechanism for an
    agent that wants to manage its own participation in the multi-agent
    chat. An agent that has finished its task can call:

        core_agent_active("translator", False)

    and the translator will be skipped on every subsequent round until
    something turns it back on:

        core_agent_active("translator", True)

    The flag is per-agent and persists across turns (and across save /
    load sessions — it is part of the agent's serialised config). It
    does NOT delete the agent or its history; the agent simply becomes
    "invisible" to the scheduler.

    Args:
        name (str): Name of the agent whose `active` flag should be
            changed. The agent must already exist (use `core_create_agent`
            first). The agent calling this tool is allowed to pass its
            OWN name — a common pattern is for an agent to deactivate
            itself once its job is done.

        active (bool): The new value for the `active` flag. Accepts a
            real Python bool, an int (0 / 1), or a string ("true" /
            "false" / "yes" / "no" / "1" / "0" / "on" / "off" — case
            insensitive). This permissiveness is deliberate: LLM tool
            calls often serialise booleans as strings, and we don't
            want the call to fail just because the model sent "true"
            instead of true.

            True  -> the agent re-enters the round-robin rotation.
            False -> the agent is skipped by the scheduler until
                     re-activated.

    Returns:
        str: A short confirmation describing what changed. Examples:
             - "Agent 'translator' is now ACTIVE (will be inferenced)."
             - "Agent 'translator' is now INACTIVE (skipped by scheduler)."
             - "No agent with name 'translator'." (no such agent)
             - "Error: Agent manager not initialized." (startup race)

    Notes:
        - This tool ONLY changes the `active` flag. It does not touch
          `silent`, `system_prompt`, `tools`, or any other config. Use
          `core_update_agent` for those.
        - The change takes effect on the next scheduler tick — if the
          agent is currently mid-inference, that inference is allowed
          to finish; the flag only controls whether a NEW turn is
          started for the agent.
        - Calling this tool with the same value as the current flag is
          safe and idempotent (it just re-emits the same confirmation).
    """
    manager = context.manager
    if manager is None:
        return "Error: Agent manager not initialized."

    name = str(name).strip() if name is not None else ""
    if not name:
        return "Error: 'name' argument is required."

    agent = manager.get_agent(name)
    if agent is None:
        # Match the wording requested by the spec for the
        # core_agent_execute tool — same phrasing here for consistency.
        return f"No agent with name {name}"

    new_value = _coerce_bool(active, default=True)
    old_value = bool(getattr(agent, "active", True))
    agent.active = new_value

    # Notify the GUI so the sidebar can re-render the row's badge.
    manager.agents_changed.emit()

    if new_value:
        return (
            f"Agent '{name}' is now ACTIVE (will be inferenced). "
            f"(was: {'ACTIVE' if old_value else 'INACTIVE'})"
        )
    else:
        return (
            f"Agent '{name}' is now INACTIVE (skipped by scheduler). "
            f"(was: {'ACTIVE' if old_value else 'INACTIVE'})"
        )


def _core_agent_execute_impl(name: str, question: str = None) -> str:
    """
    Synchronously invoke another agent for ONE full turn and return
    that agent's final answer as the tool result.

    This is the core "agent-calls-agent" primitive. When an agent A
    calls `core_agent_execute("B")`:

      1. The tool looks up agent B. If no agent with that name exists,
         the tool returns the literal string "No agent with name <name>"
         and agent A continues its turn as if nothing happened.

      2. The tool checks a recursion guard (the `agent_call_stack`
         ContextVar maintained by the framework). If agent B is ALREADY
         on the call stack — meaning B is itself paused somewhere up
         the Python call chain waiting for a tool result — the call is
         refused and an explanatory error string is returned. This
         prevents the classic infinite loop where A calls B which calls
         A which calls B ... each one paused mid-turn waiting for the
         other to finish. The error message includes the current call
         chain so the model can see exactly what went wrong.

      3. Otherwise, agent A's `Agent.call()` is blocked (it is sitting
         on the Python stack waiting for this tool to return — its
         accumulated thinking and partial context are naturally
         preserved in those local variables, so "saving the thinking
         text" is automatic). Agent B is then run for one full turn:
         it builds its own context from the shared conversation history,
         streams its own response, calls its own tools if needed, and
         produces its own final answer.

         While B is running:
           - B's live thinking stream is forwarded to the Thinking tab
             for B's endpoint (so you can watch it think in real time,
             exactly as if B's own BackendThread had picked it up).
           - B's sidebar row is flipped to "Running: <model>" and back
             to "Waiting" when B finishes.
           - B's BackendThread (if any) sees B's `_busy` flag and
             SKIPS its scheduled turn for B, so B is not inferenced
             twice in parallel.

      4. When B finishes, every assistant / tool message B produced is
         appended to the shared conversation history (so any other
         agent — including A — will see B's answer in its next context
         rebuild).

      5. B's FINAL assistant message (the last one with non-empty
         content) is returned to A as the tool result string. If B
         produced no final answer, a placeholder string is returned
         instead.

      6. A's tool-call loop receives that string as the tool result
         and continues its turn — A can now read B's answer, reason
         about it, call more tools, or produce its own final answer.

    Chat-output behaviour:
        If agent B has its `silent` flag set to True, B's final answer
        is stored in history (visible to other agents) but is NOT
        printed into the chat UI — only A sees it (as the tool result).
        If B is NOT silent (the default), B's final answer is also
        emitted to the chat as a normal message from B, so the user
        can see what B said.

    Cross-endpoint behaviour:
        B may live on a different Ollama machine than A. The tool
        automatically picks up the correct OllamaClient for B's
        endpoint via `context.client_manager`. This means you can
        have a lightweight local "router" agent that delegates heavy
        work to a big model running on a remote GPU box.

    Args:
        name (str): Name of the agent to invoke. The agent must already
            exist. It does NOT need to be `active` — `core_agent_execute`
            is a direct synchronous call and bypasses the scheduler's
            `active` check. (This is deliberate: it lets a manager
            agent invoke a worker that has been taken out of the
            rotation. If you want the scheduler's `active` check to
            apply, simply don't call this tool for inactive agents.)

    Returns:
        str: One of:
            - The called agent's final answer (its last non-empty
              assistant message content), returned as a plain string
              so the calling agent can read it as a normal tool result.
            - "No agent with name <name>" if the named agent does not
              exist.
            - "Cannot call agent '<name>' recursively: ..." if the
              named agent is already on the call stack (paused waiting
              for a tool result somewhere up the chain).
            - "Error executing agent '<name>': ..." if the called
              agent raised an exception.
            - "[Agent '<name>' produced no final answer]" if the
              called agent completed but emitted no assistant content
              (e.g. it only produced tool calls that all failed).

    Examples:
        # Simple delegation
        answer = core_agent_execute("translator")
        # `answer` now contains the translator's final reply text.

        # Manager-worker pattern
        core_agent_active("worker_a", True)        # wake the worker
        result = core_agent_execute("worker_a")    # run one turn
        core_agent_active("worker_a", False)       # put it back to sleep

    Notes:
        - This tool BLOCKS the calling agent until the called agent
          finishes. If the called agent is slow (long generation,
          many tool calls), the caller will wait. There is no
          timeout at the tool level — the per-tool 1-hour timeout
          in `Agent.call`'s ThreadPoolExecutor is the only bound.
        - The called agent runs in the SAME thread as the calling
          agent's tool execution (a worker thread of the caller's
          BackendThread). Its thinking updates are routed to the
          called agent's own endpoint's Thinking tab, not the
          caller's.
        - Recursion is detected via the `agent_call_stack`
          ContextVar, which is propagated into nested tool-execution
          worker threads by the ThreadPoolExecutor's context copy.
          This means the guard works correctly even across multiple
          levels of agent-calls-agent-calls-agent.
    """
    manager = context.manager
    if manager is None:
        return "Error: Agent manager not initialized."

    name = str(name).strip() if name is not None else ""
    if not name:
        return "Error: 'name' argument is required."

    target_agent = manager.get_agent(name)
    if target_agent is None:
        # Exact wording requested by the spec.
        return f"No agent with name {name}"

    # ------------------------------------------------------------------
    # Recursion guard.
    # ------------------------------------------------------------------
    # `agent_call_stack` is a ContextVar whose value is a tuple of agent
    # names that are currently being executed via this tool (i.e. each
    # one is paused mid-turn somewhere up the Python call chain). If the
    # target name is already in that tuple, calling it again would
    # create an infinite recursion — A waiting for B waiting for A ...
    # so we refuse and return an explanatory error.
    #
    # We also include the CURRENT caller (current_agent_name) in the
    # check, because if the caller tries to execute itself we'd have
    # the same deadlock (A waiting for A).
    caller_name = current_agent_name.get()
    current_stack = agent_call_stack.get()
    ancestors = set(current_stack) | {caller_name}
    if name in ancestors:
        chain = " -> ".join(list(current_stack) + [caller_name, name])
        return (
            f"Cannot call agent '{name}' recursively: that agent is "
            f"already paused waiting for a tool result somewhere up "
            f"the call chain (chain: {chain}). Refusing to start an "
            f"infinite loop."
        )

    # ------------------------------------------------------------------
    # Busy-flag check.
    # ------------------------------------------------------------------
    # If the target agent is already being executed (its `_busy` flag is
    # set), refuse rather than interleave two concurrent inferences on
    # the same Ollama model. This check is done EARLY — before we
    # acquire client_manager / loader / conversation — because it's a
    # cheap local check and the most likely reason to refuse.
    acquired_busy = False
    with target_agent._busy_lock:
        if target_agent._busy:
            return (
                f"Agent '{name}' is already being executed (its _busy "
                f"flag is set). Wait for the current invocation to "
                f"finish and try again."
            )
        target_agent._busy = True
        acquired_busy = True

    # ------------------------------------------------------------------
    # Acquire the dependencies the called agent needs to run.
    # ------------------------------------------------------------------
    client_manager = getattr(context, "client_manager", None)
    if client_manager is None:
        # Release the busy flag before returning so we don't leave the
        # agent permanently locked.
        if acquired_busy:
            with target_agent._busy_lock:
                target_agent._busy = False
        return "Error: Client manager not initialized (cannot run agent)."

    loader = context.loader
    if loader is None:
        if acquired_busy:
            with target_agent._busy_lock:
                target_agent._busy = False
        return "Error: Tool loader not initialized."

    conversation = context.conversation
    if conversation is None:
        if acquired_busy:
            with target_agent._busy_lock:
                target_agent._busy = False
        return "Error: Conversation not initialized."

    target_endpoint = target_agent.endpoint or "local"
    client = client_manager.get_client(target_endpoint)
    if client is None:
        if acquired_busy:
            with target_agent._busy_lock:
                target_agent._busy = False
        return f"Error: Could not obtain Ollama client for endpoint '{target_endpoint}'."

    # ------------------------------------------------------------------
    # Remote-endpoint pause trigger (NEW).
    # ------------------------------------------------------------------
    # If the target agent lives on a DIFFERENT endpoint than the caller,
    # the target endpoint's own scheduler may be mid-turn with another
    # agent — and two concurrent inference streams against the same
    # Ollama server are exactly what we want to avoid. So:
    #
    #   1. We request a pause on the target endpoint's EndpointGate.
    #   2. We wait (up to a generous timeout) until the endpoint reaches
    #      a safe pause point — this happens at the in-flight agent's
    #      next tool-call iteration boundary or at the end of its
    #      current inference ("nearest tool call or end of inference"),
    #      exactly as designed. Mid-stream generations are NEVER
    #      interrupted.
    #   3. We run the target agent (it skips the gate check itself,
    #      because it IS the reason the endpoint is paused).
    #   4. We release the pause in the finally block below so the
    #      endpoint's scheduler (and any agent paused mid-turn) resumes.
    #
    # When caller and target share an endpoint no pause is needed: the
    # caller's own scheduler thread is blocked inside this very tool
    # call, so it cannot start another turn on that endpoint anyway.
    #
    # If the quiesce wait times out we proceed anyway (with a logged
    # warning) rather than failing the call.
    caller_endpoint = None
    try:
        _caller_agent = manager.get_agent(current_agent_name.get())
        if _caller_agent is not None:
            caller_endpoint = _caller_agent.endpoint or "local"
    except Exception:
        caller_endpoint = None

    pause_requested = False
    if caller_endpoint is None or caller_endpoint != target_endpoint:
        gate = context.get_endpoint_gate(target_endpoint)
        gate.request_pause()
        pause_requested = True
        print(f"[core_agent_execute] pause requested on endpoint "
              f"'{target_endpoint}' (caller is on '{caller_endpoint}')")
        if not gate.wait_quiesced(timeout=600.0):
            print(f"[core_agent_execute] WARNING: endpoint '{target_endpoint}' "
                  f"did not reach a pause point within 600s — proceeding "
                  f"anyway (inference streams may overlap briefly).")

    # Push the target onto the call-stack ContextVar for the duration
    # of this call. Using `.set()` returns a token we can use to restore
    # the previous value exactly — important because the ContextVar
    # propagates into nested ThreadPoolExecutor workers and we want
    # each level to see its proper ancestor chain.
    stack_token = agent_call_stack.set(current_stack + (name,))

    # Set up the thinking callback that routes the called agent's
    # thinking-tab events to the correct endpoint's tab in the GUI.
    # `context.thinking_router` is a thin wrapper around a Qt signal's
    # `.emit()`, which is safe to call from any thread.
    thinking_router = getattr(context, "thinking_router", None)

    def _thinking_cb(event_type, agent_name_cb, detail, extra):
        if thinking_router is not None:
            try:
                thinking_router(target_endpoint, agent_name_cb,
                                event_type, detail, extra)
            except Exception as e:
                # Never let a GUI-side routing error abort the agent.
                print(f"[core_agent_execute] thinking_router error: {e}")

    # Flip the sidebar status of the called agent to "Running".
    status_router = getattr(context, "agent_status_router", None)
    if status_router is not None:
        try:
            status_router(target_endpoint, name, True, target_agent.model)
        except Exception as e:
            print(f"[core_agent_execute] agent_status_router error: {e}")

    try:
        # Snapshot the shared history right now and let the target run
        # one full turn against it. We deliberately use the SAME shared
        # history that the BackendThread uses — so anything the caller
        # has already said (and anything else in the conversation) is
        # visible to the called agent.
        history = conversation.get_history_snapshot()

        # A question supplied through core_agent_execute_with_question is an
        # ephemeral prompt: it participates in this inference exactly like
        # the final user message, but it is NOT written into the shared chat
        # history after the call. This keeps delegated questions private to
        # the tool invocation while still giving the target agent its normal
        # system prompt + generated chat context + final question.
        #
        # FIX (question-with-zero-context): the question is now passed to
        # Agent.call as `ephemeral_question` instead of being appended to
        # the history snapshot. build_context appends it as the FINAL user
        # message AFTER truncation — so it is visible even when the target
        # agent has max_context_messages = 0 ("no leak from chat").
        # Previously that 0-context branch returned a bare "(continue)"
        # and the question was silently lost.
        new_msgs = target_agent.call(
            history, loader, client, _thinking_cb,
            ephemeral_question=question
        )

        # Persist the called agent's produced messages into the shared
        # history so every other agent can see them on the next
        # context rebuild. This is exactly what BackendThread does
        # after a normal agent turn.
        with conversation._lock:
            conversation.add_messages(new_msgs)

        # Find the FINAL assistant message — the last message with
        # role == "assistant" and non-empty content. This is the
        # "answer" the called agent produced and what we return to
        # the caller as the tool result.
        final_answer = ""
        for msg in reversed(new_msgs):
            if (isinstance(msg, dict)
                    and msg.get("role") == "assistant"
                    and str(msg.get("content", "")).strip()):
                final_answer = str(msg["content"])
                break

        # If the called agent is NOT silent, also emit its final answer
        # to the chat UI so the human user can see it. (Silent agents
        # still produce a tool result for the caller — they just don't
        # print to chat, which is the whole point of the `silent` flag.)
        if final_answer and not getattr(target_agent, "silent", False):
            message_router = getattr(context, "message_router", None)
            if message_router is not None:
                try:
                    message_router(name, final_answer)
                except Exception as e:
                    print(f"[core_agent_execute] message_router error: {e}")

        if not final_answer:
            return f"[Agent '{name}' produced no final answer]"

        return final_answer

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[core_agent_execute] error running '{name}':\\n{tb}")
        return f"Error executing agent '{name}': {e}"
    finally:
        # Always restore the ContextVar, clear the busy flag, and flip
        # the sidebar status back to "Waiting" — even on exception.
        agent_call_stack.reset(stack_token)
        # Release the remote-endpoint pause so the target endpoint's
        # scheduler (and any agent paused mid-turn at a tool-call
        # boundary) can resume immediately.
        if pause_requested:
            try:
                context.get_endpoint_gate(target_endpoint).release_pause()
            except Exception as e:
                print(f"[core_agent_execute] gate release error: {e}")
        if acquired_busy:
            with target_agent._busy_lock:
                target_agent._busy = False
        if status_router is not None:
            try:
                status_router(target_endpoint, name, False, "")
            except Exception as e:
                print(f"[core_agent_execute] agent_status_router reset error: {e}")


def core_agent_execute(name: str) -> str:
    """
    Synchronously invoke another agent for one full turn using the normal
    shared conversation history.

    This is the original agent-to-agent execution primitive. It is kept as
    a public wrapper around the common execution implementation so existing
    agents and tools continue to work unchanged.
    """
    return _core_agent_execute_impl(name, None)


def core_agent_execute_with_question(name: str, question: str) -> str:
    """
    Synchronously invoke another agent for one full turn and append a custom
    question as the LAST user message in that agent's generated prompt.

    The target receives:
        target system prompt
        + automatically generated conversation context
        + this question as the final prompt piece

    The question itself is ephemeral: it is not added to shared conversation
    history. The called agent's own generated messages still are stored in
    shared history exactly like core_agent_execute.

    NOTE: the question is ALWAYS visible to the target — even when the
    target has max_context_messages = 0 ("no leak from the shared chat").
    With 0 the question is the ONLY user message the target sees; with a
    positive setting it is appended AFTER the truncated context, so it
    can never be cut away.

    Args:
        name (str): Name of the agent to invoke.
        question (str): The question/instruction to give that agent. It is
            inserted after the automatically generated chat context.

    Returns:
        str: The called agent's final answer, or the same error/status strings
        returned by core_agent_execute.
    """
    name = str(name).strip() if name is not None else ""
    if not name:
        return "Error: 'name' argument is required."

    question = str(question).strip() if question is not None else ""
    if not question:
        return "Error: 'question' argument is required."

    return _core_agent_execute_impl(name, question)


# ======================================================================
# Agent-to-agent DIALOGUES — private, persistent side chats
# ======================================================================
#
# core_agent_execute_dialogue is the "new era of communication between
# agents": instead of running the target against the shared main chat
# (core_agent_execute / core_agent_execute_with_question), it runs it
# inside a SEPARATE conversation that exists only between the two
# agents:
#
#   - the CALLER's messages are the "user" side; the CALLEE answers as
#     the "assistant" — to the callee it looks exactly like talking
#     with the human user;
#   - nothing from the main chat leaks in, and nothing from the side
#     chat is written back to the main chat;
#   - the side chat is PERMANENT: every message is kept in memory and
#     mirrored to <app_dir>/agent_dialogues/dialogue_<caller>__<callee>.json
#     (one file per ordered pair — A->B and B->A are separate chats);
#   - the callee keeps ALL of its own settings — system prompt (with
#     its <rules> prompt files), tools, model, endpoint, parameters.
#     One special case: max_context_messages = 0 means UNLIMITED inside
#     a dialogue (0's raison d'etre — not leaking the MAIN chat — does
#     not apply to a private side chat that never contains it);
#   - only the callee's LAST answer is returned to the caller as the
#     tool result.
#
# The companion tools core_list_dialogues / core_get_dialogue /
# core_reset_dialogue give agents (and their human) observability and
# control over the saved side chats.


def core_agent_execute_dialogue(name: str, message: str) -> str:
    """
    Talk to another agent inside a PRIVATE, persistent side chat (dialogue).

    Unlike core_agent_execute / core_agent_execute_with_question — which
    run the target against the shared main-chat context — this tool runs
    the target inside its OWN conversation with YOU:

      - Nothing from the main chat ever leaks into the side chat, and
        nothing from the side chat is written back to the main chat.
      - Your message is stored as a USER message; the target answers as
        the assistant — to the target it looks exactly like talking
        with the human user.
      - The side chat is PERMANENT: every message both of you produce
        is kept in memory and saved to the agent_dialogues folder as
        its own session file. On your next call the target SEES the
        previous messages of your side chat and continues where you
        left off.
      - For every ordered pair of agents there is exactly one side
        chat: your chat with B (you -> B) is separate from B's chat
        with you (B -> you).
      - Only the target's LAST answer of this turn is returned to you;
        its intermediate tool traffic stays in the side chat.

    The target keeps ALL of its own settings: system prompt (including
    its selected prompt files / <rules> section), tools, model,
    endpoint, temperature, max_tool_iterations, image/audio flags.
    One special case: max_context_messages = 0 means UNLIMITED inside a
    dialogue — 0 exists to keep the MAIN chat out, and a side chat never
    contains main-chat messages anyway. Set a positive value on the
    target to bound how much of the side chat it sees per turn.

    Args:
        name (str): Name of the agent to talk to. Must exist (it does
                not need to be active — same deliberate bypass as
                core_agent_execute).
        message (str): Your message, exactly as you would write it to a
                user. Appended to your side chat; the target answers it.

    Returns:
        str: The target's final answer (the last non-empty assistant
             message of this turn), or an error/status string (unknown
             agent, recursive call refused, target busy, ...).
    """
    manager = context.manager
    if manager is None:
        return "Error: Agent manager not initialized."

    name = str(name).strip() if name is not None else ""
    if not name:
        return "Error: 'name' argument is required."

    message = str(message).strip() if message is not None else ""
    if not message:
        return "Error: 'message' argument is required."

    # This tool is only meaningful from inside an agent's tool loop —
    # the CALLER is one of the two dialogue participants and gives the
    # side chat its "user" side.
    caller_name = current_agent_name.get()
    if not caller_name or caller_name == "unknown":
        return ("Error: core_agent_execute_dialogue can only be called "
                "from inside an agent's tool execution (the caller is one "
                "of the two dialogue participants).")

    target_agent = manager.get_agent(name)
    if target_agent is None:
        return f"No agent with name {name}"

    # ------------------------------------------------------------------
    # Recursion guard — same rules as core_agent_execute: the target
    # must not already be paused somewhere up the agent-call chain
    # (which also covers the caller talking to itself).
    # ------------------------------------------------------------------
    current_stack = agent_call_stack.get()
    ancestors = set(current_stack) | {caller_name}
    if name in ancestors:
        chain = " -> ".join(list(current_stack) + [caller_name, name])
        return (
            f"Cannot start a dialogue with '{name}' recursively: that agent "
            f"is already paused waiting for a tool result somewhere up the "
            f"call chain (chain: {chain})."
        )

    # ------------------------------------------------------------------
    # Busy-flag check — no second concurrent inference on the target.
    # ------------------------------------------------------------------
    with target_agent._busy_lock:
        if target_agent._busy:
            return (
                f"Agent '{name}' is already being executed. Wait for the "
                f"current invocation to finish and try again."
            )
        target_agent._busy = True

    client_manager = getattr(context, "client_manager", None)
    if client_manager is None:
        with target_agent._busy_lock:
            target_agent._busy = False
        return "Error: Client manager not initialized (cannot run agent)."

    loader = context.loader
    if loader is None:
        with target_agent._busy_lock:
            target_agent._busy = False
        return "Error: Tool loader not initialized."

    target_endpoint = target_agent.endpoint or "local"
    client = client_manager.get_client(target_endpoint)
    if client is None:
        with target_agent._busy_lock:
            target_agent._busy = False
        return f"Error: Could not obtain Ollama client for endpoint '{target_endpoint}'."

    store = _get_dialogue_store()
    if store is None:
        with target_agent._busy_lock:
            target_agent._busy = False
        return "Error: Dialogue store not initialized."

    # ------------------------------------------------------------------
    # Remote-endpoint pause (same cooperative gate as core_agent_execute):
    # when the target lives on a different endpoint, ask that endpoint
    # to pause at its nearest safe point before we run the target on it.
    # ------------------------------------------------------------------
    caller_endpoint = None
    try:
        _caller_agent = manager.get_agent(caller_name)
        if _caller_agent is not None:
            caller_endpoint = _caller_agent.endpoint or "local"
    except Exception:
        caller_endpoint = None

    pause_requested = False
    if caller_endpoint is None or caller_endpoint != target_endpoint:
        gate = context.get_endpoint_gate(target_endpoint)
        gate.request_pause()
        pause_requested = True
        print(f"[core_agent_execute_dialogue] pause requested on endpoint "
              f"'{target_endpoint}' (caller is on '{caller_endpoint}')")
        if not gate.wait_quiesced(timeout=600.0):
            print(f"[core_agent_execute_dialogue] WARNING: endpoint "
                  f"'{target_endpoint}' did not reach a pause point within "
                  f"600s — proceeding anyway.")

    stack_token = agent_call_stack.set(current_stack + (name,))

    thinking_router = getattr(context, "thinking_router", None)

    def _thinking_cb(event_type, agent_name_cb, detail, extra):
        if thinking_router is not None:
            try:
                thinking_router(target_endpoint, agent_name_cb,
                                event_type, detail, extra)
            except Exception as e:
                print(f"[core_agent_execute_dialogue] thinking_router error: {e}")

    status_router = getattr(context, "agent_status_router", None)
    if status_router is not None:
        try:
            status_router(target_endpoint, name, True, target_agent.model)
        except Exception as e:
            print(f"[core_agent_execute_dialogue] agent_status_router error: {e}")

    try:
        # Get (or resume) the side chat for this ordered pair, append the
        # caller's message, persist it, then run the target for one full
        # turn against the SIDE chat only (dialogue_mode=True).
        session = store.get(caller_name, name, create=True)
        session.add_partner_message(message)
        store.save(session)

        snapshot = session.get_history_snapshot()
        new_msgs = target_agent.call(
            snapshot, loader, client, _thinking_cb,
            dialogue_mode=True
        )

        # The callee's messages go ONLY into the side chat — never into
        # the main shared history.
        session.add_messages(new_msgs)
        store.save(session)

        # Only the LAST answer goes back to the caller.
        final_answer = ""
        for msg in reversed(new_msgs):
            if (isinstance(msg, dict)
                    and msg.get("role") == "assistant"
                    and str(msg.get("content", "")).strip()):
                final_answer = str(msg["content"])
                break

        if not final_answer:
            return f"[Agent '{name}' produced no final answer in the dialogue]"
        return final_answer

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[core_agent_execute_dialogue] error running '{name}':\\n{tb}")
        return f"Error executing dialogue with agent '{name}': {e}"
    finally:
        # Always restore the ContextVar, clear the busy flag, release a
        # remote-endpoint pause, and reset the sidebar status.
        agent_call_stack.reset(stack_token)
        if pause_requested:
            try:
                context.get_endpoint_gate(target_endpoint).release_pause()
            except Exception as e:
                print(f"[core_agent_execute_dialogue] gate release error: {e}")
        with target_agent._busy_lock:
            target_agent._busy = False
        if status_router is not None:
            try:
                status_router(target_endpoint, name, False, "")
            except Exception as e:
                print(f"[core_agent_execute_dialogue] agent status reset error: {e}")


def core_list_dialogues() -> list:
    """
    List the saved agent-to-agent dialogues (private side chats).

    Every line describes one side chat: who is the "user" side, who is
    the "assistant" side, how many messages it holds, and when its last
    message was written. Read any of them with core_get_dialogue.

    Returns:
        list[str]: One line per dialogue file, plus nothing else. When
            no dialogues exist yet, a single hint line is returned.
    """
    store = _get_dialogue_store()
    if store is None:
        return ["Error: Dialogue store not initialized."]
    entries = store.list_dialogues()
    if not entries:
        return ["(no saved dialogues yet — use core_agent_execute_dialogue "
                "to start one)"]
    caller = current_agent_name.get()
    out = []
    for e in entries:
        when = ""
        if e.get("last_time") is not None:
            try:
                when = " — last " + _format_ts(e["last_time"])
            except Exception:
                when = ""
        mine = ""
        if caller and caller != "unknown":
            if e.get("caller") == caller:
                mine = " (you are the caller / user side)"
            elif e.get("callee") == caller:
                mine = " (you are the assistant side)"
        out.append(f"{e.get('caller')} -> {e.get('callee')} — "
                   f"{e.get('messages', 0)} message(s){when}{mine}")
    return out


def core_get_dialogue(caller: str, callee: str, last_n: Optional[str] = "20") -> list:
    """
    Read messages from a saved agent-to-agent dialogue (side chat).

    Args:
        caller (str): The "user" side of the dialogue (the agent that
                opened the side chat).
        callee (str): The "assistant" side of the dialogue (the agent
                that was talked to).
        last_n (str, optional): How many RECENT messages to return.
                Default 20. Use 0 to get the whole dialogue.

    Returns:
        list[str]: The requested messages, one per line, oldest first,
            formatted as [n] [time] who (role): content — plus a final
            line with the total message count. Or an error line when no
            such dialogue exists.
    """
    store = _get_dialogue_store()
    if store is None:
        return ["Error: Dialogue store not initialized."]
    caller = str(caller).strip() if caller is not None else ""
    callee = str(callee).strip() if callee is not None else ""
    if not caller or not callee:
        return ["Error: 'caller' and 'callee' arguments are required."]
    try:
        n = _parse_index_arg(last_n)
    except ValueError:
        n = 20
    session = store.get(caller, callee, create=False)
    if session is None:
        return [f"No dialogue found between '{caller}' (caller) and "
                f"'{callee}' (callee)."]
    history = session.get_history_snapshot()
    total = len(history)
    if total == 0:
        return [f"Dialogue '{caller}' -> '{callee}' is empty."]
    if n and n > 0:
        selected = history[-n:]
        offset = total - len(selected)
    else:
        selected = history
        offset = 0
    out = [_format_history_msg(offset + i + 1, msg)
           for i, msg in enumerate(selected)]
    out.append(f"({total} message(s) in this dialogue)")
    return out


def core_reset_dialogue(name: str) -> str:
    """
    Delete YOUR side chat with another agent and start it fresh.

    Removes the dialogue in which the CALLING agent is the "user" side
    and the named agent is the "assistant" side. The opposite direction
    (their chat with you) and everybody else's dialogues are not
    touched. Use this when a side chat has accumulated noise and you
    want to restart the conversation from a clean slate.

    Args:
        name (str): The agent you have (or want) a side chat with.

    Returns:
        str: Confirmation, or a note that no such dialogue existed.
    """
    caller = current_agent_name.get()
    if not caller or caller == "unknown":
        return ("Error: core_reset_dialogue can only be called from inside "
                "an agent's tool execution.")
    name = str(name).strip() if name is not None else ""
    if not name:
        return "Error: 'name' argument is required."
    store = _get_dialogue_store()
    if store is None:
        return "Error: Dialogue store not initialized."
    if store.delete(caller, name):
        return (f"Dialogue '{caller}' -> '{name}' deleted. The next "
                f"core_agent_execute_dialogue call starts a fresh chat.")
    return (f"No dialogue existed between '{caller}' (caller) and "
            f"'{name}' (callee).")


# ======================================================================
# Agent save / load tools — portable agent configuration files
# ======================================================================
#
# These tools let agents (and the human user, via the GUI's "Save Agent
# to File" / "Load Agent from File" buttons) exchange agent definitions
# between different setups or keep a library of best agents.
#
# A saved-agent JSON contains ONLY the agent itself — system prompt,
# model, endpoint, parameters, capability flags, tool list and the
# description. It NEVER contains chat history.
#
# File format:
# {
#   "agent_file_format": "saved-agent",
#   "agent_file_version": 1,
#   "agent": { name, description, system_prompt, model, num_ctx,
#              temperature, tools, max_context_messages,
#              max_tool_iterations, endpoint, active, silent,
#              use_tools, image, audio, mcp, mcp_servers,
#              mcp_server_defs, prompts }
# }
#
# `mcp` / `mcp_servers` carry the agent's native-MCP capability flag and
# subscription list; `mcp_server_defs` embeds the DEFINITIONS of exactly
# the subscribed servers so a saved agent stays portable — import_agent
# re-registers any definition the local registry is missing.
#
# `prompts` lists the selected prompt FILES by name (no .txt) — the
# texts themselves are NOT embedded; they are re-resolved from the
# prompts/ folder on the machine that loads the agent, so a saved
# agent stays small and portable.
# ======================================================================


def core_agent_save(name: str, path: str = None) -> str:
    """
    Save an agent's configuration to a JSON file — WITHOUT any chat history.

    The file contains only the agent itself: system prompt, model, endpoint,
    parameters, capability flags (use_tools / image / audio), the tool list
    and the description. Use it to keep a library of your best agents or to
    exchange agents between different setups; load them back with
    `core_agent_load` (or the GUI's "Load Agent from File" button).

    Args:
        name (str): Name of the agent to save. The agent must exist.
        path (str, optional): Target file path. When omitted, the file is
                written as "<name>.json" in the current folder. A ".json"
                extension is appended automatically when missing.

    Returns:
        str: Confirmation with the full path written, or an error message.
    """
    manager = context.manager
    if manager is None:
        return "Error: Agent manager not initialized."

    name = str(name).strip() if name is not None else ""
    if not name:
        return "Error: 'name' argument is required."
    if manager.get_agent(name) is None:
        return f"No agent with name {name}"

    try:
        data = manager.export_agent(name)
    except Exception as e:
        return f"Error exporting agent '{name}': {e}"

    if path is None or not str(path).strip():
        target = f"{name}.json"
    else:
        target = str(path).strip()
    if not target.lower().endswith(".json"):
        target += ".json"

    try:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return (f"Agent '{name}' saved to {target} "
                f"(configuration only — no chat history included).")
    except Exception as e:
        return f"Error writing file '{target}': {e}"


def core_saved_agents_list() -> list:
    """
    List the saved-agent JSON files that exist in the current folder.

    EVERY call re-scans the folder, so files that were just saved (by
    `core_agent_save`, the GUI, or copied in from another machine) show
    up immediately. Only files carrying the saved-agent format marker
    are reported — other JSON files in the folder are ignored.

    Returns:
        list[dict]: One entry per saved agent, each with:
            file        — file name (as found in the current folder)
            name        — the agent name stored in the file
            description — the agent's description (empty string if none)
            model       — the model the agent uses
            endpoint    — the endpoint the agent is configured for
            The list is empty when no saved agents are found.
    """
    try:
        entries = os.listdir(".")
    except Exception as e:
        return [{"error": f"Could not list current folder: {e}"}]

    results = []
    for fn in sorted(entries):
        if not fn.lower().endswith(".json"):
            continue
        try:
            with open(fn, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # Unreadable / invalid JSON — not a saved agent, skip it.
            continue
        if not isinstance(data, dict):
            continue
        if data.get("agent_file_format") != "saved-agent":
            continue
        params = data.get("agent")
        if not isinstance(params, dict):
            continue
        results.append({
            "file": fn,
            "name": str(params.get("name", "")),
            "description": str(params.get("description", "")),
            "model": str(params.get("model", "")),
            "endpoint": str(params.get("endpoint", "local")),
        })
    return results


def core_agent_load(path: str, new_name: str = None) -> str:
    """
    Load a saved agent from a JSON file and add it to the current session.

    The file must be a saved-agent file created by `core_agent_save` or the
    GUI's "Save Agent to File" button. Only the agent configuration is
    loaded — no chat history ever comes with it.

    Args:
        path (str): File path of the saved agent. May be a full/relative
                path or simply a name / file name (with or without the
                ".json" extension) — the tool looks in the current folder.
                Use `core_saved_agents_list` to see the available files.
        new_name (str, optional): Import the agent under a new name. Use
                this when an agent with the stored name already exists
                in the session (otherwise the load is refused with an
                explanatory error).

    Returns:
        str: Confirmation describing the loaded agent (name, model,
             endpoint), or an error message.
    """
    manager = context.manager
    if manager is None:
        return "Error: Agent manager not initialized."

    raw = str(path).strip() if path is not None else ""
    if not raw:
        return "Error: 'path' argument is required."

    # Resolve: exact path first, then "<name>.json" in the current folder.
    resolved = None
    for candidate in (raw, raw if raw.lower().endswith(".json") else raw + ".json"):
        if candidate and os.path.isfile(candidate):
            resolved = candidate
            break
    if resolved is None:
        return (f"No saved agent file found for '{raw}' "
                f"(looked in the current folder).")

    try:
        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"Error reading '{resolved}': {e}"

    new_name = (str(new_name).strip()
                if new_name is not None and str(new_name).strip() else None)

    try:
        agent = manager.import_agent(data, new_name=new_name)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error loading agent from '{resolved}': {e}"

    return (f"Agent '{agent.name}' loaded from {resolved} "
            f"(model: {agent.model}, endpoint: {agent.endpoint or 'local'}).")
''',

    # ==================================================================
    # NATIVE MCP management tools (mcp_tools.py).
    #
    # These six tools are the self-management surface of the native MCP
    # engine (mcp_manager.py). They are AUTO-AVAILABLE to every agent
    # whose "MCP" capability checkbox is on (Agent.call injects them
    # without any selection in Available Tools), so the operator can
    # simply prompt an agent: "find an MCP server from this info and
    # subscribe" — the agent calls mcp_subscribe and every tool of that
    # server immediately becomes a first-class, directly callable tool
    # (mcp__<server>__<tool>) in the SAME tools array as the built-ins.
    #
    # They can ALSO be selected manually in Available Tools for agents
    # without the MCP checkbox — in that case subscribing automatically
    # switches the agent's `mcp` capability on (subscribe implies intent).
    # ==================================================================
    "mcp_tools.py": '''
import json
import shlex

from agent import current_agent_name


def _mgr():
    """Return the app-wide native MCP engine (None when unavailable)."""
    try:
        import mcp_manager
        return mcp_manager.get_manager()
    except Exception:
        return None


def _engine_error():
    return ("Error: the native MCP engine is not available on this machine. "
            "Make sure mcp_manager.py sits next to the app and the app was "
            "started through main.py.")


def _fresh_registry(mgr):
    """Auto re-read of mcp_servers.json before any listing — servers
    registered by the operator, by OTHER AGENTS or by other processes
    appear immediately, without restarting the app (the proven PyQt
    behaviour: the list is re-read whenever it is called)."""
    fn = getattr(mgr, "maybe_reload_registry", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass


def _me():
    """Name of the calling agent (for per-agent subscription lookups)."""
    name = current_agent_name.get()
    return name if name and name != "unknown" else ""


def _parse_args_list(value):
    """Accept a list, a JSON array string, shell-style text, or CSV text."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(a) for a in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(a) for a in parsed]
        except json.JSONDecodeError:
            pass
        try:
            return shlex.split(s)
        except ValueError:
            return [p for p in s.replace(",", " ").split() if p]
    return [str(value)]


def _parse_env(value):
    """Accept a dict, a JSON object string, or KEY=VALUE lines."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass
        out = {}
        for line in s.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
        return out
    return {}


def mcp_list_servers() -> list:
    """
    List the MCP servers available in this app (the native MCP registry).

    For every server you see: its name, transport (stdio / sse / http),
    what it runs (command or URL), its live connection state and how many
    tools it exposes. Servers marked SUBSCRIBED are the ones whose tools
    you can already call DIRECTLY as regular tools named
    mcp__<server>__<tool> (they are in your tool list — no wrapper needed).

    Use mcp_subscribe(server) to connect yourself to one of them; use
    mcp_subscribe(server=..., command=..., args=...) to register a NEW
    server from info you were given, then subscribe.

    Returns:
        list[str]: One line per registry server plus a summary line.
    """
    mgr = _mgr()
    if mgr is None:
        return [_engine_error()]
    _fresh_registry(mgr)      # registry definitions fresh before listing
    me = _me()
    lines = []
    registry = mgr.list_registry_servers()
    if not registry:
        lines.append("MCP registry is empty. Register a server with "
                     "mcp_subscribe(server=<name>, command=<cmd>, args=[...]) "
                     "or url=..., then it will be subscribed automatically.")
        return lines
    my_subs = set(mgr.get_agent_subscriptions(me)) if me else set()
    for name in sorted(registry, key=str.lower):
        cfg = registry[name]
        status = mgr.status_info_for(name)
        tag = "SUBSCRIBED (native tools available to you)" if name in my_subs \\
              else "not subscribed (use mcp_subscribe to get its tools)"
        desc = (cfg.get("description") or "").strip()
        desc_part = f" — {desc}" if desc else ""
        lines.append(f"{name} [{tag}] — {status}{desc_part}")
    lines.append(f"({len(registry)} server(s) in the registry; you are "
                 f"subscribed to {len(my_subs)}).")
    return lines


def mcp_subscribe(server: str, command: str = None, args: list = None,
                  env: dict = None, url: str = None, transport: str = None,
                  description: str = "") -> str:
    """
    Subscribe yourself to an MCP server — its tools become YOUR tools.

    TWO ways to use it:

    1) Subscribe to a server that is already in the registry:
         mcp_subscribe(server="godot")

    2) Register a NEW server from info you were given AND subscribe in one
       step. For a local command server (stdio):
         mcp_subscribe(server="filesystem",
                       command="npx",
                       args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                       description="Filesystem access")
       For a remote server (sse or streamable-http):
         mcp_subscribe(server="my-remote", url="http://192.168.3.5:3000/sse",
                       transport="sse")

    After a successful subscription every tool of that server is callable
    DIRECTLY as a normal tool named mcp__<server>__<tool> (for example
    mcp__godot__godot_get_version) — with its real arguments, no wrapper
    syntax. The subscription is saved into your agent configuration and
    survives save/load of the session and of your agent file.

    Args:
        server (str): Short name for the server (used in tool names — keep
                it short, lowercase, no spaces).
        command (str, optional): For stdio servers — the executable to run
                (e.g. "npx" or an absolute path like
                "/home/alex/.local/bin/godot-mcp"). Providing it registers
                a new server.
        args (list, optional): Command arguments (list of strings, e.g.
                ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]).
        env (dict, optional): Extra environment variables for the command
                (e.g. {"GODOT_PATH": "/usr/bin/godot"}).
        url (str, optional): For remote servers — the http(s) URL of the
                sse or streamable-http endpoint. Providing it registers a
                new remote server.
        transport (str, optional): "stdio" (default), "sse" or "http".
        description (str, optional): One line about what the server does.

    Returns:
        str: Confirmation including the exact tool names you gained, or a
        readable error (unknown server, bad definition, connection
        failure...).
    """
    mgr = _mgr()
    if mgr is None:
        return _engine_error()

    server = str(server or "").strip()
    if not server:
        return "Error: 'server' (a short name) is required."

    has_definition = (command is not None and str(command).strip()) or \\
                     (url is not None and str(url).strip())
    if has_definition:
        try:
            mgr.register_server(
                name=server,
                transport=(str(transport).strip().lower()
                           if transport and str(transport).strip() else
                           ("http" if (url and str(url).strip()) else "stdio")),
                command=str(command or "").strip(),
                args=_parse_args_list(args),
                env=_parse_env(env),
                url=str(url or "").strip(),
                description=str(description or "").strip(),
            )
        except Exception as e:
            return f"Error registering MCP server '{server}': {e}"

    me = _me()
    if not me:
        return ("Error: mcp_subscribe can only be called from inside an "
                "agent's tool execution (no calling agent identified).")
    return mgr.subscribe_agent(me, server)


def mcp_unsubscribe(server: str) -> str:
    """
    Remove your subscription to an MCP server.

    The server's native tools (mcp__<server>__*) disappear from your tool
    list; your past calls to them are dropped from your context. The
    server DEFINITION stays in the registry (other agents may still use
    it), and the server process is shut down when no agent needs it.

    Args:
        server (str): Name of the server (as shown by mcp_list_servers).

    Returns:
        str: Confirmation or an error message.
    """
    mgr = _mgr()
    if mgr is None:
        return _engine_error()
    me = _me()
    if not me:
        return ("Error: mcp_unsubscribe can only be called from inside an "
                "agent's tool execution.")
    return mgr.unsubscribe_agent(me, str(server or "").strip())


def mcp_list_tools(server: str = None) -> list:
    """
    List the tools of your subscribed MCP servers (names + arguments).

    Every listed tool is ALREADY callable directly by its full name —
    mcp__<server>__<tool> — exactly like a built-in tool, with the
    arguments shown. Use this to inspect argument shapes before calling.

    Args:
        server (str, optional): Restrict the listing to one server by
                name. When omitted, all of your subscribed servers are
                listed (or, if you have none yet, all connected servers).

    Returns:
        list[str]: Tool entries (name, short description, arguments,
        required arguments), grouped per server.
    """
    mgr = _mgr()
    if mgr is None:
        return [_engine_error()]
    _fresh_registry(mgr)      # registry definitions fresh before listing
    me = _me()
    server = str(server or "").strip() or None

    # Default: the calling agent's subscriptions; fall back to any
    # connected server when the caller has none (e.g. exploring).
    if server is None and me:
        subs = mgr.get_agent_subscriptions(me)
        if subs:
            return mgr.list_tools_info_multi(subs)
        return mgr.list_tools_info_multi([])
    return mgr.list_tools_info(server)


def mcp_call_tool(tool: str, arguments: dict = None, timeout: int = 300) -> str:
    """
    Call a tool on one of your subscribed MCP servers by name (fallback).

    Normally you do NOT need this: every tool of your subscribed servers
    is already a direct tool of yours (mcp__<server>__<tool>) — call those
    directly. Use this tool only as a fallback (e.g. to pass a longer
    timeout, or to reach a tool you know exists but that was not yet in
    your tool list).

    Accepted tool spellings:
        mcp__<server>__<tool>   (the full namespaced name — preferred)
        <server>:<tool>
        <tool>                  (only when unambiguous across your servers)

    Args:
        tool (str): The MCP tool to call (see mcp_list_tools for names).
        arguments (dict, optional): The tool's arguments as an object.
        timeout (int, optional): Per-call timeout in seconds (default 300).

    Returns:
        str: The tool's text result, or a readable error message.
    """
    mgr = _mgr()
    if mgr is None:
        return _engine_error()
    me = _me()
    if not me:
        return ("Error: mcp_call_tool can only be called from inside an "
                "agent's tool execution.")
    tool = str(tool or "").strip()
    if not tool:
        return "Error: 'tool' is required."
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return "Error: 'arguments' must be a JSON object."
    if arguments is not None and not isinstance(arguments, dict):
        return "Error: 'arguments' must be a JSON object."
    try:
        timeout_s = max(5, int(float(timeout)))
    except (TypeError, ValueError):
        timeout_s = 300
    return mgr.call_tool(me, tool, arguments or {}, timeout=timeout_s)


def mcp_status() -> list:
    """
    Health report of the native MCP engine: every registry server, its
    connection state and tool count — plus your own subscriptions.

    Returns:
        list[str]: One status line per server + a summary of your
        subscriptions.
    """
    mgr = _mgr()
    if mgr is None:
        return [_engine_error()]
    _fresh_registry(mgr)      # registry definitions fresh before reporting
    lines = mgr.status_info()
    me = _me()
    if me:
        subs = mgr.get_agent_subscriptions(me)
        lines.append(f"You ({me}) are subscribed to: "
                     f"{', '.join(subs) if subs else '(none)'}")
    return lines
'''
}
