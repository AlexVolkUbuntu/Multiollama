import importlib
import inspect
import sys
import time
from pathlib import Path
from typing import Dict, List, Callable, Optional, get_type_hints
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import re
import os
import contextvars

# Endpoint pause-gate registry (context.EndpointGate). Used by Agent.call
# to cooperatively pause at tool-call iteration boundaries while a remote
# `core_agent_execute` is running an agent on this agent's endpoint.
# NOTE: context.py imports nothing from this module, so there is no import
# cycle here.
import context

# ----------------------------------------------------------------------
# NATIVE MCP engine (mcp_manager.py).
#
# The import is deliberately defensive: mcp_manager.py only pulls in
# stdlib modules plus app_paths/context at import time (the heavy `mcp`
# package is imported lazily inside its connection coroutines), so this
# can never break app startup. `mcp_manager.get_manager()` returns None
# when the engine is not initialized (e.g. an embedding without the
# module) — every MCP code path in Agent.call treats None as "feature
# unavailable" and degrades gracefully to normal tool behaviour.
# ----------------------------------------------------------------------
try:
    import mcp_manager as _mcp_manager_module
except Exception:
    _mcp_manager_module = None


def _get_mcp_manager():
    """Return the app-wide MCPManager singleton, or None when absent."""
    if _mcp_manager_module is None:
        return None
    try:
        return _mcp_manager_module.get_manager()
    except Exception:
        return None


# ----------------------------------------------------------------------
# Tool-result prepend / prompt-injection subsystem (tool_injection.py).
# Provides:
#   - build_tool_result_prepend(): the SYSTEM section prepended to every
#     tool result handed back to the model (tool-call usage counter +
#     accumulated operator injections — delivered once, then cleared);
#   - take_pending_injections(): drains the per-agent injection queue
#     (filled by the GUI "Inject" button) when a tool result is sent;
#   - format_messages_for_debug(): renders the exact full prompt for the
#     Thinking tab (see the "prompt" thinking event in Agent.call).
# The module is standalone (stdlib only) — no import cycle possible.
# ----------------------------------------------------------------------
import tool_injection

# ----------------------------------------------------------------------
# Context variable for tracking which agent is currently executing tools.
# IMPORTANT: this MUST stay unchanged so existing tools (team_email.py,
# return_caller_name.py, and any user-added tools) continue to identify
# the calling agent correctly.
# ----------------------------------------------------------------------
current_agent_name: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_agent_name", default="unknown"
)

# ----------------------------------------------------------------------
# Context variable that tracks the chain of agents currently being
# executed via the `core_agent_execute` tool. Each entry is the name of
# an agent that was invoked from inside another agent's tool-call loop
# and is therefore "paused" (its `Agent.call()` is blocked on the
# Python stack waiting for the tool to return).
#
# Used by `core_agent_execute` (see core_tools.py) to break recursive
# cycles where agent A calls B which calls A again — without this guard
# the two agents would ping-pong forever, each one paused mid-turn
# waiting for the other to finish.
#
# Implementation note: a `ContextVar` (rather than a plain `threading.local`)
# is used because tool execution is dispatched via a `ThreadPoolExecutor`
# in `Agent._execute_tool_with_context`, which copies the current context
# to the worker thread. That means the call stack naturally inherits the
# caller's chain and any nested `core_agent_execute` will see the correct
# ancestor set without us having to pass it manually.
# ----------------------------------------------------------------------
agent_call_stack: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "agent_call_stack", default=()
)

# ----------------------------------------------------------------------
# Image path extraction helper
# ----------------------------------------------------------------------
IMAGE_EXTENSIONS = (
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.svg', '.ico'
)

LOCAL_PATH_PATTERN = re.compile(
    r'(?:'
    r'(?:[a-zA-Z]:\\|/|\./|\.\./|~/)'
    r'[^\s<>:"|?*]*(?:\\|/)?'
    r'[^\s<>:"|?*]*)'
    r'\.(?:' + '|'.join(ext[1:] for ext in IMAGE_EXTENSIONS) + r')',
    re.IGNORECASE
)

def extract_local_image_paths(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    candidates = LOCAL_PATH_PATTERN.findall(text)
    filtered = []
    seen = set()
    for path in candidates:
        if re.match(r'https?://|ftp://|www\.', path, re.IGNORECASE):
            continue
        normalized = str(Path(path))
        if normalized not in seen:
            seen.add(normalized)
            if os.path.exists(normalized):
                filtered.append(normalized)
    return filtered

# ----------------------------------------------------------------------
# Audio (.wav) path extraction helper
#
# Ollama has NO dedicated `audio` field in its chat API: audio input is
# passed through the SAME `images` field of the message (base64-encoded
# .wav data). The Ollama server detects audio by the RIFF/WAVE magic
# bytes of the payload and routes it to the model's audio channel. This
# is why audio "proceeds the same as images" and why .wav files can be
# freely COMBINED with real images in one message.
#
# Only .wav is matched here (per the spec) — other audio formats (mp3,
# ogg, ...) are NOT accepted by Ollama's audio path.
# ----------------------------------------------------------------------
WAV_LOCAL_PATH_PATTERN = re.compile(
    r'(?:'
    r'(?:[a-zA-Z]:\\|/|\./|\.\./|~/)'
    r'[^\s<>:"|?*]*(?:\\|/)?'
    r'[^\s<>:"|?*]*)'
    r'\.wav',
    re.IGNORECASE
)

def extract_local_wav_paths(text: str) -> List[str]:
    """Extract existing local .wav file paths mentioned in *text*.

    Mirrors extract_local_image_paths() but only for .wav files, which
    are the only audio format Ollama accepts through the images field.
    """
    if not isinstance(text, str):
        return []
    candidates = WAV_LOCAL_PATH_PATTERN.findall(text)
    filtered = []
    seen = set()
    for path in candidates:
        if re.match(r'https?://|ftp://|www\.', path, re.IGNORECASE):
            continue
        normalized = str(Path(path))
        if normalized not in seen:
            seen.add(normalized)
            if os.path.exists(normalized):
                filtered.append(normalized)
    return filtered

def inject_images_last_user_or_tool(messages: List[dict]) -> List[dict]:
    """Attach image paths (if any) to the last user/tool message.

    Also strips any stale ``images`` key left over from a previous iteration
    so the same dict doesn't keep accumulating image references it no longer
    needs.

    Kept unchanged for backward compatibility. The capability-gated
    injection used by Agent.call is `inject_media_last_user_or_tool`
    (see below), which generalises this to images AND audio.
    """
    for msg in reversed(messages):
        role = msg.get("role")
        if role in ("user", "tool") and "content" in msg and isinstance(msg["content"], str):
            paths = extract_local_image_paths(msg["content"])
            if paths:
                msg["images"] = paths
            else:
                msg.pop("images", None)
            break
    return messages

def inject_media_last_user_or_tool(messages: List[dict],
                                   enable_images: bool = True,
                                   enable_audio: bool = True) -> List[dict]:
    """Attach image and/or .wav audio paths to the last user/tool message.

    This is the capability-gated generalisation of
    `inject_images_last_user_or_tool`:

    - `enable_images=True`  -> local image paths are attached.
    - `enable_audio=True`   -> local .wav paths are attached.
    - Both may be True: audio and images travel through the SAME
      Ollama `images` field (the server tells them apart by the
      payload's magic bytes — RIFF/WAVE means audio), so they are
      simply combined into one list.
    - Media is attached to the last user OR tool message, so it keeps
      working in the middle of a tool-call loop (each new tool result
      that mentions a file re-attaches it for the next iteration).

    When both flags are False the function is never called (Agent.call
    skips it) and no `images` key is ever attached — this is what stops
    non-vision models from failing on image payloads.

    Stale ``images`` keys from a previous iteration are stripped when the
    current message no longer references any media, same as the original.
    """
    for msg in reversed(messages):
        role = msg.get("role")
        if role in ("user", "tool") and "content" in msg and isinstance(msg["content"], str):
            content = msg["content"]
            media = []
            seen = set()
            if enable_images:
                for p in extract_local_image_paths(content):
                    if p not in seen:
                        seen.add(p)
                        media.append(p)
            if enable_audio:
                for p in extract_local_wav_paths(content):
                    if p not in seen:
                        seen.add(p)
                        media.append(p)
            if media:
                msg["images"] = media
            else:
                msg.pop("images", None)
            break
    return messages

# ----------------------------------------------------------------------
# Function to tool conversion
# ----------------------------------------------------------------------
def function_to_tool(func) -> dict:
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    type_hints = get_type_hints(func)
    properties = {}
    required = []
    for name, param in sig.parameters.items():
        param_type = type_hints.get(name, str).__name__
        type_map = {
            'int': 'integer',
            'float': 'number',
            'str': 'string',
            'bool': 'boolean',
            'list': 'array',
            'dict': 'object',
        }
        json_type = type_map.get(param_type, 'string')
        properties[name] = {
            'type': json_type,
            'description': f'Parameter {name}',
        }
        if param.default == inspect.Parameter.empty:
            required.append(name)
    return {
        'type': 'function',
        'function': {
            'name': func.__name__,
            'description': doc.split('\n')[0] if doc else '',
            'parameters': {
                'type': 'object',
                'properties': properties,
                'required': required,
            },
        },
    }

# ----------------------------------------------------------------------
# ToolLoader
# ----------------------------------------------------------------------
class ToolLoader:
    def __init__(self, tools_dir: str = "tools"):
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(exist_ok=True)
        (self.tools_dir / "__init__.py").touch(exist_ok=True)
        self.tools: Dict[str, Callable] = {}
        self.tool_to_file: Dict[str, str] = {}
        self.tool_file_docs: Dict[str, str] = {}
        self.tool_docs: Dict[str, str] = {}
        self.file_tools: Dict[str, List[str]] = {}
        # Directory fingerprint at the time of the last scan — the
        # auto-reload baseline used by maybe_reload().
        self._scan_sig: Optional[tuple] = None
        self._load_all()

    def _scan_signature(self) -> Optional[tuple]:
        """Cheap fingerprint of the tools directory: (filename, mtime_ns,
        size) of every .py file, sorted. Detects new / edited / deleted
        tool files WITHOUT re-importing anything, so listing endpoints
        can call maybe_reload() on every request at negligible cost."""
        try:
            entries = []
            for p in self.tools_dir.glob("*.py"):
                if p.name == "__init__.py":
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                entries.append((p.name, st.st_mtime_ns, st.st_size))
            return tuple(sorted(entries))
        except Exception:
            return None

    def maybe_reload(self) -> bool:
        """Auto-reload for catalog LISTINGS (the proven PyQt behaviour:
        whenever a list is called — e.g. the agent-creation pickers — the
        tools directory is re-read, so tool files dropped in by the
        operator OR CREATED BY AN AGENT become visible immediately,
        without restarting the app and without pressing Refresh).

        Only a full re-import happens when the directory fingerprint
        actually changed; otherwise this is a handful of stat() calls.
        Never raises — a failed stat leaves the loaded catalog untouched.

        Returns:
            bool: True when a reload was performed.
        """
        try:
            sig = self._scan_signature()
            if sig is not None and sig != self._scan_sig:
                self.reload()
                return True
        except Exception:
            pass
        return False

    def _load_all(self):
        sys.path.insert(0, str(self.tools_dir.parent))
        # Build into FRESH dicts and swap at the end: in the headless
        # backend, API threads list the catalog while worker threads run
        # agents, so a concurrent get_tool() must never observe a
        # half-cleared catalog (observable behaviour is unchanged).
        tools: Dict[str, Callable] = {}
        tool_to_file: Dict[str, str] = {}
        tool_file_docs: Dict[str, str] = {}
        tool_docs: Dict[str, str] = {}
        file_tools: Dict[str, List[str]] = {}

        for pyfile in sorted(self.tools_dir.glob("*.py"), key=lambda p: p.name.lower()):
            if pyfile.name == "__init__.py":
                continue
            file_name = pyfile.stem
            file_tools[file_name] = []
            module_name = f"{self.tools_dir.name}.{file_name}"
            try:
                if module_name in sys.modules:
                    del sys.modules[module_name]
                module = importlib.import_module(module_name)
                importlib.reload(module)
                module_doc = inspect.getdoc(module) or ""
                tool_file_docs[file_name] = module_doc
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    if not name.startswith("_"):
                        tools[name] = obj
                        tool_to_file[name] = file_name
                        tool_docs[name] = inspect.getdoc(obj) or ""
                        file_tools[file_name].append(name)
            except Exception as e:
                print(f"Failed to load {pyfile.name}: {e}")

        for filename in file_tools:
            file_tools[filename].sort(key=str.lower)

        self.tools = tools
        self.tool_to_file = tool_to_file
        self.tool_file_docs = tool_file_docs
        self.tool_docs = tool_docs
        self.file_tools = file_tools
        # Record the fingerprint of the state we just loaded so
        # maybe_reload() has an accurate baseline.
        self._scan_sig = self._scan_signature()

    def reload(self):
        self._load_all()

    def get_tool_names(self) -> List[str]:
        return list(self.tools.keys())

    def get_tool(self, name: str) -> Optional[Callable]:
        return self.tools.get(name)

    def get_tool_file_names(self) -> List[str]:
        """Return loaded Python tool-file names without the .py extension."""
        return list(self.file_tools.keys())

    def get_tools_for_files(self, file_names: List[str]) -> List[str]:
        """Return unique tool/function names contributed by selected files."""
        result = []
        seen = set()
        for filename in file_names or []:
            for tool_name in self.file_tools.get(filename, []):
                if tool_name not in seen:
                    seen.add(tool_name)
                    result.append(tool_name)
        return result

    def get_tool_file_docstring(self, file_name: str) -> str:
        """Return the module-level docstring for a tool file."""
        return self.tool_file_docs.get(file_name, "")

    def get_tool_docstring(self, tool_name: str) -> str:
        """Return the main function docstring for a loaded tool."""
        return self.tool_docs.get(tool_name, "")

    def get_tool_file_for_tool(self, tool_name: str) -> Optional[str]:
        """Return the .py filename stem that defines the named tool."""
        return self.tool_to_file.get(tool_name)


# ----------------------------------------------------------------------
# PromptLoader
#
# Loads reusable prompt snippets (.txt files) from the "prompts" folder
# that sits next to the app (same pattern as the "tools" folder).
#
# WHY: when you build many agents, they often share parts of their
# system prompts (formatting rules, language rules, safety rules, ...).
# Copy-pasting those blocks into every agent is fragile — fix a rule in
# one place and every other agent keeps the stale copy. Instead, each
# shared block lives in ONE prompts/<name>.txt file, and agents select
# the files they need (multi-selection in the Edit Agent dialog, exactly
# like the tools selection). At inference time the selected texts are
# appended to the agent's system prompt inside a <rules>...</rules>
# section (see build_system_prompt below).
#
# Because the texts are resolved from disk at CALL time (not baked into
# the agent when it is created), editing a .txt file takes effect on
# the next inference for EVERY agent that selected it.
# ----------------------------------------------------------------------
class PromptLoader:
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        # Auto-create the folder so the feature is discoverable: the
        # empty "prompts" folder shows up next to the app and the user
        # only has to drop .txt files into it.
        self.prompts_dir.mkdir(exist_ok=True)
        # name (filename WITHOUT the .txt extension) -> full text
        self.prompts: Dict[str, str] = {}
        # Folder fingerprint at the time of the last scan — the
        # auto-reload baseline used by maybe_reload().
        self._scan_sig: Optional[tuple] = None
        self._load_all()

    def _scan_signature(self) -> Optional[tuple]:
        """Cheap fingerprint of the prompts folder: (filename, mtime_ns,
        size) of every .txt file, sorted. Detects new / edited / deleted
        prompt files without re-reading any file contents."""
        try:
            entries = []
            for p in self.prompts_dir.glob("*.txt"):
                try:
                    st = p.stat()
                except OSError:
                    continue
                entries.append((p.name, st.st_mtime_ns, st.st_size))
            return tuple(sorted(entries))
        except Exception:
            return None

    def maybe_reload(self) -> bool:
        """Auto-reload for prompt LISTINGS (the proven PyQt behaviour:
        whenever a list is called — e.g. the agent-creation picker — the
        prompts folder is re-scanned, so prompt files dropped in or
        edited while the app runs appear on the very next listing,
        without restarting and without pressing Refresh).

        Returns:
            bool: True when a re-scan was performed.
        """
        try:
            sig = self._scan_signature()
            if sig is not None and sig != self._scan_sig:
                self.reload()
                return True
        except Exception:
            pass
        return False

    def _load_all(self):
        prompts: Dict[str, str] = {}
        for txtfile in sorted(self.prompts_dir.glob("*.txt"),
                              key=lambda p: p.name.lower()):
            try:
                text = txtfile.read_text(encoding="utf-8")
            except Exception as e:
                print(f"Failed to load prompt file {txtfile.name}: {e}")
                continue
            # Store under the stem (no ".txt") — that is the name the
            # GUI selector shows and the name agents carry in their
            # `prompts` list.
            prompts[txtfile.stem] = text
        self.prompts = prompts
        self._scan_sig = self._scan_signature()

    def reload(self):
        """Re-scan the prompts folder (new / edited / deleted .txt files)."""
        self._load_all()

    def get_prompt_names(self) -> List[str]:
        """Return all loaded prompt names (no .txt extension)."""
        return list(self.prompts.keys())

    def get_prompt_text(self, name: str) -> Optional[str]:
        """Return the full text of one prompt file, or None if missing."""
        return self.prompts.get(name)

    def get_prompt_description(self, name: str) -> str:
        """Return the first non-empty line of a prompt file (for tooltips
        and listings). Empty string when the file or line is unavailable."""
        text = self.prompts.get(name)
        if not text:
            return ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:200]
        return ""


# ----------------------------------------------------------------------
# build_system_prompt
# ----------------------------------------------------------------------
def build_system_prompt(base_prompt: str,
                        prompt_names: Optional[List[str]],
                        prompt_loader: Optional["PromptLoader"]) -> str:
    """Combine an agent's own system prompt with its selected prompt files.

    Layout (exactly as specified):

        <original system prompt>

        <rules>
        <text of prompts/x.txt>
        <blank line>
        <text of prompts/y.txt>
        </rules>

    * When the agent has NO selected prompt files (or none of them can
      be loaded), the original system prompt is returned UNCHANGED —
      no <rules> section is added at all.
    * The <rules> opening tag comes FIRST, then every selected file's
      text (separated by a blank line), and the </rules> closing tag is
      appended after the LAST text.
    * Prompt files that no longer exist on disk are silently skipped —
      a deleted file must never break inference.
    * The order of the texts follows the agent's stored `prompts` list
      (the GUI stores it alphabetically, but a hand-edited saved-agent
      JSON may carry any order and it is respected).
    """
    base = base_prompt or ""
    names = list(prompt_names or [])
    if not names:
        return base
    if prompt_loader is None:
        return base

    texts = []
    for name in names:
        text = prompt_loader.get_prompt_text(name)
        if text is not None and text.strip():
            texts.append(text.strip())

    if not texts:
        return base

    rules_section = "<rules>\n" + "\n\n".join(texts) + "\n</rules>"

    if base.strip():
        return base.rstrip() + "\n\n" + rules_section
    return rules_section


# ----------------------------------------------------------------------
# build_context
# ----------------------------------------------------------------------
from typing import List

def build_context(agent_name: str, full_history: List[dict],
                  max_messages: int = 30,
                  allowed_tools: Optional[List[str]] = None,
                  ephemeral_question: Optional[str] = None) -> List[dict]:
    """Build the message context for an agent's next inference call.

    Only ``max_messages`` limits how many recent messages are included.
    The previous hard-coded byte-limit truncation was removed — it was a
    hidden restriction that silently dropped messages and caused
    confusing context loss in long chats. If you need tighter bounds,
    lower ``max_context_messages`` in the agent settings.

    CRITICAL (thinking-leak fix):
    The `thinking` field is NEVER copied into the rebuilt context. The
    ollama Python client's `Message` class has a `thinking` field that
    gets preserved and sent to the model in the request. If a previous
    assistant message carried a `thinking` field, the model would see
    the raw internal reasoning from earlier turns — its own AND from
    other agents (because history is shared). That caused identity
    confusion and tool-call contamination in multi-agent chats.

    Only `role`, `content`, `tool_calls`, and `name` are reconstructed
    here; any `thinking` / `sender` / internal metadata on the stored
    history is deliberately dropped.

    CRITICAL (tool-leak fix):
    If `allowed_tools` is provided, any of the agent's OWN past
    `tool_calls` (on assistant messages) or `tool` role messages for
    tools NOT in `allowed_tools` are STRIPPED from the rebuilt context.
    This ensures that if the user changes an agent's tool selection at
    runtime (e.g. removes a tool that was previously assigned and
    used), the agent's prior tool calls for now-disallowed tools don't
    leak through and teach the model about unselected tools. Combined
    with the upfront whitelist check in `Agent.call`, this guarantees
    that unselected tools are NEVER visible to the model — not in the
    API `tools` parameter, not in the context messages, not in the
    runtime tool execution path.

    EPHEMERAL QUESTION (core_agent_execute_with_question):
    When `ephemeral_question` is supplied, it is appended AFTER
    truncation as the FINAL user message of the returned context — so
    no `max_messages` setting can ever cut it away. When
    `max_messages == 0` it REPLACES the "(continue)" placeholder and is
    the ONLY user message the agent sees. The question never enters
    the shared history; it exists solely inside this inference call.
    """
    transformed = []

    # FIX: when max_messages == 0, return a minimal user message instead
    # of an empty list. Qwen3.x Jinja templates require at least one
    # user-role message or the server returns a 500.
    #
    # FIX (question-with-zero-context): when an ephemeral question was
    # supplied (core_agent_execute_with_question), it REPLACES the
    # "(continue)" placeholder. "0 = don't leak the shared chat into the
    # agent" must not swallow the direct question — with 0 context the
    # question is the ONLY thing the agent sees besides its system
    # prompt.
    if max_messages == 0:
        q = str(ephemeral_question).strip() if ephemeral_question is not None else ""
        if q:
            return [{"role": "user", "content": q}]
        return [{"role": "user", "content": "(continue)"}]

    # Pre-compute the allowed-tools set for O(1) membership tests.
    # `None` means "no filtering" (backward-compatible behaviour for
    # callers that don't pass allowed_tools).
    allowed_set = set(allowed_tools) if allowed_tools is not None else None

    for msg in full_history:
        sender = msg.get("sender", "unknown")
        role = msg["role"]
        content = msg["content"]

        if sender == agent_name:
            if role == "assistant":
                # Skip empty-content assistant messages with no tool calls.
                # These pollute context and teach the model to produce
                # empty responses. Only keep assistant turns that actually
                # said something or called a tool.
                if not content.strip() and not msg.get("tool_calls"):
                    continue
                # Own assistant replies stay as-is.
                # NOTE: `thinking` is intentionally NOT copied — see the
                # docstring above for the full rationale.
                new_msg = {"role": "assistant", "content": content}
                if "tool_calls" in msg:
                    # Normalise tool_calls before sending to Ollama:
                    # - ensure each has a "function" dict with "name" and "arguments"
                    # - ensure arguments is a DICT (not a JSON string), because
                    #   some Ollama versions mis-serialise string arguments
                    #   into malformed XML when they convert the history back
                    #   into the model's chat template. This is a likely
                    #   contributor to the "XML syntax error: element <function>
                    #   closed by </parameter>" error.
                    clean_calls = []
                    for tc in msg["tool_calls"]:
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get("function", {})
                        if not isinstance(fn, dict):
                            continue
                        name = fn.get("name", "")
                        if not name:
                            continue
                        # TOOL-LEAK FIX: if the agent's current allowed
                        # tools don't include this tool, drop the
                        # tool_call entirely. The model must not see
                        # references to tools it shouldn't have — not
                        # even in its own past assistant turns.
                        if allowed_set is not None and name not in allowed_set:
                            continue
                        args = fn.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        if not isinstance(args, dict):
                            args = {}
                        clean_calls.append({
                            "function": {"name": name, "arguments": args}
                        })
                    if clean_calls:
                        new_msg["tool_calls"] = clean_calls
                transformed.append(new_msg)

            elif role == "tool":
                # Own tool results – preserve BOTH `name` (Ollama API
                # standard) and `tool_name` (legacy field some versions
                # used) so the tool result is recognised regardless of
                # which Ollama version is running.
                tool_id = msg.get("tool_name", msg.get("name", "unknown"))
                # TOOL-LEAK FIX: if this tool result is for a tool not
                # in the agent's current allowed list, skip it entirely.
                # The model must not see results from tools it shouldn't
                # have — not even its own past tool results for tools
                # that have since been removed from the allowed list.
                if allowed_set is not None and tool_id not in allowed_set:
                    continue
                transformed.append({
                    "role": "tool",
                    "content": content,
                    "name": tool_id,
                    "tool_name": tool_id,
                })

            else:
                # Own system/user messages – show as a self-note
                transformed.append({
                    "role": "user",
                    "content": f"{sender}: {content}"
                })

        else:
            # Messages from other agents – always become user messages.
            # NOTE: `thinking` and `tool_calls` from other agents are
            # deliberately NOT forwarded. Other agents' tool results
            # (role == "tool") are skipped entirely; their assistant
            # content is wrapped as "sender: content" so the model can
            # tell who said what without seeing internal reasoning or
            # raw tool-call structures that don't belong to it.
            if role == "tool":
                continue
            else:
                content = f"{sender}: {content}"
                transformed.append({
                    "role": "user",
                    "content": content,
                    "name": sender
                })

    # Truncation by message count only (no byte limit)
    if max_messages > 0:
        truncated = transformed[-max_messages:]
    else:
        truncated = transformed

    # FIX (issue: "last message was from assistant" / template errors):
    # Drop leading ORPHAN tool results — their requesting assistant
    # message fell outside the truncation window. A tool result with no
    # matching assistant tool_calls turn upsets several Ollama chat
    # templates. (Same guard build_dialogue_context has always had;
    # build_context was missing it.)
    while truncated and truncated[0].get("role") == "tool":
        truncated.pop(0)

    # If after truncation the first message isn't a user role, insert an identity reminder
    if truncated and truncated[0]["role"] != "user":
        truncated.insert(0, {
            "role": "user",
            "content": f"Reminder: You are {agent_name}. (Conversation continued)"
        })

    # EPHEMERAL QUESTION, normalised once (used by the two guards below).
    q = str(ephemeral_question).strip() if ephemeral_question is not None else ""

    # FIX: Qwen3.x template guard — ensure at least one user message
    # exists after all transformations and truncation. Covers the case
    # where the history is empty, or contains only tool messages from
    # other agents (which are skipped), leaving zero user-role entries.
    # Skipped when an ephemeral question is about to be appended — the
    # question itself satisfies the template's user-message requirement.
    if not q and not any(m.get("role") == "user" for m in truncated):
        truncated.insert(0, {"role": "user", "content": "(continue)"})

    # EPHEMERAL QUESTION (core_agent_execute_with_question): appended
    # AFTER truncation as the FINAL user message, so no max_messages
    # setting can ever cut it away. The question is never written into
    # the shared history — it exists only in this inference call.
    if q:
        truncated.append({"role": "user", "content": q})

    return truncated


def build_dialogue_context(agent_name: str, full_history: List[dict],
                            max_messages: int = 30,
                            allowed_tools: Optional[List[str]] = None) -> List[dict]:
    """Build the message context for an agent's turn inside a DIALOGUE.

    A dialogue (see the `core_agent_execute_dialogue` tool) is a PRIVATE
    side chat between exactly two agents: the CALLER — whose messages
    are stored with role "user" — and this agent (the CALLEE, the
    "assistant" side). The `full_history` passed in is the dialogue's
    OWN history; nothing from the main shared chat is ever present in
    it, and nothing from it is ever written back to the main chat.

    Differences from `build_context`:

      1. The partner's messages are presented as PLAIN user messages —
         content verbatim, WITHOUT the "sender: " prefix. The callee
         believes it is talking with the human user, which keeps its
         system prompt and conversational behaviour fully natural.

      2. `max_messages == 0` means UNLIMITED here. In the main chat, 0
         exists to prevent leaking the shared conversation into the
         agent; a dialogue history contains nothing from the main chat,
         so that rationale does not apply — and the dialogue feature
         promises that the called agent sees the previous messages of
         the side chat. A positive value still truncates to the last N
         messages (the agent's own setting remains in force).

      3. The partner's latest message is the LAST entry of the history,
         so count-based truncation can never remove it — the callee
         always sees the message it must answer.

    Same safety rules as `build_context` apply:
      - the `thinking` field is NEVER copied (thinking-leak fix);
      - own past tool_calls / tool results for tools NOT in
        `allowed_tools` are stripped (tool-leak fix);
      - empty-content assistant messages with no tool calls are skipped.
    """
    transformed = []

    allowed_set = set(allowed_tools) if allowed_tools is not None else None

    for msg in full_history:
        sender = msg.get("sender", "unknown")
        role = msg["role"]
        content = msg["content"]

        if sender == agent_name:
            # Own messages — identical handling to build_context.
            if role == "assistant":
                if not content.strip() and not msg.get("tool_calls"):
                    continue
                new_msg = {"role": "assistant", "content": content}
                if "tool_calls" in msg:
                    # Normalise tool_calls (dict arguments, allowed tools
                    # only) — same rules as build_context.
                    clean_calls = []
                    for tc in msg["tool_calls"]:
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get("function", {})
                        if not isinstance(fn, dict):
                            continue
                        name = fn.get("name", "")
                        if not name:
                            continue
                        # TOOL-LEAK FIX: drop tool_calls for tools that are
                        # not in the agent's current allowed list.
                        if allowed_set is not None and name not in allowed_set:
                            continue
                        args = fn.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        if not isinstance(args, dict):
                            args = {}
                        clean_calls.append({
                            "function": {"name": name, "arguments": args}
                        })
                    if clean_calls:
                        new_msg["tool_calls"] = clean_calls
                transformed.append(new_msg)

            elif role == "tool":
                tool_id = msg.get("tool_name", msg.get("name", "unknown"))
                # TOOL-LEAK FIX: skip results for disallowed tools.
                if allowed_set is not None and tool_id not in allowed_set:
                    continue
                transformed.append({
                    "role": "tool",
                    "content": content,
                    "name": tool_id,
                    "tool_name": tool_id,
                })

            else:
                # Defensive: own user/system messages shown as a self-note
                # (dialogues normally only contain own assistant/tool
                # messages, but stay consistent with build_context).
                transformed.append({
                    "role": "user",
                    "content": f"{sender}: {content}"
                })

        else:
            # Partner ("user" side) messages — PLAIN user messages with
            # the content VERBATIM: no "sender: " prefix, so the callee
            # thinks it is talking with the human user. The partner only
            # ever produces role "user" messages in a dialogue; anything
            # else is defensively skipped.
            if role == "tool":
                continue
            transformed.append({
                "role": "user",
                "content": content,
            })

    # Truncation: a positive max_messages keeps the last N messages
    # (the partner's latest message is the tail — always survives).
    # max_messages == 0 -> NO truncation (see docstring point 2).
    if max_messages > 0:
        truncated = transformed[-max_messages:]
    else:
        truncated = transformed

    # Drop leading orphan tool results — their requesting assistant
    # message fell outside the truncation window, and Ollama is happier
    # without a tool result that has no matching tool_calls turn.
    while truncated and truncated[0].get("role") == "tool":
        truncated.pop(0)

    # If after truncation the first message isn't a user role, insert an
    # identity reminder (same guard as build_context).
    if truncated and truncated[0]["role"] != "user":
        truncated.insert(0, {
            "role": "user",
            "content": f"Reminder: You are {agent_name}. (Conversation continued)"
        })

    # Qwen3.x template guard — at least one user message must exist.
    if not any(m.get("role") == "user" for m in truncated):
        truncated.insert(0, {"role": "user", "content": "(continue)"})

    return truncated

# ----------------------------------------------------------------------
# Agent class – fully synchronous
# ----------------------------------------------------------------------
class Agent:
    def __init__(self, name: str, system_prompt: str,
                 model: str = "qwen3.5:9b",
                 num_ctx: int = 32000,
                 temperature: float = 1.0,
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
                 prompts: Optional[List[str]] = None):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.tools = tools or []
        self.max_context_messages = max_context_messages
        # 0 = unlimited. This is a USER SETTING (exposed in the GUI),
        # not a hidden restriction.
        self.max_tool_iterations = max_tool_iterations

        # NEW: which Ollama machine this agent runs on.
        # "local" = local Ollama; "ip:port" (e.g. "192.168.3.3:11434") = remote.
        self.endpoint = endpoint if endpoint else "local"

        # ------------------------------------------------------------------
        # Description — a short, human-readable note about what this agent
        # is for. It exists purely for agent MANAGEMENT (the Edit dialog,
        # the agent dropdown tooltips, the sidebar, the saved-agent JSON
        # and the core_saved_agents_list tool). It is NEVER sent to the
        # model and has zero effect on inference.
        # ------------------------------------------------------------------
        self.description = description or ""

        # ------------------------------------------------------------------
        # Capability flags (per-agent, exposed as checkboxes in the GUI):
        #
        # `use_tools` (default True) — when False, NO tools are shown to
        #   the model at inference time (the API `tools` parameter is not
        #   sent), past tool_calls / tool results are stripped from the
        #   rebuilt context, and any (hallucinated) tool call is rejected
        #   by the whitelist check.
        #
        # `image` (default False) — when True, image paths found in the
        #   last user/tool message are attached to the model call
        #   (including mid tool-call loop iterations). When False (the
        #   default), images are NEVER attached — this fixes non-vision
        #   models failing on image payloads.
        #
        # `audio` (default False) — same as `image`, but for .wav files.
        #   Ollama receives audio through the same `images` message field
        #   (RIFF/WAVE magic-byte detection), so audio can be combined
        #   with images. Only .wav files are attached.
        # ------------------------------------------------------------------
        self.use_tools = bool(use_tools)
        self.image = bool(image)
        self.audio = bool(audio)

        # ------------------------------------------------------------------
        # NATIVE MCP capability (the "MCP" checkbox in the GUI, next to
        # Tools / Image / Audio).
        #
        # `mcp` (default False) — when True, the agent may SUBSCRIBE to
        #   any number of MCP servers. Every tool of every subscribed
        #   server is then exposed to the model DIRECTLY as a first-class
        #   tool definition (namespaced "mcp__<server>__<tool>") in the
        #   same `tools` array as the built-in tools — no wrapper syntax,
        #   no meta-tools, no retry loops hunting for the right call
        #   shape. Tool calls to those names are routed straight to the
        #   MCP server by Agent.call and results flow back through the
        #   normal tool-result pipeline (thinking tab, usage counter,
        #   operator injections, shared history).
        #
        #   Additionally, the MCP MANAGEMENT core tools (mcp_list_servers,
        #   mcp_subscribe, mcp_unsubscribe, mcp_list_tools, mcp_call_tool,
        #   mcp_status) become auto-available to this agent (without
        #   selecting them in Available Tools), so you can simply prompt:
        #   "find an MCP server from this info and subscribe".
        #
        # `mcp_servers` — the agent's subscription list (registry server
        #   NAMES, order preserved, duplicates dropped). It is part of the
        #   agent's serialised config: it is saved/loaded with the session,
        #   with the agent file (Save Agent to File / core_agent_save) and
        #   restored on import. Server DEFINITIONS live in the shared
        #   registry (mcp_servers.json) and travel inside session and
        #   saved-agent files as a snapshot.
        #
        # The master `use_tools` flag still gates everything: when Tools
        # is unchecked, no MCP tools are shown either (MCP is a tool
        # source, not a bypass of the Tools switch).
        # ------------------------------------------------------------------
        self.mcp = bool(mcp)
        self.mcp_servers: List[str] = []
        if mcp_servers:
            seen = set()
            for s in mcp_servers:
                s = str(s).strip()
                if s and s not in seen:
                    seen.add(s)
                    self.mcp_servers.append(s)

        # GUI-only tool-file filter. None means all files are shown in the
        # agent editor's Available Tools catalog. This field is deliberately
        # ignored by Agent.call(), so it has no effect on model behaviour.
        self.tool_files = None if tool_files is None else list(tool_files)

        # ------------------------------------------------------------------
        # Selected prompt files (names WITHOUT the ".txt" extension).
        #
        # At inference time (Agent.call) the texts of these files are read
        # from the prompts/ folder via the global PromptLoader
        # (context.prompt_loader) and appended to the agent's system
        # prompt inside a <rules>...</rules> section — see
        # build_system_prompt() above for the exact layout.
        #
        # Just like `tools`, this list stores only NAMES: the content is
        # resolved fresh on every call, so editing a prompts/*.txt file
        # takes effect for every agent that selected it on its next
        # inference, without re-editing each agent.
        #
        # The list may contain names whose files no longer exist — they
        # are silently skipped at call time (and re-appear automatically
        # if the file is restored).
        # ------------------------------------------------------------------
        self.prompts = [str(p) for p in prompts] if prompts else []

        # ------------------------------------------------------------------
        # Scheduling / visibility flags (added for the agentic-core work).
        # ------------------------------------------------------------------
        # `active` — when False, the BackendThread's round-robin loop SKIPS
        #            this agent entirely (it is not inferenced). The agent
        #            still exists, keeps its config and history, and can be
        #            re-activated at any time (via the `core_agent_active`
        #            tool or via the Edit Agent dialog). This is the primary
        #            mechanism for an agent to "take itself offline" at
        #            runtime — for example, an agent that has finished its
        #            task can call `core_agent_active(my_name, False)` and
        #            step out of the rotation, leaving the floor to others.
        #
        # `silent` — when True, the agent's final answer is still produced
        #            and stored in shared history (other agents CAN see it,
        #            and the Thinking tab still shows the live stream), but
        #            it is NOT printed into the chat UI. Useful for "worker"
        #            agents whose output is meant for other agents to consume
        #            rather than for the human user. Unlike `active`, this
        #            flag is NOT exposed as a runtime tool — it can only be
        #            flipped from the Edit Agent dialog (by design, so a
        #            misbehaving agent can't silently mute itself forever).
        # ------------------------------------------------------------------
        self.active = bool(active)
        self.silent = bool(silent)

        # Per-call transient state — _last_call_time is the only thing
        # that persists across calls, and it's just for rate-limiting.
        self._last_call_time = 0.0

        # ------------------------------------------------------------------
        # In-use flag for `core_agent_execute`.
        #
        # When an agent is being run inside another agent's tool-call loop
        # (via `core_agent_execute`), this flag is set to True so that the
        # agent's own BackendThread — which may wake up on its next
        # round-robin tick — sees the agent is busy and skips the turn
        # instead of trying to inference it concurrently (which would
        # interleave two streams on the same Ollama model and corrupt
        # both turns).
        #
        # The flag is protected by `_busy_lock` so the check-and-set is
        # atomic against the BackendThread's check.
        # ------------------------------------------------------------------
        self._busy = False
        self._busy_lock = __import__("threading").Lock()

    def get_tool_definitions(self, loader: ToolLoader) -> List[dict]:
        tools = []
        for name in self.tools:
            func = loader.get_tool(name)
            if func:
                try:
                    tools.append(function_to_tool(func))
                except Exception:
                    pass
        return tools

    def is_tool_allowed(self, tool_name: str) -> bool:
        # Enforces the user-configured per-agent tool whitelist.
        # This is a USER SETTING (each agent has its own tools), not a
        # hidden restriction — keep it.
        #
        # NEW: when the agent's `use_tools` flag is False (the "Tools"
        # checkbox in the GUI), NO tool is allowed — the model was not
        # shown any tools, so any tool call it produces is rejected by
        # this guard.
        #
        # NEW (native MCP): when the agent's `mcp` flag is True, the
        # MCP whitelist extends past `self.tools`:
        #   * the MCP MANAGEMENT core tools (mcp_list_servers,
        #     mcp_subscribe, ...), and
        #   * every namespaced native tool (mcp__<server>__<tool>) of
        #     every server the agent is subscribed to.
        # Both are checked through the MCP engine (mcp_manager), which
        # knows the live subscription list and each server's current
        # tool catalog.
        if not getattr(self, "use_tools", True):
            return False
        if tool_name in self.tools:
            return True
        if getattr(self, "mcp", False):
            mgr = _get_mcp_manager()
            if mgr is not None and mgr.is_tool_allowed(self.name, tool_name):
                return True
        return False

    def call(self, full_history: List[dict], loader: ToolLoader, client,
             thinking_callback=None, ephemeral_question: Optional[str] = None,
             dialogue_mode: bool = False):
        """
        Run one full agent turn (streaming + tool-call loop).

        ``client`` is the OllamaClient bound to this agent's endpoint – the
        caller (BackendThread) is responsible for passing the correct one.

        ``ephemeral_question`` (optional) is a one-shot FINAL user message
        used by the core_agent_execute_with_question tool. build_context
        appends it AFTER truncation, so it stays visible even when
        max_context_messages is 0; it is never stored in any history.

        ``dialogue_mode`` (optional) is used by the
        core_agent_execute_dialogue tool: the context is built from the
        agent-to-agent side chat's OWN history instead of the main shared
        history (see build_dialogue_context). Everything else — system
        prompt (with selected prompt files / <rules>), tools,
        capabilities — is unchanged.

        ALL per-call state is LOCAL to this method. No instance state
        leaks across calls (the old ``self._content_started`` hack that
        caused thinking-text accumulation across calls has been removed).

        CRITICAL (thinking-leak fix):
        The `thinking` channel from the streaming response is forwarded
        ONLY to the thinking_callback (UI Thinking tab + console). It is
        NEVER stored on the assistant message that gets appended to
        `messages` (the local list sent to Ollama on the next iteration)
        or to `new_messages` (the list returned to the BackendThread and
        merged into the shared conversation.history).

        Why: the ollama Python client's `Message` model has a `thinking`
        field, so any dict carrying that key is faithfully serialised
        into the HTTP request body. The Ollama server then includes the
        thinking text in the model's chat template, which means:

        * On the next tool-call iteration of the SAME turn, the model
          sees its own previous raw thinking.
        * Because conversation.history is shared across all agents on
          all machines, any agent whose context is rebuilt from that
          history would also ingest the thinking text if build_context
          ever forwarded it.

        Storing thinking on the assistant message is therefore both an
        in-turn leak and a latent cross-agent leak. The fix is to simply
        not attach the `thinking` key. The text is still shown live in
        the Thinking tab via thinking_callback, and the
        thinking-as-content fallback below preserves the model's actual
        answer when it never transitioned from <thinking> to content.
        """
        # ---- Build the initial message list ----
        # Pass `self.tools` to build_context so the agent's OWN past
        # tool_calls / tool results for tools no longer in the allowed
        # list are stripped from the rebuilt context (tool-leak fix).
        #
        # NEW (use_tools flag): when the agent has tools DISABLED, we pass
        # an EMPTY allowed list instead — this strips ALL of the agent's
        # past tool_calls / tool results from the rebuilt context so no
        # tool usage is ever shown to the model on inference.
        # build_context itself is UNCHANGED.
        #
        # NEW (dialogue mode): when running inside an agent-to-agent side
        # chat (core_agent_execute_dialogue), the context comes from the
        # DIALOGUE's own history via build_dialogue_context — the
        # partner's messages appear as plain user messages, and
        # max_context_messages == 0 means "unlimited" there (a side chat
        # never contains main-chat messages, so the 0-setting's
        # no-main-chat-leak rationale does not apply).
        #
        # NEW (ephemeral question): core_agent_execute_with_question
        # passes its question through to build_context, which appends it
        # as the FINAL user message after truncation — visible even when
        # max_context_messages is 0.
        #
        # NEW (native MCP): the allowed-tools list handed to the context
        # builders is EXTENDED with the MCP surface (management tools +
        # every namespaced native tool of every subscribed server) when
        # the agent's `mcp` flag is on. Without this, build_context's
        # tool-leak filter would strip the agent's own past MCP tool
        # calls / results from the rebuilt context — the model would
        # forget it ever called an MCP tool.
        _mcp_mgr = _get_mcp_manager()
        _mcp_enabled = bool(getattr(self, "mcp", False)) and _mcp_mgr is not None
        if getattr(self, "use_tools", True):
            _allowed_tools_list = list(self.tools)
        else:
            _allowed_tools_list = []
        if _mcp_enabled and getattr(self, "use_tools", True):
            try:
                _allowed_tools_list = _mcp_mgr.extend_allowed_tools(
                    self.name, _allowed_tools_list
                )
            except Exception as e:
                print(f"[{self.name}] MCP allowed-tools extension failed: {e}")

        if dialogue_mode:
            context_messages = build_dialogue_context(
                self.name, full_history,
                max_messages=self.max_context_messages,
                allowed_tools=_allowed_tools_list
            )
        else:
            context_messages = build_context(
                self.name, full_history,
                max_messages=self.max_context_messages,
                allowed_tools=_allowed_tools_list,
                ephemeral_question=ephemeral_question
            )

        # ---- Build the initial message list ----
        # The system message is the agent's own system prompt PLUS the
        # texts of its selected prompt files (prompts/*.txt), wrapped in
        # a <rules>...</rules> section:
        #
        #     <original system prompt>
        #
        #     <rules>
        #     text of prompts/a.txt
        #
        #     text of prompts/b.txt
        #     </rules>
        #
        # When no prompt files are selected (the default), the system
        # message is exactly the original system prompt — no <rules>
        # section is added at all.
        #
        # The prompt texts are resolved from the global PromptLoader
        # (context.prompt_loader) on EVERY call, so an edit to a .txt
        # file in the prompts folder takes effect on the next inference
        # of every agent that selected it. If the loader is unavailable
        # (e.g. an embedding without a prompts folder) the prompt files
        # are simply skipped — inference must never break because of it.
        prompt_loader = getattr(context, "prompt_loader", None)
        selected_prompts = list(getattr(self, "prompts", []) or [])
        if selected_prompts and prompt_loader is not None:
            loaded = [n for n in selected_prompts
                      if (prompt_loader.get_prompt_text(n) or "").strip()]
            if loaded:
                print(f"[{self.name}] Appending {len(loaded)} prompt file(s) "
                      f"to system prompt as <rules>: {', '.join(loaded)}")
        system_content = build_system_prompt(
            self.system_prompt, selected_prompts, prompt_loader
        )
        messages = [{"role": "system", "content": system_content}] + context_messages

        # ---- Tool capability gating (NEW) ----
        # When `use_tools` is False, no tool definitions are built at all:
        #   - the `tools` parameter is NOT sent to Ollama (empty list ->
        #     `tools=None` in chat_stream), so the model never sees any
        #     tools at inference time;
        #   - any tool call the model hallucinates anyway is rejected by
        #     is_tool_allowed() (see above).
        if getattr(self, "use_tools", True):
            tools = self.get_tool_definitions(loader)
        else:
            tools = []

        # ------------------------------------------------------------------
        # ---- NATIVE MCP TOOL MERGE (the whole point of this feature) ----
        # ------------------------------------------------------------------
        # When the agent's `mcp` flag is on (and the Tools master switch
        # is on too), the tool surface grows by TWO groups — both are
        # REAL first-class tool definitions in the very same `tools`
        # array, so the model calls them exactly like built-in tools:
        #
        #   1. NATIVE MCP TOOLS: every tool of every subscribed MCP
        #      server, namespaced as "mcp__<server>__<tool>" with the
        #      server's own JSON Schema. The model no longer has to wrap
        #      anything or guess meta-tool syntax — one call, one result.
        #
        #   2. MCP MANAGEMENT core tools (mcp_list_servers, mcp_subscribe,
        #      mcp_unsubscribe, mcp_list_tools, mcp_call_tool, mcp_status)
        #      — auto-available WITHOUT selecting them in Available
        #      Tools, so you can simply prompt the agent: "find an MCP
        #      server from this info and subscribe".
        #
        # Subscribed servers that are not yet connected (fresh session
        # load, app restart) are revived here so the model sees its full
        # tool surface on turn one.
        # ------------------------------------------------------------------
        if _mcp_enabled and getattr(self, "use_tools", True):
            try:
                _mcp_mgr.ensure_subscribed_connected(self.name)
            except Exception as e:
                print(f"[{self.name}] MCP connect-on-turn failed: {e}")
            try:
                mcp_native_defs = _mcp_mgr.native_tool_defs_for_agent(self.name)
            except Exception as e:
                print(f"[{self.name}] MCP tool definitions failed: {e}")
                mcp_native_defs = []
            if mcp_native_defs:
                existing_names = {t["function"]["name"] for t in tools
                                  if isinstance(t, dict) and "function" in t}
                for d in mcp_native_defs:
                    if d["function"]["name"] not in existing_names:
                        tools.append(d)
                        existing_names.add(d["function"]["name"])
            # Management core tools: auto-injected from the loader catalog.
            mcp_mgmt_names = getattr(
                _mcp_manager_module, "MANAGEMENT_TOOL_NAMES", ()
            ) if _mcp_manager_module is not None else ()
            existing_names = {t["function"]["name"] for t in tools
                              if isinstance(t, dict) and "function" in t}
            for mgmt_name in mcp_mgmt_names:
                if mgmt_name in existing_names:
                    continue
                mgmt_func = loader.get_tool(mgmt_name)
                if mgmt_func:
                    try:
                        tools.append(function_to_tool(mgmt_func))
                        existing_names.add(mgmt_name)
                    except Exception:
                        pass

        new_messages = []
        tool_iterations = 0
        # NEW: cumulative count of TOOL CALLS executed this turn (1-based).
        # Reported to the model in the [SYSTEM INFO] prepend of every tool
        # result ("call #N, M left") — see tool_injection.
        # NOTE: this counts individual tool calls, which can be MORE than
        # tool_iterations when the model emits parallel tool calls in one
        # batch. The turn budget itself is still bounded by
        # max_tool_iterations (inference iterations).
        tool_calls_used = 0
        # NEW (issue #8): has the tools-limit-exceeded notice already
        # been delivered to the model this turn? Set by BOTH delivery
        # paths (the per-tool-call refusal gate below and the
        # iteration-budget path at the top of the loop) so the notice
        # is never delivered twice in the same turn.
        tools_limit_notice_delivered = False
        consecutive_empty_responses = 0  # LOCAL — resets every call, no leak
        consecutive_stream_errors = 0    # LOCAL — resets every call, no leak

        # `effective_tools` starts as the full tool list. If a streaming
        # error looks like an XML/parse error (model generated malformed
        # tool-call tags), we set this to None on the 2nd retry so the
        # model is forced to produce a plain text answer instead of
        # repeating the same malformed XML tool call.
        effective_tools = tools

        print(f"[{self.name}] Starting inference on {self.endpoint}...")
        print(f"[{self.name}] Allowed tools: {self.tools}")
        _mcp_cap_extra = ""
        if getattr(self, "mcp", False):
            _mcp_cap_extra = (f", mcp=True (servers: "
                              f"{', '.join(getattr(self, 'mcp_servers', []) or [])})")
        print(f"[{self.name}] Capabilities: use_tools={getattr(self, 'use_tools', True)}, "
              f"image={getattr(self, 'image', False)}, audio={getattr(self, 'audio', False)}"
              f"{_mcp_cap_extra}")

        start_time = time.time()

        # max_tool_iterations == 0 means unlimited
        while True:
            tool_iterations += 1
            if self.max_tool_iterations > 0 and tool_iterations > self.max_tool_iterations:
                if tool_iterations > self.max_tool_iterations + 1:
                    # The ONE allowed post-limit "final answer" iteration
                    # is already used up. Only reachable if the model kept
                    # returning truly-empty responses after the notice
                    # (those retries are capped at 3 anyway). End the turn
                    # here — the post-loop code emits the "done" event.
                    print(f"[{self.name}] Reached max_tool_iterations="
                          f"{self.max_tool_iterations} (final-answer "
                          f"iteration included); ending turn.")
                    break

                # ---- FIX (issue #8): tools limit exhausted — NO MORE
                # SILENT STOPS ----
                # OLD behavior: `break` right here. The model never saw
                # the results of its last (legitimate) tool calls and
                # never produced a final answer — the agent just went
                # silent, which is exactly the reported bug.
                # NEW behavior: deliver the tools-limit-exceeded notice,
                # strip tools for exactly ONE extra iteration, and let
                # the model generate its final answer. The turn then
                # ends through the normal no-tool-calls exit.
                effective_tools = None  # text-only — final answer forced
                if not tools_limit_notice_delivered:
                    tools_limit_notice_delivered = True
                    notice = tool_injection.tools_limit_exceeded_notice()
                    # user-role notice, LOCAL messages only (NOT
                    # new_messages): the limit renews next turn, so future
                    # contexts must not learn "tools limit exceeded" as a
                    # permanent fact. `messages` currently ends with tool
                    # results (role "tool") or assistant text — appending
                    # a user message is template-safe for every Ollama
                    # chat template (and the last-role guard below only
                    # has to not see an assistant/system tail).
                    messages.append({
                        "role": "user",
                        "content": notice
                    })
                    print(f"[{self.name}] Tools limit exceeded "
                          f"({tool_calls_used}/{self.max_tool_iterations} "
                          f"tool calls used, iteration budget "
                          f"{self.max_tool_iterations} done) — delivering "
                          f"limit notice, forcing one final text-only "
                          f"answer.")
                    if thinking_callback:
                        thinking_callback(
                            "status", self.name,
                            "Tools limit exceeded — notice delivered, "
                            "forcing final answer (no tools on this "
                            "iteration).",
                            ""
                        )
                # Fall through: run the ONE final text-only iteration.

            # ---- Remote-execution pause gate (NEW) ----
            # If a remote `core_agent_execute` is currently running an
            # agent on THIS agent's endpoint, pause here (cooperatively)
            # until it finishes. This is the "nearest tool call or end of
            # inference" pause point: the current streaming iteration is
            # never interrupted — we wait at the TOP of the next
            # iteration, i.e. right before we would issue a new inference
            # request (typically after the model asked for a tool call,
            # or at the start of a turn).
            #
            # Agents being executed via core_agent_execute themselves
            # (agent_call_stack non-empty) SKIP this check — they are the
            # reason the endpoint is paused and would otherwise deadlock.
            if not agent_call_stack.get():
                try:
                    _gate = context.get_endpoint_gate(self.endpoint)
                    if _gate.is_paused():
                        print(f"[{self.name}] Pausing turn: remote agent execution "
                              f"in progress on endpoint {self.endpoint}...")
                        if thinking_callback:
                            thinking_callback(
                                "status", self.name,
                                f"Pausing turn — endpoint {self.endpoint} is busy "
                                f"with a remote agent execution. Will resume "
                                f"automatically.",
                                ""
                            )
                        # Tell remote executors we reached a safe pause point,
                        # then wait until the endpoint re-opens.
                        _gate.mark_quiesced()
                        _gate.wait_open()
                        print(f"[{self.name}] Resuming turn on {self.endpoint}.")
                        if thinking_callback:
                            thinking_callback(
                                "status", self.name,
                                f"Resuming turn — endpoint {self.endpoint} is free "
                                f"again.",
                                ""
                            )
                except Exception:
                    # Never let a gate problem break the inference loop.
                    pass

            # ---- Light rate-limit (avoids hammering Ollama) ----
            time_since_last = time.time() - self._last_call_time
            if time_since_last < 0.5:
                time.sleep(0.5 - time_since_last)
            self._last_call_time = time.time()

            # ---- Media (image / audio) attachment, capability-gated (NEW) ----
            # Images are attached ONLY when the agent is flagged as
            # vision-capable (the "Image" checkbox). Audio (.wav) likewise
            # via the "Audio" checkbox, and it can be combined with images
            # (both travel through the same `images` message field).
            # When both flags are off, NOTHING is attached — this is the
            # fix for non-vision models failing on image payloads that
            # used to be injected unconditionally.
            if getattr(self, "image", False) or getattr(self, "audio", False):
                messages = inject_media_last_user_or_tool(
                    messages,
                    enable_images=bool(getattr(self, "image", False)),
                    enable_audio=bool(getattr(self, "audio", False)),
                )

            # ---- FIX: Qwen3.x template guard (mid-loop) ----
            # Ollama truncates from the BEGINNING of the messages list
            # when total tokens exceed num_ctx. After several large tool
            # results, the original user message(s) from build_context
            # can be silently dropped. Qwen3.8's Jinja template then
            # throws "No user query found in messages." Re-inject a
            # minimal user message if none survived.
            # This runs before EVERY streaming call, not just the first.
            if not any(
                m.get("role") == "user" and m.get("content", "").strip()
                for m in messages
            ):
                messages.insert(1, {
                    "role": "user",
                    "content": "(continue)"
                })

            # ---- FIX (issue #2): "last message was from assistant" ----
            # When agents run WITHOUT user participation (round-robin
            # scheduler ticks with "Pause User Input" off, autonomous
            # multi-turn chains, dialogues), the rebuilt context can
            # legitimately END with this agent's own last assistant
            # message: it was the last one to speak and nobody has said
            # anything since. Many Ollama chat templates refuse to START
            # generation after an assistant turn — the server then
            # returns the "... last message was from assistant ..." error
            # the user kept seeing.
            #
            # The guard runs before EVERY streaming call (the only place
            # messages are sent to Ollama), so it also covers dialogue
            # turns and any future path that rebuilds the context. A
            # trailing `tool` message is VALID (mid tool-call loop), so
            # only assistant / system / unknown roles are fixed.
            last_role = messages[-1].get("role") if messages else None
            if last_role not in ("user", "tool"):
                messages.append({
                    "role": "user",
                    "content": "(continue)"
                })
                print(f"[{self.name}] Context ended with role "
                      f"'{last_role}' (no user/tool turn to answer) — "
                      f"appended a '(continue)' user message so Ollama "
                      f"can generate.")

            # ---- FULL PROMPT DUMP (issue #6) ----
            # Emit the EXACT, COMPLETE message list (system + rebuilt
            # history + injected guards) to the Thinking tab of this
            # agent's machine — not to the terminal (print) — so the
            # operator can see precisely what the model receives on
            # every inference call, including history and tools.
            if thinking_callback:
                try:
                    prompt_dump = tool_injection.format_messages_for_debug(
                        messages,
                        agent_name=self.name,
                        model=self.model,
                        endpoint=self.endpoint,
                        tools=(effective_tools if effective_tools else None),
                        num_ctx=self.num_ctx,
                        temperature=self.temperature,
                        iteration=tool_iterations,
                    )
                    thinking_callback("prompt", self.name, prompt_dump, "")
                except Exception as dump_err:
                    # Debugging aid must NEVER break inference.
                    print(f"[{self.name}] Prompt dump failed: {dump_err}")

            # ---------- STREAMING RESPONSE ----------
            print(f"\n[{self.name}] Streaming response (iter {tool_iterations}):")

            accumulated_content = ""
            accumulated_thinking = ""
            accumulated_tool_calls = {}  # Robust accumulation by index

            # LOCAL flag — resets every iteration, no leak across calls
            # (the old `self._content_started` was the accumulated-state bug)
            stream = client.chat_stream(
                model=self.model,
                messages=messages,
                tools=effective_tools if effective_tools else None,
                num_ctx=self.num_ctx,
                temperature=self.temperature
            )

            try:
                for chunk in stream:
                    if chunk.get('thinking'):
                        thinking_text = chunk['thinking']
                        if thinking_callback:
                            thinking_callback("thinking", self.name, thinking_text, "")
                        print(thinking_text, end='', flush=True)
                        accumulated_thinking += thinking_text

                    if chunk.get('content'):
                        content_text = chunk['content']
                        if thinking_callback:
                            thinking_callback("stream", self.name, content_text, "")
                        print(content_text, end='', flush=True)
                        accumulated_content += content_text

                    # --- ROBUST TOOL CALL ACCUMULATION ---
                    # NOTE: client now yields RAW tool_calls per chunk
                    # (no pre-accumulation), so each tc here is a fresh
                    # fragment — appending args is correct.
                    raw_tool_calls = chunk.get('tool_calls')
                    if raw_tool_calls:
                        for tc in raw_tool_calls:
                            idx = tc.get('index', 0) if isinstance(tc, dict) else getattr(tc, 'index', 0)
                            if idx not in accumulated_tool_calls:
                                accumulated_tool_calls[idx] = {"name": "", "arguments": ""}
                            if isinstance(tc, dict):
                                func_info = tc.get('function', {})
                                name = func_info.get('name', '')
                                args = func_info.get('arguments', '')
                            else:
                                name = getattr(tc.function, 'name', '') or ''
                                args = getattr(tc.function, 'arguments', '') or ''
                            if name:
                                accumulated_tool_calls[idx]["name"] = name
                            if isinstance(args, str):
                                accumulated_tool_calls[idx]["arguments"] += args
                            elif args is not None:
                                accumulated_tool_calls[idx]["arguments"] += json.dumps(args)

                    if chunk.get('done'):
                        print(f"done in chunk.get ({self.name})")
                        break

            except Exception as e:
                error_str = str(e)
                print(f"\n[{self.name}] Streaming error: {e}")
                if thinking_callback:
                    thinking_callback("error", self.name, error_str, "")
                consecutive_stream_errors += 1

                # ---- SALVAGE: if we already accumulated some content or
                # thinking before the error, keep it rather than throwing
                # the whole turn away. This is common when the model
                # generates good text output and then trips on a malformed
                # tool-call XML tag at the very end.
                salvage = accumulated_content.strip() or accumulated_thinking.strip()
                if salvage:
                    print(f"[{self.name}] Salvaging partial response after "
                          f"streaming error ({len(salvage)} chars)")
                    # NOTE: `thinking` is intentionally NOT attached to the
                    # salvaged assistant message. The thinking text is only
                    # used as a content fallback (when no content channel
                    # was produced) — see the thinking-as-content fallback
                    # below. Attaching it as a separate `thinking` field
                    # would re-introduce the in-turn / cross-agent leak.
                    assistant_msg = {
                        "role": "assistant",
                        "content": salvage,
                        "sender": self.name
                    }
                    new_messages.append(assistant_msg)
                    break  # exit the tool-iteration loop with partial content

                # ---- RETRY: no content to salvage. Try again, up to 2 times.
                if consecutive_stream_errors <= 2:
                    # Detect XML / parse errors — these are the ones where
                    # the model generated malformed tool-call tags like
                    # "<function> closed by </parameter>". On the 2nd
                    # retry, disable tools entirely so the model can't
                    # repeat the same XML mistake.
                    is_parse_error = any(kw in error_str.lower() for kw in [
                        'xml', 'syntax', 'parse', 'unexpected', 'malformed',
                        'token', 'status code: -1', 'element', 'closed by'
                    ])
                    if is_parse_error and consecutive_stream_errors >= 2:
                        effective_tools = None  # force text-only response
                        print(f"[{self.name}] Parse error #{consecutive_stream_errors}; "
                              f"retrying WITHOUT tools to avoid XML tool-call issue.")
                    else:
                        print(f"[{self.name}] Retrying after streaming error "
                              f"(attempt {consecutive_stream_errors}/2)")

                    messages.append({
                        "role": "user",
                        "content": (
                            f"[System: Your previous response caused a "
                            f"formatting error: {error_str[:150]}. "
                            f"Please respond again, simply and carefully.]"
                        )
                    })
                    continue  # re-enter the while loop for another attempt

                # ---- GIVE UP: max retries exhausted.
                error_msg = f"[System: Inference failed for {self.name}: {e}]"
                new_messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sender": self.name
                })
                return new_messages

            # --- Convert accumulator to final list format ---
            # The stream completed without throwing — reset the error counter
            # so a future transient error gets a fresh retry budget.
            consecutive_stream_errors = 0

            # Also restore tools if they were disabled for a retry — now
            # that the model is responding properly, let it use tools again.
            # EXCEPT (issue #8): once the per-turn tool budget is burned,
            # tools stay OFF for the rest of the turn — otherwise this
            # restore would quietly re-enable tools right after the
            # limit notice / refusal gate forced text-only mode, and the
            # model could keep tool-calling past its limit.
            if (effective_tools is None and tools
                    and not (self.max_tool_iterations > 0
                             and tool_calls_used >= self.max_tool_iterations)):
                effective_tools = tools

            if accumulated_tool_calls:
                final_tool_calls = []
                for idx in sorted(accumulated_tool_calls.keys()):
                    tc_data = accumulated_tool_calls[idx]
                    if tc_data["name"]:
                        args_str = tc_data["arguments"]
                        if not args_str:
                            args_str = "{}"
                        else:
                            try:
                                json.loads(args_str)
                            except json.JSONDecodeError:
                                args_str = "{}"
                        final_tool_calls.append({
                            "function": {
                                "name": tc_data["name"],
                                "arguments": args_str
                            }
                        })
            else:
                final_tool_calls = None

            # Emit stream_end so the thinking tab gets a newline between
            # the streamed text and the "Finished turn" line.
            if thinking_callback and (accumulated_content or accumulated_thinking):
                thinking_callback("stream_end", self.name, "", "")

            final_content = accumulated_content.strip()
            accumulated_thinking = accumulated_thinking.strip()



            # ---------- TRULY-EMPTY-RESPONSE HANDLING ----------
            # If we STILL have no content, no thinking, and no tool calls,
            # the model produced absolutely nothing. Don't store an empty
            # message (it pollutes history and teaches the model to stay
            # silent). Instead, retry with a reminder, up to 3 times.
            if not final_content and not final_tool_calls:
                consecutive_empty_responses += 1
                if consecutive_empty_responses <= 3:
                    print(f"[{self.name}] Truly empty response #"
                          f"{consecutive_empty_responses}; injecting reminder "
                          f"and retrying.")
                    reminder = {
                        "role": "user",
                        "content": (
                            f"[System: Your last response contained no text. "
                            f"You MUST respond with actual content right now, "
                            f"{self.name}. Please answer.]"
                        )
                    }
                    messages.append(reminder)
                    continue  # re-enter the while loop for another attempt
                else:
                    print(f"[{self.name}] {consecutive_empty_responses} "
                          f"consecutive empty responses; ending turn.")
                    if thinking_callback:
                        thinking_callback("done", self.name, "", "")
                    break  # give up, exit the tool-iteration loop
            else:
                # We got something — reset the empty counter.
                consecutive_empty_responses = 0

            # Build assistant message.
            #
            # CRITICAL (thinking-leak fix): the `thinking` field is NOT
            # attached to this dict. It used to be, and because the ollama
            # Python client's `Message` model has a `thinking` field, the
            # raw reasoning text was being:
            #   1. sent back to Ollama on the next tool-call iteration of
            #      THIS turn (via `messages.append(assistant_msg)` below),
            #      so the model kept reading its own previous thinking; and
            #   2. persisted into conversation.history via `new_messages`,
            #      from where it could leak into any agent's context if
            #      build_context ever forwarded it.
            # The thinking text is still shown live in the Thinking tab
            # (via thinking_callback above) and printed to the console, so
            # no observability is lost by not storing it on the message.
            assistant_msg = {
                "role": "assistant",
                "content": final_content,
                "sender": self.name
            }
            # `thinking` is intentionally omitted — see comment above.
            # Defense-in-depth: explicitly pop it in case a future code
            # change accidentally re-introduces it.
            assistant_msg.pop("thinking", None)

            # ---------- STRICT TOOL WHITELIST ENFORCEMENT ----------
            # Check ALL tool calls against the agent's allowed tools
            # BEFORE persisting anything. If ANY tool call is for a
            # tool not in the agent's allowed list, REJECT the entire
            # batch and force text-only mode for the rest of the turn.
            #
            # This is the user's primary defense against tool leakage:
            # an agent whose tools list is empty (or doesn't include a
            # particular tool) must NEVER have that tool available —
            # not in the API `tools` parameter (already filtered by
            # get_tool_definitions), not in the context messages
            # (filtered by build_context's allowed_tools parameter),
            # not in the assistant message's `tool_calls` field
            # (filtered here), not in a `tool` role response (we
            # don't create one for rejected calls), and not in the
            # error/reminder text (we use a generic message that
            # mentions NO tool names — neither the rejected tool's
            # name nor the list of allowed tools).
            #
            # Unselected tools must be completely invisible to the
            # model — even during the prompt, and even if the model
            # tries to use them.
            if final_tool_calls:
                disallowed_count = sum(
                    1 for tc in final_tool_calls
                    if not self.is_tool_allowed(tc["function"]["name"])
                )
                if disallowed_count > 0:
                    # Reject the entire batch. Don't attach tool_calls
                    # to the assistant message — the model's tool call
                    # attempt is silently dropped from history so it
                    # can't leak into future contexts.
                    if not assistant_msg["content"].strip():
                        assistant_msg["content"] = (
                            "[Tool call rejected — responding in text mode]"
                        )
                    new_messages.append(assistant_msg)
                    messages.append(assistant_msg)

                    # Add a generic reminder to LOCAL messages only
                    # (not to new_messages, which becomes shared
                    # history). The reminder deliberately does NOT
                    # mention any tool names. This prevents the model
                    # from learning about tools it shouldn't know
                    # about, and avoids creating a `tool` role message
                    # (which would carry the rejected tool's name in
                    # its `name` field and reinforce the tool's
                    # existence in the model's context).
                    messages.append({
                        "role": "user",
                        "content": (
                            "[System: One or more of the tools you attempted "
                            "to use is not available. Please respond in plain "
                            "text only — do not attempt to call any tools.]"
                        )
                    })

                    # Force text-only mode for the rest of this turn
                    # so the model can't keep trying disallowed tools.
                    effective_tools = None
                    print(f"[{self.name}] Rejected {disallowed_count} "
                          f"disallowed tool call(s); forcing text-only "
                          f"mode for the rest of this turn.")
                    if thinking_callback:
                        thinking_callback(
                            "error", self.name,
                            "Disallowed tool call rejected. "
                            "Forcing text-only mode.",
                            ""
                        )
                    continue  # re-enter the while loop for another attempt

                # All tool calls are allowed — parse and attach them
                # to the assistant message.
                parsed_tool_calls = []
                for tc in final_tool_calls:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    parsed_tool_calls.append({
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": args
                        }
                    })
                assistant_msg["tool_calls"] = parsed_tool_calls

            new_messages.append(assistant_msg)
            messages.append(assistant_msg)

            if not final_tool_calls:
                break

            # ---------- TOOL EXECUTION ----------
            # All tool_calls have already passed the upfront whitelist
            # check above. No per-tool check is needed here — the
            # whitelist is enforced once, atomically, before any tool
            # is executed or any tool message is persisted. This is
            # the only place where unselected tools could previously
            # leak: the old per-tool check created a `tool` role
            # message carrying the rejected tool's name in its `name`
            # field, which both taught the model about the tool and
            # persisted the reference into shared history.
            for tool_call in final_tool_calls:
                tool_name = tool_call["function"]["name"]
                raw_args = tool_call["function"]["arguments"]

                # ----------------------------------------------------------
                # NEW (issue #8): TOOLS-LIMIT EXHAUSTION GATE.
                #
                # The turn's tool budget is already exhausted, but the
                # model is asking for ANOTHER tool call. OLD behavior:
                # the tool executed (parallel batches) or the loop broke
                # — and either way the agent silently stopped without
                # any final answer. NEW behavior: this tool call is NOT
                # executed at all. The agent receives ONLY the
                # tools-limit-exceeded notice as this call's tool
                # result — phrased so the model understands it must
                # generate its final answer now.
                #
                # The notice is delivered as a `tool` role message so the
                # assistant tool_calls message stays properly answered
                # (required by Ollama chat templates), and it is
                # persisted to history (new_messages) so future
                # contexts stay well-formed too. No prepend section, no
                # counter, no operator injections — per the spec, the
                # agent gets ONLY the notice text (anything else would
                # dilute the "generate final answer" instruction).
                # ----------------------------------------------------------
                if (self.max_tool_iterations > 0
                        and tool_calls_used >= self.max_tool_iterations):
                    notice = tool_injection.tools_limit_exceeded_notice()
                    print(f"\n[{self.name}] Tools limit exceeded "
                          f"({tool_calls_used}/{self.max_tool_iterations} "
                          f"tool calls used) — tool '{tool_name}' will NOT "
                          f"be executed. Returning the limit notice to "
                          f"force a final answer.")
                    if thinking_callback:
                        thinking_callback(
                            "status", self.name,
                            f"Tools limit exceeded — tool '{tool_name}' "
                            f"NOT executed, notice returned instead.",
                            ""
                        )
                        # Show the notice in the Thinking tab exactly as
                        # the agent receives it (same rendering as a
                        # normal tool result).
                        thinking_callback(
                            "tool_result", self.name, tool_name, notice)

                    # History copy — the refused call IS answered in
                    # history, keeping the assistant tool_calls message
                    # well-formed for future context rebuilds.
                    notice_tool_msg = {
                        "role": "tool",
                        "content": notice,
                        "name": tool_name,
                        "tool_name": tool_name,
                        "sender": self.name,
                    }
                    notice_tool_msg.pop("thinking", None)
                    new_messages.append(notice_tool_msg)

                    # Local copy — identical content (the notice is the
                    # entire tool result; there is deliberately no
                    # composed variant).
                    notice_tool_msg_local = {
                        "role": "tool",
                        "content": notice,
                        "name": tool_name,
                        "tool_name": tool_name,
                        "sender": self.name,
                    }
                    notice_tool_msg_local.pop("thinking", None)
                    messages.append(notice_tool_msg_local)

                    # The rest of this turn is text-only: tools are
                    # removed from the API call, so further tool calls
                    # are impossible; any that somehow arrive hit this
                    # same gate again. The model's only remaining option
                    # is to generate its final answer.
                    tools_limit_notice_delivered = True
                    effective_tools = None
                    continue  # next tool call in this batch (if any)

                # NEW: cumulative tool-call counter for this turn. The
                # number is reported to the model in the [SYSTEM INFO]
                # prepend below ("call #N, M left").
                tool_calls_used += 1
                print(f"\n[{self.name}] Calling tool: {tool_name} "
                      f"(call #{tool_calls_used} this turn)")
                try:
                    args_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    args_str = json.dumps(args_dict, indent=2)
                except Exception:
                    args_str = str(raw_args)
                print(f"[{self.name}] Arguments: {args_str}")

                if thinking_callback:
                    thinking_callback("tool_call", self.name, tool_name, args_str)

                func = loader.get_tool(tool_name)
                if func:
                    try:
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                self._execute_tool_with_context,
                                self.name, func, raw_args
                            )
                            # 1-hour timeout — generous enough for any
                            # reasonable tool, including LLM-calling ones.
                            result = future.result(timeout=3600.0)

                        result_str = str(result)
                        # No aggressive truncation — keep full result so
                        # the model can reason about it. Only cap at 50K
                        # chars to prevent pathological cases from
                        # blowing out the context window.
                        if len(result_str) > 50000:
                            result_str = result_str[:50000] + "\n...[truncated at 50000 chars]"

                        print(f"[{self.name}] Tool result: {result_str[:500]}")

                    except FutureTimeoutError:
                        result = f"Error: Tool '{tool_name}' timed out after 3600s"
                        result_str = str(result)
                        print(f"[{self.name}] {result_str}")
                        if thinking_callback:
                            thinking_callback("error", self.name, result_str, "")

                    except Exception as e:
                        result = f"Error executing tool: {e}"
                        result_str = str(result)
                        print(f"[{self.name}] {result_str}")
                        if thinking_callback:
                            thinking_callback("error", self.name, result_str, "")

                elif _mcp_enabled and _mcp_mgr.handles_tool(self.name, tool_name):
                    # ------------------------------------------------------
                    # NATIVE MCP TOOL EXECUTION.
                    #
                    # The model called a namespaced MCP tool
                    # (mcp__<server>__<tool>) of one of this agent's
                    # subscribed servers. The MCP engine routes the call
                    # to the server process; everything AFTER this branch
                    # is IDENTICAL to a built-in tool: the result flows
                    # through the same 50K cap, the same [SYSTEM INFO]
                    # prepend (usage counter + operator injections), the
                    # same thinking-tab events and the same shared-
                    # history persistence. This is what makes MCP usage
                    # "native" — the model sees no difference between a
                    # built-in tool and an MCP tool.
                    # ------------------------------------------------------
                    try:
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                self._execute_mcp_with_context,
                                _mcp_mgr, self.name, tool_name, raw_args
                            )
                            result = future.result(timeout=3700.0)

                        result_str = str(result)
                        if len(result_str) > 50000:
                            result_str = result_str[:50000] + "\n...[truncated at 50000 chars]"

                        print(f"[{self.name}] MCP tool result: {result_str[:500]}")

                    except FutureTimeoutError:
                        result_str = (f"Error: MCP tool '{tool_name}' "
                                      f"timed out after 3700s")
                        print(f"[{self.name}] {result_str}")
                        if thinking_callback:
                            thinking_callback("error", self.name, result_str, "")

                    except Exception as e:
                        result_str = f"Error executing MCP tool '{tool_name}': {e}"
                        print(f"[{self.name}] {result_str}")
                        if thinking_callback:
                            thinking_callback("error", self.name, result_str, "")
                else:
                    result = f"Tool '{tool_name}' not found in loader"
                    result_str = str(result)
                    print(f"[{self.name}] {result_str}")
                    if thinking_callback:
                        thinking_callback("error", self.name, result_str, "")

                # ----------------------------------------------------------
                # NEW (feature #3): tool-result PREPEND section.
                #
                # build_tool_result_prepend() is the single "endpoint"
                # that decides what extra text the agent receives with
                # this tool result:
                #   * the tool-call usage counter (call #N, M left), and
                #   * any OPERATOR INJECTIONS queued for this agent (the
                #     "Inject" button in the GUI). The queue is DRAINED
                #     here (take_pending_injections = read + clear), so
                #     everything the operator passed while the agent was
                #     working is delivered TOGETHER with this tool
                #     result, and the buffer starts empty for the next
                #     gathering cycle.
                #
                # Delivery is one-shot by construction: the COMPOSED
                # content (prepend + result) goes into the LOCAL
                # `messages` list (what the model sees for the rest of
                # this turn), while the copy stored into shared history
                # via `new_messages` is CLEAN — future turns will NOT
                # see the injected text again.
                # ----------------------------------------------------------
                pending = tool_injection.take_pending_injections(self.name)
                prepend_text = tool_injection.build_tool_result_prepend(
                    tool_name=tool_name,
                    tool_call_number=tool_calls_used,
                    max_tool_calls=self.max_tool_iterations,
                    pending_injections=pending,
                    agent_name=self.name,
                )
                if pending:
                    print(f"[{self.name}] Delivering {len(pending)} queued "
                          f"operator injection(s) with this tool result "
                          f"(buffer cleared).")
                composed_result = prepend_text + result_str

                if thinking_callback:
                    # The Thinking tab sees the COMPOSED text — exactly
                    # what the agent receives for this tool result.
                    thinking_callback("tool_result", self.name,
                                      tool_name, composed_result)

                # History copy — CLEAN result (no prepend), so injected
                # text and per-turn counters never persist into future
                # contexts. Uses the 50K-capped string, matching what
                # the model actually saw this turn.
                tool_msg = {
                    "role": "tool",
                    "content": result_str,
                    "name": tool_name,
                    "tool_name": tool_name,
                    "sender": self.name,
                }
                # Defense-in-depth: ensure no thinking field sneaks in
                # (should never happen, but guards against future
                # regressions that might accidentally attach thinking
                # to tool result messages).
                tool_msg.pop("thinking", None)
                new_messages.append(tool_msg)

                # Local copy — COMPOSED (prepend + result). This is the
                # version sent to Ollama for the remaining iterations of
                # THIS turn.
                tool_msg_local = {
                    "role": "tool",
                    "content": composed_result,
                    "name": tool_name,
                    "tool_name": tool_name,
                    "sender": self.name,
                }
                tool_msg_local.pop("thinking", None)
                messages.append(tool_msg_local)

        elapsed = time.time() - start_time
        print(f"[{self.name}] Inference complete in {elapsed:.1f}s, {tool_iterations} iterations")
        if thinking_callback:
            thinking_callback("done", self.name, "", "")

        return new_messages

    def _execute_tool_with_context(self, agent_name, func, arguments):
        # IMPORTANT: this ContextVar set is what lets user tools (team_email,
        # return_caller_name, ...) know which agent called them. Unchanged.
        current_agent_name.set(agent_name)
        return self._execute_tool_sync(func, arguments)

    def _execute_mcp_with_context(self, mcp_mgr, agent_name, tool_name, arguments):
        """Execute one NATIVE MCP tool call on the engine (worker thread).

        Mirrors _execute_tool_with_context: it runs inside the same kind
        of ThreadPoolExecutor worker, sets the current-agent ContextVar
        (so MCP servers could inspect it via user tools if ever needed)
        and normalises the arguments (JSON string -> dict) before handing
        the call to the MCP engine, which routes it to the subscribed
        server. Errors come back as readable strings — the caller already
        knows how to surface those to the model.
        """
        current_agent_name.set(agent_name)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON in MCP tool arguments: {e}"
        if not isinstance(arguments, dict):
            arguments = {}
        return mcp_mgr.call_tool(agent_name, tool_name, arguments)

    def _execute_tool_sync(self, func, arguments):
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON in tool arguments: {e}"
        try:
            return func(**arguments)
        except Exception as e:
            return f"Error executing tool: {e}"
