import threading
import ollama
from typing import List, Dict, Optional, Generator, Any
import time
import subprocess
import requests


class OllamaClient:
    """
    Synchronous, thread-safe Ollama client bound to a specific endpoint.

    - endpoint="local" uses the default local Ollama server (ollama.Client()).
    - endpoint="ip:port" (e.g. "192.168.3.3:11434") uses a remote Ollama server
      via ollama.Client(host="http://ip:port").

    - Streaming yields content, thinking, tool_calls, and done flag.
    - Stop flag for cooperative cancellation.
    - Retry logic with exponential backoff.

    IMPORTANT (thinking-mode fix):
    `think=True` is now passed explicitly to every chat call. The Ollama
    Python client's `chat()` method has a `think` parameter whose default
    is `None`, which means the Ollama HTTP API default applies — and per
    the Ollama API docs that default is `false`. When `think` is not
    explicitly enabled, even thinking-capable models (qwen3, qwen3.5,
    deepseek-r1, ...) are NOT guaranteed to wrap their internal reasoning
    in `<think>...</think>` tags.

    When the model omits the tags, Ollama cannot tell reasoning apart
    from the final answer, so the entire reasoning stream is emitted on
    the `content` channel. That text then ends up in the chat window as
    if it were the agent's actual response — exactly the symptom where
    the agent writes paragraphs of "Let me call the tool..." / "Wait,
    looking at the conversation structure..." analysis and never actually
    calls the tool or produces a short answer.

    Explicitly setting `think=True` forces the model to use the tags, so
    Ollama routes the reasoning to the `thinking` field (which we forward
    to the Thinking tab only — see agent.Agent.call) and the actual
    answer to the `content` field (which is what gets stored in history
    and shown in the chat). This keeps reasoning out of the chat history
    and out of other agents' context.

    The `think` parameter is exposed on every public chat method so an
    individual agent can opt out (`think=False`) if it is using a
    non-thinking model. The default is `True` because every model in the
    user's setup (qwen3.5:9b) is a thinking model.

    IMPORTANT (bug fix): chat_stream now yields RAW tool_calls from each
    chunk only — it no longer pre-accumulates them. The previous
    `accumulated_tool_calls.extend(tool_calls)` + `yield copy` pattern
    caused every yielded chunk to re-deliver ALL previously seen
    tool_call fragments. The Agent.call loop then appended arguments
    on every chunk, producing arguments strings like
    `{"key":"val"}{"key":"val"}{"key":"val"}` — JSON parse failed,
    tools were called with empty `{}` args, and long chats eventually
    ground to a halt as the model kept getting empty-arg errors.
    """

    def __init__(self, endpoint: str = "local",
                 request_timeout: int = 120,
                 max_retries: int = 3):
        self.endpoint = endpoint if endpoint else "local"
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self._client: Optional[ollama.Client] = None
        self._stop_flag = threading.Event()
        self._last_success = time.time()
        self._consecutive_failures = 0

    # ------------------------------------------------------------------
    # Endpoint / host helpers
    # ------------------------------------------------------------------
    def _get_host(self) -> Optional[str]:
        """Return the host URL for ollama.Client(host=...) or None for local default."""
        if self.endpoint == "local" or not self.endpoint:
            return None
        ep = self.endpoint.strip()
        if ep.startswith("http://") or ep.startswith("https://"):
            return ep
        return f"http://{ep}"

    def _get_client(self) -> ollama.Client:
        if self._client is None:
            host = self._get_host()
            if host:
                self._client = ollama.Client(host=host)
            else:
                self._client = ollama.Client()
        return self._client

    def _close_client(self):
        self._client = None

    def _safe_stop_model(self, model: str):
        """Best-effort model unload on the endpoint.

        For local: shells out to `ollama stop <model>`.
        For remote: HTTP DELETE on /api/generate with {"model": <model>}.
        """
        try:
            if self.endpoint == "local" or not self.endpoint:
                subprocess.Popen(
                    ['ollama', 'stop', model],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                host = self._get_host()
                requests.delete(
                    f"{host}/api/generate",
                    json={"model": model},
                    timeout=5
                )
        except Exception as e:
            print(f"Model stop warning for {model} on {self.endpoint}: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def stop(self):
        """Signal the current stream to stop at the next chunk."""
        self._stop_flag.set()

    def stop_model(self, model: str):
        """Public helper to unload a specific model from this endpoint."""
        self._safe_stop_model(model)

    def ensure_ready(self, model: str = None) -> bool:
        """Check if Ollama server at this endpoint is reachable."""
        try:
            client = self._get_client()
            client.list()
            return True
        except Exception as e:
            print(f"Ollama server at {self.endpoint} not reachable: {e}")
            self._close_client()
            return False

    # ------------------------------------------------------------------
    # Non-streaming chat
    # ------------------------------------------------------------------
    def chat(self, model: str, messages: List[Dict],
             tools: Optional[List[Dict]] = None,
             num_ctx: int = 32000,
             temperature: float = 0.5,
             think: bool = True) -> Optional[Dict]:
        """
        Non-streaming chat call.

        `think=True` (default) forces the model to use `<think>...</think>`
        tags so reasoning is routed to the `thinking` field of the
        response message, NOT the `content` field. Set `think=False`
        only when calling a non-thinking model.
        """
        return self._chat_with_retry(model, messages, tools, num_ctx, temperature, think)

    def _chat_with_retry(self, model: str, messages: List[Dict],
                         tools: Optional[List[Dict]],
                         num_ctx: int, temperature: float,
                         think: bool) -> Optional[Dict]:
        for attempt in range(self.max_retries):
            try:
                result = self._execute_chat(model, messages, tools, num_ctx, temperature, think)
                if result is not None:
                    self._last_success = time.time()
                    self._consecutive_failures = 0
                    return result
            except Exception as e:
                self._consecutive_failures += 1
                print(f"Attempt {attempt + 1} failed for {model} on {self.endpoint}: {e}")
                self._close_client()
                self._safe_stop_model(model)
                if attempt < self.max_retries - 1:
                    wait_time = min(2 ** attempt, 10)
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        print(f"All retries exhausted for {model} on {self.endpoint}")
        return None

    def _execute_chat(self, model: str, messages: List[Dict],
                      tools: Optional[List[Dict]],
                      num_ctx: int, temperature: float,
                      think: bool) -> Optional[Dict]:
        client = self._get_client()
        options = {"num_ctx": num_ctx, "temperature": temperature}
        params = {
            "model": model,
            "messages": messages,
            "options": options,
            # Force thinking mode on. Without this, the Ollama API
            # default (`think=false`) applies and thinking-capable models
            # are not guaranteed to wrap reasoning in <think> tags —
            # the reasoning then leaks into `content` and ends up in
            # chat. See the class docstring for the full rationale.
            "think": think,
        }
        if tools:
            params["tools"] = tools
        return client.chat(**params)

    # ------------------------------------------------------------------
    # Streaming chat
    # ------------------------------------------------------------------
    def chat_stream(self, model: str, messages: List[Dict],
                    tools: Optional[List[Dict]] = None,
                    num_ctx: int = 32000,
                    temperature: float = 0.5,
                    think: bool = True) -> Generator[Dict[str, Any], None, None]:
        """
        Synchronous streaming generator.
        Yields dicts with keys: content, thinking, tool_calls, done.

        tool_calls in each yielded dict are the RAW tool_calls from
        that single chunk only — NOT accumulated. The caller
        (Agent.call) is responsible for accumulating them by index
        across chunks. This is the fix for the args-doubling bug.

        `think=True` (default) forces the model to use `<think>...</think>`
        tags so reasoning is routed to the `thinking` field of each
        streamed chunk, NOT the `content` field. Without this, the
        Ollama API default (`think=false`) applies and thinking-capable
        models are not guaranteed to wrap reasoning in <think> tags —
        the reasoning then leaks into `content` and shows up in chat as
        if it were the agent's actual answer. See the class docstring
        for the full rationale.
        """
        self._stop_flag.clear()
        client = self._get_client()
        options = {"num_ctx": num_ctx, "temperature": temperature}
        params = {
            "model": model,
            "messages": messages,
            "options": options,
            "stream": True,
            # Force thinking mode on. Same rationale as in _execute_chat.
            "think": think,
        }
        if tools:
            params["tools"] = tools

        try:
            inner_stream = client.chat(**params)
            for chunk in inner_stream:
                if self._stop_flag.is_set():
                    break

                message = chunk.get('message', {})
                content = message.get('content', '')
                thinking = message.get('thinking', '')
                # RAW — just this chunk's tool_calls, no accumulation
                tool_calls = message.get('tool_calls', None)
                done = chunk.get('done', False)

                yield {
                    'content': content,
                    'thinking': thinking,
                    'tool_calls': tool_calls,
                    'done': done,
                }
                if done:
                    break
        except Exception as e:
            print(f"Streaming error for {model} on {self.endpoint}: {e}")
            raise

    def get_model_usage(self) -> Dict[str, dict]:
        """Return Ollama /api/ps model memory usage for this endpoint.

        The returned mapping is:
            {model_name: {"size": total_bytes, "size_vram": vram_bytes}}

        ``size_vram`` is the GPU-resident portion. The GUI derives the CPU
        portion as ``max(size - size_vram, 0)`` so partially-offloaded models
        are displayed correctly.
        """
        base_url = self._get_host() or "http://127.0.0.1:11434"
        session = requests.Session()
        # /api/ps is a direct LAN/local control-plane request. Do not let
        # HTTP(S)_PROXY / ALL_PROXY environment variables divert it.
        session.trust_env = False
        response = session.get(
            f"{base_url.rstrip('/')}/api/ps",
            timeout=min(float(self.request_timeout), 5.0),
        )
        response.raise_for_status()
        payload = response.json()
        usage = {}
        for model_info in payload.get("models", []) or []:
            if not isinstance(model_info, dict):
                continue
            name = str(model_info.get("name") or model_info.get("model") or "").strip()
            if not name:
                continue
            try:
                usage[name] = {
                    "size": int(model_info.get("size", 0) or 0),
                    "size_vram": int(model_info.get("size_vram", 0) or 0),
                }
            except (TypeError, ValueError):
                continue
        return usage

    def health_check(self, model: str = "qwen3.5:9b") -> bool:
        return False  # Placeholder

    def close(self):
        self._close_client()


class ClientManager:
    """
    Manages one OllamaClient per endpoint ("local" or "ip:port").

    Thread-safe: multiple BackendThreads (one per endpoint) call get_client(ep)
    to obtain a singleton client for their endpoint. Stop / shutdown operations
    are also routed through here.
    """

    def __init__(self, request_timeout: int = 180, max_retries: int = 2):
        self._clients: Dict[str, OllamaClient] = {}
        self._lock = threading.Lock()
        self._request_timeout = request_timeout
        self._max_retries = max_retries

    def get_client(self, endpoint: str) -> OllamaClient:
        """Get or create an OllamaClient for the given endpoint."""
        ep = endpoint if endpoint else "local"
        with self._lock:
            if ep not in self._clients:
                self._clients[ep] = OllamaClient(
                    endpoint=ep,
                    request_timeout=self._request_timeout,
                    max_retries=self._max_retries
                )
            return self._clients[ep]

    def stop_endpoint(self, endpoint: str):
        """Signal stop on the client for the given endpoint."""
        ep = endpoint if endpoint else "local"
        with self._lock:
            client = self._clients.get(ep)
            if client:
                client.stop()

    def stop_all(self):
        """Signal stop on all clients (used on app shutdown)."""
        with self._lock:
            for client in self._clients.values():
                client.stop()

    def shutdown(self):
        """Close all clients (used on app shutdown)."""
        with self._lock:
            for client in self._clients.values():
                try:
                    client.close()
                except Exception:
                    pass
            self._clients.clear()
