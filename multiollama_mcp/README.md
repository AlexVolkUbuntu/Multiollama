# MultiOllama MCP Server (`MultiOllama_mcp`)

A **standalone MCP (Model Context Protocol) server** that exposes **every
feature** of a running [MultiOllama headless backend](../MultiOllama_backend/)
as first-class MCP tools — so **any MCP-capable agentic app** can drive a
multi-agent LLM team:

> Claude Desktop · Cursor · Cline / Roo Code · Windsurf · VS Code (MCP) ·
> Zed · your own agents · **and the MultiOllama backend's own agents**
> (they can subscribe to this very server — §7).

```
   ┌──────────────────────┐  JSON-RPC over stdio (default)   ┌───────────────────────────┐
   │  MCP client app      │ ────────────────────────────►  │  ml_mcp_server.py          │
   │  (any agentic app)   │ ◄────────────────────────────   │  63 tools · thin HTTP      │
   └──────────────────────┘                                 │  client · zero own state  │
                                                            └─────────────┬─────────────┘
                                                              plain HTTP + │ X-API-Key
                                                                           ▼
                                                 ┌────────────────────────────────────────┐
                                                 │  MultiOllama backend                     │
                                                 │  python run_server.py --port 8200       │
                                                 │  FastAPI: /state /control /queue        │
                                                 │  /output /files /health /docs           │
                                                 └────────────────────────────────────────┘
```

**Why standalone:** the server is a pure *adapter*. It holds no sessions,
agents or files — the backend stays the single authority. It runs as its own
process with its own two dependencies (`mcp`, `httpx`), can be deployed on
any machine (the backend may live elsewhere), and one instance serves one
backend URL — run a second copy with `--backend-url` for a second backend.

---

## 1. Package contents

| File | Purpose |
| --- | --- |
| `ml_mcp_server.py` | The whole server — single file, stdlib + `mcp` + `httpx` |
| `requirements.txt` | The only two dependencies (for the machine that runs this process) |
| `README.md` | This file — install, configure, connect, workflows, troubleshooting |
| `test_ml_mcp.py` | Smoke-test harness: raw JSON-RPC over stdio against a live backend (§10) |
| `examples/claude_desktop_config.example.json` | Claude Desktop wiring |
| `examples/cursor_mcp.example.json` | Cursor wiring (stdio variant) |
| `examples/cline_mcp_settings.example.json` | Cline / Roo Code wiring |

Tested with: MCP SDK **1.27.0** (any ≥ 1.2), backend **1.0.0**, Python
**3.10+** (3.12 in CI), Linux and Windows.

---

## 2. Installation

```bash
# any machine that will run the MCP server process (usually your own PC)
cd MultiOllama_mcp
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python ml_mcp_server.py --version     # prints: ml_mcp_server 1.0.0 (mcp SDK …)
```

The backend itself has its own `requirements.txt` and needs **none** of
this — the two sides share nothing but HTTP.

---

## 3. Quick start (4 steps)

```bash
# 1) start the backend somewhere (it may be another machine)
cd MultiOllama_backend && python run_server.py --port 8200

# 2) verify THIS server can reach it
python ml_mcp_server.py --check --backend-url http://127.0.0.1:8200
#    -> health {...} / state {...} / "OK — the backend is reachable"

# 3) see the whole tool surface
python ml_mcp_server.py --list-tools        # 63 tools, one line each

# 4) wire it into your MCP client (copy-paste configs in §5)
```

---

## 4. Configuration

Precedence: **CLI flag > environment variable > default**.

| Setting | CLI flag | Environment variable | Default |
| --- | --- | --- | --- |
| Backend URL | `--backend-url URL` | `ML_BACKEND_URL` | `http://127.0.0.1:8200` |
| API key | `--api-key KEY` | `ML_MCP_API_KEY` (fallback `ML_API_KEY`) | none |
| Per-call timeout | `--timeout SEC` | `ML_MCP_TIMEOUT` | 30 |
| Transport | `--transport stdio\|sse\|streamable-http` | `ML_MCP_TRANSPORT` | `stdio` |
| HTTP bind host | `--host` | `ML_MCP_HOST` | `127.0.0.1` |
| HTTP bind port | `--port` | `ML_MCP_PORT` | 8090 |

Notes:

* **API key.** If the backend runs with `ML_API_KEY` set, every call needs
  it. The header (`X-API-Key`) is used for all tools; `?token=` is appended
  automatically when a tool builds a clickable URL (`ml_files_url`).
  Setting `ML_MCP_API_KEY` on this server (or just re-using `ML_API_KEY`)
  configures both directions in one shot.
* **Timeout.** 30 s covers everything except `ml_endpoints_models`
  (120 s, it blocks while querying Ollama) and `ml_mcp_server_connect`
  (`wait` + 15 s). Raise `ML_MCP_TIMEOUT` if your backend is slow (huge
  histories, slow disks).
* **stdout discipline.** In stdio mode the server never writes to stdout
  (that channel is the JSON-RPC wire). Logs go to stderr — safe for
  Claude Desktop / Cursor, which treat stray stdout as a protocol error.
* **Transport choices.**
  * `stdio` (default) — the client spawns this file; best for local apps.
  * `streamable-http` — run it as a small service for *remote* agents:
    ```bash
    python ml_mcp_server.py --transport streamable-http \
           --host 0.0.0.0 --port 8090 --backend-url http://10.0.0.4:8200
    # MCP endpoint: http://<this-machine>:8090/mcp
    ```
  * `sse` — the legacy HTTP variant (endpoint `/sse`), for older clients.
  * Both HTTP variants were verified end-to-end (initialize → tools/call).

---

## 5. Connecting MCP clients

All examples assume the server file at
`/absolute/path/to/MultiOllama_mcp/ml_mcp_server.py` — **use absolute paths
in client configs** (the client may spawn the process from any working
directory). `python` must be on PATH, or give the full interpreter path
(Windows: `C:\\Python312\\python.exe`, venvs: `…\\.venv\\Scripts\\python.exe`).

### 5.1 Claude Desktop

Edit the config file (Claude → Settings → Developer → Edit Config):

| OS | Path |
| --- | --- |
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

```json
{
  "mcpServers": {
    "MultiOllama": {
      "command": "python",
      "args": ["/absolute/path/to/MultiOllama_mcp/ml_mcp_server.py"],
      "env": { "ML_BACKEND_URL": "http://127.0.0.1:8200" }
    }
  }
}
```

Restart Claude Desktop; the tool icon shows **63 tools**. If the backend
is down you'll still connect — every call returns the `unreachable` error
contract with the fix hint. (Ready-made file:
`examples/claude_desktop_config.example.json`.)

### 5.2 Cursor

`~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (project):

```json
{
  "mcpServers": {
    "MultiOllama": {
      "command": "python",
      "args": ["/absolute/path/to/MultiOllama_mcp/ml_mcp_server.py"],
      "env": { "ML_BACKEND_URL": "http://127.0.0.1:8200" }
    }
  }
}
```

Cursor also speaks streamable-http — run the server as in §4 and use
`{"mcpServers": {"MultiOllama": {"url": "http://127.0.0.1:8090/mcp"}}}`.

### 5.3 Cline / Roo Code (VS Code)

Open the MCP settings (Cline sidebar → MCP Servers → Configure) — it
edits `cline_mcp_settings.json`:

```json
{
  "mcpServers": {
    "MultiOllama": {
      "command": "python",
      "args": ["/absolute/path/to/MultiOllama_mcp/ml_mcp_server.py"],
      "env": { "ML_BACKEND_URL": "http://127.0.0.1:8200" },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### 5.4 Windsurf

`~/.codeium/windsurf/mcp_config.json` — same shape as the Cursor config
above.

### 5.5 Any other stdio MCP client

Speak JSON-RPC 2.0 over the process's stdin/stdout, one message per line:

```
→ {"jsonrpc":"2.0","id":1,"method":"initialize","params":{
     "protocolVersion":"2025-06-18","capabilities":{},
     "clientInfo":{"name":"my-client","version":"1.0"}}}
← {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18",
     "serverInfo":{"name":"MultiOllama-backend","version":"1.0.0"}, …}}
→ {"jsonrpc":"2.0","method":"notifications/initialized"}
→ {"jsonrpc":"2.0","id":2,"method":"tools/list"}          # 63 tools
→ {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
     "name":"ml_backend_health","arguments":{}}}
← {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text",
     "text":"{ \"status\": \"ok\", …}"}],"isError":false}}
```

The server also answers `ping` and honors protocol-version negotiation.
`test_ml_mcp.py` is a working reference implementation of exactly this
exchange (§10).

### 5.6 Remote agents over HTTP (streamable-http)

```bash
# on the machine that can reach BOTH the agents and the backend:
python ml_mcp_server.py --transport streamable-http --host 0.0.0.0 --port 8090 \
       --backend-url http://10.0.0.4:8200 --api-key "$ML_API_KEY"
```

Clients connect to `http://<host>:8090/mcp`. Raw check with curl:

```bash
curl -s http://127.0.0.1:8090/mcp \
     -H "Content-Type: application/json" \
     -H "Accept: application/json, text/event-stream" \
     -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
            "protocolVersion":"2025-06-18","capabilities":{},
            "clientInfo":{"name":"curl","version":"1.0"}}}'
# -> event: message / data: {"jsonrpc":"2.0","id":1,"result":{…}}
# keep the returned Mcp-Session-Id header on subsequent requests
```

**Security:** binding `0.0.0.0` exposes the whole backend control plane
(create agents, read files, run commands *through agents*) to everyone on
the network. Use an API key, keep the bind on a VPN/interface you trust,
or front it with a reverse proxy with auth.

---

## 6. The ONE backend rule (read this first)

> **Agents live INSIDE a session.** The backend boots with *no session
> loaded*, so every session-scoped call answers **409** until you start
> one.

```
server starts ──► modules only, queue idle (session tools → 409)
      │
      ├─ ml_session_new      ─► interactive session (WAITS for your first message)
      ├─ ml_session_load     ─► interactive session from a file
      └─ ml_queue_enqueue   ─► queued unattended run
      │
      ▼
   NOW agents, chat, inject, tools, MCP subscriptions … all work
   (they target the ACTIVE session) until it finishes/aborts → back to 409.
```

Every MCP tool surfaces this cleanly: a 409 comes back as
`{"ok": false, "error_kind": "http", "status": 409, "detail": …,
"hint": "no ACTIVE session: agents live inside a session — call
ml_session_new (or ml_session_load / ml_queue_enqueue) first"}` — so even
an agent that has never read this file self-corrects on its first mistake.

---

## 7. Dogfood — let the backend's OWN agents use it

The backend is itself an MCP *client* (its `mcp_manager`, native tools).
Register this server in its registry and subscribe an agent — now the
backend's agents can drive their own backend (checked end-to-end; the
status line reports `connected — 63 tools`):

```bash
B=http://127.0.0.1:8200

# 1) register THIS server (absolute path! the backend spawns it)
curl -s -X POST $B/control/mcp/servers -H "Content-Type: application/json" -d '{
  "name": "MultiOllama-self",
  "transport": "stdio",
  "command": "python3",
  "args": ["/absolute/path/to/MultiOllama_mcp/ml_mcp_server.py"],
  "env": {"ML_BACKEND_URL": "http://127.0.0.1:8200"},
  "description": "this backend exposed as MCP (dogfood)"
}'

# 2) connect (spawns the server process + MCP handshake + tool listing)
curl -s -X POST $B/control/mcp/servers/MultiOllama-self/connect \
     -H "Content-Type: application/json" -d '{"wait":30}'

# 3) subscribe an agent of the active session — its tool list now
#    includes mcp__MultiOllama-self__ml_backend_health, …, all 63
curl -s -X POST $B/control/mcp/subscribe -H "Content-Type: application/json" \
     -d '{"agent_name":"watcher","server":"MultiOllama-self"}'
```

The same three steps work from MCP tools themselves (`ml_mcp_server_add`
→ `ml_mcp_server_connect` → `ml_mcp_subscribe`) — i.e. *any* MCP client
can set up the dogfood loop through the tools.

---

## 8. Tool catalog (63 tools)

Every tool returns the **backend's JSON verbatim** on success, and a
structured error object on failure (§11). Grouped exactly as in
`ml_mcp_server.py`; run `--list-tools` anytime for the same list.

### Backend / system (4)

| Tool | What it does |
| --- | --- |
| `ml_backend_health` | Cheap probe (GET /health, no auth) — call this FIRST |
| `ml_backend_state` | Full snapshot (GET /state); `brief=true` (default) compacts it to the decision-relevant fields |
| `ml_backend_overview` | Machine-readable map of every backend REST endpoint |
| `ml_system_shutdown` | Stop the BACKEND process (needs `confirm=true`; destructive) |

### Sessions (8)

| Tool | What it does |
| --- | --- |
| `ml_session_new` | Create the ACTIVE interactive session (default: waits for your first message) |
| `ml_session_load` | Load a session file (find paths with `ml_sessions_on_disk`) |
| `ml_session_get` | Active-session summary — never 409s, safe first call |
| `ml_session_save` | Manual save (default `<name>.json` next to the backend) |
| `ml_session_rename` | Rename the active session |
| `ml_session_finish` | Graceful finish (turns complete, final autosave) |
| `ml_session_abort` | Hard stop (final autosave still happens) |
| `ml_sessions_on_disk` | Every loadable session file (autosave/ + sessions/) |

### Agents (12)

| Tool | What it does |
| --- | --- |
| `ml_agents_list` | All agents + running flags |
| `ml_agent_get` | One agent's full config |
| `ml_agent_create` | Create (system prompt, model, endpoint, tools, limits, flags, MCP subs, prompts) |
| `ml_agent_edit` | PATCH — only provided fields change |
| `ml_agent_rename` / `ml_agent_delete` | Rename / delete |
| `ml_agent_set_active` | In/out of the round-robin |
| `ml_agent_stop` | Stop the running stream (sidebar Stop) |
| `ml_agent_inject` | Operator text → prepended to the agent's NEXT tool result (never persists in history) |
| `ml_agent_injections` | Pending injection count + texts |
| `ml_agent_save` / `ml_agent_load` | Portable agent JSON files |

### Chat & turns (9)

| Tool | What it does |
| --- | --- |
| `ml_chat_send` | Send a message — `send_now` (drain now) or `queue` (engines pick it up later) |
| `ml_chat_send_now` | Send-Now button: unblock every engine's user-turn wait |
| `ml_chat_history` | Full history; message entries may carry the **`files` download array** (§9) |
| `ml_chat_pending` / `ml_chat_pending_clear` | View / clear the pending user-message queue |
| `ml_chat_user` | Get/set the user name |
| `ml_chat_wait_user` | Get/set "Wait user input" (ON = round-robin waits for you) |
| `ml_chat_attach` | "Add file" — queue a file path as the next user message |
| `ml_ask` | ★ One-shot turn: send → wait for idle → return ONLY the appended messages (with their `files` arrays) |

### Output / waiting (2)

| Tool | What it does |
| --- | --- |
| `ml_output_recent` | Last N live events (chat, thinking/tool streams, lifecycle) from the ring buffer |
| `ml_wait_idle` | Block until the session is idle (or timeout, with what's still running) |

### Files — the chat download-links feature (2)

| Tool | What it does |
| --- | --- |
| `ml_files_url` | Absolute view + download URLs for one local file (token appended when a key is set) |
| `ml_files_fetch` | Fetch a file THROUGH the backend → base64 + MIME + size (agents "read" images/audio/documents without filesystem access; 2 MiB default cap, 64 MiB max) |

### The session run queue (7)

| Tool | What it does |
| --- | --- |
| `ml_queue_enqueue` | Unattended run: load newest autosave → pre-queue a message → run to quiescence → autosave → next |
| `ml_queue_list` | Pending + running + recent finished entries + injections |
| `ml_queue_move` / `ml_queue_cancel` | Reorder (0 = next) / cancel one or all |
| `ml_queue_inject` | Inject into an agent's next TOOL RESULT — never enqueues the session |
| `ml_queue_inject_list` / `ml_queue_inject_cancel` | Manage staged injections |

### Backend tools / prompts / names (6)

| Tool | What it does |
| --- | --- |
| `ml_tools_list` / `ml_tools_reload` | The agents' toolbox (per file, with docstrings) / re-scan |
| `ml_prompts_list` / `ml_prompts_reload` / `ml_prompt_get` | Prompt snippets (for agent `prompts` fields) |

plus `ml_names_list` — the `names.txt` agent-name pool (auto-naming).

### The backend's own MCP registry (9)

| Tool | What it does |
| --- | --- |
| `ml_mcp_servers_list` / `ml_mcp_status` | Registry + live connection status |
| `ml_mcp_server_add` / `_connect` / `_disconnect` / `_delete` | Manage servers the backend's agents can use |
| `ml_mcp_server_tools` | Tools a connected server exposes to agents |
| `ml_mcp_subscribe` / `ml_mcp_unsubscribe` | Per-agent subscriptions (tools become `mcp__<server>__<tool>`) |

### Ollama endpoints (4)

| Tool | What it does |
| --- | --- |
| `ml_endpoints_list` / `ml_endpoints_usage` | Endpoint status / loaded-model memory |
| `ml_endpoints_models` | Installed models of one endpoint (feeds agent `model` fields) |
| `ml_endpoints_scan` | LAN scan for Ollama servers (psutil on the backend) |

---

## 9. The download-links feature through MCP

Every chat message that mentions a **real local file** is enriched by the
backend with a `files` array (the previous milestone in this project):

```json
{"role": "assistant", "sender": "alpha", "content": "Saved it as ./chart.png",
 "files": [{"name": "chart.png", "kind": "image", "size": 12345,
            "path": "/home/me/backend/chart.png",
            "url": "/files?path=%2Fhome%2Fme%2Fbackend%2Fchart.png",
            "url_download": "/files?path=%2F…&download=1"}]}
```

From an MCP client:

1. `ml_chat_history` (or `ml_ask`) → read the `files` arrays of the reply.
2. `ml_files_url {"path": "/home/me/backend/chart.png"}` → absolute,
   clickable, token-carrying download URLs.
3. `ml_files_fetch {"path": …}` → base64 + MIME, if the client wants the
   bytes themselves (e.g. attach the chart to a vision model).

`ml_files_url` never checks existence (URL building only);
`ml_files_fetch` returns a 404 error contract when nothing resolves.
Bare names resolve exactly like the chat scan: working directory first,
then the backend folder.

---

## 10. Testing

`test_ml_mcp.py` needs **no** `mcp` package (it speaks raw JSON-RPC over
the spawned server's stdio) — run it wherever the server will run:

```bash
cd MultiOllama_backend && python run_server.py --port 8200    # terminal 1
cd ../MultiOllama_mcp && python test_ml_mcp.py \
    --backend-url http://127.0.0.1:8200 \
    --file-dir  /absolute/path/to/MultiOllama_backend \      # optional: enables the bare-name file checks
    --event-port 8288                                        # only if backend started with --tcp-port
#   -> 46 PASS  0 WARN  0 FAIL  — ALL GREEN
```

Covered: handshake, tools/list, health/state/overview, the 409 contract,
session lifecycle, agent CRUD + save/load round-trip, chat basics,
`ml_files_url`/`ml_files_fetch` round-trip + 404, **the `files` download
arrays in history AND in live events**, `ml_wait_idle`, `ml_ask` (real
generation with Ollama up; the structured timeout path when it is down),
queue + inject, tools/prompts/names/endpoints (incl. the 502 contract),
the MCP registry, and the shutdown refusal. Generation-dependent checks
WARN instead of FAIL when Ollama is absent — everything else is green
without it.

---

## 11. Error contract & troubleshooting

Every failure is a structured object:

```json
{"ok": false, "error_kind": "http", "status": 409,
 "detail": "no active session", "hint": "agents live inside a session — …"}
```

| Symptom | Meaning / fix |
| --- | --- |
| `error_kind: "unreachable"` | The backend is not running (or wrong URL). Start it: `cd MultiOllama_backend && python run_server.py --port 8200`; then check `--backend-url` / `ML_BACKEND_URL`. |
| `status 409` | No ACTIVE session — see §6. |
| `status 401` | Backend runs with `ML_API_KEY`; pass `--api-key` / `ML_MCP_API_KEY` / `ML_API_KEY`. |
| `status 404` | Unknown name/path — list valid ones (`ml_agents_list`, `ml_sessions_on_disk`, …). |
| `status 502` | Upstream Ollama call failed — check the endpoint/model (`ml_endpoints_list`, `ml_endpoints_models`). |
| `status 503` | Optional backend feature missing — `pip install psutil` / `mcp` **on the backend machine**. |
| `error_kind: "timeout"` | Backend too slow — raise `ML_MCP_TIMEOUT` / `--timeout`. |
| `error_kind: "too_large"` | `ml_files_fetch` cap — re-call with a bigger `max_bytes` (≤ 64 MiB). |
| Claude Desktop shows 0 tools / "connection failed" | Wrong `command` (use an absolute python path), missing deps in that interpreter, or stray stdout. Check the client's MCP logs; run `--check` with the same interpreter. |
| `ml_ask` returns `appended: 0`, `timed_out: true` | Engines never finished — usually Ollama down (502 events in `ml_output_recent`) or agents kept chatting (`ml_chat_wait_user` off). Read `wait.running_agents`. |
| User messages stuck out of history | Pending messages enter history only when engines drain them (Ollama must be up). `ml_chat_pending` shows them; `ml_chat_pending_clear` discards. |

---

## 12. Security notes

* **stdio (default)** is the safest wiring: the server runs as your user,
  next to your client, and talks to `127.0.0.1`.
* The backend's `GET /files` can read **any file on the backend machine**
  (that is the point of the download-links feature). Protect it with
  `ML_API_KEY` and bind the backend to `127.0.0.1` (`ML_BACKEND_HOST`)
  unless it must serve a LAN.
* `ml_files_url` embeds `?token=…` when a key is configured — the URL
  then carries the secret; treat it like a password (prefer `include_token:
  false` when the consumer can send headers).
* `ml_system_shutdown` and agent command tools are **destructive**; the
  former refuses without `confirm=true`, consider approval modes in your
  MCP client for the rest.
* Exposing the server over `streamable-http` exposes the whole control
  plane — see §5.6.

---

## 13. FAQ

**Can one MCP server drive several backends?**
No — one instance per backend URL (the backend is addressed in every
call). Run a second copy with `--backend-url http://other:8200`; clients
show them side by side (name them differently in the client config).

**Does it work while the backend is down?**
The MCP connection stays up; every tool returns the `unreachable`
contract with the start-up hint. Nothing crashes.

**Where do agent-written files land?**
In the BACKEND's working directory (its core file tools write there) —
that is why bare names resolve cwd-first. `ml_chat_history`'s `files`
arrays carry absolute `path`s; use those.

**Which backend versions?**
Built against backend 1.0.0's documented API (README.md / API_GUIDE.md in
`MultiOllama_backend/`). The server only uses stable, documented
endpoints; new backend features appear via `ml_backend_overview` /
raw REST until wrapped here.

**MCP protocol version?** The SDK negotiates (2025-06-18 at time of
writing) — clients on older spec versions are handled by the SDK itself.
