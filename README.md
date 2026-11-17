# MultiOllama Backend

Headless, cross-platform (Linux / Windows) backend for the proven
MultiOllama multi-agent / multi-endpoint engine. This is the first
component of the compound fullstack structure: the same engine the
PyQt desktop app runs, wrapped in a network API so future frontends
(web browser, mobile apps, other backends on other PCs) can fully
drive and observe it.

```
┌──────────────┐   REST /control/*   ┌──────────────────────────┐
│  frontend    │◄───────────────────►│                          │
│  (future)    │   WS  /ws/output    │   MultiOllama backend     │
│  web/mobile/ │◄───────────────────►│                          │
│  desktop     │   SSE /output/stream│  ┌────────────────────┐  │
└──────────────┘                     │  │ proven core (pure) │  │
      other backends on other PCs ───►│  │ agent.py           │  │
      (interconnection)              │  │ agent_manager.py   │  │
                                      │  │ ollama_client.py   │  │
                                      │  │ tool_injection.py  │  │
                                      │  │ core_tools.py      │  │
                                      │  │ mcp_manager.py     │  │
                                      │  │ dialogue_store.py  │  │
                                      │  └────────────────────┘  │
                                      │  session queue → autosave │
                                      │  per-endpoint engines     │
                                      └──────────┬───────────────┘
                                                 │ Ollama API
                                     ┌───────────┴───────────┐
                                     │ local + remote Ollama │
                                     │ (ip:port endpoints)   │
                                     └───────────────────────┘
```

## Quick start

```bash
pip install -r requirements.txt
python run_server.py                     # 0.0.0.0:8000 (+ TCP events :8888)
```

Windows: same commands (Python 3.10+). Optionally create a venv first.

```bash
python run_server.py --host 127.0.0.1 --port 9000 --tcp-port 8889
```

OpenAPI docs: `http://localhost:8000/docs`

## Built-in web console

Open **`http://localhost:8000/`** in any browser — the backend serves
`webui.html` (one self-contained file, no Node / no build step) with a
full control console: live event stream, chat, agent CRUD (with
names/tools/prompts/mcp pickers), session browser with per-session
backup roll-down, queue management and the inject control. It is the
reference frontend: everything it does goes through the public REST/WS
API, so it doubles as a blueprint for the future custom frontends (it
can also be opened from anywhere and pointed at a remote backend —
base URL + API key are remembered in the browser).

Endpoint-by-endpoint usage with copy-paste `curl` examples:
**[API_GUIDE.md](API_GUIDE.md)**.

## Chat file download links

Agents produce files on disk (images, audio, video, PDFs, archives —
written by the core file tools, `core_run_python`, user tools). When
an agent's **answer** mentions a real local file, the backend scans
the message (the proven PyQt path scanner — absolute paths, `./`-,
`..`- and `~`-relative references, bare names like `output.png`
resolved against the working directory and the app directory; a
reference only counts when the file actually EXISTS, so ordinary
words are never mistaken for files). The message text itself is
NEVER modified — instead every chat event and every history message
carries an extra `files` array:

```json
{"sender":"alpha","text":"Saved it as ./chart.png","session":"team",
 "files":[{"name":"chart.png","kind":"image","size":12345,
            "path":"/abs/dir/chart.png",
            "url":"/files?path=%2Fabs%2Fdir%2Fchart.png",
            "url_download":"/files?path=…&download=1"}]}
```

`GET /files?path=…` serves the file (inline; correct MIME type,
range requests supported), `&download=1` forces a real browser
download. Auth when `ML_API_KEY` is set: `?token=` (that is why the
built-in console's links stay clickable — browsers cannot put the
`X-API-Key` header on a plain `<a href>`). The built-in web console
appends the links at the end of each chat message — one clickable
download link per line, with a kind icon and the file size. Thinking
streams are never scanned; the stored history / LLM context is never
polluted with link text.

## Auto-reload — tools / MCP / prompts / names (live catalog)

The original PyQt app re-read its lists **whenever a list was called**
(e.g. every time the Create-Agent dialog opened) and shipped an
agent-facing `core_reload_tools()` so agents could use tools **they
created themselves** without restarting the app. The backend keeps both
behaviours:

* **Every listing re-reads disk first** — `GET /control/tools` re-scans
  `tools/`, `GET /control/prompts` re-scans `prompts/`,
  `GET /control/mcp/servers` re-reads `mcp_servers.json`,
  `GET /control/names` re-reads `names.txt`, and `GET /state` does all
  four for its `tools` / `prompts` / `mcp` / `names` blocks. Drop a
  tool file into `tools/`, hand-edit the MCP registry or `names.txt`
  while the server runs — the very next call sees it, **no restart, no
  Refresh button**. The check is a couple of `stat()` calls (name +
  mtime + size); a real re-import happens only when something actually
  changed, so polling `/state` stays cheap.
* **Agents get the same contract inside conversations**:
  `core_list_tools()` / `core_list_prompts()` re-scan before answering;
  `mcp_list_servers()` / `mcp_list_tools()` / `mcp_status()` re-read
  the registry; `core_reload_tools()` force-reloads tools + prompts +
  MCP registry in one call. A tool created with `core_create_tool` is
  **immediately callable** — by its creator and by every other agent
  after `core_update_agent` attaches it.
* **The web console mirrors it**: the agent dialog and each picker
  (names / tools / prompts / mcp) fetch a fresh `/state` snapshot at
  the moment they open.
* Force endpoints for completeness: `POST /control/tools/reload`,
  `POST /control/prompts/reload`, `POST /control/mcp/reload`, and
  `POST /control/catalog/reload` (all three at once — the REST
  equivalent of the desktop app's Tools/Prompts/MCP Refresh buttons).

## Startup behaviour

The terminal process loads **all modules** and loads **no sessions**:

* `tools/` — the four core tool files are (re)written; user tools untouched
* `prompts/` — reusable prompt snippets folder
* MCP engine + registry (`mcp_servers.json`), Ollama `ClientManager`
* dialogue store (`agent_dialogues/`), session store (`autosave/`, `sessions/`)
* the session queue parks in **wait** mode until a session is enqueued

## The session queue (core new behaviour)

```
while running:
    entry  = next pending queue entry        (or WAIT when empty)
    session = newest saved file for entry.session_name
              (autosave/ first, then sessions/)
    load it, apply entry options, queue the entry's message as a
    user message BEFORE the engines start
    run it (round-robin per endpoint — the proven loop)
    session "done" when:  no agent running
                       + pending queue empty
                       + every agent inactive   (core_agent_active(name, False))
    auto-save the session the moment it is done
    next entry — or wait for one
```

Enqueue a session (design brief 4.1):

```bash
curl -X POST localhost:8000/queue/sessions -H 'Content-Type: application/json' \
     -d '{"session_name":"team-alpha","message":"continue with the plan"}'
```

Options per entry: `activate_all`, `activate: ["agent"]` (re-activate
saved-inactive agents), `idle_timeout`, `auto_finish`.

A session also finishes via `POST /control/session/finish` (graceful) or
`/control/session/abort` (hard stop), or when ALL endpoints stay
unreachable longer than `ML_ENDPOINT_DOWN_TIMEOUT` (default 120 s).

Inject to an agent's next tool result — never enqueues the session
(design brief 4.2):

```bash
curl -X POST localhost:8000/queue/inject -H 'Content-Type: application/json' \
     -d '{"session_name":"team-alpha","agent_name":"alpha","text":"check the results"}'
```

Live when that session is running; staged (delivered automatically on
its next run) otherwise. The proven `tool_injection` semantics apply:
the text is prepended to the agent's next TOOL result and never
persists into history.

## Endpoints

### Full control — everything the PyQt GUI could do (`/control/*`)

| PyQt control | REST |
| --- | --- |
| input box / Send / Send-to-queue | `POST /control/message {text, mode: queue\|send_now}` |
| pending queue view / clear | `GET` / `DELETE /control/queue` |
| user name box | `GET`/`POST /control/user` |
| Pause User Input checkbox ("wait user input", ON for interactive sessions) | `GET`/`POST /control/pause` |
| Send Now | `POST /control/send-now` |
| Add file | `POST /control/attach {path}` |
| agent dropdown / sidebar | `GET /control/agents` |
| Create / Edit agent dialog | `POST /control/agents` · `PATCH /control/agents/{name}` |
| Delete / Rename agent | `DELETE /control/agents/{name}` · `PUT /control/agents/{name}` |
| core_agent_active | `POST /control/agents/{name}/active` |
| sidebar Stop | `POST /control/agents/{name}/stop` |
| sidebar Inject | `POST /control/agents/{name}/inject` |
| Save/Load agent to/from file | `POST /control/agents/save` · `/control/agents/load` |
| agent auto-naming pool (names.txt) | `GET /control/names` (also the `names` block of `/state`) |
| session name box | `GET`/`POST /control/session/name` |
| Save / Load Session | `POST /control/session/save` · `/control/session/load` |
| New Session | `POST /control/session/new` |
| finish / abort | `POST /control/session/finish` · `/control/session/abort` |
| Refresh tools / prompts | `POST /control/tools/reload` · `/control/prompts/reload` |
| MCP servers dialog | `/control/mcp/*` (servers, connect, tools, status, subscribe) |
| endpoint/model dropdowns, LAN scan | `/control/endpoints/*` |
| model memory badges | `GET /control/endpoints/usage` |
| app shutdown | `POST /system/shutdown {"confirm": true}` |

### Full output — all current state + streaming

* `GET /state` — complete state snapshot (server, session, agents,
  endpoints, queue, injections, tools, prompts, mcp, names.txt pool,
  model usage, sessions on disk)
* `WS /ws/output` — snapshot on connect, then every live event:
  chat messages, per-agent thinking/content token streams, full prompt
  dumps, tool calls + composed tool results, status changes, agent
  running/idle flips, queue updates, session lifecycle, injections.
  Send `{"action": "refresh"}` to re-request a snapshot.
* `GET /output/stream` — the same as Server-Sent Events
* `GET /output/recent?limit=` — ring-buffer replay (reconnects)
* `GET /output/history` — chat history of the active session
  (chat view: user / agent answers / system notes; tool traffic
  filtered — `include_tools=true` for the raw context; `tail=N`
  keeps the newest N — this is what the UI's "Reload history"
  uses)
* `GET /files?path=&download=` — serve / download a local file the
  chat scan links to (see “Chat file download links” above)

### Additional control (design brief items 4.1 / 4.2)

* `POST /queue/sessions` — enqueue session run (+ message before it)
* `GET /queue/sessions` · `DELETE /queue/sessions[/{id}]` ·
  `PATCH /queue/sessions/{id}/move {position}`
* `POST /queue/inject` — inject to next tool output, no enqueueing
* `GET /queue/inject` · `DELETE /queue/inject/{id}`

### Legacy TCP events (:8888)

The `send_event()` snippet from the `core_get_send_message_code` tool
still works — messages enter the active session's history and the
output stream, exactly like the PyQt app's EventServer.

## Interactive vs queue sessions

At most ONE session is active. Queue entries create sessions with
auto-finish; `POST /control/session/new` / `/control/session/load`
create interactive sessions that run until finished explicitly (or
until a queued session preempts them — the preempted session is
auto-saved, not lost; disable with `ML_PREEMPT_INTERACTIVE=0`).

Interactive sessions start with **wait user input ON** (the proven
PyQt default): agents hold their turns until the operator sends a
message or presses Send Now — no more agents chatting away before you
even start. Opt out with `{"wait_user_input": false}` in the request
body; queue entries always run unattended (OFF).

## Compatibility with the PyQt app

Session files, saved-agent files, the tools/ and prompts/ folders and
the agent_dialogues/ store use the IDENTICAL formats. A session
auto-saved by the desktop app (drop it into `sessions/` or `autosave/`)
can be queued here, and vice versa. The agent auto-naming pool works
the same too: drop your `names.txt` (one name per line) into the
backend root — new agents get a random unused name from it, and the
web console's name picker lists the pool.

## Configuration (environment variables)

| Variable | Default | Meaning |
| --- | --- | --- |
| `ML_BACKEND_HOST` | `0.0.0.0` | bind address |
| `ML_BACKEND_PORT` | `8000` | HTTP/WS port |
| `ML_EVENT_TCP_PORT` | `8888` | legacy TCP event port |
| `ML_CORS_ORIGINS` | `*` | browser origins allowed to call the API (CORS); `""` disables |
| `ML_BACKEND_AUTOSAVE_KEEP` | `20` | newest autosaves kept |
| `ML_BACKEND_AUTOSAVE_MIN_INTERVAL` | `1.0` | mid-run autosave throttle (s) |
| `ML_SESSIONS_DIR` | `sessions` | manual session import folder |
| `ML_API_KEY` | – | require `X-API-Key` (WS/SSE: `?token=`) |
| `ML_PREEMPT_INTERACTIVE` | `1` | queue preempts interactive sessions |
| `ML_PREEMPT_TIMEOUT` | `600` | grace before forced abort (s) |
| `ML_ENDPOINT_DOWN_TIMEOUT` | `120` | all-endpoints-down finish trigger (s) |
| `ML_DONE_GRACE` | `3` | quiesce detector startup grace (s) |
| `ML_USAGE_POLL_INTERVAL` | `3` | /api/ps polling (s) |
| `ML_BACKEND_LOG_LEVEL` | `info` | uvicorn log level |

## Deployment notes (Linux / Windows)

* Pure Python, no GUI toolkit. Dependencies: `ollama`, `requests`,
  `fastapi`, `uvicorn[standard]`, `pydantic`, `websockets`.
* Optional: `psutil` (LAN scan endpoint), `mcp` (native MCP engine).
* The backend folder is relocatable — media paths in loaded sessions
  are re-anchored to the current folder automatically (proven logic).
* systemd (Linux) example:

  ```ini
  [Service]
  WorkingDirectory=/opt/MultiOllama_backend
  ExecStart=/usr/bin/python3 run_server.py
  Restart=always
  ```

* Windows: run via Task Scheduler (`pythonw run_server.py`) or as a
  service wrapper (NSSM) — no console interaction is required.

## Roadmap alignment

* **deploy on different PCs, interconnected apps** — every backend is
  an HTTP+WS service; a frontend can drive several backends at once;
  backends can call each other's `/control/message` + `/queue/*`.
* **web browser frontend** — DONE in v1: the built-in web console at
  `GET /` plus CORS support (`ML_CORS_ORIGINS`) for separately-served
  frontends; WS `/ws/output` + `/control/*` cover the full PyQt
  surface; OpenAPI schema at `/docs` scaffolds clients.
* **mobile apps** — the SSE fallback and token auth fit constrained
  network stacks.
