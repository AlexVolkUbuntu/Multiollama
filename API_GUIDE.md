# MultiOllama Backend — API Usage Guide

Everything you need to drive the backend from curl, a script, another
backend, or a browser frontend. The companion files:

| File | Purpose |
| --- | --- |
| `run_server.py` | start the server (`python run_server.py --port 8000`) |
| `webui.html` | built-in web console, served at `GET /` |
| `README.md` | architecture, queue semantics, deployment |
| `API_GUIDE.md` | **this file** — every endpoint with copy-paste examples |

Interactive OpenAPI docs are always at **`/docs`** (Swagger UI).

All examples below assume the backend runs at
`http://127.0.0.1:8000` (your port; the default is 8000) and use the
shell variable style:

```bash
BASE=http://127.0.0.1:8000
KEY=YOUR_KEY          # only needed when ML_API_KEY is set on the server
AUTH=(-H "X-API-Key: $KEY")
```

If `ML_API_KEY` is **not** set on the server, the `X-API-Key` header is
ignored (that's why your very first test got `409`, not `401`).

---

## 1. The one rule that explains your 409

> **Agents live INSIDE a session.** The backend deliberately boots with
> **no session loaded** ("load all modules, no sessions"), so every
> session-scoped endpoint answers `409 Conflict` until you start one.

This is the intended lifecycle:

```
server starts ──► modules only, queue waits (idle)
      │
      ├─ POST /control/session/new      ─► interactive session (no auto-finish)
      │                                    starts WAITING for your first
      │                                    message (wait_user_input: true)
      ├─ POST /control/session/load     ─► interactive session from a file
      │                                    (also starts waiting by default)
      └─ POST /queue/sessions           ─► queued run (loads from autosave,
                                           runs, autosaves, next entry…)
                │
                ▼
   NOW agent CRUD, chat, inject, tools, MCP … all work (they target
   the ACTIVE session) until it finishes/aborts — then back to 409.
```

Your failing call, fixed (three commands, in order):

```bash
# 1) create a session (becomes the ACTIVE session)
curl -s -X POST $BASE/control/session/new "${AUTH[@]}" \
     -H "Content-Type: application/json" -d '{"name":"probe"}'
# -> {"ok":true,"session_name":"probe"}

# 2) NOW creating the agent works — your original command, unchanged
curl -s -X POST $BASE/control/agents "${AUTH[@]}" \
     -H "Content-Type: application/json" \
     -d '{"name":"probe","system_prompt":"","model":"qwen3.5:9b","endpoint":"local"}'
# -> {"ok":true,"agent":{"name":"probe","endpoint":"local"}}

# 3) talk to it
curl -s -X POST $BASE/control/message "${AUTH[@]}" \
     -H "Content-Type: application/json" \
     -d '{"text":"hello, introduce yourself","mode":"send_now"}'
```

Endpoints that need an active session and will answer `409` when idle:
`/control/message`, `/control/queue`, `/control/user`, `/control/pause`,
`/control/send-now`, `/control/attach`, **all of `/control/agents/*`**,
`/control/session/name|save|finish|abort`, `/control/mcp/subscribe|unsubscribe`.

Endpoints that work with **no** session: `/state`, `/health`,
`/queue/*`, `/control/session/new|load`, `/control/session` (GET),
`/control/tools*`, `/control/prompts*`, `/control/mcp/servers*` (registry),
`/control/endpoints*`, `/output/recent`.

---

## 2. The 405 on OPTIONS — CORS (now fixed)

You sent a browser CORS **preflight**:

```
OPTIONS /control/agents  +  Origin: http://127.0.0.1:3000
                           +  Access-Control-Request-Method: POST
```

The backend had no CORS middleware, so the OPTIONS request hit the
route table (which only knows GET/POST/…) and died with
`405 Method Not Allowed`. Browsers abort the real request after that.

`server/app.py` now adds FastAPI's `CORSMiddleware`, so:

* `OPTIONS` preflight → `204` + `Access-Control-Allow-*` headers
* cross-origin `fetch()` with `X-API-Key` works
* the WebSocket authenticates with `?token=` (browsers cannot set
  custom WS headers — that's a browser rule, not a backend rule)

Control it with one env var (default allows every origin):

```bash
ML_CORS_ORIGINS="*"                                              # default
ML_CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"    # lock down
ML_CORS_ORIGINS=""                                               # disable
```

Verification (your exact command, now passing):

```bash
curl -i -X OPTIONS $BASE/control/agents \
  -H "Origin: http://127.0.0.1:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: content-type,x-api-key"
# HTTP/1.1 204 No Content
# access-control-allow-origin: http://127.0.0.1:3000
```

Note: `curl`, Python scripts, and backend-to-backend calls never send
preflights — CORS is purely a browser mechanism.

---

## 3. Authentication

| Server setting | What clients must do |
| --- | --- |
| `ML_API_KEY` unset (your case) | nothing — every call is open |
| `ML_API_KEY=secret123` | REST: `X-API-Key: secret123` header (or `?token=`) · WS/SSE: `?token=secret123` |

Wrong/missing key → `401 {"detail":"invalid or missing API key"}`.
The WS closes with code `4401` instead.

---

## 4. Full output — state + streaming

### `GET /state` — one-shot complete snapshot

```bash
curl -s $BASE/state "${AUTH[@]}" | python3 -m json.tool | head -40
```

Returns: `server` (identity, config, folders), `active_session`
(name, agents, endpoints, pending messages, history length),
`session_queue`, `recent_queue`, `staged_injections`,
`live_injections`, `tools`, `prompts`, `mcp`, `names` (the names.txt
pool — file path + list), `model_usage`, `sessions_on_disk`. The chat
history itself is **not** embedded (it can be huge) — pull it
separately.

### `WS /ws/output` — the primary live channel (browsers, mobile, other backends)

```js
// browser / Node
const ws = new WebSocket("ws://127.0.0.1:8000/ws/output?token=" + KEY);
ws.onmessage = (m) => console.log(JSON.parse(m.data));
ws.send(JSON.stringify({action: "refresh"}));   // ask for a new snapshot
ws.send(JSON.stringify({action: "ping"}));      // -> {"type":"pong"}
```

Server → client frames:

```json
{"type":"snapshot","seq":0,"ts":1762…,"data":{ …same as GET /state… }}
{"type":"chat","data":{"sender":"alpha","text":"…",
 "role":"assistant","session":"team"}}
   role = user | assistant | system | error — the display role, so
   frontends can render proper chat bubbles that match a history
   reload exactly (derived from the sender by the engine rules)
{"type":"thinking","data":{"endpoint":"local","agent":"alpha",
 "event_type":"stream|thinking|prompt|tool_call|tool_result|status|done|error",
 "detail":"…","extra":…}}
{"type":"status","data":{"endpoint":"local","text":"Running: alpha","color":"blue"}}
{"type":"agent_status","data":{"endpoint":"local","agent":"alpha","running":true,"model":"…"}}
{"type":"queue_updated","data":{"pending_user_messages":["…"]}}
{"type":"session","data":{"event":"started|finished|aborted|…","session_name":"…"}}
{"type":"injection","data":{"session_name":"…","agent_name":"…","staged":true,…}}
{"type":"agents_changed","data":{"agents":["alpha","beta"]}}
{"type":"log","data":{"text":"…"}}
{"type":"ping"}   // keepalive every 20 s of silence
```

A snapshot arrives on connect, after every `{"action":"refresh"}`, and
a plain GET `/state` any time — reconnecting clients resync with one
round-trip.

### `GET /output/stream` — the same events as Server-Sent Events

```bash
curl -N "$BASE/output/stream?token=$KEY"        # Ctrl-C to stop
# data: {"type":"snapshot", …}
# data: {"type":"chat", …}
```

Keepalive comments every 15 s. Ideal for mobile / constrained stacks.

### `GET /output/recent?limit=200` — reconnect gap-filler

The last N events from the in-memory ring buffer (default size 1000).

### `GET /output/history?tail=0&offset=0&limit=0&include_tools=false` — active session's chat history

```bash
curl -s "$BASE/output/history?tail=50" "${AUTH[@]}"
# {"active":true,"session_name":"probe","total":12,"total_raw":31,
#  "offset":0,"messages":[…]}
```

Messages are `{role, sender, content, timestamp}`; newest last.

* **Chat view by default**: only user messages, agent ANSWERS and
  system notes — exactly what the engines broadcast live and what the
  chat tab shows. Raw tool traffic (`role:"tool"` results and
  content-less assistant tool-call requests) is filtered OUT; pass
  `include_tools=true` for the raw LLM context (debugging).
* **Slicing** (applied to the filtered list): `tail>0` → only the
  newest `tail` messages (what a chat reload wants — plain `limit`
  keeps the OLDEST N and silently hides the recent end on long
  tool-heavy sessions); otherwise `offset` skips the oldest N, then
  `limit` caps the count (`0` = everything).
* `total` counts the filtered chat messages; `total_raw` counts
  everything stored including tool traffic.

### Chat file download links — `files` array + `GET /files`

Every chat message that mentions a REAL local file is enriched with
a `files` array — on the live `chat` WS/SSE events AND on the
history messages above. The scan is the proven PyQt path scanner
(absolute paths, `./` `../` `~` references, bare names like
`output.png` resolved against the working directory then the app
directory), gated on a recognised extension (image / audio / video /
document / archive) AND the file actually existing, so ordinary
words are never mistaken for files. Only final answers are scanned —
thinking/token streams travel on the separate `thinking` events and
are never touched, and the stored history text is NEVER modified
(only display copies gain the field).

```json
{"sender":"alpha","text":"Saved it as ./chart.png","role":"assistant",
 "session":"probe",
 "files":[{"name":"chart.png","kind":"image","size":12345,
            "path":"/home/me/backend/chart.png",
            "url":"/files?path=%2Fhome%2Fme%2Fbackend%2Fchart.png",
            "url_download":"/files?path=%2Fhome%2Fme%2Fbackend%2Fchart.png&download=1"}]}
```

Serve / download one of them:

```bash
# view inline (correct MIME type, range requests supported)
curl -s "$BASE/files?path=%2Fhome%2Fme%2Fbackend%2Fchart.png" -o chart.png

# force a browser download (Content-Disposition: attachment + filename)
curl -s "$BASE/files?path=%2Fhome%2Fme%2Fbackend%2Fchart.png&download=1" -o chart.png

# bare names resolve exactly like the chat scan (cwd, then app dir)
curl -s "$BASE/files?path=chart.png&download=1" -o chart.png

# with auth: the token rides in the query string (<a> links can't set headers)
curl -s "$BASE/files?path=chart.png&token=$KEY" -o chart.png
```

`404` when nothing exists at the reference; `401` with the wrong
token when `ML_API_KEY` is set (the endpoint accepts `X-API-Key`
header or `?token=`, like the rest of the API).

The built-in web console renders this as a "files — download" block
appended at the end of the message — one clickable link per file,
each on its own line, with a kind icon, the filename and its size.

---

## 5. Full control — the PyQt surface (`/control/*`)

Machine-readable map: `GET /control/overview`.

### 5.1 Chat & user

| Purpose | Call |
| --- | --- |
| input box + **Send to queue** | `POST /control/message {"text":"…","mode":"queue"}` |
| input box + **Send** (drain now) | `POST /control/message {"text":"…","mode":"send_now"}` |
| **Send Now** button (no text) | `POST /control/send-now` |
| pending queue display | `GET /control/queue` |
| Clear pending queue | `DELETE /control/queue` |
| User name box | `GET` / `POST /control/user {"name":"Alex"}` |
| Wait user input (Pause User Input) checkbox | `GET` / `POST /control/pause {"paused":true}` |
| Add file | `POST /control/attach {"path":"/home/alex/notes.txt"}` |

Note the checkbox semantics (same as the desktop app): **ON** = the
round-robin waits for your message (or `POST /control/send-now`)
before the next agent turn. Interactive sessions created via
`/control/session/new` / `/control/session/load` now start with it **ON**
(`wait_user_input` body field, default `true`) — agents no longer start
chatting before you do. Queue entries always run with it OFF.

```bash
curl -s -X POST $BASE/control/message "${AUTH[@]}" -H "Content-Type: application/json" \
     -d '{"text":"summarize the plan so far","mode":"send_now"}'
```

`mode` semantics: `"queue"` parks the text in
`pending_user_messages` until the engines naturally come around;
`"send_now"` additionally unblocks every engine's user-turn wait so
they drain immediately.

### 5.2 Agents (active session only — see §1)

| Purpose | Call |
| --- | --- |
| list agents (+running flags) | `GET /control/agents` |
| one agent | `GET /control/agents/{name}` |
| **create** (Edit dialog defaults) | `POST /control/agents {…}` |
| **edit** (only listed fields change) | `PATCH /control/agents/{name} {…}` |
| rename | `PUT /control/agents/{name} {"new_name":"beta"}` |
| delete | `DELETE /control/agents/{name}` |
| in/out of round-robin | `POST /control/agents/{name}/active {"active":false}` |
| sidebar **Stop** | `POST /control/agents/{name}/stop` |
| sidebar **Inject** (live session) | `POST /control/agents/{name}/inject {"text":"…"}` |
| pending injections of an agent | `GET /control/agents/{name}/injections` |
| Save agent to file | `POST /control/agents/save {"name":"alpha","path":"alpha.json"} (path optional)` |
| Load agent from file | `POST /control/agents/load {"path":"alpha.json","new_name":null}` |

Full field set for create/edit (all optional except `name` on create;
defaults shown):

```bash
curl -s -X POST $BASE/control/agents "${AUTH[@]}" -H "Content-Type: application/json" -d '{
  "name": "alpha",
  "system_prompt": "You are the team lead. Be concise.",
  "description": "team lead",
  "model": "qwen3.5:9b",
  "endpoint": "local",
  "temperature": 0.5,
  "num_ctx": 32000,
  "tools": ["core_file_write", "core_execute_command"],
  "max_context_messages": 30,
  "max_tool_iterations": 1000,
  "active": true,
  "silent": false,
  "use_tools": true,
  "image": false,
  "audio": false,
  "mcp": false,
  "mcp_servers": [],
  "prompts": []
}'
```

`endpoint` is `"local"` or `"ip:port"` of another Ollama server — that's
the multi-endpoint part of the engine. Available tool names are listed
in `GET /state` → `tools.tools[].name`.

### 5.3 Sessions

| Purpose | Call |
| --- | --- |
| active session summary | `GET /control/session` (no 409 when idle — returns `{"active":false,…}`) |
| **New Session** | `POST /control/session/new {"name":"team","user_name":"Alex","wait_user_input":true} (all optional)` |
| **Load Session** | `POST /control/session/load {"path":"/abs/path/session.json","wait_user_input":true}` |
| **Save Session** | `POST /control/session/save {"path":null}` (null → `<session name>.json`) |
| session Name box | `GET` / `POST /control/session/name {"name":"team-alpha"}` |
| graceful finish (turns complete, then final autosave) | `POST /control/session/finish` |
| hard stop (final autosave still happens) | `POST /control/session/abort` |

`wait_user_input` (default `true`) = the session starts with the
"Wait user input" checkbox ON: agents hold their turns until your
first message / Send Now. Pass `false` to start chatting immediately.

At most ONE session is active. A queued run may preempt an interactive
session (it is autosaved first, never lost; `ML_PREEMPT_INTERACTIVE=0`
disables).

### 5.4 Tools / prompts / names — AUTO-RELOAD on every call

```bash
curl -s $BASE/control/tools "${AUTH[@]}"          # every tool file + tool + docstring
curl -s -X POST $BASE/control/tools/reload "${AUTH[@]}"          # force (rarely needed)
curl -s $BASE/control/prompts "${AUTH[@]}"        # prompt snippets in prompts/
curl -s $BASE/control/prompts/{name} "${AUTH[@]}"
curl -s $BASE/control/names "${AUTH[@]}"          # names.txt pool for agent auto-naming
curl -s -X POST $BASE/control/catalog/reload "${AUTH[@]}"        # force-reload ALL lists at once
curl -s -X POST $BASE/control/mcp/reload "${AUTH[@]}"           # force re-read of mcp_servers.json
```

**Every list call re-reads its sources from disk first** (the proven
PyQt behaviour — the desktop app re-scanned whenever a list was
opened, e.g. on agent creation):

* `GET /control/tools` re-scans `tools/` — tool files dropped in by
  hand **or created by an agent via `core_create_tool`** appear on the
  very next call, without restarting the backend and without pressing
  any reload;
* `GET /control/prompts` (and `/state`'s `prompts` block) re-scans
  `prompts/`;
* `GET /control/mcp/servers` re-reads `mcp_servers.json` — servers
  hand-edited into the file (or registered by another process) appear
  immediately;
* `GET /control/names` re-reads `names.txt` fresh every call;
* `GET /state` does all of the above for its `tools` / `prompts` /
  `mcp` / `names` blocks, so every snapshot is live.

The check is a couple of `stat()` calls (file name + mtime + size);
the actual re-import only happens when something really changed. The
`POST .../reload` endpoints remain for explicit force-reloads, and
`POST /control/catalog/reload` does tools + prompts + MCP registry in
one call (the REST equivalent of the desktop app's three Refresh
buttons). The web console applies the same contract on its side: the
agent dialog and every picker (names / tools / prompts / mcp) fetch a
fresh `/state` snapshot at the moment they open.

`GET /control/names` (also the `names` block of every `/state`
snapshot) returns `{"file":"…/names.txt","names":["Alpha","Bravo",…]}`.
Drop a `names.txt` (one name per line) into the backend root — or the
directory you launch the server from — and:

* the web console pre-fills every **New agent** dialog with a random
  UNUSED name from it (the proven desktop behaviour), and
* the **names ▾** button opens a single-selection picker of the pool
  (names already in use are shown but not selectable).

Missing file → empty list; auto-naming simply stays off.

**Agents get the same auto-reload from inside a conversation** (tools
available to every agent):

* `core_list_tools()` — re-scans `tools/` before answering; a tool
  another agent just created is listed (and callable after
  `core_update_agent(tools=[...])` attaches it);
* `core_list_prompts()` — re-scans `prompts/` before answering;
* `mcp_list_servers()` / `mcp_list_tools()` / `mcp_status()` — re-read
  `mcp_servers.json` before answering;
* `core_reload_tools()` — force-reloads everything (tools + prompts +
  MCP registry) and reports counts; agents that just created a tool
  with `core_create_tool` can call it immediately in the same
  conversation (the loader is reloaded as part of the creation).

### 5.5 MCP

```bash
curl -s $BASE/control/mcp/servers "${AUTH[@]}"                    # registry (auto re-reads mcp_servers.json)
curl -s -X POST $BASE/control/mcp/servers "${AUTH[@]}" -H "Content-Type: application/json" \
     -d '{"name":"fs","transport":"stdio","command":"npx","args":["-y","mcp-server-filesystem","/tmp"]}'
curl -s -X POST $BASE/control/mcp/servers/fs/connect "${AUTH[@]}" -H "Content-Type: application/json" -d '{"wait":60}'
curl -s $BASE/control/mcp/servers/fs/tools "${AUTH[@]}"
curl -s $BASE/control/mcp/status "${AUTH[@]}"
curl -s -X POST $BASE/control/mcp/subscribe "${AUTH[@]}" -H "Content-Type: application/json" \
     -d '{"agent_name":"alpha","server":"fs"}'
curl -s -X POST $BASE/control/mcp/unsubscribe "${AUTH[@]}" -H "Content-Type: application/json" \
     -d '{"agent_name":"alpha","server":"fs"}'
curl -s -X DELETE $BASE/control/mcp/servers/fs "${AUTH[@]}"
```

(Requires the optional `mcp` package; everything degrades gracefully
with `available:false` otherwise.) Subscribed servers' tool lists are
re-fetched automatically when the cache is older than the TTL, and the
registry file is re-read on every listing — a server added to
`mcp_servers.json` while the backend runs is immediately usable.

### 5.6 Endpoints / models / usage

```bash
curl -s $BASE/control/endpoints "${AUTH[@]}"                       # per-endpoint status + agents
curl -s "$BASE/control/endpoints/models?endpoint=local" "${AUTH[@]}"   # model dropdown (blocks a few s)
curl -s -X POST $BASE/control/endpoints/scan "${AUTH[@]}" -H "Content-Type: application/json" \
     -d '{"port":11434}'                                           # LAN Ollama scan (psutil)
curl -s $BASE/control/endpoints/usage "${AUTH[@]}"                 # /api/ps memory per endpoint
```

### 5.7 System

```bash
curl -s $BASE/health                       # no auth: status, uptime, active session, queue length
curl -s $BASE/system/info "${AUTH[@]}"     # same payload as /state
curl -s -X POST $BASE/system/shutdown "${AUTH[@]}" -H "Content-Type: application/json" \
     -d '{"confirm":true}'                 # graceful exit (final autosave)
```

---

## 6. The queue — your two special control endpoints

### 6.1 `POST /queue/sessions` — run a session (+ message before it)

```bash
curl -s -X POST $BASE/queue/sessions "${AUTH[@]}" -H "Content-Type: application/json" -d '{
  "session_name": "team-alpha",
  "message": "continue with the plan",
  "sender": "user",
  "activate_all": false,
  "activate": ["alpha"],
  "idle_timeout": null,
  "auto_finish": true
}'
```

Only `session_name` is required. When its turn comes, the backend:

1. finds the **newest autosave** for that name (then `sessions/`),
2. loads it, applies `activate`/`activate_all`,
3. queues `message` as a user message **before** the engines start,
4. runs the proven round-robin until the session quiesces
   (no agent running, pending queue empty, every agent inactive),
5. **autosaves** it and moves to the next entry — or waits for one.

Management:

```bash
curl -s $BASE/queue/sessions "${AUTH[@]}"                    # queue + active + recent + injections
curl -s $BASE/queue/state "${AUTH[@]}"                       # queue-focused state block
curl -s -X PATCH $BASE/queue/sessions/{id}/move "${AUTH[@]}" -H "Content-Type: application/json" \
     -d '{"position":0}'                                     # 0 = runs next
curl -s -X DELETE $BASE/queue/sessions/{id} "${AUTH[@]}"     # cancel one entry
curl -s -X DELETE $BASE/queue/sessions "${AUTH[@]}"          # cancel all pending
```

### 6.2 `POST /queue/inject` — inject into the next tool output (never enqueues)

```bash
curl -s -X POST $BASE/queue/inject "${AUTH[@]}" -H "Content-Type: application/json" -d '{
  "session_name": "team-alpha",
  "agent_name": "alpha",
  "text": "check the results and continue"
}'
# {"ok":true,"delivered_live":false,"injection":{…},"staged":[…]}
```

* session **running** → `delivered_live:true`; the text is prepended to
  that agent's **next tool result** (proven `tool_injection` semantics —
  it never persists into the history).
* session **not running** → staged (`delivered_live:false`); delivered
  automatically the next time that session runs.
* Either way the session is **never** added to the run queue.

```bash
curl -s $BASE/queue/inject "${AUTH[@]}"                      # staged + live counts
curl -s -X DELETE $BASE/queue/inject/{id} "${AUTH[@]}"       # remove a staged one
```

Shortcut while a session is running (the sidebar Inject button):

```bash
curl -s -X POST $BASE/control/agents/alpha/inject "${AUTH[@]}" -H "Content-Type: application/json" \
     -d '{"text":"check the results"}'
```

---

## 7. The built-in web console (`GET /`)

Open **`http://127.0.0.1:8000/`** in any browser — that's the whole
frontend. One self-contained `webui.html`, no Node, no build step,
works identically on Linux and Windows, and it uses exactly the API
described above (so it doubles as a reference implementation for your
future custom frontends):

* **Console** — every live event (chat, thinking/tool streams, status,
  lifecycle), filterable, with streaming token aggregation
* **Chat** — history + message box (`queue` / `send now`), user name,
  wait-user-input toggle (ON by default for new/loaded sessions),
  attach, pending-queue view. Every message that mentions a real
  local file ends with a "files — download" block: one clickable
  download link per file (`GET /files?…&download=1`), appended after
  the message text, each on its own line — the desktop app's inline
  multimedia behaviour, translated for the browser
* **Agents** — create / edit / rename / delete / activate / stop /
  inject, with a model picker fed by `/control/endpoints/models`.
  The New/Edit dialog has small side-box pickers:
  **names ▾** (single-select from `names.txt`, auto-random pre-fill),
  **tools ▾** (tree multi-select — tool files expand to their tools),
  **prompts ▾** and **mcp ▾** (multi-selects with descriptions)
* **Sessions** — new / load / everything on disk. One row per session
  NAME showing only its newest save; older saves roll down from the
  **Backups** dropdown in the row (pick one → its details and the Load
  button target it)
* **Queue** — active session card (finish/abort), enqueue form,
  reorder / cancel entries, recent results
* **Inject** — the §6.2 control with live/staged feedback
* **State** — the full `/state` JSON

It remembers the base URL + API key in `localStorage`, so you can also
open it from anywhere (e.g. served by backend A, pointing at backend B)
— handy for your multi-PC plans. `GET /info` shows the old static
directory page.

---

## 8. Status codes you may see

| Code | Meaning | Fix |
| --- | --- | --- |
| `400` | bad body (empty text, unknown agent field, invalid file…) | read `detail` |
| `401` | wrong/missing `X-API-Key` (or `?token=`) | set the header to `ML_API_KEY`'s value |
| `404` | unknown agent / session file / queue entry | check names via `/state` |
| `405` on OPTIONS | ~~no CORS~~ (fixed) — now only if you hit a wrong path+method | check `/control/overview` |
| `409` | no active session (see §1) — or a queued run blocks `session/new` | create/load/queue a session; finish the queued one |
| `4401` (WS) | WS auth failed — handshake rejected (HTTP 403) | put the key in the `?token=` query param |
| `502` | upstream Ollama call failed | check the endpoint/model |
| `503` | optional feature missing (`psutil` LAN scan, `mcp` engine) | `pip install psutil` / `pip install mcp` |

---

## 9. Legacy TCP events

The `send_event()` snippet produced by the core
`core_get_send_message_code` tool still works — point it at the TCP
port (default 8888). Messages enter the active
session's history with the `[Event]` prefix and are broadcast to every
output client.

---

## 10. Configuration (env vars)

| Variable | Default | Meaning |
| --- | --- | --- |
| `ML_BACKEND_HOST` | `0.0.0.0` | bind address |
| `ML_BACKEND_PORT` | `8000` | HTTP/WS port (default: 8000) |
| `ML_EVENT_TCP_PORT` | `8888` | legacy TCP event port (default: 8888) |
| `ML_CORS_ORIGINS` | `*` | allowed browser origins (§2) |
| `ML_API_KEY` | — | require auth |
| `ML_BACKEND_AUTOSAVE_KEEP` | `20` | autosaves kept |
| `ML_BACKEND_AUTOSAVE_MIN_INTERVAL` | `1.0` | mid-run autosave throttle (s) |
| `ML_SESSIONS_DIR` | `sessions` | manual session import folder |
| `ML_PREEMPT_INTERACTIVE` | `1` | queue may preempt interactive sessions |
| `ML_PREEMPT_TIMEOUT` | `600` | grace before forced abort (s) |
| `ML_ENDPOINT_DOWN_TIMEOUT` | `120` | all-endpoints-down finish trigger (s) |
| `ML_DONE_GRACE` | `3` | quiesce detector startup grace (s) |
| `ML_USAGE_POLL_INTERVAL` | `3` | `/api/ps` polling (s) |
| `ML_BACKEND_LOG_LEVEL` | `info` | uvicorn log level |

---

## 11. Multi-PC / multi-frontend recipes

* **Backend A drives backend B** (interconnection): A's agent calls a
  tool that does `requests.post(B + "/control/message", …)` or enqueues
  sessions on B via `/queue/sessions`. Every backend is symmetric.
* **Web frontend elsewhere**: serve your own page from any origin and
  rely on CORS (§2), or just use the built-in console (§7) pointed at
  the remote backend's URL.
* **Mobile**: the SSE endpoint (`/output/stream?token=`) plus plain
  REST works on every HTTP stack; WS when available.
* **Health monitoring**: poll `GET /health` (cheap, no auth).
