#!/usr/bin/env python3
"""
test_ml_mcp.py — smoke-test the standalone MCP server against a RUNNING
multillama backend.

It speaks raw JSON-RPC over the server's stdio (exactly what an MCP
client does — no `mcp` package needed HERE), runs a full workflow
through the tools, and prints a PASS / FAIL / WARN report:

    # 1) start a backend somewhere
    cd multillama_backend && python run_server.py --port 8200

    # 2) run this test
    python test_ml_mcp.py --backend-url http://127.0.0.1:8200

What is covered
===============
  * MCP initialize handshake + tools/list (63 tools expected)
  * health / state / overview
  * the 409-before-session error contract (the ONE rule)
  * session new / get / save / finish lifecycle
  * agent create / list / get / edit / save / load / delete
  * chat user name, wait-user-input toggle, send, pending queue,
    pending clear, history — including the message 'files' download
    arrays (the chat download-links feature) via MCP
  * files URL build + base64 fetch round-trip
  * ml_ask one-shot turn (real generation when Ollama is up; the
    structured timeout/error path when it is not — both valid)
  * run queue list / inject / inject list / inject cancel
  * tools / prompts / names / endpoints / endpoints models
  * the backend's own MCP registry (list / status)
  * ml_system_shutdown refusal without confirm=true

Exit code 0 = no FAILs. Ollama-dependent checks WARN (never FAIL) when
no Ollama answers on --ollama-url.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
PROTOCOL_VERSION = "2025-06-18"


# ======================================================================
# Minimal line-based JSON-RPC client for the MCP stdio transport
# ======================================================================
class McpStdioClient:
    """Spawn the MCP server and speak newline-delimited JSON-RPC."""

    def __init__(self, python: str, script: str, env: Dict[str, str]):
        self._next_id = 0
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.proc = subprocess.Popen(
            [python, script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
            env={**os.environ, **env},
        )
        self._reader = threading.Thread(target=self._read_loop,
                                        daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue          # stray non-JSON output; surfaced via stderr
            with self._lock:
                rid = msg.get("id")
                if rid is not None:
                    self._pending[rid] = msg

    def request(self, method: str, params: Optional[dict] = None,
                timeout: float = 30.0) -> Dict[str, Any]:
        self._next_id += 1
        rid = self._next_id
        msg = {"jsonrpc": "2.0", "id": rid, "method": method,
               "params": params or {}}
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if rid in self._pending:
                    return self._pending.pop(rid)
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server died while waiting for {method} "
                    f"(id={rid}); stderr:\n{self._stderr()}")
            time.sleep(0.02)
        raise TimeoutError(f"no reply within {timeout}s for {method}")

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def initialize(self) -> Dict[str, Any]:
        resp = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "ml-mcp-test", "version": "1.0"},
        })
        if "error" in resp:
            raise RuntimeError(f"initialize failed: {resp['error']}")
        self.notify("notifications/initialized")
        return resp["result"]

    def list_tools(self) -> List[Dict[str, Any]]:
        resp = self.request("tools/list")
        return resp.get("result", {}).get("tools", [])

    def call_tool(self, name: str, args: Optional[dict] = None,
                  timeout: float = 30.0) -> Dict[str, Any]:
        try:
            resp = self.request("tools/call",
                                {"name": name, "arguments": args or {}},
                                timeout=timeout)
        except TimeoutError as exc:
            return {"ok": False, "error_kind": "test_timeout",
                    "detail": str(exc), "tool": name}
        if "error" in resp:
            return {"ok": False, "error_kind": "protocol",
                    "detail": json.dumps(resp["error"])[:400]}
        result = resp.get("result", {})
        if result.get("isError"):
            texts = [c.get("text", "") for c in result.get("content", [])
                     if c.get("type") == "text"]
            return {"ok": False, "error_kind": "tool",
                    "detail": " ".join(texts)[:400]}
        # FastMCP: dict returns arrive both as JSON text content and as
        # structuredContent — prefer the structured one.
        if "structuredContent" in result:
            sc = result["structuredContent"]
            return sc if isinstance(sc, dict) else {"ok": True, "data": sc}
        for c in result.get("content", []):
            if c.get("type") == "text":
                try:
                    return json.loads(c.get("text", ""))
                except json.JSONDecodeError:
                    return {"ok": True, "data": c.get("text", "")}
        return {"ok": True, "data": None}

    def close(self) -> None:
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def _stderr(self) -> str:
        try:
            assert self.proc.stderr is not None
            data = self.proc.stderr.read() or ""
            return data[-4000:]
        except Exception:
            return "<unreadable>"


# ======================================================================
# Test driver
# ======================================================================
class TestReport:
    def __init__(self) -> None:
        self.passed = 0
        self.warned = 0
        self.failed = 0

    def check(self, ok: bool, label: str, detail: str = "",
              warn_only: bool = False) -> bool:
        if ok:
            self.passed += 1
            print(f"  PASS  {label}")
        elif warn_only:
            self.warned += 1
            print(f"  WARN  {label}" + (f"  ({detail})" if detail else ""))
        else:
            self.failed += 1
            print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))
        return ok


def _ollama_up(url: str) -> bool:
    host = (url.split("://", 1)[-1] if "://" in url else url).split("/")[0]
    host, _, port = host.partition(":")
    port = port or "11434"
    try:
        with socket.create_connection((host, int(port)), timeout=1.5):
            return True
    except OSError:
        return False


def _tcp_event(port: int, sender: str, message: str) -> Dict[str, Any]:
    """One legacy TCP event push (the send_event() snippet protocol:
    one JSON per connection, reply is JSON + newline)."""
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(json.dumps({"sender": sender,
                              "message": message}).encode("utf-8"))
        # the server reads until EOF before it replies — half-close the
        # write side exactly like the proven send_event() snippet does
        # by exiting its `with socket` block after sendall
        s.shutdown(socket.SHUT_WR)
        s.settimeout(5)
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
    try:
        return json.loads(data.decode("utf-8", "replace").strip())
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--backend-url",
                        default=os.environ.get("ML_BACKEND_URL",
                                                "http://127.0.0.1:8200"))
    parser.add_argument("--ollama-url",
                        default="http://127.0.0.1:11434")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--server-script",
                        default=str(HERE / "ml_mcp_server.py"))
    parser.add_argument("--file-dir", default=None,
                        help="a folder the BACKEND resolves bare file names "
                             "against (its working directory or app "
                             "folder) — enables the bare-name checks")
    parser.add_argument("--session-name", default="mcp-probe")
    parser.add_argument("--event-port", type=int,
                        default=int(os.environ.get("ML_EVENT_TCP_PORT",
                                                   "8888")),
                        help="backend legacy TCP event port (the "
                             "engine-free history check)")
    args = parser.parse_args()

    report = TestReport()
    ollama = _ollama_up(args.ollama_url)
    print(f"backend: {args.backend_url}   "
          f"ollama: {'up' if ollama else 'DOWN (generation tests WARN)'}")

    env = {"ML_BACKEND_URL": args.backend_url, "ML_MCP_API_KEY": "",
           "ML_API_KEY": ""}
    client = McpStdioClient(args.python, args.server_script, env)

    try:
        # ---- 1. MCP handshake -----------------------------------------
        try:
            init = client.initialize()
            server_name = init.get("serverInfo", {}).get("name", "")
            report.check(server_name == "multillama-backend",
                         "MCP initialize handshake",
                         f"serverInfo={init.get('serverInfo')}")
        except Exception as exc:
            report.check(False, "MCP initialize handshake", str(exc))
            print("\nserver stderr tail:\n" + client._stderr())
            return 1

        tools = client.list_tools()
        names = sorted(t.get("name", "") for t in tools)
        report.check(len(tools) >= 60, "tools/list exposes the full surface",
                     f"{len(tools)} tools")
        report.check(all(n.startswith("ml_") for n in names) and
                     all(t.get("description") for t in tools),
                     "every tool is ml_-prefixed and documented")
        print(f"        ({len(tools)} tools: "
              f"{', '.join(names[:6])}, …)")

        # ---- 2. health / state / overview -----------------------------
        h = client.call_tool("ml_backend_health")
        report.check(h.get("status") == "ok", "ml_backend_health",
                     json.dumps(h)[:200])

        st = client.call_tool("ml_backend_state")
        report.check("server" in st and "active_session" in st
                     and "tools" in st, "ml_backend_state (brief)",
                     json.dumps(st)[:200])

        ov = client.call_tool("ml_backend_overview")
        report.check(isinstance(ov, dict) and len(ov) >= 5,
                     "ml_backend_overview", json.dumps(ov)[:200])

        # ---- 3. the 409 rule + safe session probe ----------------------
        sg = client.call_tool("ml_session_get")
        report.check(sg.get("active") is False or
                     sg.get("active") is True and "session_name" in sg,
                     "ml_session_get never 409s (idle-safe)",
                     json.dumps(sg)[:200])

        # if a session is somehow already active, finish it first so the
        # 409 contract can be demonstrated deterministically
        if sg.get("active"):
            client.call_tool("ml_session_finish")
            time.sleep(1)

        al409 = client.call_tool("ml_agents_list")
        report.check(al409.get("ok") is False and
                     al409.get("status") == 409 and
                     "session" in str(al409.get("hint", "")).lower(),
                     "409-before-session error contract carries a hint",
                     json.dumps(al409)[:200])

        # ---- 4. session lifecycle --------------------------------------
        sn = client.call_tool("ml_session_new", {
            "name": args.session_name, "user_name": "McpTester",
            "wait_user_input": True})
        report.check(sn.get("ok") is True, "ml_session_new",
                     json.dumps(sn)[:200])

        # ---- 5. agents -------------------------------------------------
        ac = client.call_tool("ml_agent_create", {
            "name": "alpha", "system_prompt": "You are the team lead.",
            "description": "lead", "model": "qwen3.5:9b",
            "endpoint": "local"})
        report.check(ac.get("ok") is True, "ml_agent_create alpha",
                     json.dumps(ac)[:200])
        ac2 = client.call_tool("ml_agent_create", {
            "name": "beta", "system_prompt": "You are the coder.",
            "description": "coder"})
        report.check(ac2.get("ok") is True, "ml_agent_create beta",
                     json.dumps(ac2)[:200])

        al = client.call_tool("ml_agents_list")
        agent_names = [a.get("name") for a in al.get("agents", [])] \
            if isinstance(al.get("agents"), list) else list(
                (al.get("agents") or {}).keys() or [])
        report.check(set(agent_names) >= {"alpha", "beta"},
                     "ml_agents_list", str(agent_names))

        ag = client.call_tool("ml_agent_get", {"name": "alpha"})
        agent = ag.get("agent") or {}
        report.check(agent.get("name") == "alpha"
                     and "system_prompt" in agent,
                     "ml_agent_get", json.dumps(ag)[:200])

        ae = client.call_tool("ml_agent_edit",
                              {"name": "alpha", "temperature": 0.7})
        report.check(ae.get("ok") is True, "ml_agent_edit (PATCH)",
                     json.dumps(ae)[:200])

        # ---- 6. chat basics --------------------------------------------
        cu = client.call_tool("ml_chat_user", {"name": "McpTester"})
        report.check(cu.get("user_name") == "McpTester", "ml_chat_user set",
                     json.dumps(cu)[:200])
        cu2 = client.call_tool("ml_chat_user")
        report.check(cu2.get("user_name") == "McpTester", "ml_chat_user get",
                     json.dumps(cu2)[:200])
        pw = client.call_tool("ml_chat_wait_user", {"paused": True})
        report.check(pw.get("paused_user_input") is True,
                     "ml_chat_wait_user set", json.dumps(pw)[:200])

        # ---- 7. files: URL build + base64 fetch round-trip --------------
        # absolute paths are recognised by the chat scan and the /files
        # resolver regardless of the backend's working directory; the
        # bare-name variant is additionally tested when --file-dir points
        # at a folder the backend resolves bare names against.
        payload = base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
            b"nGNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==")
        probe_dir = Path(args.file_dir or
                          Path(args.server_script).resolve().parent)
        probe_file = probe_dir / "mcp_probe_media.png"
        probe_file.write_bytes(payload)
        abs_path = str(probe_file.resolve())
        try:
            fu = client.call_tool("ml_files_url", {"path": abs_path})
            report.check(fu.get("ok") is True and
                         "/files?path=" in str(fu.get("url_download", "")),
                         "ml_files_url (absolute path)",
                         json.dumps(fu)[:200])

            ff = client.call_tool("ml_files_fetch", {"path": abs_path})
            same = (base64.b64encode(payload).decode()
                    == str(ff.get("base64", "")))
            report.check(same and ff.get("ok") is True,
                         "ml_files_fetch base64 round-trip",
                         f"size={ff.get('size')} mime={ff.get('mime')}")

            nf = client.call_tool("ml_files_fetch",
                                  {"path": str(probe_dir /
                                               "no_such_file_xyz.png")})
            report.check(nf.get("ok") is False and nf.get("status") == 404,
                         "ml_files_fetch 404 contract",
                         json.dumps(nf)[:200])

            if args.file_dir:
                fb = client.call_tool("ml_files_fetch",
                                      {"path": "mcp_probe_media.png"})
                report.check(fb.get("ok") is True,
                             "ml_files_fetch bare name (--file-dir)",
                             json.dumps(fb)[:200])

            # land a file-mentioning message in history WITHOUT engines
            # (the proven send_event() TCP snippet), then read it back
            # through MCP: the history copy AND the live chat event both
            # carry the files download array
            ev = _tcp_event(args.event_port, "mcp-tester",
                            f"[probe] chart saved at {abs_path}")
            report.check(ev.get("ok") is True,
                         "legacy TCP event accepted (engine-free "
                         "history write)", json.dumps(ev)[:200])
            ch = client.call_tool("ml_chat_history", {"limit": 50})
            msgs = ch.get("messages", [])
            with_files = [m for m in msgs if m.get("files")]
            report.check(bool(msgs), "ml_chat_history returns messages",
                         f"total={ch.get('total')}")
            report.check(any("mcp_probe_media.png"
                             in json.dumps(m.get("files", []))
                             for m in with_files),
                         "history message carries the 'files' download array",
                         "the chat download-links feature, seen through MCP")
            rec = client.call_tool("ml_output_recent", {"limit": 60})
            evs = rec.get("events", [])
            chat_with_files = [e for e in evs
                               if isinstance(e, dict)
                               and e.get("type") == "chat"
                               and isinstance(e.get("data"), dict)
                               and e.get("data", {}).get("files")]
            report.check(bool(chat_with_files),
                         "live chat event carries the 'files' array "
                         "(ml_output_recent)",
                         f"{len(evs)} events buffered")
        finally:
            probe_file.unlink(missing_ok=True)

        # ---- 8. chat send / pending / history (with files arrays) ------
        cs = client.call_tool("ml_chat_send", {
            "text": "please look at ./mcp_probe_media.png and describe it",
            "mode": "queue"})
        report.check(cs.get("ok") is True, "ml_chat_send (queue)",
                     json.dumps(cs)[:200])
        pq = client.call_tool("ml_chat_pending")
        report.check(isinstance(pq.get("pending_user_messages"), list),
                     "ml_chat_pending", json.dumps(pq)[:200])
        pc = client.call_tool("ml_chat_pending_clear")
        report.check(pc.get("ok") is True, "ml_chat_pending_clear",
                     json.dumps(pc)[:200])

        cs2 = client.call_tool("ml_chat_send", {
            "text": f"i saved the chart as {abs_path} — check it",
            "mode": "send_now"})
        report.check(cs2.get("ok") is True, "ml_chat_send (2nd, send_now)",
                     json.dumps(cs2)[:200])

        # ---- 9. wait_idle / ml_ask -------------------------------------
        wi = client.call_tool("ml_wait_idle",
                              {"timeout_seconds": 15, "poll_seconds": 0.5},
                             timeout=30)
        report.check(wi.get("idle") is True, "ml_wait_idle",
                     json.dumps(wi)[:300])

        ask_args = {"text": "hello team — introduce yourselves in one line",
                    "timeout_seconds": 60 if ollama else 20,
                    "poll_seconds": 1,
                    "mode": "send_now"}
        ask = client.call_tool("ml_ask", ask_args, timeout=90 if ollama else 40)
        if ollama:
            report.check(ask.get("ok") is True and
                         int(ask.get("appended") or 0) >= 2,
                         "ml_ask one-shot turn (live Ollama)",
                         json.dumps({k: v for k, v in ask.items()
                                     if k != "messages"})[:300])
        else:
            report.check("wait" in ask or ask.get("ok") is False,
                         "ml_ask returns a structured result without Ollama",
                         json.dumps({k: v for k, v in ask.items()
                                     if k != "messages"})[:300],
                         warn_only=True)

        # ---- 10. run queue + inject ------------------------------------
        ql = client.call_tool("ml_queue_list")
        report.check(isinstance(ql, dict), "ml_queue_list",
                     json.dumps(ql)[:200])
        qi = client.call_tool("ml_queue_inject", {
            "session_name": args.session_name, "agent_name": "alpha",
            "text": "operator check-in"})
        report.check(qi.get("ok") is True and
                     qi.get("delivered_live") in (True, False),
                     "ml_queue_inject (live or staged)",
                     json.dumps(qi)[:200])
        qil = client.call_tool("ml_queue_inject_list")
        staged = qil.get("staged") or []
        live = qil.get("live") or {}
        report.check(bool(staged) or bool(live),
                     "ml_queue_inject_list (staged or live)",
                     json.dumps(qil)[:200])
        if staged:
            inj_id = staged[0].get("id") or staged[0].get("injection_id")
            if inj_id:
                qc = client.call_tool("ml_queue_inject_cancel",
                                      {"injection_id": inj_id})
                report.check(qc.get("ok") is True,
                             "ml_queue_inject_cancel",
                             json.dumps(qc)[:200])

        # ---- 11. tools / prompts / names / endpoints -------------------
        tl = client.call_tool("ml_tools_list")
        report.check("tool_files" in tl, "ml_tools_list",
                     json.dumps(tl)[:200])
        pl = client.call_tool("ml_prompts_list")
        report.check("prompts" in pl, "ml_prompts_list",
                     json.dumps(pl)[:200])
        nl = client.call_tool("ml_names_list")
        report.check("names" in nl, "ml_names_list", json.dumps(nl)[:200])
        el = client.call_tool("ml_endpoints_list")
        report.check(isinstance(el, dict) and el, "ml_endpoints_list",
                     json.dumps(el)[:200])
        em = client.call_tool("ml_endpoints_models", {"endpoint": "local"},
                             timeout=20)
        report.check(bool(em.get("models")) or
                     (em.get("ok") is False and em.get("status") == 502),
                     "ml_endpoints_models (or its 502 contract)",
                     json.dumps(em)[:200], warn_only=not ollama)

        # ---- 12. the backend's own MCP registry -------------------------
        ms = client.call_tool("ml_mcp_servers_list")
        report.check(ms.get("available") is True or
                     ms.get("ok") is False,
                     "ml_mcp_servers_list", json.dumps(ms)[:300],
                     warn_only=True)
        mst = client.call_tool("ml_mcp_status")
        report.check(isinstance(mst, dict), "ml_mcp_status",
                     json.dumps(mst)[:200])

        # ---- 13. agent save / load round-trip ---------------------------
        asv = client.call_tool("ml_agent_save", {"name": "alpha"})
        report.check(asv.get("ok") is True, "ml_agent_save",
                     json.dumps(asv)[:200])
        path = asv.get("path") or asv.get("saved_to") or ""
        if path:
            al2 = client.call_tool("ml_agent_load",
                                   {"path": path, "new_name": "alpha2"})
            report.check(al2.get("ok") is True, "ml_agent_load",
                         json.dumps(al2)[:200])
            if al2.get("ok") is True:
                client.call_tool("ml_agent_delete", {"name": "alpha2"})

        # ---- 14. session save + finish + shutdown refusal ---------------
        sv = client.call_tool("ml_session_save")
        report.check(sv.get("ok") is True, "ml_session_save",
                     json.dumps(sv)[:200])
        sh = client.call_tool("ml_system_shutdown")
        report.check(sh.get("ok") is False and
                     sh.get("error_kind") == "usage",
                     "ml_system_shutdown refuses without confirm",
                     json.dumps(sh)[:200])
        fin = client.call_tool("ml_session_finish")
        report.check(fin.get("ok") is True, "ml_session_finish",
                     json.dumps(fin)[:200])

    finally:
        client.close()

    total = report.passed + report.warned + report.failed
    print(f"\n{report.passed} PASS  {report.warned} WARN  "
          f"{report.failed} FAIL  ({total} checks)")
    if report.failed:
        print("\nSOME CHECKS FAILED — rerun with the backend log visible "
              "and inspect the FAIL lines above.")
        return 1
    print("\nALL GREEN — the MCP server drives the backend correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
