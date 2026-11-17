"""
ollama_scan.py

LAN discovery of Ollama servers + per-endpoint model listing.

This module is an adaptation of the author's standalone `ollama-scan.py`
script, turned into a reusable library for the main application:

  * `get_local_subnet()`          - first non-loopback IPv4 network (psutil).
  * `check_ollama(ip, port)`      - async TCP probe of one host.
  * `scan_ollama_hosts(network)`  - concurrent probe of the whole subnet.
  * `get_models(ip, port)`        - model names from /api/tags (legacy helper).
  * `fetch_models_for_endpoint()` - model names for "local" / "ip:port" /
                                    "http://..." endpoint strings (GUI helper).
  * `scan_lan_ollama(port)`       - synchronous one-shot LAN scan returning
                                    a list of "ip:port" strings.

The GUI (EditAgentDialog) uses `scan_lan_ollama()` + `fetch_models_for_endpoint()`
in a background thread to populate the Endpoint / Model dropdowns so the user
can pick a machine and a model instead of typing them by hand. Manual typing
remains fully supported — the dropdowns are editable.

psutil is OPTIONAL: if it is not installed, `scan_lan_ollama()` raises a
clear RuntimeError and the GUI falls back to manual endpoint entry (and still
auto-fetches the model list for whatever endpoint is typed/selected).

Running this file directly reproduces the behaviour of the original
ollama-scan.py script:

    python ollama_scan.py
    Scanning 192.168.1.0/24 for Ollama on port 11434...
    Ollama hosts found: ['192.168.1.5', '192.168.1.9']
    192.168.1.5:
      - qwen3.5:9b
      - ...
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import socket
import urllib.request

# psutil is only needed to auto-detect the local subnet. Keep the import
# guarded so the rest of the module (model fetching for a typed endpoint)
# works even without it.
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    psutil = None
    HAVE_PSUTIL = False

DEFAULT_PORT = 11434
SCAN_TIMEOUT = 0.5        # seconds per TCP probe
SCAN_CONCURRENCY = 200    # parallel probes (same as the original script)
MAX_HOSTS = 8192          # safety cap for very large subnets (e.g. /16)


# ----------------------------------------------------------------------
# Subnet detection
# ----------------------------------------------------------------------
def get_local_subnet() -> ipaddress.IPv4Network:
    """Return the first non-loopback IPv4 network, e.g. 192.168.1.0/24.

    Same logic as the original ollama-scan.py: iterate network interfaces,
    take the first AF_INET address that is not 127.* and combine it with
    its netmask.
    """
    if not HAVE_PSUTIL:
        raise RuntimeError(
            "psutil is not installed - cannot auto-detect the local subnet. "
            "Install it with 'pip install psutil' or type the endpoint manually."
        )
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                ip = addr.address
                netmask = addr.netmask
                if netmask:
                    return ipaddress.IPv4Network(f"{ip}/{netmask}", strict=False)
    raise RuntimeError("Could not determine local subnet")


# ----------------------------------------------------------------------
# Host probing
# ----------------------------------------------------------------------
async def check_ollama(ip, port: int = DEFAULT_PORT, timeout: float = SCAN_TIMEOUT):
    """Probe a single host for an open Ollama port. Returns str(ip) or None."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(str(ip), port),
            timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return str(ip)
    except Exception:
        return None


async def scan_ollama_hosts(network: ipaddress.IPv4Network,
                            port: int = DEFAULT_PORT,
                            concurrency: int = SCAN_CONCURRENCY,
                            max_hosts: int = MAX_HOSTS):
    """Probe every host in *network* for Ollama, *concurrency* at a time.

    Returns the list of responding IP strings (in network order).
    """
    sem = asyncio.Semaphore(concurrency)

    async def limited_check(ip):
        async with sem:
            return await check_ollama(ip, port)

    tasks = []
    for i, ip in enumerate(network.hosts()):
        if i >= max_hosts:
            break
        tasks.append(asyncio.ensure_future(limited_check(ip)))
    results = await asyncio.gather(*tasks)
    return [ip for ip in results if ip]


# ----------------------------------------------------------------------
# Model listing
# ----------------------------------------------------------------------
def get_models(ip, port: int = DEFAULT_PORT, timeout: float = 2):
    """Legacy helper kept from the original script: model names for a raw IP.

    Returns a list of model name strings; on error the list contains a single
    "error: ..." entry (exactly like ollama-scan.py did).
    """
    url = f"http://{ip}:{port}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.load(resp)
        return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        return [f"error: {e}"]


def endpoint_to_base_url(endpoint: str) -> str:
    """Convert an endpoint string to a base URL for HTTP calls.

    Accepts:
      - "local" / "" / None  -> http://127.0.0.1:11434 (or OLLAMA_HOST env)
      - "ip:port"            -> http://ip:port
      - "host:port"          -> http://host:port
      - "http://..." / "https://..." -> unchanged
    """
    ep = (endpoint or "local").strip()
    if not ep or ep == "local":
        host = os.environ.get("OLLAMA_HOST", "").strip()
        if host:
            if not host.startswith(("http://", "https://")):
                host = "http://" + host
            return host.rstrip("/")
        return "http://127.0.0.1:11434"
    if ep.startswith(("http://", "https://")):
        return ep.rstrip("/")
    return f"http://{ep}"


def fetch_models_for_endpoint(endpoint: str, timeout: float = 3) -> list:
    """Model names advertised by the Ollama server at *endpoint*.

    *endpoint* may be "local", "ip:port" or a full "http://..." URL —
    anything the EditAgentDialog endpoint field accepts. Raises on
    connection errors so callers can decide how to report them.
    """
    base = endpoint_to_base_url(endpoint)
    url = f"{base}/api/tags"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.load(resp)
    names = []
    for m in data.get("models", []):
        name = m.get("name") or m.get("model")
        if name:
            names.append(name)
    return names


# ----------------------------------------------------------------------
# One-shot synchronous LAN scan (used by the GUI background thread)
# ----------------------------------------------------------------------
def scan_lan_ollama(port: int = DEFAULT_PORT,
                    concurrency: int = SCAN_CONCURRENCY) -> list:
    """Scan the local subnet for Ollama servers.

    Returns a list of "ip:port" endpoint strings (the same format the
    EditAgentDialog endpoint field uses). Raises RuntimeError when psutil
    is missing or the subnet cannot be determined.
    """
    network = get_local_subnet()
    hosts = asyncio.run(scan_ollama_hosts(network, port=port,
                                          concurrency=concurrency))
    return [f"{ip}:{port}" for ip in hosts]


# ----------------------------------------------------------------------
# CLI behaviour (identical in spirit to the original ollama-scan.py)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    network = get_local_subnet()
    print(f"Scanning {network} for Ollama on port {DEFAULT_PORT}...")

    hosts = asyncio.run(scan_ollama_hosts(network, port=DEFAULT_PORT))
    print("Ollama hosts found:", hosts)

    for ip in hosts:
        models = get_models(ip)
        print(f"\n{ip}:")
        for model in models:
            print(f"  - {model}")
