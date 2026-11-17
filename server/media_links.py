"""
server/media_links.py — chat-text file scan -> downloadable links.

WHY THIS MODULE EXISTS
======================
The proven PyQt app rendered agent answers through
markdown_renderer, which auto-detected local media paths in the
message text and turned them into inline image/audio/video widgets
(the "multimedia shown directly in the chat" behaviour).

The headless backend serves frontends over HTTP, where inline
file:// widgets are impossible (browsers block them). Instead, every
chat message that mentions a real file on disk is enriched with a
"files" array, and every entry carries a ready-made download URL
served by GET /files (api_files.py):

    {"sender": "alpha", "text": "Saved it as ./chart.png",
     "session": "team",                       <- unchanged
     "files": [{
         "name": "chart.png",
         "kind": "image",                     # image|audio|video|document|archive|file
         "size": 12345,                       # bytes
         "path": "/abs/dir/chart.png",        # resolved absolute path
         "url": "/files?path=%2Fabs%2F...",   # inline view
         "url_download": "/files?path=...&download=1"}]}

The message TEXT is never modified — history stays exactly what the
agents said (no link noise ever reaches the LLM context); only the
transport metadata gains the "files" array.

Scan provenance
===============
The two scanning passes and their regexes are copied VERBATIM from
markdown_renderer (the battle-tested PyQt scanner) and extended with
a document/archive extension set:

  1. head pass  — absolute Unix / Windows-drive / ./ ../ ~/ references;
  2. bare pass  — relative names like "output.png" or "media/clip.mp3",
                 accepted ONLY when the extension is recognised AND
                 the reference resolves to an EXISTING file, so plain
                 words ("see foo.py") are never mistaken for files.

Resolution (cwd first, then the app dir) is the proven
app_paths.resolve_media_path — bare names written by the core file
tools land in the process working directory, media sitting next to
the app is still found when the server was launched from elsewhere.

Where it is wired
=================
  * runner.py   — both chat publish sites (engine message hook +
                  context.message_router) call enrich_chat_event().
  * app.py      — the legacy TCP event handler calls enrich_chat_event().
  * api_output.py — GET /output/history attaches files to COPIES of
                  the stored messages (the shared history dicts are
                  never mutated — the LLM context must stay clean).

Thinking streams are NEVER scanned: they travel on the separate
"thinking" event route, so the scan naturally covers only final
answers (and user messages, which may also mention files).
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple
from urllib.parse import quote as _url_quote

import app_paths

# ----------------------------------------------------------------------
# Extension sets (media sets verbatim from markdown_renderer.py)
# ----------------------------------------------------------------------
IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".svg", ".ico", ".xbm", ".xpm",
}
AUDIO_EXTS = {
    ".mp3", ".wav", ".ogg", ".oga", ".flac", ".m4a", ".aac",
    ".wma", ".opus", ".aiff", ".aif",
}
VIDEO_EXTS = {
    ".mp4", ".m4v", ".webm", ".ogv", ".mkv", ".mov", ".avi",
    ".wmv", ".flv", ".mpg", ".mpeg", ".ts",
}
# Documents agents actually produce on disk (reports, data, exports).
# Bare-name matching still requires file EXISTENCE, so ordinary text
# ("version 2.0" — extension ".0") can never false-positive here.
DOCUMENT_EXTS = {
    ".pdf", ".txt", ".md", ".markdown", ".rst", ".rtf",
    ".json", ".csv", ".tsv", ".xml", ".yaml", ".yml", ".ini", ".toml",
    ".html", ".htm", ".css", ".js", ".ts", ".py", ".log",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".odp", ".epub",
}
ARCHIVE_EXTS = {
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".tgz", ".7z", ".rar",
}

_ALL_DOWNLOADABLE_EXTS = (set(IMAGE_EXTS) | set(AUDIO_EXTS)
                          | set(VIDEO_EXTS) | set(DOCUMENT_EXTS)
                          | set(ARCHIVE_EXTS))

# ----------------------------------------------------------------------
# Scanning regexes — VERBATIM copies from markdown_renderer.py
# ----------------------------------------------------------------------
# A local path is one of:
#   - absolute Unix path       /home/user/x.png
#   - absolute Windows path    C:\Users\me\x.png  or  C:/Users/me/x.png
#   - explicit relative path   ./x.png  ../x.png
#   - home-relative            ~/x.png
# Bare "x.png" is deliberately NOT matched here (it would over-match
# running text); the separate bare-name pass below handles it with an
# extension + existence gate.
_LOCAL_PATH_HEAD = re.compile(
    r'(?:'
    r'(?:[A-Za-z]:[\\/])'                # Windows drive C:\ or C:/
    r'|(?:/[A-Za-z0-9._\-+@~]+)'          # absolute Unix /something
    r'|(?:\.{1,2}/)'                     # ./ or ../
    r'|(?:~/)'                           # ~/
    r')'
)
# A path body char is anything that's commonly part of a path.
# Excludes whitespace and the usual "punctuation that ends a sentence".
_PATH_BODY = r'[^\s<>:"|?*\)\]\}]'
_PATH_BODY_RE = re.compile(_PATH_BODY)

# Bare RELATIVE reference — "output.png" or "media/clip.mp3" without
# any ./ ../ ~/ drive or /-root prefix. Negative lookbehind blocks
# matches inside a longer path / word / after a dot; negative
# lookahead blocks partial matches at the end of longer names. The
# extension + existence filters in scan_file_links make sure only
# REAL files are accepted.
_BARE_NAME_RE = re.compile(
    r'(?<![\w/\\.\-])'
    r'[A-Za-z0-9_\-][A-Za-z0-9._\-]*(?:/[A-Za-z0-9._\-]+)*'
    r'(?![\w/\\])'
)


def _downloadable_kind(name_or_path: str) -> str:
    """'image'/'audio'/'video'/'document'/'archive' for a download-
    able extension, '' otherwise. Does NOT check existence."""
    ext = os.path.splitext(name_or_path)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in DOCUMENT_EXTS:
        return "document"
    if ext in ARCHIVE_EXTS:
        return "archive"
    return ""


# ----------------------------------------------------------------------
# The scan
# ----------------------------------------------------------------------
def scan_file_links(text: str, max_links: int = 64) -> List[Dict[str, object]]:
    """Find local file references in *text* that point to real files.

    Returns a list of link descriptors (in first-appearance order,
    de-duplicated by resolved absolute path):

        {"name", "kind", "size", "path", "url", "url_download"}

    where ``url`` / ``url_download`` are RELATIVE URLs on this backend
    (``/files?path=…``); any frontend prefixes its own base URL.
    Non-media kinds are included too (documents, archives, data) —
    the gate is: recognised extension + the file EXISTS. Text with no
    file references (the overwhelming majority) returns [].
    """
    if not text:
        return []
    results: List[Dict[str, object]] = []
    seen_paths: set = set()
    seen_raw: set = set()

    def _add(candidate: str) -> None:
        if len(results) >= max_links:
            return
        resolved = app_paths.resolve_media_path(candidate)
        if not resolved or resolved in seen_paths:
            return
        kind = _downloadable_kind(resolved)
        if not kind:
            return
        try:
            size = os.path.getsize(resolved)
        except OSError:
            return
        seen_paths.add(resolved)
        quoted = _url_quote(resolved, safe="")
        results.append({
            "name": os.path.basename(resolved),
            "kind": kind,
            "size": size,
            "path": resolved,
            "url": f"/files?path={quoted}",
            "url_download": f"/files?path={quoted}&download=1",
        })

    # ---- pass 1: head-regex (absolute / ./ ../ ~/ references) --------
    i = 0
    n = len(text)
    while i < n:
        m = _LOCAL_PATH_HEAD.match(text, i)
        if not m:
            i += 1
            continue
        j = m.end()
        while j < n and _PATH_BODY_RE.match(text[j]):
            j += 1
        candidate = text[m.start():j]
        # Trim trailing sentence punctuation + backticks/quotes (the
        # proven trimmers — LLMs wrap paths in `backticks`).
        candidate = candidate.rstrip(".,;:!?)`\"'")
        candidate = candidate.lstrip("`\"'")
        if candidate and candidate not in seen_raw:
            seen_raw.add(candidate)
            _add(candidate)
        i = j

    # ---- pass 2: bare relative names (extension + existence gated) --
    for m in _BARE_NAME_RE.finditer(text):
        raw = m.group(0)
        raw = raw.rstrip(".,;:!?)`\"'")
        raw = raw.lstrip("`\"'")
        if not raw or raw in seen_raw:
            continue
        seen_raw.add(raw)
        if not _downloadable_kind(raw):
            continue          # fast reject: no downloadable extension
        _add(raw)

    return results


# ----------------------------------------------------------------------
# Enrichment helpers (used by runner / app / api_output)
# ----------------------------------------------------------------------
def enrich_chat_event(data: Dict[str, object]) -> Dict[str, object]:
    """Add a 'files' array to a chat EVENT dict (in place).

    The dict is the freshly-built broadcast payload (never shared
    history), so in-place mutation is safe. Thinking streams never
    come through here — only final chat messages.
    """
    try:
        files = scan_file_links(str(data.get("text") or ""))
    except Exception as exc:          # a scan failure must never kill a message
        print(f"[media_links] scan failed: {exc}")
        return data
    if files:
        data["files"] = files
    return data


def enrich_history_messages(messages: List[Dict[str, object]],
                            max_messages: int = 0) -> List[Dict[str, object]]:
    """Return COPIES of history messages with 'files' attached.

    The stored message dicts are shared with the conversation object
    (and therefore with the LLM context builder), so they must NEVER
    be mutated here — each message is shallow-copied first.

    max_messages <= 0 (default) enriches EVERY message — the caller
    (api_output) already sliced what it wants to send. A positive
    value caps at the oldest N (kept for direct callers that want a
    cheap scan of just the head of a huge list).
    """
    out: List[Dict[str, object]] = []
    if max_messages and max_messages > 0:
        messages = messages[:max_messages]
    for m in messages:
        try:
            msg = dict(m)
            files = scan_file_links(str(msg.get("content") or ""))
            if files:
                msg["files"] = files
        except Exception as exc:
            print(f"[media_links] history scan failed: {exc}")
            msg = dict(m)
        out.append(msg)
    return out


def resolve_downloadable_path(ref: str) -> str:
    """Resolve a /files?path=… reference to an existing absolute path.

    Same resolution order as the scan (proven app_paths resolver), so
    a frontend that only got the bare name from somewhere can still
    fetch it. Returns '' when nothing exists at the reference.
    """
    ref = (ref or "").strip().strip("\"'`")
    if not ref:
        return ""
    return app_paths.resolve_media_path(ref)
