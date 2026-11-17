"""
server/api_files.py — GET /files: download / view any local file the
chat scan links to.

    GET /files?path=/home/me/out/chart.png                 (inline view)
    GET /files?path=/home/me/out/chart.png&download=1     (force download)
    GET /files?path=chart.png                             (bare name —
                                      resolved like the chat scan)

Auth follows the shared gate (ML_API_KEY): X-API-Key header OR
?token=… — the ?token= form exists precisely because <a href> links
in a browser page cannot set headers, so every download link the
web console renders stays clickable even when a key is set.

The reference is resolved with the proven app_paths resolver (cwd
first, then the app directory), exactly like the scan in
media_links.py — so a bare "output.png" written by an agent lands on
the right file even when the server was launched from another folder.

Content-Type is guessed from the extension; unknown types fall back
to application/octet-stream (which browsers download rather than try
to render — the safe default for arbitrary agent-produced files).
Range requests (video/audio seeking, resumable downloads) work: the
underlying FileResponse implements them.

Note on access scope: this endpoint can read ANY file on the machine
the backend runs on — exactly like the proven core_read_file tool any
agent already exposes. It is protected by the same optional API key
as the whole control plane; binding the server to 127.0.0.1
(ML_BACKEND_HOST) keeps it LAN/local-only when desired.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from .api_control import require_api_key
from .media_links import resolve_downloadable_path

router = APIRouter(tags=["files"])


@router.get("/files")
def serve_file(path: str = Query(..., min_length=1,
                                 description="absolute, ./-relative or bare file reference"),
              download: Optional[int] = Query(default=None, ge=0, le=1,
                                              description="1 = force attachment download"),
              app=Depends(require_api_key)):
    """Serve one local file (inline by default, attachment with ?download=1)."""
    resolved = resolve_downloadable_path(path)
    if not resolved or not os.path.isfile(resolved):
        raise HTTPException(status_code=404,
                            detail=f"file not found: {path}")
    filename = os.path.basename(resolved)
    attachment = bool(download)
    return FileResponse(
        resolved,
        filename=filename,
        content_disposition_type="attachment" if attachment else "inline",
    )
