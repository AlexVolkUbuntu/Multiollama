"""
app_paths.py — application-directory detection and media-path resolution.

Why this module exists
======================
The app is a portable folder that gets moved / renamed between versions.
Media files produced by agents (images, audio, video) are often referenced
in chat with bare relative names ("output.png") instead of full paths, and
absolute paths saved inside session files break once the app folder moves.

This module is the ONE central place that answers:

  * "Where does the app currently live?"        -> get_app_dir()
  * "Where do auto-saves live?"                 -> get_autosave_dir()
  * "Does this media reference point to a real
     file, and if so, at which absolute path?"  -> resolve_media_path()

Resolution order for a relative reference:
  1. the process working directory (where relative writes performed by
     the core file tools actually land), then
  2. the application directory (so bare names of files sitting next to
     the app are found even when the app was launched from elsewhere).

The detected app directory is computed ONCE at import time — the app
never moves while it is running (it only moves between versions), which
is exactly the scenario that session re-anchoring handles: session files
remember the app dir they were saved with (see gui_main
`_build_session_data` / `_reanchor_loaded_history`), and on load the
recorded absolute paths are remapped onto the CURRENT app dir.
"""

from __future__ import annotations

import os
import sys

# Sub-folder (inside the app dir) that holds automatic session saves.
# Auto-created on demand by get_autosave_dir().
AUTOSAVE_FOLDER_NAME = "autosave"

_APP_DIR: str = ""


def detect_app_dir() -> str:
    """Best-effort detection of the directory the app runs from.

    Order of preference:
      1. directory of the packaged executable (PyInstaller-style frozen
         apps — `sys.frozen`);
      2. directory containing this source file — i.e. the app folder
         itself, regardless of the working directory the app was
         launched from;
      3. the working directory as a last resort.
    """
    # 1. Frozen / packaged executable
    if getattr(sys, "frozen", False):
        try:
            base = os.path.dirname(os.path.abspath(sys.executable))
            if os.path.isdir(base):
                return base
        except Exception:
            pass
    # 2. Directory containing this source file — the app folder itself.
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        if os.path.isdir(base):
            return base
    except Exception:
        pass
    # 3. Last resort: current working directory.
    return os.getcwd()


def get_app_dir() -> str:
    """Return the app's own directory (detected once, then cached).

    The app path never changes *during* a session — only between
    versions — so caching is safe and keeps the detection logic from
    being re-run on every rendered message.
    """
    global _APP_DIR
    if not _APP_DIR:
        _APP_DIR = detect_app_dir()
    return _APP_DIR


def get_autosave_dir() -> str:
    """Return ``<app_dir>/autosave``, creating it when missing.

    Never raises — on failure the path is still returned and callers
    deal with the write error (an auto-save problem must never take
    the app down).
    """
    path = os.path.join(get_app_dir(), AUTOSAVE_FOLDER_NAME)
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as exc:
        print(f"[autosave] could not create folder {path}: {exc}")
    return path


def resolve_media_path(candidate: str) -> str:
    """Map a file reference found in chat text to an existing file.

    Returns the ABSOLUTE path of the file the reference points to, or
    ``''`` when nothing exists at that reference. (Whether the file is
    a media type at all is gated by the CALLER — the path scanner in
    markdown_renderer — so this resolver stays generally useful.)

    * Absolute references are checked as-is (after ``~`` expansion).
    * Relative references are tried against the working directory
      first, then against the app directory — this is what makes a
      bare ``output.png`` written by an agent next to the app show up
      in the chat / media list even though no full path was provided.
    * ``~``-relative references are expanded like a shell would.
    """
    ref = (candidate or "").strip()
    # Defensive: strip quotes / backticks that sometimes glue onto
    # paths inside LLM output. (The path scanner in markdown_renderer
    # already does this, but the resolver is also called directly.)
    ref = ref.strip("\"'`")
    if not ref:
        return ""
    ref = os.path.expanduser(ref)
    if not ref:
        return ""
    if os.path.isabs(ref):
        return ref if os.path.isfile(ref) else ""
    for base in (os.getcwd(), get_app_dir()):
        if not base:
            continue
        probe = os.path.normpath(os.path.join(base, ref))
        if os.path.isfile(probe):
            return probe
    return ""
