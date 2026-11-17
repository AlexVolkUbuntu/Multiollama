"""
markdown_renderer.py

Converts plain-text LLM output into rich HTML suitable for QTextBrowser.

Features:
  - Fenced code blocks (```lang\\n...``` or ~~~lang\\n...~~~) with monospace
    styling and a soft background. The language hint is preserved as a small
    label in the corner of the block so the user can see what language the
    model thought it was writing.
  - Inline code (`code`) — rendered with the same monospace style.
  - Markdown bold (**text** / __text__), italic (*text* / _text_),
    strikethrough (~~text~~), inline link [text](url).
  - ATX headers (# H1 .. ###### H6).
  - Bullet lists (- / * / +) and numbered lists (1. / 2.) rendered as
    indented blocks with the marker preserved (so a long item still lines
    up under its bullet, which Qt's <ul>/<li> doesn't always do well).
  - Block quotes (> ...).
  - Horizontal rules (--- / *** on their own line).
  - Auto-linking of bare URLs (http://, https://, ftp://).
  - Auto-detection of local file paths that point to existing files of
    known image / audio / video types. Those are turned into clickable
    inline widgets:
        * Images      -> <a href=imageview://path><img src=file://path width=200></a>
        * Audio files -> <a href=audio://path>[Play audio: name]</a>
        * Video files -> <a href=video://path>[Play video: name]</a>
  - Bare RELATIVE media names ("output.png", "media/clip.mp3" — no ./ ~/
    or absolute prefix) are resolved against the app directory (see
    app_paths.resolve_media_path) and rendered the same way, so media
    written next to the app shows up even when the agent never gave a
    full path. Only references that resolve to an EXISTING file count,
    so ordinary words are never mistaken for media.
  - Plain-text paths that point to existing files of unknown type are
    left alone (they may be code, source files, etc. — turning them
    into clickable links would be misleading).

The renderer is deliberately dependency-free (no `markdown` library) so
the user does not have to install anything extra.

Public API:
    render_message_to_html(text: str) -> str

Returned HTML is a fragment (no <html>/<body> wrapper) suitable for
QTextBrowser.append() or QTextBrowser.setHtml().

The anchor schemes used for media are:
    imageview://<absolute-path>     - left-click opens the image viewer dialog
    audio://<absolute-path>         - left-click opens the audio player dialog
    video://<absolute-path>         - left-click opens the video player dialog
                                       (silent fallback to OS player if Qt
                                        cannot decode the format)

Session path re-anchoring (app moved between versions):
    reanchor_text_paths(text, old_base, new_base)
        rewrites absolute MEDIA paths recorded under a previous app
        location onto the current one, so media references inside saved
        sessions keep working after the app folder moves / is renamed.

For external http(s)/ftp links a normal <a href="..."> tag is emitted —
QTextBrowser will emit anchorClicked and our handler routes it to the
OS browser via QDesktopServices.
"""

from __future__ import annotations

import html as _html
import os
import re
from typing import List, Tuple
from urllib.parse import quote as _url_quote

import app_paths

# ----------------------------------------------------------------------
# Media type definitions
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

# All recognised media extensions in one set — used by the bare-name
# pass in _extract_local_paths and by reanchor_text_paths.
_ALL_MEDIA_EXTS = set(IMAGE_EXTS) | set(AUDIO_EXTS) | set(VIDEO_EXTS)


def _has_media_extension(name: str) -> bool:
    """True when *name* ends with a recognised image/audio/video ext."""
    return os.path.splitext(name)[1].lower() in _ALL_MEDIA_EXTS


# A local path is one of:
#   - absolute Unix path       /home/user/x.png
#   - absolute Windows path    C:\Users\me\x.png  or  C:/Users/me/x.png
#   - explicit relative path   ./x.png  ../x.png
#   - home-relative            ~/x.png
# We deliberately do NOT match bare "x.png" (no leading ./ or ~/) in THIS
# regex because it would over-match things like "see foo.py" inside
# running text. Bare media names are handled by a SEPARATE, extension-
# gated pass in _extract_local_paths (see _BARE_MEDIA_NAME_RE) which only
# accepts names that resolve to an existing media file.
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

# Bare RELATIVE media reference — "output.png" or "media/clip.mp3"
# without any ./ ../ ~/ drive or /-root prefix. The negative lookbehind
# blocks matches that sit inside a longer path (…/output.png), inside a
# word (myoutput.png) or right after a dot (hidden files / "e.g.");
# the negative lookahead blocks partial matches at the end of longer
# names. The extension + existence filters in _extract_local_paths then
# make sure only REAL media files are accepted.
_BARE_MEDIA_NAME_RE = re.compile(
    r'(?<![\w/\\.\-])'
    r'[A-Za-z0-9_\-][A-Za-z0-9._\-]*(?:/[A-Za-z0-9._\-]+)*'
    r'(?![\w/\\])'
)

# URL regex — match http(s)://, ftp:// and www. bare links.
_URL_RE = re.compile(
    r'(?P<url>'
    r'(?:https?://|ftp://)'                    # explicit scheme
    r'[^\s<>"]+'
    r'|'
    r'www\.[^\s<>"]+\.[^\s<>"]+'              # bare www.example.com
    r')',
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Code-block fence extraction
# ----------------------------------------------------------------------
# Match a complete fenced code block: opening fence (``` or ~~~), optional
# language tag, content (lazy), closing fence.
_FENCE_RE = re.compile(
    r'(?P<fence>```|~~~)(?P<lang>[^\n`]*)\n(?P<code>.*?)^(?P=fence)[ \t]*\n?',
    re.DOTALL | re.MULTILINE,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _file_url(path: str) -> str:
    """Convert a local path to a file:// URL safe to embed in HTML."""
    abs_path = os.path.abspath(os.path.expanduser(path))
    # QUrl.fromLocalFile would do this, but we want a plain string here.
    return "file://" + _url_quote(abs_path, safe="/:\\")


def _media_url(scheme: str, path: str) -> str:
    """Build a custom-scheme anchor URL carrying an absolute local path.

    Examples:
        imageview:///home/user/pic.png
        audio:///home/me/clip.mp3
        video:///home/me/clip.mp4
    """
    abs_path = os.path.abspath(os.path.expanduser(path))
    # Custom-scheme URLs need the path to be encoded so spaces / unicode
    # survive QTextBrowser's QUrl round-trip.
    return scheme + "://" + _url_quote(abs_path, safe="/:\\")


def _detect_media_kind(path: str) -> str:
    """Return 'image' / 'audio' / 'video' / '' for a local file path.

    Returns '' when the file does not exist or its extension is not a
    recognised media type.
    """
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        return ""
    ext = os.path.splitext(expanded)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    return ""


def _extract_local_paths(text: str) -> List[Tuple[str, str, str]]:
    """Find local file references in *text* that point to real media files.

    Returns a list of ``(raw, resolved, kind)`` tuples in source order,
    de-duplicated by the raw reference:

      * *raw*      — the reference exactly as it appears in the text
                     (``./x.png``, ``/tmp/a.wav``, ``output.png`` …).
                     Used to build the text-replacement regex.
      * *resolved* — the EXISTING ABSOLUTE path the reference maps to
                     (see app_paths.resolve_media_path). Used to build
                     the viewer / player links, so relative references
                     keep working even when the process working
                     directory differs from the app directory.
      * *kind*     — 'image' / 'audio' / 'video'.

    Two scanning passes run over the text:

      1. the classic head-regex pass (absolute / ./ ../ ~/ references);
      2. a bare-name pass for relative references like ``output.png``
         or ``media/clip.mp3`` — accepted ONLY when the name has a
         recognised media extension AND resolves to an existing file
         (working directory first, then the app directory), so plain
         words such as "see foo.py" are never mistaken for media.
    """
    results: List[Tuple[str, str, str]] = []
    seen = set()
    i = 0
    n = len(text)
    while i < n:
        m = _LOCAL_PATH_HEAD.match(text, i)
        if not m:
            i += 1
            continue
        # Walk forward collecting path-body chars.
        j = m.end()
        while j < n and _PATH_BODY_RE.match(text[j]):
            j += 1
        candidate = text[m.start():j]
        # Trim trailing punctuation that often attaches to paths in
        # natural-language sentences ("see /tmp/foo.png," or "(see ./x.png)").
        # ALSO strip backticks and quotes — LLMs love to wrap file paths
        # in backticks like `` `/tmp/clip.wav` `` and without stripping,
        # the trailing backtick would glue onto the extension and break
        # the ``.wav`` -> ``audio`` lookup.
        candidate = candidate.rstrip(".,;:!?)`\"'")
        # And strip leading backticks / quotes too (rare but possible
        # if the head regex somehow caught one).
        candidate = candidate.lstrip("`\"'")
        if candidate and candidate not in seen:
            seen.add(candidate)
            resolved = app_paths.resolve_media_path(candidate)
            if resolved:
                kind = _detect_media_kind(resolved)
                if kind:
                    results.append((candidate, resolved, kind))
        i = j

    # Second pass: bare relative media names ("output.png"). Gated by
    # media extension + file existence, so ordinary running text is
    # never touched. Trailing sentence punctuation ("…output.png.",
    # "…clip.mp3!") is stripped the same way the head pass does, so a
    # period at the end of a sentence cannot glue onto the extension
    # and break the kind lookup.
    for m in _BARE_MEDIA_NAME_RE.finditer(text):
        raw = m.group(0)
        # Only the chars that the regex class itself can over-capture
        # need trimming here ('.' and '-'); the rest are harmless.
        raw = raw.rstrip(".,;:!?)`\"'")
        raw = raw.lstrip("`\"'")
        if not raw or raw in seen:
            continue
        seen.add(raw)
        if not _has_media_extension(raw):
            continue
        resolved = app_paths.resolve_media_path(raw)
        if resolved:
            kind = _detect_media_kind(resolved)
            if kind:
                results.append((raw, resolved, kind))

    return results


def _render_image(path: str) -> str:
    """HTML fragment for an image: clickable thumbnail that opens the
    full-size viewer. Width is capped at 200px so chat stays readable."""
    fileurl = _file_url(path)
    media = _media_url("imageview", path)
    name = os.path.basename(path)
    return (
        f'<a href="{media}" style="text-decoration:none;">'
        f'<img src="{fileurl}" width="200" '
        f'style="border:1px solid #ccc; border-radius:3px; vertical-align:middle;" '
        f'alt="{_html.escape(name)}" />'
        f'</a>'
        f' <span style="color:#888; font-size:11px;">[click to view full size]</span>'
    )


def _render_audio(path: str) -> str:
    """HTML fragment for an audio file: clickable link that opens the
    in-app audio player dialog."""
    media = _media_url("audio", path)
    name = os.path.basename(path)
    return (
        f'<a href="{media}" style="background:#e3f2fd; color:#1565c0; '
        f'padding:2px 6px; border-radius:3px; text-decoration:none; '
        f'font-weight:bold;">&#9654; Play audio: {_html.escape(name)}</a>'
    )


def _render_video(path: str) -> str:
    """HTML fragment for a video file: clickable link that opens the
    in-app video player. Falls back silently to the OS default player
    if Qt cannot decode the format (handled by the click handler)."""
    media = _media_url("video", path)
    name = os.path.basename(path)
    return (
        f'<a href="{media}" style="background:#ede7f6; color:#4527a0; '
        f'padding:2px 6px; border-radius:3px; text-decoration:none; '
        f'font-weight:bold;">&#9658; Play video: {_html.escape(name)}</a>'
    )


def _render_url(url: str) -> str:
    """HTML fragment for an external URL.

    The *url* is taken from text that has already been HTML-escaped by
    ``render_message_to_html`` (so ``&`` is already ``&amp;`` etc.).
    We therefore must NOT re-escape it here — doing so would produce
    double-escaped entities like ``&amp;amp;``. We just wrap it in an
    <a> tag verbatim.
    """
    # ``url`` here is already escaped. Decide whether to prepend http://
    # by looking at the original (case-insensitive) — but we have to
    # check against the escaped form, which is identical for these
    # ASCII prefixes.
    if url.lower().startswith(("http://", "https://", "ftp://")):
        href = url
    elif url.lower().startswith("www."):
        href = "http://" + url
    else:
        href = "http://" + url
    return f'<a href="{href}" style="color:#1976d2;">{url}</a>'


# ----------------------------------------------------------------------
# In-text media replacement
# ----------------------------------------------------------------------
def _replace_media_in_text(text: str) -> str:
    """Replace occurrences of local media references in *text* with the
    appropriate HTML fragments. URLs are NOT touched here (handled later
    by _autolink_urls). Code blocks must already have been extracted."""
    media = _extract_local_paths(text)
    if not media:
        return text

    # raw reference -> (resolved absolute path, kind). Rendering always
    # uses the RESOLVED path so relative references produce working
    # imageview:// / audio:// / video:// anchors.
    media_map = {raw: (resolved, kind) for raw, resolved, kind in media}

    # Build a regex that matches any of the (escaped) raw references.
    # Use re.escape so special chars (e.g. '+', '.') are literal.
    # Word-boundary-ish: we use lookahead to not match a trailing path char.
    # Sort by length DESC so longer references are matched first (avoids
    # accidentally matching a prefix of a longer path).
    parts = sorted((re.escape(raw) for raw in media_map),
                   key=len, reverse=True)
    pattern = re.compile(
        r'(?<![\w/\\])(' + '|'.join(parts) + r')(?![\w/\\])'
    )

    def _sub(m):
        entry = media_map.get(m.group(1))
        if not entry:
            return m.group(0)
        resolved, kind = entry
        if kind == "image":
            return _render_image(resolved)
        if kind == "audio":
            return _render_audio(resolved)
        if kind == "video":
            return _render_video(resolved)
        return m.group(0)

    return pattern.sub(_sub, text)


def _autolink_urls(text: str) -> str:
    """Wrap bare URLs in <a> tags. Text is assumed to be already
    HTML-escaped at this point — we therefore escape the URL inside the
    link text but not the href."""
    out = []
    last = 0
    for m in _URL_RE.finditer(text):
        out.append(text[last:m.start()])
        url = m.group("url")
        out.append(_render_url(url))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


# ----------------------------------------------------------------------
# Markdown (subset) processing on a single non-code segment
# ----------------------------------------------------------------------
def _apply_inline_markdown(text: str) -> str:
    """Apply inline markdown rules to *text*. The text is assumed to be
    already HTML-escaped; this function inserts raw HTML tags at the
    appropriate positions.

    Order matters: inline code is stashed as a placeholder FIRST, so its
    content is protected from bold/italic/strike regexes and from URL
    autolinking. Markdown links are NOT processed here because they
    were already stashed at the top level (render_message_to_html step
    0a) before this function runs.
    """
    # 1. Inline code  `...` — stash with placeholders so the content
    # is protected from further substitution. We use \x02 (STX) rather
    # than \x00 (NUL) because some HTML parsers strip NUL bytes.
    code_spans: List[str] = []
    placeholder_re = re.compile(r'`([^\n`]+?)`')

    def _stash_code(m):
        code_spans.append(m.group(1))
        return f"\x02CODE{len(code_spans) - 1}\x02"
    text = placeholder_re.sub(_stash_code, text)

    # 2. Bold **text** or __text__
    text = re.sub(r'\*\*([^\n*]+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<![A-Za-z0-9_])__([^\n_]+?)__', r'<b>\1</b>', text)

    # 3. Italic *text* or _text_ — careful not to break **bold** which
    # we already converted above.
    text = re.sub(r'(?<!\*)\*([^\n*]+?)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'(?<![A-Za-z0-9_])_([^\n_]+?)_(?![A-Za-z0-9_])', r'<i>\1</i>', text)

    # 4. Strikethrough ~~text~~
    text = re.sub(r'~~([^\n~]+?)~~', r'<s>\1</s>', text)

    # 5. Restore inline code spans as <code> elements.
    def _restore_code(m):
        idx = int(m.group(1))
        return (
            f'<code style="background:#f4f4f4; padding:1px 4px; '
            f'border-radius:3px; font-family:Consolas,'
            f'\'Courier New\',monospace;">{code_spans[idx]}</code>'
        )
    text = re.sub(r'\x02CODE(\d+)\x02', _restore_code, text)

    # 6. Autolink URLs that survive in the text. Markdown links were
    # already stashed at the top level so they're not visible here.
    text = _autolink_urls(text)

    return text


def _apply_block_markdown(text: str) -> str:
    """Apply block-level markdown rules: headers, lists, blockquotes,
    horizontal rules, paragraph breaks.

    NOTE: *text* is HTML-escaped on entry — ``>`` has become ``&gt;``,
    so blockquote detection looks for the escaped form. Inline markdown
    (bold, italic, code) is applied to text content too.
    """
    lines = text.split("\n")
    out: List[str] = []
    in_ul = False  # unordered list (we render as indented divs, not <ul>)
    in_ol = False
    list_buffer: List[str] = []

    def flush_list():
        nonlocal in_ul, in_ol, list_buffer
        if not list_buffer:
            return
        # We render the list items as <div>s with manual markers so long
        # items still align under their bullet. Qt's <ul>/<li> rendering
        # tends to wrap badly inside QTextBrowser.
        marker = "&bull;" if in_ul else "1."
        items_html = "".join(
            f'<div style="margin-left:18px; text-indent:-13px;">'
            f'<span style="color:#888;">{marker}</span>&nbsp;'
            f'{_apply_inline_markdown(_replace_media_in_text(item))}</div>'
            for item in list_buffer
        )
        out.append(f'<div style="margin:2px 0;">{items_html}</div>')
        list_buffer = []
        in_ul = in_ol = False

    for line in lines:
        stripped = line.rstrip()

        # Empty line ends any list.
        if not stripped:
            flush_list()
            out.append("<br/>")
            continue

        # Horizontal rule  ---  or  ***  on its own line
        if re.match(r'^(-{3,}|\*{3,}|_{3,})$', stripped):
            flush_list()
            out.append('<hr style="border:none; border-top:1px solid #ccc; '
                      'margin:6px 0;"/>')
            continue

        # ATX header: # H1 .. ###### H6
        m = re.match(r'^(#{1,6})\s+(.*)$', stripped)
        if m:
            flush_list()
            level = len(m.group(1))
            content = m.group(2).strip()
            content = _apply_inline_markdown(_replace_media_in_text(content))
            sizes = {1: 22, 2: 19, 3: 17, 4: 15, 5: 13, 6: 12}
            sz = sizes.get(level, 14)
            out.append(
                f'<div style="font-size:{sz}px; font-weight:bold; '
                f'margin:6px 0 3px 0;">{content}</div>'
            )
            continue

        # Block quote:  > ... (after escaping, looks like  &gt; ...)
        if stripped.startswith("&gt;"):
            flush_list()
            # Strip leading &gt; markers (one per line; nested quotes
            # may have multiple).
            content = stripped
            while content.startswith("&gt;"):
                content = content[4:]
            content = content.lstrip()
            content = _apply_inline_markdown(_replace_media_in_text(content))
            out.append(
                f'<div style="border-left:3px solid #b0bec5; padding-left:8px; '
                f'color:#555; margin:3px 0;">{content}</div>'
            )
            continue

        # Unordered list:  - / * / +  item
        m = re.match(r'^\s*[-*+]\s+(.*)$', stripped)
        if m:
            in_ol = False
            in_ul = True
            list_buffer.append(m.group(1).strip())
            continue

        # Ordered list:  1. item
        m = re.match(r'^\s*\d+\.\s+(.*)$', stripped)
        if m:
            in_ul = False
            in_ol = True
            list_buffer.append(m.group(1).strip())
            continue

        # Regular paragraph line.
        flush_list()
        line_html = _apply_inline_markdown(_replace_media_in_text(line))
        out.append(f'<div style="margin:1px 0;">{line_html}</div>')

    flush_list()
    return "".join(out)


# ----------------------------------------------------------------------
# Top-level render
# ----------------------------------------------------------------------
def render_message_to_html(text: str) -> str:
    """Convert a single message's plain-text content to HTML.

    Steps:
      0a. Stash markdown links [text](url) — protects their text and URL
          from media-path replacement. Without this, a path inside the
          (url) part of a markdown link would get replaced with an
          \x00MEDIA<n>\x00 placeholder, then the placeholder would
          get re-inserted INTO the href attribute, producing
          nested <a> tags and broken HTML.
      0b. Pre-scan for media paths (images/audio/video) ANYWHERE in the
          remaining text and replace each with a placeholder. This MUST
          happen before code-block extraction so that media inside code
          blocks is also detected. Markdown-link URLs are protected by
          step 0a.
      1. Pull fenced code blocks out and replace them with placeholders
         so the markdown processor never sees the code content.
      2. Process the remaining text:
         a. HTML-escape.
         b. Apply block-level markdown (headers, lists, quotes, hr, p).
         c. Inside each block, apply inline markdown, autolink URLs.
      3. Splice the rendered code blocks back in.
      4. Splice the rendered media HTML back in.
      5. Restore the markdown links LAST so their HTML is not touched
         by any subsequent processing.
    """
    if not text:
        return ""
    if not text.strip():
        # Whitespace-only — return a single non-breaking space so Qt
        # still renders a blank line (avoids "empty paragraph" issues
        # with QTextBrowser.append()).
        return "&nbsp;"

    # --- 0a. Stash markdown links FIRST ---
    # Match [text](url) or [text](url "title"). We need to protect BOTH
    # the link text and the URL from the media-replacement pass below,
    # because either might contain a file path that we don't want
    # auto-converted to a player widget (the markdown link is the
    # user/agent's explicit formatting choice).
    md_links: List[tuple] = []  # list of (link_text, link_url)
    md_link_re = re.compile(r'\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)')

    def _stash_md_link(m):
        md_links.append((m.group(1), m.group(2)))
        return f"\x01MDLINK{len(md_links) - 1}\x01"
    work = md_link_re.sub(_stash_md_link, text)

    # --- 0b. Pre-scan for media paths ---
    # Build a list of (placeholder_id, html_fragment) pairs and replace
    # each media path in the text with a ``\x02MEDIA<n>\x02`` placeholder.
    # We use \x02 (STX) instead of \x00 (NUL) because some HTML parsers
    # (including Qt's rich text engine in older versions) silently strip
    # NUL bytes — which would corrupt our placeholders. \x02 is safe.
    media_stash: List[str] = []
    media_paths = _extract_local_paths(work)
    if media_paths:
        # raw reference -> (resolved absolute path, kind); rendering uses
        # the resolved path so relative references yield working anchors.
        media_map = {raw: (resolved, kind)
                     for raw, resolved, kind in media_paths}
        # Sort by length DESC so longer references are matched first
        # (avoids accidentally matching a prefix of a longer path).
        parts = sorted((re.escape(raw) for raw in media_map),
                       key=len, reverse=True)
        media_pattern = re.compile(
            r'(?<![\w/\\])(' + '|'.join(parts) + r')(?![\w/\\])'
        )

        def _stash_media(m):
            entry = media_map.get(m.group(1))
            if not entry:
                return m.group(0)
            resolved, kind = entry
            if kind == "image":
                media_stash.append(_render_image(resolved))
            elif kind == "audio":
                media_stash.append(_render_audio(resolved))
            elif kind == "video":
                media_stash.append(_render_video(resolved))
            else:
                return m.group(0)  # leave as-is (shouldn't happen)
            return f"\x02MEDIA{len(media_stash) - 1}\x02"

        work = media_pattern.sub(_stash_media, work)

    # --- 1. Extract code blocks ---
    code_blocks: List[Tuple[str, str]] = []  # list of (language, code)

    def _stash_block(m):
        lang = m.group("lang").strip()
        code = m.group("code")
        # Strip a single trailing newline (common from models) but keep
        # internal structure intact.
        if code.endswith("\n"):
            code = code[:-1]
        code_blocks.append((lang, code))
        return f"\x02BLOCK{len(code_blocks) - 1}\x02"

    work = _FENCE_RE.sub(_stash_block, work)

    # --- 2. Escape + block markdown ---
    # NOTE: We escape the *non-code-segment* text, then run block/inline
    # markdown on the escaped text. Code blocks are stashed raw and will
    # be re-escaped individually when spliced back.
    # The media placeholders are also in this text — they survive
    # HTML-escaping unchanged (\x01 and \x02 are control chars that
    # html.escape() does NOT touch).
    escaped = _html.escape(work)
    body = _apply_block_markdown(escaped)

    # --- 3. Splice code blocks back in ---
    def _restore_block(m):
        idx = int(m.group(1))
        lang, code = code_blocks[idx]
        esc = _html.escape(code)
        lang_label = (
            f'<div style="font-size:10px; color:#888; '
            f'margin-bottom:2px; font-family:Consolas,monospace;">'
            f'{_html.escape(lang)}</div>'
            if lang else ""
        )
        return (
            f'<div style="background:#1e1e1e; color:#d4d4d4; '
            f'font-family:Consolas,\'Courier New\',monospace; '
            f'font-size:12px; padding:8px 10px; margin:6px 0; '
            f'border-radius:4px; border:1px solid #3a3a3a; '
            f'white-space:pre-wrap; word-break:break-word;">'
            f'{lang_label}{esc}</div>'
        )

    body = re.sub(r'\x02BLOCK(\d+)\x02', _restore_block, body)

    # --- 4. Splice media back in ---
    # Restore media placeholders LAST so the inserted HTML doesn't get
    # touched by any subsequent processing.
    def _restore_media(m):
        idx = int(m.group(1))
        return media_stash[idx]

    body = re.sub(r'\x02MEDIA(\d+)\x02', _restore_media, body)

    # --- 5. Restore markdown links ---
    # The link text and URL were stashed BEFORE media scanning and
    # HTML escaping, so they're in their original (unescaped) form.
    # We escape them now (consistent with how the rest of the text is
    # treated) and emit a clean <a> tag.
    def _restore_md_link(m):
        idx = int(m.group(1))
        link_text, link_url = md_links[idx]
        return (
            f'<a href="{_html.escape(link_url, quote=True)}" '
            f'style="color:#1976d2;">{_html.escape(link_text)}</a>'
        )
    body = re.sub(r'\x01MDLINK(\d+)\x01', _restore_md_link, body)

    # Tidy up: collapse 3+ consecutive <br/> into 2.
    body = re.sub(r'(<br/>\s*){3,}', '<br/><br/>', body)

    return body


# ----------------------------------------------------------------------
# Convenience: list all media items in a piece of text (for the media list)
# ----------------------------------------------------------------------
def extract_media_items(text: str) -> List[Tuple[str, str, str]]:
    """Return list of (kind, path, display) tuples for every media item
    found in *text* (including inside code blocks). kind ∈
    {'image','audio','video','url'}.

    Used by the GUI's Media List panel. Items are returned in the order
    they appear; duplicates (by path/url) are kept because two messages
    might legitimately reference the same file — the user asked for
    "all media files from chat window".
    """
    items: List[Tuple[str, str, str]] = []

    # URLs first.
    for m in _URL_RE.finditer(text):
        url = m.group("url")
        items.append(("url", url, url))

    # Local media paths. The reported *target* is the RESOLVED absolute
    # path, so the media-list panel keeps working entries even when the
    # original message only contained a bare relative name.
    for raw, resolved, kind in _extract_local_paths(text):
        items.append((kind, resolved, os.path.basename(resolved)))

    return items


# ----------------------------------------------------------------------
# Session path re-anchoring — survives the app folder moving between
# versions. gui_main stores the app dir inside every saved session; on
# load it calls reanchor_text_paths() on each history message with the
# OLD recorded base and the CURRENT app dir as the new base.
# ----------------------------------------------------------------------
def _scan_absolute_path_tokens(text: str) -> List[str]:
    """Collect absolute-looking path tokens from *text* WITHOUT any
    existence or media-type filtering.

    Unlike _extract_local_paths, this must be able to see paths that no
    longer exist on disk (that's exactly the case re-anchoring fixes).
    Only Unix-root (/…) and Windows-drive (C:\\… / C:/…) forms count as
    absolute here.
    """
    tokens: List[str] = []
    seen = set()
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
        candidate = candidate.rstrip(".,;:!?)`\"'")
        candidate = candidate.lstrip("`\"'")
        if candidate and candidate not in seen:
            seen.add(candidate)
            if candidate.startswith("/") or re.match(r'^[A-Za-z]:[\\/]',
                                                     candidate):
                tokens.append(candidate)
        i = j
    return tokens


def reanchor_text_paths(text: str, old_base: str,
                        new_base: str) -> Tuple[str, int]:
    """Re-map absolute MEDIA paths from a previous app location onto the
    current one, so sessions saved before the app folder moved keep
    their media working.

    Rules (deliberately conservative):
      * A path is only rewritten when the ORIGINAL path no longer exists
        on disk — working paths are never touched.
      * Only paths with a recognised media extension are considered.
      * When *old_base* is known and the path lives under it, the path is
        re-anchored as ``<new_base>/<relative-rest>`` — but only when
        that file actually exists at the new location.
      * When *old_base* is empty (legacy sessions saved before app_dir
        was recorded), a basename-only fallback maps a missing media
        path to ``<new_base>/<basename>`` when such a file exists.
      * When old_base and new_base are the same, nothing happens (the
        app did not move — guessing would risk false positives).

    Returns ``(new_text, number_of_replacements)``.
    """
    if not isinstance(text, str) or not text.strip():
        return text, 0
    new_base = (new_base or "").strip()
    if not new_base:
        return text, 0

    def _norm(p: str) -> str:
        # normcase makes the comparison case-insensitive on Windows
        # and is a no-op elsewhere.
        return os.path.normcase(os.path.normpath(p))

    norm_new = _norm(new_base)
    norm_old = _norm(os.path.expanduser(old_base)) if old_base else ""

    if norm_old and norm_old == norm_new:
        # App did not move — nothing to re-anchor.
        return text, 0

    count = 0
    for tok in _scan_absolute_path_tokens(text):
        try:
            if os.path.isfile(os.path.expanduser(tok)):
                # Still valid — never touch working paths.
                continue
        except Exception:
            continue
        if os.path.splitext(tok)[1].lower() not in _ALL_MEDIA_EXTS:
            # Only re-anchor media references.
            continue

        replacement = None
        if norm_old:
            norm_tok = _norm(tok)
            prefix = norm_old.rstrip("/\\") + os.sep
            if norm_tok == norm_old:
                # The path IS the old app dir itself — nothing to map.
                continue
            if norm_tok.startswith(prefix):
                rel = norm_tok[len(prefix):]
                cand = os.path.normpath(os.path.join(new_base, rel))
                if os.path.isfile(cand):
                    replacement = cand
        elif not old_base:
            # Legacy session (no recorded app dir): conservative basename
            # fallback for missing media files.
            cand = os.path.normpath(
                os.path.join(new_base, os.path.basename(tok))
            )
            if os.path.isfile(cand):
                replacement = cand

        if replacement is not None:
            # Replace with boundary guards so we never rewrite a path
            # that merely CONTAINS the token as a substring.
            pattern = re.compile(
                r'(?<![\w/\\])' + re.escape(tok) + r'(?![\w/\\])'
            )
            text, n = pattern.subn(replacement, text)
            count += n
    return text, count
