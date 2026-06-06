"""Optional diagram rendering for fenced ```mermaid / ```dot blocks.

Renders to a base64 PNG using whatever local CLI is available:
- Mermaid via `mmdc` (npm i -g @mermaid-js/mermaid-cli)
- Graphviz via `dot` (winget install graphviz)

Returns None when no renderer is installed (or the source is incomplete), so the
caller can simply show the diagram source as a normal code block. Results are
cached by content so repeated renders (streaming / scrolling) never re-run a
subprocess for the same diagram.
"""

import base64
import hashlib
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

LOGGER = logging.getLogger(__name__)

MERMAID_LANGUAGES = {"mermaid"}
DOT_LANGUAGES = {"dot", "graphviz"}

_CACHE: dict[str, str | None] = {}


def is_diagram_language(language: str) -> bool:
    lang = language.strip().lower()
    return lang in MERMAID_LANGUAGES or lang in DOT_LANGUAGES


def diagram_renderer_available(language: str) -> bool:
    lang = language.strip().lower()
    if lang in MERMAID_LANGUAGES:
        return shutil.which("mmdc") is not None
    if lang in DOT_LANGUAGES:
        return shutil.which("dot") is not None
    return False


def render_diagram(language: str, code: str, *, timeout: float = 15.0) -> str | None:
    lang = language.strip().lower()
    code = code.strip()
    if not code or not is_diagram_language(lang):
        return None
    key = hashlib.sha1(f"{lang}\n{code}".encode("utf-8")).hexdigest()
    if key in _CACHE:
        return _CACHE[key]
    try:
        result = _render_mermaid(code, timeout) if lang in MERMAID_LANGUAGES else _render_dot(code, timeout)
    except Exception:
        LOGGER.debug("diagram render failed", exc_info=True)
        result = None
    _CACHE[key] = result
    return result


def _render_dot(code: str, timeout: float) -> str | None:
    dot = shutil.which("dot")
    if not dot:
        return None
    proc = subprocess.run([dot, "-Tpng"], input=code.encode("utf-8"), capture_output=True, timeout=timeout)
    if proc.returncode != 0 or not proc.stdout:
        return None
    return base64.b64encode(proc.stdout).decode("ascii")


def _render_mermaid(code: str, timeout: float) -> str | None:
    mmdc = shutil.which("mmdc")
    if not mmdc:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "diagram.mmd"
        output = Path(tmp) / "diagram.png"
        source.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            [mmdc, "-i", str(source), "-o", str(output), "-b", "white"],
            capture_output=True,
            timeout=timeout,
        )
        if proc.returncode != 0 or not output.exists():
            return None
        return base64.b64encode(output.read_bytes()).decode("ascii")
