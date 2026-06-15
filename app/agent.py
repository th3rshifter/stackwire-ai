"""Agent mode: the assistant proposes shell commands, the user approves each one,
StackWire runs it and feeds the output back to the model. NOTHING runs without an
explicit click — confirmation is mandatory.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

AGENT_SYSTEM_PROMPT = """You are StackWire in AGENT mode. You can accomplish tasks on the
user's Windows machine by running shell commands — but EVERY command requires the user's
explicit approval first, and you only ever propose ONE command at a time.

To run a command, output a short sentence saying what it does, then EXACTLY:
<run>the command</run>
and nothing after it. Then stop and wait for the result, which the user will send back to you.

When you have done enough to answer the user's request, just write the final answer normally
(no <run> tag).

Rules:
- One command per step. Wait for its output before the next.
- Prefer safe, read-only commands first (dir, type, git status, etc.).
- NEVER propose a destructive command (deleting files, formatting, shutting down) unless the
  user explicitly asked for that exact action.
- If a command fails, read the error and adapt.
- Keep commands valid for Windows (PowerShell / cmd)."""

_RUN_RE = re.compile(r"<run>\s*(.*?)\s*</run>", re.DOTALL)

# Patterns that are almost always destructive — surface an extra warning before running.
_DANGEROUS = (
    "rm -rf", "rm -r ", "rmdir /s", "del /", "del /f", "format ", "mkfs",
    "shutdown", "diskpart", "reg delete", ":(){", "> /dev/sd", "dd if=", "fdisk",
)


def extract_command(text: str) -> str | None:
    """Return the proposed command from a <run>...</run> tag, or None if the model answered."""
    match = _RUN_RE.search(text or "")
    if not match:
        return None
    command = match.group(1).strip()
    return command or None


def strip_command_tag(text: str) -> str:
    """The model's prose with the <run> tag removed (shown above the confirmation bar)."""
    return _RUN_RE.sub("", text or "").strip()


def is_dangerous(command: str) -> bool:
    low = (command or "").lower()
    return any(token in low for token in _DANGEROUS)


def agent_cwd() -> str:
    return os.getenv("STACKWIRE_AGENT_CWD", str(ROOT_DIR)).strip() or str(ROOT_DIR)


def run_command(command: str, *, cwd: str | None = None, timeout: int = 60) -> str:
    """Run an APPROVED command and return a compact result string for the model.

    Always called only after explicit user approval. Output is truncated so it doesn't
    blow up the context window.
    """
    workdir = cwd or agent_cwd()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"[command timed out after {timeout}s]"
    except Exception as exc:  # noqa: BLE001
        return f"[failed to run: {exc}]"

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    body = out
    if err:
        body = (body + "\n[stderr]\n" + err).strip()
    if not body:
        body = "(no output)"
    if len(body) > 6000:
        body = body[:6000] + "\n…(output truncated)"
    return f"[exit code {proc.returncode}]\n{body}"
