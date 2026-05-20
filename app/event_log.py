import json
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any


_LOCK = Lock()


def append_client_event(event: str, details: dict[str, Any] | None = None) -> str:
    server_time = datetime.now().astimezone().replace(microsecond=0)
    date = server_time.strftime("%Y-%m-%d")
    clock = server_time.strftime("%H:%M:%S")
    logged_at = server_time.isoformat()
    log_path = Path(os.getenv("STACKWIRE_EVENT_LOG_PATH", "logs/stackwire_client_events.md"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    safe_event = event.strip()[:80] or "client_event"
    payload = details or {}
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)

    with _LOCK:
        existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        chunks: list[str] = []
        if f"## {date}" not in existing:
            if existing and not existing.endswith("\n"):
                chunks.append("\n")
            chunks.append(f"\n## {date}\n")
        chunks.append(f"\n### {clock} {safe_event}\n")
        chunks.append(f"- server_time: {logged_at}\n")
        chunks.append(f"- details: `{serialized}`\n")
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(chunks))

    return logged_at
