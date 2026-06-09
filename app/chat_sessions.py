import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import ROOT_DIR


Message = tuple[str, str]


@dataclass(frozen=True)
class ChatSessionSummary:
    id: str
    title: str
    updated_at: str
    message_count: int


def _store_path() -> Path:
    configured = os.getenv("STACKWIRE_CHAT_STORE", "").strip()
    return Path(configured) if configured else ROOT_DIR / "data" / "chat_sessions.json"


def _now() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _read_store() -> dict[str, Any]:
    path = _store_path()
    if not path.exists():
        return {"current_id": "", "sessions": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"current_id": "", "sessions": []}
    if not isinstance(data, dict):
        return {"current_id": "", "sessions": []}
    sessions = data.get("sessions")
    if not isinstance(sessions, list):
        data["sessions"] = []
    data["current_id"] = str(data.get("current_id", ""))
    return data


def _write_store(data: dict[str, Any]) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_messages(messages: list[Message] | list[tuple[str, str]]) -> list[list[str]]:
    normalized: list[list[str]] = []
    for role, content in messages:
        role_text = str(role).strip()
        if role_text not in {"user", "assistant"}:
            continue
        normalized.append([role_text, str(content)])
    return normalized


def _clean_title_text(text: str) -> str:
    text = re.sub(r"\[\[(?:screenshot|generated_image):[A-Za-z0-9+/=]*\]\]", " ", text)
    text = re.sub(r"\[\[file:([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"```[a-zA-Z0-9_.+-]*\n.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[*_#>\[\]()]+" , " ", text)
    text = text.replace("\\", "/")
    text = re.sub(r"\s+", " ", text).strip(" -:,.!?;")
    return text


def _title_candidate_score(candidate: str) -> int:
    if not candidate:
        return -10_000
    words = re.findall(r"[\wА-Яа-яЁё]+", candidate, flags=re.UNICODE)
    slash_penalty = min(16, candidate.count("/") * 3)
    file_tree_penalty = 12 if re.search(r"(^|\s)[├└│]", candidate) else 0
    extension_penalty = 8 if re.search(r"\.(ya?ml|json|py|js|ts|tsx|jsx|md|txt|conf|j2)\b", candidate, re.IGNORECASE) else 0
    cyrillic_bonus = 8 if re.search(r"[А-Яа-яЁё]", candidate) else 0
    question_bonus = 6 if re.search(r"\b(что|как|почему|зачем|где|покажи|сделай|объясни|напиши)\b", candidate, re.IGNORECASE) else 0
    return len(words) + cyrillic_bonus + question_bonus - slash_penalty - file_tree_penalty - extension_penalty


def _short_title(text: str, *, max_chars: int = 58, max_words: int = 8) -> str:
    words = re.findall(r"[^\s]+", text)
    if not words:
        return "New chat"
    title = " ".join(words[:max_words]).strip(" -:,.!?;")
    if len(title) > max_chars:
        title = title[:max_chars].rsplit(" ", 1)[0].strip(" -:,.!?;") or title[:max_chars].strip(" -:,.!?;")
    if title:
        title = title[0].upper() + title[1:]
    return title or "New chat"


def _title_from_messages(messages: list[Message] | list[tuple[str, str]]) -> str:
    for role, content in messages:
        if role != "user":
            continue
        text = _clean_title_text(str(content))
        if not text:
            continue
        chunks = [
            _clean_title_text(chunk)
            for chunk in re.split(r"(?:\n+|[.!?]\s+|\s+[–—-]\s+)", text)
            if _clean_title_text(chunk)
        ]
        if not chunks:
            continue
        best = max(chunks, key=_title_candidate_score)
        return _short_title(best)
    return "New chat"


def list_sessions() -> list[ChatSessionSummary]:
    data = _read_store()
    summaries: list[ChatSessionSummary] = []
    for raw in data.get("sessions", []):
        if not isinstance(raw, dict):
            continue
        messages = raw.get("messages") if isinstance(raw.get("messages"), list) else []
        summaries.append(
            ChatSessionSummary(
                id=str(raw.get("id", "")),
                title=str(raw.get("title", "") or "New chat"),
                updated_at=str(raw.get("updated_at", "")),
                message_count=len(messages),
            )
        )
    summaries = [item for item in summaries if item.id]
    summaries.sort(key=lambda item: item.updated_at, reverse=True)
    return summaries


def current_session_id() -> str:
    return str(_read_store().get("current_id", ""))


def load_session(session_id: str) -> list[Message]:
    data = _read_store()
    for raw in data.get("sessions", []):
        if not isinstance(raw, dict) or raw.get("id") != session_id:
            continue
        messages = raw.get("messages")
        if not isinstance(messages, list):
            return []
        result: list[Message] = []
        for item in messages:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                role, content = str(item[0]), str(item[1])
                if role in {"user", "assistant"}:
                    result.append((role, content))
        return result
    return []


def save_session(session_id: str, messages: list[Message] | list[tuple[str, str]], *, title: str | None = None) -> str:
    data = _read_store()
    sessions = [raw for raw in data.get("sessions", []) if isinstance(raw, dict)]
    if not session_id:
        session_id = uuid4().hex
    now = _now()
    normalized = _normalize_messages(messages)
    fallback_title = _title_from_messages(messages)
    explicit_title = title.strip() if title is not None else ""

    found = False
    for raw in sessions:
        if raw.get("id") != session_id:
            continue
        if explicit_title:
            raw["title"] = explicit_title
        elif not raw.get("title_manual"):
            raw["title"] = fallback_title
        raw["messages"] = normalized
        raw["updated_at"] = now
        found = True
        break
    if not found:
        sessions.append(
            {
                "id": session_id,
                "title": explicit_title or fallback_title,
                "title_manual": False,
                "created_at": now,
                "updated_at": now,
                "messages": normalized,
            }
        )
    data["sessions"] = sessions
    data["current_id"] = session_id
    _write_store(data)
    return session_id


def create_session() -> str:
    return save_session("", [], title="New chat")


def set_current_session(session_id: str) -> None:
    data = _read_store()
    data["current_id"] = session_id
    _write_store(data)


def rename_session(session_id: str, title: str) -> bool:
    clean_title = " ".join(str(title).strip().split())
    if not session_id or not clean_title:
        return False
    data = _read_store()
    sessions = [raw for raw in data.get("sessions", []) if isinstance(raw, dict)]
    changed = False
    for raw in sessions:
        if raw.get("id") != session_id:
            continue
        raw["title"] = clean_title[:80]
        raw["title_manual"] = True
        raw["updated_at"] = _now()
        changed = True
        break
    if changed:
        data["sessions"] = sessions
        _write_store(data)
    return changed


def delete_session(session_id: str) -> str:
    data = _read_store()
    sessions = [raw for raw in data.get("sessions", []) if isinstance(raw, dict) and raw.get("id") != session_id]
    data["sessions"] = sessions
    if data.get("current_id") == session_id:
        data["current_id"] = str(sessions[0].get("id", "")) if sessions else ""
    _write_store(data)
    return str(data.get("current_id", ""))
