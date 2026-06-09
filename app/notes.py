import os
from pathlib import Path

from app.config import ROOT_DIR


def notes_path() -> Path:
    configured = os.getenv("STACKWIRE_NOTES_PATH", "").strip()
    return Path(configured) if configured else ROOT_DIR / "data" / "notes.md"


def load_notes(path: Path | None = None) -> str:
    resolved = path or notes_path()
    if not resolved.exists():
        return ""
    return resolved.read_text(encoding="utf-8")


def save_notes(text: str, path: Path | None = None) -> None:
    resolved = path or notes_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")
