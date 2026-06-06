import base64
import html
import json
import logging
import math
import os
import queue
import random
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zipfile import ZipFile

import requests
from PySide6.QtCore import QByteArray, QBuffer, QEasingCurve, QEvent, QIODevice, QLineF, QObject, QPoint, QPointF, QRect, QSize, QPropertyAnimation, QThread, QTimer, QUrl, Qt, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QFont, QFontDatabase, QIcon, QKeyEvent, QKeySequence, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QShortcut, QTextCursor, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QSizePolicy,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from requests import RequestException

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import APP_NAME, LOCAL_ENV_FILE, get_stt_settings, is_cuda_whisper_error, load_local_env, update_stt_language_lock, whisper_language, whisper_model_attempts, whisper_vad_parameters  # noqa: E402
from app.diagrams import is_diagram_language, render_diagram  # noqa: E402
from app.event_log import append_client_event  # noqa: E402

load_local_env()

LOGGER = logging.getLogger(__name__)

from app.llm import ANSWER_MODE, DEFAULT_STACKWIRE_MODEL, OLLAMA_URL, AskResult, ExpandResult, OllamaClient, current_answer_model, current_vision_model  # noqa: E402
from app.question_recovery import RecoveryResult, STACKWIRE_MODE, current_recovery_model  # noqa: E402
from app.storage import create_session, log_feedback, save_good_answer  # noqa: E402
from app.tech_terms import WHISPER_TECHNICAL_PROMPT, normalize_spoken_technical_terms  # noqa: E402
from app.transcript_repair import clean_stt_output, collapse_repeated_phrases, condense_spoken_question, is_probable_stt_hallucination, repair_live_transcript  # noqa: E402


def _env_float_raw(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _rgba(red: int, green: int, blue: int, opacity: float) -> str:
    alpha = round(_clamp(opacity, 0.0, 1.0) * 255)
    return f"rgba({red}, {green}, {blue}, {alpha})"


STACKWIRE_PANEL_OPACITY = _clamp(_env_float_raw("STACKWIRE_PANEL_OPACITY", 0.70), 0.45, 0.92)
STACKWIRE_RAIL_OPACITY = _clamp(STACKWIRE_PANEL_OPACITY + 0.10, 0.55, 0.95)
STACKWIRE_BUBBLE_OPACITY = _clamp(STACKWIRE_PANEL_OPACITY + 0.08, 0.55, 0.94)

ACCENT = "#9ad6bd"          # primary mint
ACCENT2 = "#8ab4f0"         # secondary cool — role, links, selection
CORAL = "#e8896b"           # warm — active recording indicator
GOLD = "#a98cff"            # (unused, kept for back-compat)
BLUE = "#8ab4f0"            # links / selection
# Layered surfaces: translucent per-widget, so text stays opaque while the app
# still works as an overlay. Tune with STACKWIRE_PANEL_OPACITY=0.45..0.92.
BG = _rgba(17, 21, 27, STACKWIRE_PANEL_OPACITY)
SURFACE = _rgba(25, 30, 38, STACKWIRE_PANEL_OPACITY)
ELEVATED = _rgba(43, 50, 61, STACKWIRE_BUBBLE_OPACITY)
HAIRLINE = "rgba(154, 214, 189, 0.08)"
PANEL = BG                          # back-compat alias
PANEL_LIGHT = "rgba(10, 13, 18, 230)"
RAIL = _rgba(9, 12, 16, STACKWIRE_RAIL_OPACITY)
TEXT = "#dbeee7"
MUTED = "#8295a0"

FONT_STACK = '"Space Grotesk", "Manrope", "Segoe UI", Arial, sans-serif'
FONT_DISPLAY = '"Space Grotesk", "Manrope", "Segoe UI", sans-serif'
FONTS_DIR = ROOT_DIR / "assets" / "fonts"
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_VOSK_MODEL_NAME = os.getenv("VOSK_MODEL_NAME", "vosk-model-ru-0.54")
DEFAULT_VOSK_MODEL_DIR = MODELS_DIR / DEFAULT_VOSK_MODEL_NAME
DEFAULT_VOSK_MODEL_ZIP = MODELS_DIR / f"{DEFAULT_VOSK_MODEL_NAME}.zip"
DEFAULT_VOSK_MODEL_URL = os.getenv(
    "VOSK_MODEL_URL",
    f"https://alphacephei.com/vosk/models/{DEFAULT_VOSK_MODEL_NAME}.zip",
)
FALLBACK_VOSK_MODEL_NAME = "vosk-model-small-ru-0.22"
FALLBACK_VOSK_MODEL_DIR = MODELS_DIR / FALLBACK_VOSK_MODEL_NAME
FALLBACK_VOSK_MODEL_ZIP = MODELS_DIR / f"{FALLBACK_VOSK_MODEL_NAME}.zip"
FALLBACK_VOSK_MODEL_URL = f"https://alphacephei.com/vosk/models/{FALLBACK_VOSK_MODEL_NAME}.zip"
STT_SETTINGS = get_stt_settings()
STT_BACKEND = STT_SETTINGS.backend
STT_ALLOW_VOSK_FALLBACK = STT_SETTINGS.allow_vosk_fallback
STT_MIC_SIGNAL_THRESHOLD = STT_SETTINGS.mic_signal_threshold
STT_LOOPBACK_SIGNAL_THRESHOLD = STT_SETTINGS.loopback_signal_threshold
STT_PROBE_LOOPBACK_DEVICES = STT_SETTINGS.probe_loopback_devices
STT_LIVE_MAX_WORDS = STT_SETTINGS.live_max_words
STT_CONTEXT_LINES = STT_SETTINGS.context_lines
WHISPER_MODEL = STT_SETTINGS.model
WHISPER_CHUNK_SECONDS = STT_SETTINGS.chunk_seconds
WHISPER_CHUNK_OVERLAP_SECONDS = STT_SETTINGS.chunk_overlap_seconds
WHISPER_SAMPLE_RATE = STT_SETTINGS.sample_rate
WHISPER_VAD_FILTER = STT_SETTINGS.vad_filter
WHISPER_RETRY_WITHOUT_VAD = STT_SETTINGS.retry_without_vad
WHISPER_HOTWORDS = STT_SETTINGS.hotwords
WHISPER_INITIAL_PROMPT = WHISPER_TECHNICAL_PROMPT
STACKWIRE_API_URL = os.getenv("STACKWIRE_API_URL", "").strip().rstrip("/")
# Authentication is required before chatting unless explicitly disabled.
STACKWIRE_REQUIRE_AUTH = os.getenv("STACKWIRE_REQUIRE_AUTH", "1").strip().lower() in {"1", "true", "yes", "on"}
# Bearer token for the current signed-in user (set after login); sent on remote calls.
CURRENT_AUTH_TOKEN = ""


def set_auth_token(token: str) -> None:
    global CURRENT_AUTH_TOKEN
    CURRENT_AUTH_TOKEN = (token or "").strip()


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {CURRENT_AUTH_TOKEN}"} if CURRENT_AUTH_TOKEN else {}


STACKWIRE_API_CONNECT_TIMEOUT = float(os.getenv("STACKWIRE_API_CONNECT_TIMEOUT", "5"))
STACKWIRE_API_TIMEOUT = float(os.getenv("STACKWIRE_API_TIMEOUT", "300"))
STACKWIRE_REMOTE_STT = os.getenv("STACKWIRE_REMOTE_STT", "1" if STACKWIRE_API_URL else "0").strip() == "1"
STACKWIRE_STT_TIMEOUT = float(os.getenv("STACKWIRE_STT_TIMEOUT", "120"))
STACKWIRE_HIDE_FROM_CAPTURE = os.getenv("STACKWIRE_HIDE_FROM_CAPTURE", "1").strip() == "1"
STACKWIRE_HIDE_TASKBAR = os.getenv("STACKWIRE_HIDE_TASKBAR", "1").strip() == "1"
STACKWIRE_ACRYLIC = os.getenv("STACKWIRE_ACRYLIC", "0").strip().lower() in {"1", "true", "yes", "on"}
MIN_UI_ZOOM = 0.75
MAX_UI_ZOOM = 1.55
ZOOM_STEP = 0.1
UI_ZOOM = 1.0
DEFAULT_MODEL_CHOICES: tuple[str, ...] = (
    DEFAULT_STACKWIRE_MODEL,
    "qwen2.5-coder:7b",
    "qwen2.5:7b",
    "gemma3:4b",
    "gemma4:latest",
)

EXPAND_LABELS: dict[str, str] = {
    "details": "Расширение: Подробнее",
    "components": "Расширение: С компонентами",
    "example": "Расширение: Пример",
    "compare": "Расширение: Сравнение",
    "troubleshoot": "Расширение: Troubleshooting",
}

EXPAND_MENU_ITEMS: tuple[tuple[str, str], ...] = (
    ("details", "Подробнее"),
    ("components", "С компонентами"),
    ("example", "С примером кода/конфига"),
    ("compare", "Сравнить с аналогами"),
    ("troubleshoot", "Troubleshooting"),
)

LIGHTWEIGHT_STT_CORRECTIONS: tuple[tuple[str, str], ...] = (
    (r"\bдев\s+и\s+прок\b", "/dev и /proc"),
    (r"\bдев\b", "/dev"),
    (r"\bпрок\b", "/proc"),
    (r"\bетс\b", "/etc"),
    (r"\bвар\s+лог\b", "/var/log"),
)

LIVE_FILLER_WORDS = (
    "окей",
    "хм",
    "хмм",
    "мм",
    "ммм",
    "м",
    "аа",
    "ааа",
    "ээ",
    "эээ",
    "ладно",
    "слушай",
    "смотри",
    "значит",
    "короче",
    "типа",
    "ну",
    "вот",
    "вообще",
    "пожалуйста",
    "давай",
    "давайте",
)


def _dedupe_models(models: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for model in models:
        value = model.strip()
        if not value or value.lower() in seen:
            continue
        seen.add(value.lower())
        result.append(value)
    return result


def _ollama_tags_url() -> str:
    if OLLAMA_URL.endswith("/api/chat"):
        return f"{OLLAMA_URL[:-len('/api/chat')]}/api/tags"
    return OLLAMA_URL.replace("/api/chat", "/api/tags")


def _installed_ollama_models() -> list[str]:
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(_ollama_tags_url(), timeout=1.5)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []
    models = payload.get("models") if isinstance(payload, dict) else []
    if not isinstance(models, list):
        return []
    names = [str(item.get("name", "")).strip() for item in models if isinstance(item, dict)]
    return _dedupe_models(names)


def _model_choices() -> list[str]:
    return _dedupe_models(
        [
            current_answer_model(),
            current_recovery_model(),
            current_vision_model(),
            *_installed_ollama_models(),
            *DEFAULT_MODEL_CHOICES,
        ]
    )


def _save_local_env_values(values: dict[str, str]) -> None:
    existing = LOCAL_ENV_FILE.read_text(encoding="utf-8-sig").splitlines() if LOCAL_ENV_FILE.exists() else []
    remaining = {key: value.strip() for key, value in values.items() if value.strip()}
    lines: list[str] = []
    pattern = re.compile(r"^\s*#?\s*([A-Z0-9_]+)\s*=")
    for line in existing:
        match = pattern.match(line)
        key = match.group(1) if match else ""
        if key in remaining:
            lines.append(f"{key}={remaining.pop(key)}")
        else:
            lines.append(line)
    if remaining:
        if lines and lines[-1].strip():
            lines.append("")
        for key, value in remaining.items():
            lines.append(f"{key}={value}")
    LOCAL_ENV_FILE.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

LIVE_FILLER_PHRASES = (
    "в общем",
    "на самом деле",
    "можешь рассказать",
    "можешь объяснить",
    "расскажи пожалуйста",
    "объясни пожалуйста",
)

LIVE_TRAILING_NOISE = (
    "знаешь",
    "знаешь нет",
    "понимаешь",
    "да",
    "нет",
)


def normalize_transcript(text: str) -> str:
    normalized = text.strip()
    for pattern, replacement in LIGHTWEIGHT_STT_CORRECTIONS:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    normalized = normalize_spoken_technical_terms(normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _is_cuda_whisper_error(exc: BaseException) -> bool:
    return is_cuda_whisper_error(exc)


def _short_error(exc: BaseException, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(exc)).strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def clean_live_transcript(text: str) -> str:
    cleaned = repair_live_transcript(normalize_transcript(text))
    if not cleaned:
        return ""

    for phrase in LIVE_FILLER_PHRASES:
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b[,\s]*", " ", cleaned, flags=re.IGNORECASE)
    filler_pattern = r"\b(?:" + "|".join(re.escape(word) for word in LIVE_FILLER_WORDS) + r")\b[,\s]*"
    cleaned = re.sub(filler_pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:а|и)\s+(?=(что|как|чем|когда|почему|зачем|расскажи|объясни)\b)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:расскажи|рассказать|объясни|объяснить)\s+(?=что такое\b)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;")

    for phrase in LIVE_TRAILING_NOISE:
        cleaned = re.sub(rf"(?:,?\s*\b{re.escape(phrase)}\b\??)+$", "", cleaned, flags=re.IGNORECASE).strip(" ,.;")

    cleaned = _merge_definition_fragments(cleaned)
    cleaned = re.sub(r"\s+([?!,.:;])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;")

    if cleaned and looks_like_question(cleaned) and cleaned[-1] not in ".?!":
        cleaned = f"{cleaned}?"
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _merge_definition_fragments(text: str) -> str:
    parts = [part.strip(" ,.?") for part in re.split(r"\bчто такое\b", text, flags=re.IGNORECASE)]
    if len(parts) <= 2:
        return text

    terms: list[str] = []
    for part in parts[1:]:
        part = re.sub(r"\b(?:знаешь|нет|да|пожалуйста)\b", "", part, flags=re.IGNORECASE)
        part = re.sub(r"\s+", " ", part).strip(" ,.?")
        if part:
            terms.append(part)

    if not terms:
        return text
    if len(terms) == 1:
        return f"что такое {terms[0]}"
    return "что такое " + ", ".join(terms[:-1]) + " и " + terms[-1]


QUESTION_MARKERS = (
    "что",
    "как",
    "почему",
    "зачем",
    "когда",
    "где",
    "чем",
    "какой",
    "какая",
    "какие",
    "объясни",
    "объяснить",
    "расскажи",
    "рассказать",
    "опиши",
    "сравни",
    "разница",
    "отличается",
    "диагностировать",
    "починить",
    "debug",
    "troubleshoot",
)


def looks_like_question(text: str) -> bool:
    lowered = text.lower()
    return "?" in lowered or any(marker in lowered for marker in QUESTION_MARKERS)


def append_transcript_segment(current: str, addition: str, max_words: int | None = None) -> str:
    max_words = max_words or STT_LIVE_MAX_WORDS
    addition = addition.strip()
    if not addition:
        return current.strip()
    current = current.strip()
    if not current:
        return addition

    current_words = current.split()
    addition_words = addition.split()
    max_overlap = min(len(current_words), len(addition_words), 12)
    for size in range(max_overlap, 0, -1):
        if current_words[-size:] == addition_words[:size]:
            merged_words = current_words + addition_words[size:]
            break
    else:
        if addition.lower() in current.lower():
            merged_words = current_words
        else:
            merged_words = current_words + addition_words

    if len(merged_words) > max_words:
        merged_words = merged_words[-max_words:]
    merged = collapse_repeated_phrases(" ".join(merged_words))
    merged_words = merged.split()
    if len(merged_words) > max_words:
        merged_words = merged_words[-max_words:]
    return " ".join(merged_words)


def _px(value: int | float, scale: float | None = None) -> int:
    return max(1, round(float(value) * (UI_ZOOM if scale is None else scale)))


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def icon_pixmap(kind: str, size: int, color: str = TEXT) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen_width = max(1, round(size * 0.105))
    pen = QPen(QColor(color), pen_width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    s = float(size)
    if kind == "mark":
        painter.setBrush(QColor(154, 214, 189, 34))
        painter.drawRoundedRect(int(s * 0.12), int(s * 0.12), int(s * 0.76), int(s * 0.76), int(s * 0.18), int(s * 0.18))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(int(s * 0.30), int(s * 0.70), int(s * 0.48), int(s * 0.28))
        painter.drawLine(int(s * 0.48), int(s * 0.28), int(s * 0.72), int(s * 0.70))
    elif kind == "listen":
        painter.drawRoundedRect(int(s * 0.36), int(s * 0.16), int(s * 0.28), int(s * 0.46), int(s * 0.14), int(s * 0.14))
        painter.drawLine(int(s * 0.50), int(s * 0.68), int(s * 0.50), int(s * 0.84))
        painter.drawLine(int(s * 0.35), int(s * 0.84), int(s * 0.65), int(s * 0.84))
    elif kind == "stop":
        painter.setBrush(QColor(color))
        painter.drawRoundedRect(int(s * 0.30), int(s * 0.30), int(s * 0.40), int(s * 0.40), int(s * 0.08), int(s * 0.08))
        painter.setBrush(Qt.BrushStyle.NoBrush)
    elif kind == "clear":
        painter.drawLine(int(s * 0.28), int(s * 0.32), int(s * 0.72), int(s * 0.32))
        painter.drawLine(int(s * 0.35), int(s * 0.42), int(s * 0.65), int(s * 0.72))
        painter.drawLine(int(s * 0.65), int(s * 0.42), int(s * 0.35), int(s * 0.72))
    elif kind == "debug":
        painter.drawEllipse(int(s * 0.28), int(s * 0.25), int(s * 0.44), int(s * 0.50))
        painter.drawLine(int(s * 0.28), int(s * 0.46), int(s * 0.16), int(s * 0.38))
        painter.drawLine(int(s * 0.72), int(s * 0.46), int(s * 0.84), int(s * 0.38))
        painter.drawLine(int(s * 0.28), int(s * 0.62), int(s * 0.16), int(s * 0.72))
        painter.drawLine(int(s * 0.72), int(s * 0.62), int(s * 0.84), int(s * 0.72))
    elif kind == "capture":
        painter.drawLine(int(s * 0.18), int(s * 0.34), int(s * 0.18), int(s * 0.18))
        painter.drawLine(int(s * 0.18), int(s * 0.18), int(s * 0.34), int(s * 0.18))
        painter.drawLine(int(s * 0.66), int(s * 0.18), int(s * 0.82), int(s * 0.18))
        painter.drawLine(int(s * 0.82), int(s * 0.18), int(s * 0.82), int(s * 0.34))
        painter.drawLine(int(s * 0.82), int(s * 0.66), int(s * 0.82), int(s * 0.82))
        painter.drawLine(int(s * 0.82), int(s * 0.82), int(s * 0.66), int(s * 0.82))
        painter.drawLine(int(s * 0.34), int(s * 0.82), int(s * 0.18), int(s * 0.82))
        painter.drawLine(int(s * 0.18), int(s * 0.82), int(s * 0.18), int(s * 0.66))
    elif kind == "ask":
        painter.drawLine(int(s * 0.22), int(s * 0.50), int(s * 0.72), int(s * 0.50))
        painter.drawLine(int(s * 0.54), int(s * 0.30), int(s * 0.74), int(s * 0.50))
        painter.drawLine(int(s * 0.54), int(s * 0.70), int(s * 0.74), int(s * 0.50))
    elif kind == "expand":
        painter.drawLine(int(s * 0.24), int(s * 0.50), int(s * 0.76), int(s * 0.50))
        painter.drawLine(int(s * 0.50), int(s * 0.24), int(s * 0.50), int(s * 0.76))
    elif kind == "actions":
        painter.drawEllipse(int(s * 0.22), int(s * 0.44), int(s * 0.12), int(s * 0.12))
        painter.drawEllipse(int(s * 0.44), int(s * 0.44), int(s * 0.12), int(s * 0.12))
        painter.drawEllipse(int(s * 0.66), int(s * 0.44), int(s * 0.12), int(s * 0.12))
    elif kind == "settings":
        painter.drawEllipse(int(s * 0.34), int(s * 0.34), int(s * 0.32), int(s * 0.32))
        painter.drawLine(int(s * 0.50), int(s * 0.16), int(s * 0.50), int(s * 0.28))
        painter.drawLine(int(s * 0.50), int(s * 0.72), int(s * 0.50), int(s * 0.84))
        painter.drawLine(int(s * 0.16), int(s * 0.50), int(s * 0.28), int(s * 0.50))
        painter.drawLine(int(s * 0.72), int(s * 0.50), int(s * 0.84), int(s * 0.50))
        painter.drawLine(int(s * 0.26), int(s * 0.26), int(s * 0.34), int(s * 0.34))
        painter.drawLine(int(s * 0.66), int(s * 0.66), int(s * 0.74), int(s * 0.74))
        painter.drawLine(int(s * 0.74), int(s * 0.26), int(s * 0.66), int(s * 0.34))
        painter.drawLine(int(s * 0.34), int(s * 0.66), int(s * 0.26), int(s * 0.74))
    elif kind == "close":
        painter.drawLine(int(s * 0.30), int(s * 0.30), int(s * 0.70), int(s * 0.70))
        painter.drawLine(int(s * 0.70), int(s * 0.30), int(s * 0.30), int(s * 0.70))
    elif kind == "copy":
        painter.drawRoundedRect(int(s * 0.22), int(s * 0.22), int(s * 0.40), int(s * 0.40), int(s * 0.07), int(s * 0.07))
        painter.drawRoundedRect(int(s * 0.38), int(s * 0.38), int(s * 0.40), int(s * 0.40), int(s * 0.07), int(s * 0.07))
    elif kind == "edit":
        painter.drawLine(int(s * 0.26), int(s * 0.74), int(s * 0.64), int(s * 0.36))
        painter.drawLine(int(s * 0.64), int(s * 0.36), int(s * 0.76), int(s * 0.48))
        painter.drawLine(int(s * 0.76), int(s * 0.48), int(s * 0.38), int(s * 0.86))
        painter.drawLine(int(s * 0.26), int(s * 0.74), int(s * 0.38), int(s * 0.86))
        painter.drawLine(int(s * 0.24), int(s * 0.88), int(s * 0.40), int(s * 0.88))
    elif kind == "attach":
        # paperclip
        painter.drawArc(int(s * 0.34), int(s * 0.16), int(s * 0.32), int(s * 0.30), 0, 180 * 16)
        painter.drawLine(int(s * 0.34), int(s * 0.31), int(s * 0.34), int(s * 0.66))
        painter.drawLine(int(s * 0.66), int(s * 0.31), int(s * 0.66), int(s * 0.62))
        painter.drawArc(int(s * 0.34), int(s * 0.50), int(s * 0.32), int(s * 0.30), 180 * 16, 180 * 16)
        painter.drawLine(int(s * 0.50), int(s * 0.31), int(s * 0.50), int(s * 0.72))
    elif kind == "file":
        painter.drawRoundedRect(int(s * 0.28), int(s * 0.18), int(s * 0.44), int(s * 0.64), int(s * 0.06), int(s * 0.06))
        painter.drawLine(int(s * 0.38), int(s * 0.40), int(s * 0.62), int(s * 0.40))
        painter.drawLine(int(s * 0.38), int(s * 0.52), int(s * 0.62), int(s * 0.52))
        painter.drawLine(int(s * 0.38), int(s * 0.64), int(s * 0.54), int(s * 0.64))

    painter.end()
    return pixmap


def make_icon(kind: str, size: int, color: str = TEXT) -> QIcon:
    return QIcon(icon_pixmap(kind, size, color))


def pixmap_to_base64_png(pixmap: QPixmap) -> str:
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    pixmap.save(buffer, "PNG")
    return buffer.data().toBase64().data().decode("ascii")


def thinking_bar_png(phase: int, width: int, height: int) -> str:
    """An indeterminate 'thinking' bar: a soft mint highlight sliding back and forth."""
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    radius = height / 2
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(154, 214, 189, 18))
    painter.drawRoundedRect(0, 0, width, height, radius, radius)

    segment = width * 0.34
    travel = max(1.0, width - segment)
    position = abs(((phase % 48) / 24.0) - 1.0)  # 0..1 ping-pong
    x = position * travel
    gradient = QLinearGradient(x, 0, x + segment, 0)
    gradient.setColorAt(0.0, QColor(154, 214, 189, 0))
    gradient.setColorAt(0.5, QColor(160, 224, 200, 220))
    gradient.setColorAt(1.0, QColor(154, 214, 189, 0))
    painter.setBrush(gradient)
    painter.drawRoundedRect(int(x), 0, int(segment), height, radius, radius)
    painter.end()
    return pixmap_to_base64_png(pixmap)


def balance_streaming_markdown(text: str) -> str:
    """Keep partially-streamed markdown well-formed so it never renders as a giant
    bold/code block or shows raw ``**``/`` ` `` tokens until the model finishes.

    Splitting on ``` gives alternating outside/inside-code segments; we only ever
    balance bold and inline code in the trailing outside-code segment (where the
    cursor is), so code that legitimately contains a backtick is never disturbed."""
    segments = text.split("```")
    inside_open_fence = len(segments) % 2 == 0  # an odd number of ``` ⇒ inside a block
    if inside_open_fence:
        return f"{text}\n```"
    tail = segments[-1]  # text after the last closed fence (plain markdown context)
    suffix = ""
    if tail.count("**") % 2:
        suffix += "**"
    if tail.count("`") % 2:
        suffix += "`"
    return text + suffix


def build_html_style() -> str:
    return f"""
<style>
body {{
  margin: 0;
  color: #c7d1db;
  font-family: {FONT_STACK};
  font-size: {_px(16)}px;
  line-height: 1.6;
}}
h2 {{
  margin: {_px(13)}px 0 {_px(7)}px;
  color: #e4e9f0;
  font-size: {_px(17)}px;
  font-weight: 700;
}}
p {{
  margin: 0 0 {_px(9)}px;
}}
ul {{
  margin: 0 0 {_px(11)}px {_px(20)}px;
  padding: 0;
}}
li {{
  margin: {_px(5)}px 0;
}}
code {{
  padding: {_px(2)}px {_px(6)}px;
  border-radius: {_px(6)}px;
  background: rgba(154, 214, 189, 0.08);
  color: #d8e0ea;
  font-family: Consolas, Courier New, monospace;
}}
strong {{
  color: #f1f5f9;
  font-weight: 700;
}}
.code-card {{
  margin: {_px(12)}px 0 {_px(14)}px;
  border: 1px solid rgba(154, 214, 189, 0.08);
  border-radius: {_px(12)}px;
}}
.code-head-cell {{
  padding: {_px(7)}px {_px(13)}px;
  border-bottom: 1px solid rgba(154, 214, 189, 0.06);
  background: #20242d;
  font-size: {_px(11)}px;
}}
.code-body {{
  background: #14171d;
}}
.code-lang {{
  color: #9ad6bd;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}}
pre {{
  margin: 0;
  padding: {_px(16)}px {_px(18)}px;
  color: #dfe6ef;
  font-family: Consolas, Courier New, monospace;
  font-size: {_px(14)}px;
  line-height: 1.55;
  white-space: pre-wrap;
}}
.code-comment {{
  color: #7f94b8;
}}
.code-keyword {{
  color: #8fc6ff;
  font-weight: 800;
}}
.code-string {{
  color: #c7d9ff;
}}
.code-number {{
  color: #acc4c0;
}}
.code-flag {{
  color: #d6dbe6;
}}
.code-op {{
  color: #d1d5db;
  font-weight: 800;
}}
</style>
"""

def build_chat_style() -> str:
    return f"""
<style>
.msg {{
  margin: 0 0 {_px(8)}px;
  padding: {_px(8)}px {_px(14)}px;
}}
.msg-assistant {{
  background: transparent;
}}
.user-text {{
  color: #eef3f9;
}}
.role {{
  margin-bottom: {_px(6)}px;
  color: #7d8a99;
  font-size: {_px(11)}px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}}
.msg-actions {{
  text-align: right;
}}
.msg-actions a {{
  text-decoration: none;
}}
.welcome {{
  min-height: {_px(300)}px;
  padding-top: {_px(78)}px;
  text-align: center;
}}
.welcome-mark {{
  margin-bottom: {_px(10)}px;
  color: #9ad6bd;
  font-size: {_px(34)}px;
  font-weight: 800;
}}
.welcome-title {{
  font-family: {FONT_DISPLAY};
  color: #e6f4ee;
  font-size: {_px(42)}px;
  font-weight: 700;
  letter-spacing: 0.01em;
}}
.welcome-sub {{
  margin-top: {_px(10)}px;
  color: #88a096;
  font-size: {_px(14)}px;
}}
.welcome-model {{
  margin-top: {_px(26)}px;
  color: #5d7480;
  font-size: {_px(12)}px;
  letter-spacing: 0.04em;
}}
.shot {{
  margin: {_px(2)}px 0;
  background: transparent;
  line-height: 1.0;
}}
.msg-actions {{
  margin-top: {_px(9)}px;
}}
.thinking {{
  margin: {_px(2)}px 0;
}}
.thinking-label {{
  margin-bottom: {_px(6)}px;
  color: #8b97a6;
  font-size: {_px(13)}px;
}}
.file-chip {{
  margin: {_px(2)}px 0 {_px(6)}px;
  color: #c7d1db;
  font-size: {_px(13)}px;
}}
.diagram-card {{
  margin: {_px(10)}px 0 {_px(12)}px;
  padding: {_px(10)}px;
  border: 1px solid rgba(154, 214, 189, 0.08);
  border-radius: {_px(12)}px;
  background: #ffffff;
  text-align: center;
}}
.diagram-card img {{
  max-width: 100%;
}}
</style>
"""


# Raw code per fenced block of the current render, so the in-card copy button works.
# Cleared at the start of each render_chat pass; ids are stable within one render.
CODE_SNIPPETS: list[str] = []


def _document_css() -> str:
    """All chat CSS as a raw stylesheet (no <style> tags) for setDefaultStyleSheet."""
    raw = f"{build_html_style()}{build_chat_style()}"
    return raw.replace("<style>", "").replace("</style>", "")

# Diagram rendering is skipped while a message is streaming (incomplete source) and
# enabled for the final render, so we never shell out to a renderer per token.
_DIAGRAM_RENDER = {"enabled": True}


def markdown_to_html(markdown: str) -> str:
    markdown = normalize_unfenced_code_blocks(markdown)
    parts: list[str] = []
    pattern = re.compile(r"```([a-zA-Z0-9_.+-]*)\n(.*?)```", re.DOTALL)
    cursor = 0

    for match in pattern.finditer(markdown):
        parts.append(text_to_html(markdown[cursor : match.start()]))
        raw_language = match.group(1) or "code"
        language = html.escape(raw_language)
        raw_code = match.group(2).strip("\n")

        if _DIAGRAM_RENDER["enabled"] and is_diagram_language(raw_language):
            diagram_png = render_diagram(raw_language, raw_code)
            if diagram_png:
                parts.append(
                    f'<div class="diagram-card"><img src="data:image/png;base64,{diagram_png}" /></div>'
                )
                cursor = match.end()
                continue

        code = highlight_code(raw_language, raw_code)
        snippet_id = len(CODE_SNIPPETS)
        CODE_SNIPPETS.append(raw_code)
        copy_png = pixmap_to_base64_png(icon_pixmap("copy", _px(13), "#8b97a6"))
        copy_link = (
            f"<a href='copycode:{snippet_id}'>"
            f"<img src='data:image/png;base64,{copy_png}' width='{_px(13)}' height='{_px(13)}' /></a>"
        )
        parts.append(
            f'<table class="code-card" width="100%" cellspacing="0" cellpadding="0">'
            f'<tr><td class="code-head-cell">'
            f'<table width="100%" cellspacing="0" cellpadding="0"><tr>'
            f'<td class="code-lang">{language}</td>'
            f'<td align="right">{copy_link}</td>'
            f"</tr></table>"
            f"</td></tr>"
            f'<tr><td class="code-body"><pre>{code}</pre></td></tr>'
            f"</table>"
        )
        cursor = match.end()

    parts.append(text_to_html(markdown[cursor:]))
    return f"<html><head>{build_html_style()}</head><body>{''.join(parts)}</body></html>"


CODE_LABEL_LANGUAGES: dict[str, str] = {
    "dockerfile": "dockerfile",
    "docker-compose.yml": "yaml",
    "compose.yml": "yaml",
    "compose.yaml": "yaml",
    "values.yaml": "yaml",
    "values.yml": "yaml",
    "chart.yaml": "yaml",
    "deployment.yaml": "yaml",
    "service.yaml": "yaml",
    "ingress.yaml": "yaml",
    "configmap.yaml": "yaml",
    "secret.yaml": "yaml",
    "yaml": "yaml",
    "yml": "yaml",
    "nginx.conf": "nginx",
    "default.conf": "nginx",
    "jenkinsfile": "groovy",
    "main.tf": "hcl",
    "variables.tf": "hcl",
    "outputs.tf": "hcl",
}

SECTION_HEADINGS = {
    "коротко:",
    "как работает:",
    "практика:",
    "пример:",
    "нюанс:",
    "best practices:",
    "основной ответ:",
    "подробный ответ:",
}


def normalize_unfenced_code_blocks(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    in_fence = False
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            index += 1
            continue

        language = _language_from_code_label(stripped)
        if (
            not in_fence
            and language
            and index + 1 < len(lines)
            and _looks_like_code_line(lines[index + 1].strip(), language)
        ):
            block: list[str] = []
            index += 1
            while index < len(lines):
                current = lines[index]
                current_stripped = current.strip()
                next_language = _language_from_code_label(current_stripped)
                if next_language and index + 1 < len(lines) and _looks_like_code_line(lines[index + 1].strip(), next_language):
                    break
                if _is_section_heading(current_stripped):
                    break
                if not current_stripped:
                    if index + 1 < len(lines) and _looks_like_code_line(lines[index + 1].strip(), language):
                        block.append(current)
                        index += 1
                        continue
                    break
                if not _looks_like_code_line(current_stripped, language) and _looks_like_plain_text(current_stripped):
                    break
                block.append(current)
                index += 1

            out.append(f"```{language}")
            out.extend(block)
            out.append("```")
            continue

        out.append(line)
        index += 1

    return "\n".join(out)


def _language_from_code_label(label: str) -> str:
    normalized = label.strip().lower().rstrip(":")
    return CODE_LABEL_LANGUAGES.get(normalized, "")


def _is_section_heading(text: str) -> bool:
    return text.strip().lower() in SECTION_HEADINGS


def _looks_like_code_line(text: str, language: str) -> bool:
    if not text:
        return False
    if language == "dockerfile":
        return bool(re.match(r"^(FROM|RUN|COPY|ADD|WORKDIR|USER|CMD|ENTRYPOINT|EXPOSE|ENV|ARG|LABEL|HEALTHCHECK|VOLUME|SHELL)\b", text, re.IGNORECASE))
    if language in {"yaml", "yml"}:
        return bool(re.match(r"^[-\w.\"']+:\s*|^-\s+\w|^[\w.-]+:\s*$|^[\"']?\d+:\d+[\"']?\s*(#.*)?$", text))
    if language == "nginx":
        return bool(re.match(r"^(server|location|listen|server_name|proxy_pass|proxy_set_header|root|index|upstream|return)\b|^[{}]$", text, re.IGNORECASE))
    if language in {"hcl", "terraform"}:
        return bool(re.match(r"^(resource|module|variable|output|provider|data|locals)\b|^[\w.-]+\s*=", text, re.IGNORECASE))
    if language in {"groovy", "jenkinsfile"}:
        return bool(re.match(r"^(pipeline|agent|stages|stage|steps|environment|post)\b|^[{}]$", text, re.IGNORECASE))
    return bool(re.search(r"[{}:=#;/\\[\\]$]|^\s*[-\w.]+\s", text))


def _looks_like_plain_text(text: str) -> bool:
    if len(text) < 45:
        return False
    return bool(re.search(r"[а-яА-Я]", text)) and not bool(re.search(r"[{}:=#;/\\[\\]$]", text))


def highlight_code(language: str, code: str) -> str:
    lang = language.lower().strip()
    lang = CODE_LABEL_LANGUAGES.get(lang, lang)
    return "\n".join(_highlight_code_line(lang, line) for line in code.splitlines())


def _highlight_code_line(language: str, line: str) -> str:
    code_part, comment_part = _split_code_comment(language, line)
    highlighted = html.escape(code_part)

    keyword_patterns = _keyword_patterns(language)
    if keyword_patterns:
        highlighted = re.sub(
            r"\b(" + "|".join(keyword_patterns) + r")\b",
            r'<span class="code-keyword">\1</span>',
            highlighted,
            flags=re.IGNORECASE,
        )

    highlighted = re.sub(r"(&quot;.*?&quot;|&#x27;.*?&#x27;)", r'<span class="code-string">\1</span>', highlighted)
    highlighted = re.sub(r"(?<![\w.-])(\d+)(?![\w.-])", r'<span class="code-number">\1</span>', highlighted)
    highlighted = re.sub(r"(?<!\w)(--[\w-]+)", r'<span class="code-flag">\1</span>', highlighted)
    highlighted = highlighted.replace("&amp;&amp;", '<span class="code-op">&amp;&amp;</span>')
    highlighted = highlighted.replace("||", '<span class="code-op">||</span>')

    if comment_part:
        highlighted += f'<span class="code-comment">{html.escape(comment_part)}</span>'
    return highlighted


def _split_code_comment(language: str, line: str) -> tuple[str, str]:
    comment_markers = ("#",)
    if language in {"js", "javascript", "ts", "typescript", "go", "java", "c", "cpp"}:
        comment_markers = ("//", "#")

    quote: str | None = None
    index = 0
    while index < len(line):
        char = line[index]
        if char in {'"', "'"} and (index == 0 or line[index - 1] != "\\"):
            quote = None if quote == char else char if quote is None else quote
        if quote is None:
            for marker in comment_markers:
                if line.startswith(marker, index):
                    return line[:index], line[index:]
        index += 1
    return line, ""


def _keyword_patterns(language: str) -> tuple[str, ...]:
    if language in {"dockerfile", "docker"}:
        return ("FROM", "RUN", "COPY", "ADD", "WORKDIR", "USER", "CMD", "ENTRYPOINT", "EXPOSE", "ENV", "ARG", "AS")
    if language in {"yaml", "yml"}:
        return ("apiVersion", "kind", "metadata", "spec", "containers", "image", "ports", "env", "values", "resources")
    if language == "nginx":
        return ("server", "location", "listen", "server_name", "proxy_pass", "proxy_set_header", "upstream", "root", "index")
    if language in {"bash", "sh", "shell", "powershell", "ps1"}:
        return ("sudo", "systemctl", "journalctl", "docker", "kubectl", "helm", "terraform", "ansible", "ollama", "python")
    if language in {"hcl", "terraform", "tf"}:
        return ("resource", "module", "variable", "output", "provider", "locals", "data")
    if language in {"groovy", "jenkinsfile"}:
        return ("pipeline", "agent", "stages", "stage", "steps", "environment", "post")
    return ()


def markdown_fragment(markdown: str) -> str:
    rendered = markdown_to_html(markdown)
    return rendered.split("<body>", 1)[1].rsplit("</body>", 1)[0]


def render_message_fragment(markdown: str) -> str:
    stripped = markdown.strip()
    thinking = re.fullmatch(r"\[\[thinking:(\d+)\]\]", stripped)
    if thinking:
        phase = int(thinking.group(1))
        bar = thinking_bar_png(phase, _px(200), _px(6))
        return (
            "<div class='thinking'>"
            "<div class='thinking-label'>Думаю</div>"
            f"<img src='data:image/png;base64,{bar}' width='{_px(200)}' height='{_px(6)}' />"
            "</div>"
        )

    file_pattern = re.compile(r"\[\[file:([^\]]+)\]\]")
    files = file_pattern.findall(markdown)
    screenshot_pattern = re.compile(r"\[\[screenshot:([A-Za-z0-9+/=]+)\]\]")
    screenshots = screenshot_pattern.findall(markdown)
    text = file_pattern.sub("", screenshot_pattern.sub("", markdown)).strip()
    fragments: list[str] = []
    for name in files:
        icon = pixmap_to_base64_png(icon_pixmap("file", _px(18), "#9ad6bd"))
        fragments.append(
            "<div class='file-chip'>"
            f"<img src='data:image/png;base64,{icon}' width='{_px(18)}' height='{_px(18)}' /> "
            f"{html.escape(name)}"
            "</div>"
        )
    for screenshot in screenshots:
        safe_src = html.escape(screenshot, quote=True)
        # Compact thumbnail (GPT/Claude style): a plain inline image in a tight
        # paragraph — no table/card wrapper, so Qt does not add big block spacing
        # around it. A modest fixed width keeps it a neat preview.
        img_w = _px(240)
        fragments.append(
            f"<p class='shot'><img src='data:image/png;base64,{safe_src}' width='{img_w}' /></p>"
        )
    if text:
        fragments.append(markdown_fragment(text))
    return "".join(fragments)


def render_user_message_fragment(markdown: str) -> str:
    # Older sessions stored a "Вопрос N" prefix; strip it so no label/stripe is shown.
    match = re.match(r"^Вопрос\s+\d+\s*\n\n(.+)$", markdown.strip(), flags=re.DOTALL)
    body = match.group(1).strip() if match else markdown
    return render_message_fragment(body)


def text_to_html(text: str) -> str:
    escaped = html.escape(text.strip())
    if not escaped:
        return ""

    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(
        r"(https?://[^\s<]+)",
        lambda m: f'<a href="{m.group(1)}" style="color:{BLUE};text-decoration:none">{m.group(1)}</a>',
        escaped,
    )
    lines = escaped.splitlines()
    out: list[str] = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                out.append("</ul>")
                in_list = False
            continue

        if stripped.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{stripped[2:]}</li>")
            continue

        numbered = re.match(r"^\d+\.\s+(.+)$", stripped)
        if numbered:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{numbered.group(1)}</li>")
            continue

        if in_list:
            out.append("</ul>")
            in_list = False

        if stripped.endswith(":") and len(stripped) < 80:
            out.append(f"<h2>{stripped}</h2>")
        else:
            out.append(f"<p>{stripped}</p>")

    if in_list:
        out.append("</ul>")

    return "".join(out)


class PromptEdit(QTextEdit):
    submitted = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.keep_arrow_cursor()

    def keep_arrow_cursor(self) -> None:
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.viewport().setCursor(Qt.CursorShape.ArrowCursor)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.submitted.emit()
            return
        super().keyPressEvent(event)

    def enterEvent(self, event) -> None:  # noqa: ANN001
        self.keep_arrow_cursor()
        super().enterEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        self.keep_arrow_cursor()
        super().mouseMoveEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.accept()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        event.accept()


class LoginDialog(QDialog):
    """Sign-in / registration gate. Talks to the auth server via app.auth_client."""

    def __init__(self, parent: QWidget | None, *, default_username: str = "") -> None:
        super().__init__(parent)
        from app import auth_client

        self._auth_client = auth_client
        self.token = ""
        self.username = ""
        self._mode = "login"  # or "register"

        self.setWindowTitle("Stackwire — вход")
        self.setModal(True)
        self.setObjectName("settingsDialog")
        self.setMinimumWidth(_px(420))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(_px(22), _px(20), _px(22), _px(18))
        layout.setSpacing(_px(12))

        title = QLabel("Вход в Stackwire")
        title.setObjectName("dialogTitle")
        self._subtitle = QLabel("Войдите в аккаунт, чтобы пользоваться чатом.")
        self._subtitle.setObjectName("dialogNote")
        self._subtitle.setWordWrap(True)

        form = QFormLayout()
        form.setHorizontalSpacing(_px(12))
        form.setVerticalSpacing(_px(10))

        self.username_edit = QLineEdit()
        self.username_edit.setObjectName("settingsCombo")
        self.username_edit.setPlaceholderText("логин")
        self.username_edit.setText(default_username)
        self.password_edit = QLineEdit()
        self.password_edit.setObjectName("settingsCombo")
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText("пароль")
        self.password_edit.returnPressed.connect(self._submit)

        server_label = self._auth_client.auth_base_url()
        self._server_hint = QLabel(f"Сервер: {server_label}")
        self._server_hint.setObjectName("dialogNote")
        self._server_hint.setWordWrap(True)

        form.addRow("Логин", self.username_edit)
        form.addRow("Пароль", self.password_edit)

        self._error = QLabel("")
        self._error.setObjectName("dialogError")
        self._error.setWordWrap(True)
        self._error.setVisible(False)

        actions = QHBoxLayout()
        self._toggle_button = QPushButton("Создать аккаунт")
        self._toggle_button.setObjectName("ghostButton")
        self._toggle_button.clicked.connect(self._toggle_mode)
        actions.addWidget(self._toggle_button)
        actions.addStretch(1)
        self._primary = QPushButton("Войти")
        self._primary.setObjectName("dialogPrimaryButton")
        self._primary.clicked.connect(self._submit)
        actions.addWidget(self._primary)

        layout.addWidget(title)
        layout.addWidget(self._subtitle)
        layout.addLayout(form)
        layout.addWidget(self._server_hint)
        layout.addWidget(self._error)
        layout.addLayout(actions)

    def _toggle_mode(self) -> None:
        self._mode = "register" if self._mode == "login" else "login"
        if self._mode == "register":
            self._primary.setText("Зарегистрироваться")
            self._toggle_button.setText("У меня уже есть аккаунт")
            self._subtitle.setText("Создайте локальный аккаунт (мин. 6 символов в пароле).")
        else:
            self._primary.setText("Войти")
            self._toggle_button.setText("Создать аккаунт")
            self._subtitle.setText("Войдите в аккаунт, чтобы пользоваться чатом.")
        self._error.setVisible(False)

    def _show_error(self, message: str) -> None:
        self._error.setText(message)
        self._error.setVisible(True)

    def _submit(self) -> None:
        username = self.username_edit.text().strip()
        password = self.password_edit.text()
        if not username or not password:
            self._show_error("Введите логин и пароль.")
            return
        self._primary.setEnabled(False)
        try:
            if self._mode == "register":
                creds = self._auth_client.register(username, password)
            else:
                creds = self._auth_client.login(username, password)
        except self._auth_client.AuthClientError as exc:
            self._primary.setEnabled(True)
            self._show_error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self._primary.setEnabled(True)
            self._show_error(_short_error(exc))
            return
        self.token = creds.token
        self.username = creds.username
        self.accept()


class SettingsDialog(QDialog):
    """Detailed, tabbed settings — everything editable from the UI, no env hand-editing."""

    def __init__(self, parent: QWidget, models: list[str], audio_devices: list[str], current_audio_device: str) -> None:
        super().__init__(parent)
        self._window = parent
        self.setWindowTitle("Stackwire settings")
        self.setModal(True)
        self.setObjectName("settingsDialog")
        self.setMinimumWidth(_px(560))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(_px(18), _px(16), _px(18), _px(16))
        layout.setSpacing(_px(12))

        heading = QLabel("Настройки")
        heading.setObjectName("dialogTitle")

        tabs = QTabWidget()
        tabs.setObjectName("settingsTabs")
        tabs.addTab(self._build_account_tab(), "Аккаунт")
        tabs.addTab(self._build_models_tab(models), "Модели")
        tabs.addTab(self._build_speech_tab(audio_devices, current_audio_device), "Речь")
        tabs.addTab(self._build_knowledge_tab(), "База и поиск")

        actions = QHBoxLayout()
        actions.addStretch(1)
        cancel = QPushButton("Отмена")
        cancel.setObjectName("ghostButton")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Сохранить")
        save.setObjectName("dialogPrimaryButton")
        save.clicked.connect(self.accept)
        actions.addWidget(cancel)
        actions.addWidget(save)

        layout.addWidget(heading)
        layout.addWidget(tabs)
        layout.addLayout(actions)

    # -- tabs ----------------------------------------------------------- #
    def _form_page(self) -> tuple[QWidget, QFormLayout]:
        page = QWidget()
        form = QFormLayout(page)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(_px(12))
        form.setVerticalSpacing(_px(10))
        form.setContentsMargins(_px(6), _px(10), _px(6), _px(6))
        return page, form

    def _build_account_tab(self) -> QWidget:
        page, form = self._form_page()
        authed = bool(getattr(self._window, "authenticated", False))
        username = str(getattr(self._window, "auth_username", "") or "")
        required = bool(getattr(self._window, "auth_required", False))

        status = "не требуется" if not required else (f"вошёл как {username}" if authed else "не выполнен вход")
        self._account_status = QLabel(status)
        self._account_status.setObjectName("dialogNote")
        form.addRow("Статус", self._account_status)

        self.auth_url_edit = QLineEdit()
        self.auth_url_edit.setObjectName("settingsCombo")
        from app import auth_client

        self.auth_url_edit.setText(os.getenv("STACKWIRE_AUTH_URL", "").strip() or auth_client.auth_base_url())
        self.auth_url_edit.setPlaceholderText("http://127.0.0.1:8000")
        form.addRow("Сервер авторизации", self.auth_url_edit)

        buttons = QHBoxLayout()
        self._login_button = QPushButton("Выйти" if authed else "Войти")
        self._login_button.setObjectName("ghostButton")
        self._login_button.clicked.connect(self._account_action)
        buttons.addWidget(self._login_button)
        buttons.addStretch(1)
        holder = QWidget()
        holder.setLayout(buttons)
        form.addRow("", holder)

        note = QLabel("Аккаунт нужен один раз — токен запоминается и подставляется автоматически.")
        note.setObjectName("dialogNote")
        note.setWordWrap(True)
        form.addRow("", note)
        return page

    def _account_action(self) -> None:
        window = self._window
        if getattr(window, "authenticated", False):
            if hasattr(window, "logout"):
                window.logout()
        elif hasattr(window, "prompt_login"):
            window.prompt_login()
        authed = bool(getattr(window, "authenticated", False))
        username = str(getattr(window, "auth_username", "") or "")
        self._login_button.setText("Выйти" if authed else "Войти")
        self._account_status.setText(f"вошёл как {username}" if authed else "не выполнен вход")

    def _build_models_tab(self, models: list[str]) -> QWidget:
        page, form = self._form_page()
        self.answer_model = self._model_combo(models, current_answer_model())
        self.recovery_model = self._model_combo(models, current_recovery_model())
        self.vision_model = self._model_combo(models, current_vision_model())
        self.answer_mode = NoWheelComboBox()
        self.answer_mode.setObjectName("settingsCombo")
        self.answer_mode.addItems(["normal", "deep"])
        self.answer_mode.setCurrentText(os.getenv("ANSWER_MODE", os.getenv("STACKWIRE_ANSWER_MODE", "normal")).strip().lower() or "normal")
        form.addRow("Ответы", self.answer_model)
        form.addRow("Распознавание вопроса", self.recovery_model)
        form.addRow("Зрение (скриншоты)", self.vision_model)
        form.addRow("Режим ответа", self.answer_mode)
        return page

    def _build_speech_tab(self, devices: list[str], current_device: str) -> QWidget:
        page, form = self._form_page()
        self.audio_device = self._audio_combo(devices, current_device)
        self.language_mode = NoWheelComboBox()
        self.language_mode.setObjectName("settingsCombo")
        self.language_mode.addItems(["auto", "ru", "en"])
        self.language_mode.setCurrentText((os.getenv("STT_LANGUAGE_MODE", "auto").strip().lower() or "auto"))
        self.beam_size = NoWheelComboBox()
        self.beam_size.setObjectName("settingsCombo")
        self.beam_size.addItems(["1", "3", "5", "8"])
        self.beam_size.setCurrentText(os.getenv("WHISPER_BEAM_SIZE", "5").strip() or "5")
        self.vad_filter = QCheckBox("Фильтр тишины (VAD)")
        self.vad_filter.setChecked(os.getenv("WHISPER_VAD_FILTER", "1").strip().lower() in {"1", "true", "yes", "on"})
        form.addRow("Аудио-устройство", self.audio_device)
        form.addRow("Язык речи", self.language_mode)
        form.addRow("Точность (beam)", self.beam_size)
        form.addRow("", self.vad_filter)
        hint = QLabel("Зафиксируйте язык (ru/en), если речь на одном языке — точность выше.")
        hint.setObjectName("dialogNote")
        hint.setWordWrap(True)
        form.addRow("", hint)
        return page

    def _build_knowledge_tab(self) -> QWidget:
        page, form = self._form_page()
        self.web_search = QCheckBox("Искать в интернете при неуверенности (DuckDuckGo)")
        self.web_search.setChecked(os.getenv("STACKWIRE_WEB_SEARCH", "1").strip().lower() not in {"0", "false", "no", "off"})
        self.remember_answers = QCheckBox("Запоминать ответы в локальную базу")
        self.remember_answers.setChecked(os.getenv("STACKWIRE_REMEMBER_ANSWERS", "1").strip().lower() not in {"0", "false", "no", "off"})
        form.addRow("", self.web_search)
        form.addRow("", self.remember_answers)

        try:
            from app import vectorstore

            info = vectorstore.stats()
            if info.get("available"):
                state = f"Qdrant: {info.get('points', 0)} записей · модель {str(info.get('model','')).split('/')[-1]}"
            else:
                state = "Векторная база выключена (нет qdrant-client/fastembed)."
        except Exception:
            state = "Векторная база недоступна."
        self._vector_state = QLabel(state)
        self._vector_state.setObjectName("dialogNote")
        self._vector_state.setWordWrap(True)
        form.addRow("Локальная база", self._vector_state)

        reindex = QPushButton("Переиндексировать знания")
        reindex.setObjectName("ghostButton")
        reindex.clicked.connect(self._reindex)
        holder = QWidget()
        hl = QHBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(reindex)
        hl.addStretch(1)
        form.addRow("", holder)
        return page

    def _reindex(self) -> None:
        try:
            from app import vectorstore

            vectorstore.ensure_indexed(force=True)
            info = vectorstore.stats()
            self._vector_state.setText(f"Готово · {info.get('points', 0)} записей в базе.")
        except Exception as exc:  # noqa: BLE001
            self._vector_state.setText(f"Ошибка переиндексации: {_short_error(exc)}")

    # -- helpers -------------------------------------------------------- #
    def _model_combo(self, models: list[str], current: str) -> NoWheelComboBox:
        combo = NoWheelComboBox()
        combo.setEditable(False)
        combo.setObjectName("settingsCombo")
        choices = _dedupe_models([current, *models])
        combo.addItems(choices)
        combo.setCurrentText(current)
        return combo

    def _audio_combo(self, devices: list[str], current: str) -> NoWheelComboBox:
        combo = NoWheelComboBox()
        combo.setEditable(False)
        combo.setObjectName("settingsCombo")
        choices = _dedupe_models([current, *devices])
        combo.addItems(choices)
        combo.setCurrentText(current)
        return combo

    def values(self) -> dict[str, str]:
        recovery = self.recovery_model.currentText().strip()
        return {
            "ANSWER_MODEL": self.answer_model.currentText().strip(),
            "RECOVERY_MODEL": recovery,
            "FAST_RECOVERY_MODEL": recovery,
            "VISION_MODEL": self.vision_model.currentText().strip(),
            "ANSWER_MODE": self.answer_mode.currentText().strip(),
            "STACKWIRE_AUDIO_DEVICE": self.audio_device.currentText().strip(),
            "STT_LANGUAGE_MODE": self.language_mode.currentText().strip(),
            "WHISPER_BEAM_SIZE": self.beam_size.currentText().strip(),
            "WHISPER_VAD_FILTER": "1" if self.vad_filter.isChecked() else "0",
            "STACKWIRE_WEB_SEARCH": "1" if self.web_search.isChecked() else "0",
            "STACKWIRE_REMEMBER_ANSWERS": "1" if self.remember_answers.isChecked() else "0",
            "STACKWIRE_AUTH_URL": self.auth_url_edit.text().strip(),
        }


class AnswerBrowser(QTextBrowser):
    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            event.accept()
            return
        super().wheelEvent(event)


class NeuralBackground(QWidget):
    """Animated particle / neural-network backdrop shown on the welcome screen.

    A drifting set of nodes connected by fading lines, plus a sparse dust layer.
    Painted natively with QPainter, transparent to mouse events, and only ticks
    while visible so it costs nothing during a conversation.
    """

    NODE_COUNT = 40
    DUST_COUNT = 48
    LINK_DISTANCE = 150.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._nodes: list[list[float]] = []  # [x, y, vx, vy]
        self._dust: list[list[float]] = []  # [x, y, vx, vy, radius]
        self._w = 0.0
        self._h = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60fps for smooth motion
        self._timer.timeout.connect(self._tick)

    def _seed(self) -> None:
        w = max(1, self.width())
        h = max(1, self.height())
        self._w, self._h = float(w), float(h)
        self._nodes = [
            [random.uniform(0, w), random.uniform(0, h), random.uniform(-0.14, 0.14), random.uniform(-0.14, 0.14)]
            for _ in range(self.NODE_COUNT)
        ]
        self._dust = [
            [random.uniform(0, w), random.uniform(0, h), random.uniform(-0.06, 0.06), random.uniform(-0.10, 0.02), random.uniform(0.6, 1.9)]
            for _ in range(self.DUST_COUNT)
        ]

    def start(self) -> None:
        if not self._nodes:
            self._seed()
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _tick(self) -> None:
        w = max(1.0, self._w)
        h = max(1.0, self._h)
        for node in self._nodes:
            node[0] += node[2]
            node[1] += node[3]
            if node[0] <= 0 or node[0] >= w:
                node[2] *= -1
                node[0] = min(max(node[0], 0.0), w)
            if node[1] <= 0 or node[1] >= h:
                node[3] *= -1
                node[1] = min(max(node[1], 0.0), h)
        for dust in self._dust:
            dust[0] = (dust[0] + dust[2]) % (w + 6)
            dust[1] = (dust[1] + dust[3]) % (h + 6)
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        new_w = float(max(1, self.width()))
        new_h = float(max(1, self.height()))
        # Rescale existing points instead of reseeding so the field does not
        # shimmer/jump while the window is being dragged.
        if self._nodes and self._w > 0 and self._h > 0:
            sx = new_w / self._w
            sy = new_h / self._h
            for node in self._nodes:
                node[0] *= sx
                node[1] *= sy
            for dust in self._dust:
                dust[0] *= sx
                dust[1] *= sy
        self._w, self._h = new_w, new_h
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        if not self._nodes:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(154, 214, 189, 22))
        for dust in self._dust:
            painter.drawEllipse(QPointF(dust[0], dust[1]), dust[4], dust[4])

        link = self.LINK_DISTANCE
        for i, a in enumerate(self._nodes):
            ax, ay = a[0], a[1]
            for b in self._nodes[i + 1 :]:
                dx = ax - b[0]
                dy = ay - b[1]
                dist2 = dx * dx + dy * dy
                if dist2 >= link * link:
                    continue
                alpha = int(44 * (1.0 - (dist2 ** 0.5) / link))
                if alpha <= 0:
                    continue
                painter.setPen(QPen(QColor(154, 214, 189, alpha), 1.0))
                painter.drawLine(QLineF(ax, ay, b[0], b[1]))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(154, 214, 189, 130))
        for node in self._nodes:
            painter.drawEllipse(QPointF(node[0], node[1]), 2.2, 2.2)
        painter.end()


class ThinkingDots(QWidget):
    """Three softly pulsing mint dots shown while the assistant is thinking."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._phase = 0.0
        self.setFixedHeight(_px(20))
        self.setMinimumWidth(_px(60))
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        self._phase += 0.16
        self.update()

    def stop(self) -> None:
        self._timer.stop()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        radius = _px(4)
        gap = _px(15)
        cx = radius + _px(2)
        cy = self.height() / 2
        for index in range(3):
            wave = math.sin(self._phase - index * 0.7) * 0.5 + 0.5
            alpha = int(90 + 150 * wave)
            scale = 0.65 + 0.5 * wave
            painter.setBrush(QColor(154, 214, 189, alpha))
            r = radius * scale
            painter.drawEllipse(QPointF(cx + index * gap, cy), r, r)
        painter.end()


class ChatMessageBrowser(QTextBrowser):
    """A per-message rich-text view that auto-sizes to its content height so the
    outer scroll area (not the browser) does the scrolling."""

    def __init__(self, on_anchor) -> None:  # noqa: ANN001
        super().__init__()
        self.setObjectName("msgBrowser")
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.viewport().setAutoFillBackground(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.anchorClicked.connect(on_anchor)

    def set_html(self, markup: str) -> None:
        self.setHtml(markup)
        self._fit()

    def _fit(self) -> None:
        doc = self.document()
        doc.setTextWidth(max(1, self.viewport().width()))
        self.setFixedHeight(int(doc.size().height()) + _px(4))

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._fit()

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()  # let the outer scroll area handle scrolling


class AssistantRow(QWidget):
    """Assistant message: 'Ассистент' label + content (thinking dots or rich text) + copy."""

    def __init__(self, index: int, on_anchor, on_copy) -> None:  # noqa: ANN001
        super().__init__()
        self.index = index
        self._on_anchor = on_anchor
        self.browser: ChatMessageBrowser | None = None
        self.thinking: ThinkingDots | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(_px(2), _px(2), _px(2), _px(2))
        layout.setSpacing(_px(4))
        # Role line: small brand avatar + "Ассистент".
        role_row = QHBoxLayout()
        role_row.setContentsMargins(0, 0, 0, 0)
        role_row.setSpacing(_px(7))
        avatar = QLabel()
        avatar.setPixmap(icon_pixmap("mark", _px(16), ACCENT))
        role = QLabel("Ассистент")
        role.setObjectName("roleLabel")
        role_row.addWidget(avatar)
        role_row.addWidget(role)
        role_row.addStretch(1)
        layout.addLayout(role_row)
        self._holder = QWidget()
        self._hl = QVBoxLayout(self._holder)
        self._hl.setContentsMargins(0, 0, 0, 0)
        self._hl.setSpacing(0)
        layout.addWidget(self._holder)
        self._copy = _flat_icon_button("copy", "Скопировать", lambda: on_copy(self.index))
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addWidget(self._copy)
        actions.addStretch(1)
        self._actions_holder = QWidget()
        self._actions_holder.setLayout(actions)
        layout.addWidget(self._actions_holder)
        self._actions_holder.setVisible(False)

    def _clear_holder(self) -> None:
        while self._hl.count():
            item = self._hl.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)  # remove from view immediately (deleteLater is async)
                widget.deleteLater()
        self.thinking = None
        self.browser = None

    def show_thinking(self) -> None:
        self._clear_holder()
        self.thinking = ThinkingDots()
        self._hl.addWidget(self.thinking)
        self._actions_holder.setVisible(False)

    def show_html(self, markup: str, *, final: bool = False) -> None:
        if self.browser is None:
            self._clear_holder()
            self.browser = ChatMessageBrowser(self._on_anchor)
            self._hl.addWidget(self.browser)
        self.browser.set_html(markup)
        self._actions_holder.setVisible(final)


class ChatArea(QWidget):
    """Modern chat surface: an animated welcome backdrop when empty, and a scrollable
    column of message-bubble widgets when there is a conversation."""

    def __init__(self, background: QWidget) -> None:
        super().__init__()
        self.setObjectName("chatArea")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._background = background
        background.setParent(self)

        self.welcome = QWidget(self)
        self.welcome.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        wl = QVBoxLayout(self.welcome)
        wl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wl.setSpacing(_px(8))
        logo = QLabel()
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setPixmap(icon_pixmap("mark", _px(56), ACCENT))
        title = QLabel("Stackwire")
        title.setObjectName("welcomeTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub = QLabel("Think faster. Work locally.")
        sub.setObjectName("welcomeSub")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wl.addWidget(logo)
        wl.addWidget(title)
        wl.addWidget(sub)

        self.scroll = QScrollArea(self)
        self.scroll.setObjectName("chatScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.container = QWidget()
        self.container.setObjectName("chatContainer")
        # Full-width column: assistant rows hug the left, user bubbles hug the right.
        self.col = QVBoxLayout(self.container)
        self.col.setContentsMargins(_px(16), _px(12), _px(16), _px(12))
        self.col.setSpacing(_px(8))
        self.col.addStretch(1)
        self.scroll.setWidget(self.container)

        background.lower()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        rect = self.rect()
        self._background.setGeometry(rect)
        self.welcome.setGeometry(rect)
        self.scroll.setGeometry(rect)
        super().resizeEvent(event)

    def add_row(self, widget: QWidget) -> None:
        # Insert before the trailing stretch so messages stack top-to-bottom.
        self.col.insertWidget(self.col.count() - 1, widget)

    def clear_rows(self) -> None:
        while self.col.count() > 1:  # keep the trailing stretch
            item = self.col.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)  # remove from view immediately (deleteLater is async)
                widget.deleteLater()

    def show_welcome(self) -> None:
        self._background.show()
        self._background.start()
        self.welcome.show()
        self.welcome.raise_()
        self.scroll.hide()

    def show_list(self) -> None:
        self._background.stop()
        self._background.hide()
        self.welcome.hide()
        self.scroll.show()
        self.scroll.raise_()

    def scroll_to_bottom(self) -> None:
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())


def _flat_icon_button(kind: str, tooltip: str, on_click) -> QPushButton:  # noqa: ANN001
    button = QPushButton()
    button.setObjectName("msgActionButton")
    button.setToolTip(tooltip)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setFixedSize(_px(26), _px(26))
    button.setIcon(make_icon(kind, _px(15), "#8290a0"))
    button.clicked.connect(lambda: on_click())
    return button


def _soft_shadow(widget: QWidget, *, blur: int = 28, dy: int = 8, alpha: int = 130) -> None:
    """Attach a soft drop shadow. NOTE: a widget can hold only one QGraphicsEffect,
    so never combine this with a fade-in opacity effect on the same widget."""
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setOffset(0, dy)
    effect.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(effect)


def _animate_in(widget: QWidget, *, duration: int = 170) -> None:
    """Fade a freshly added row in (opacity 0→1)."""
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    # Drop the effect when done so it doesn't keep intercepting paints.
    anim.finished.connect(lambda: widget.setGraphicsEffect(None))
    widget._fade_anim = anim  # keep a reference so it isn't GC'd  # type: ignore[attr-defined]
    anim.start()


_CARET_CACHE: dict[str, str] = {}


def _caret_png() -> str:
    """Cached base64 PNG of a small mint caret bar (the bundled fonts lack a block glyph)."""
    if "png" not in _CARET_CACHE:
        pixmap = QPixmap(6, 26)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(ACCENT))
        painter.drawRoundedRect(1, 2, 3, 22, 2, 2)
        painter.end()
        _CARET_CACHE["png"] = pixmap_to_base64_png(pixmap)
    return _CARET_CACHE["png"]


def _with_stream_caret(markup: str) -> str:
    """Insert a small caret image at the end of the last paragraph of rendered HTML."""
    caret = f"<img src='data:image/png;base64,{_caret_png()}' width='{_px(3)}' height='{_px(15)}' />"
    idx = markup.rfind("</p>")
    if idx != -1:
        return markup[:idx] + caret + markup[idx:]
    return markup.replace("</body>", caret + "</body>", 1)


def _rounded_pixmap(pixmap: QPixmap, radius: int) -> QPixmap:
    """Return a copy of pixmap with rounded corners (mask via QPainterPath)."""
    if pixmap.isNull():
        return pixmap
    rounded = QPixmap(pixmap.size())
    rounded.fill(Qt.GlobalColor.transparent)
    painter = QPainter(rounded)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(0, 0, pixmap.width(), pixmap.height(), radius, radius)
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, pixmap)
    painter.end()
    return rounded


class ActionPopup(QFrame):
    def __init__(self, parent: QWidget, items: tuple[tuple[str, str], ...], callback) -> None:  # noqa: ANN001
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("actionPopup")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        for value, label in items:
            button = QPushButton(label)
            button.setObjectName("popupButton")
            button.clicked.connect(lambda _checked=False, selected=value: self._select(callback, selected))
            layout.addWidget(button)

    def _select(self, callback, value: str) -> None:  # noqa: ANN001
        self.hide()
        callback(value)

    def show_below(self, anchor: QWidget) -> None:
        self.adjustSize()
        point = anchor.mapToGlobal(anchor.rect().bottomLeft())
        self.move(point.x(), point.y() + _px(4))
        self.show()
        self.raise_()


def _request_response_text(exc: RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return ""
    try:
        return str(response.text or "")
    except Exception:
        return ""


def _remote_request_error(prefix: str, api_url: str, exc: RequestException) -> str:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    response_text = _request_response_text(exc)
    error_text = f"{exc} {response_text}".lower()
    if status_code == 403 and "requires a subscription" in response_text.lower():
        return (
            f"{prefix}: Ollama cloud model access was denied. "
            "The selected cloud model manifest exists locally, but the chat call requires an Ollama account/subscription. "
            "Run `ollama signin`, check https://ollama.com/upgrade, or switch ANSWER_MODEL/RECOVERY_MODEL back to local models. "
            f"Details: {response_text or exc}"
        )

    if not api_url:
        if "remote end closed connection without response" in error_text or "connection aborted" in error_text:
            return (
                f"{prefix}: local Ollama accepted the connection but closed the chat request without a response. "
                "This usually happens when the selected model is too heavy, still loading, or the Ollama runner crashed. "
                "Try the request again after the model finishes loading, restart Ollama, or switch ANSWER_MODEL/RECOVERY_MODEL to a smaller local model. "
                f"Current details: {exc}"
            )
        return (
            f"{prefix}: local Ollama is not available at 127.0.0.1:11434. "
            "Start Ollama, or run start_client.bat <SERVER_IP> <PORT> to use remote Stackwire API. "
            "If Ollama is installed on this PC, run: ollama serve. "
            f"Details: {exc}"
        )

    return (
        f"{prefix}: remote Stackwire API is unavailable at {api_url}. "
        "Start start_server.bat on the server PC, check SERVER_IP/firewall, "
        "or unset STACKWIRE_API_URL to use local mode. "
        f"Details: {exc}"
    )

class AskWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, raw_text: str, context: list[str], *, trusted_text: bool = False, storage_session_id: int | None = None) -> None:
        super().__init__()
        self.raw_text = raw_text
        self.context = context
        self.trusted_text = trusted_text
        self.storage_session_id = storage_session_id
        self.api_url = STACKWIRE_API_URL
        self.client = None if self.api_url else OllamaClient(storage_session_id=storage_session_id)
        self.session = requests.Session()
        self.session.trust_env = False

    @Slot()
    def run(self) -> None:
        try:
            if self.api_url:
                self.finished.emit(self._ask_remote())
                return
            if self.client is None:
                raise RuntimeError("Local Ollama client is not initialized")
            self.finished.emit(self.client.ask(self.raw_text, self.context, trusted_text=self.trusted_text))
        except RequestException as exc:
            self.failed.emit(_remote_request_error("Processing request failed", self.api_url, exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _ask_remote(self) -> AskResult:
        payload = json.dumps(
            {"text": self.raw_text, "context": self.context, "trusted_text": self.trusted_text},
            ensure_ascii=False,
        ).encode("utf-8")

        response = self.session.post(
            f"{self.api_url}/ask",
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8", **_auth_headers()},
            timeout=(STACKWIRE_API_CONNECT_TIMEOUT, STACKWIRE_API_TIMEOUT),
        )
        self._raise_for_status(response, "/ask")
        data = response.json()
        recovery_payload = data.get("recovery") or {}
        technical_entities = recovery_payload.get("technical_entities") or []
        ambiguities = recovery_payload.get("ambiguities") or []
        candidate_questions = recovery_payload.get("candidate_questions") or []
        candidate_details = recovery_payload.get("candidate_details") or []
        recovery = RecoveryResult(
            confidence=self._as_float(recovery_payload.get("confidence"), 0.0),
            recovered_question=str(recovery_payload.get("recovered_question", "")),
            detected_topic=str(recovery_payload.get("detected_topic", "NEED_CLARIFICATION")),
            reason=str(recovery_payload.get("reason", "")),
            technical_entities=[str(item) for item in technical_entities if str(item).strip()],
            ambiguities=[str(item) for item in ambiguities if str(item).strip()],
            needs_manual_fix=bool(recovery_payload.get("needs_manual_fix", False)),
            candidate_questions=[str(item) for item in candidate_questions if str(item).strip()],
            candidate_quality=str(recovery_payload.get("candidate_quality", "unclear")),
            candidate_details=[item for item in candidate_details if isinstance(item, dict)],
        )
        return AskResult(
            raw_text=str(data.get("raw_text", self.raw_text)),
            recovery=recovery,
            answer=str(data.get("answer", "")),
            answered=bool(data.get("answered", False)),
            recovery_latency=self._as_float(data.get("recovery_latency"), 0.0),
            answer_latency=self._as_float(data.get("answer_latency"), 0.0),
            total_latency=self._as_float(data.get("total_latency"), 0.0),
            question_id=self._as_int(data.get("question_id")),
            answer_id=self._as_int(data.get("answer_id")),
            plan_domain=str(data.get("plan_domain") or "") or None,
            plan_intent=str(data.get("plan_intent") or "") or None,
        )

    def _as_int(self, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, int | float | str):
            try:
                parsed = int(value)
            except ValueError:
                return None
            return parsed if parsed > 0 else None
        return None

    def _as_float(self, value: object, default: float) -> float:
        if isinstance(value, int | float | str):
            try:
                return float(value)
            except ValueError:
                return default
        return default

    def _raise_for_status(self, response: requests.Response, endpoint: str) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text
            try:
                payload = response.json()
                detail = str(payload.get("detail", payload))
            except ValueError:
                pass
            raise RuntimeError(
                f"Remote API {endpoint} returned {response.status_code}: {detail[:500]}"
            ) from exc


class AskStreamWorker(QObject):
    """Streams the answer token-by-token from local Ollama; falls back to a single
    non-streamed response in remote API mode (server has no streaming endpoint)."""

    recovered = Signal(str)
    delta = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, raw_text: str, context: list[str], *, trusted_text: bool = False, storage_session_id: int | None = None) -> None:
        super().__init__()
        self.raw_text = raw_text
        self.context = context
        self.trusted_text = trusted_text
        self.storage_session_id = storage_session_id
        self.api_url = STACKWIRE_API_URL
        self.client = None if self.api_url else OllamaClient(storage_session_id=storage_session_id)
        self.session = requests.Session()
        self.session.trust_env = False

    @Slot()
    def run(self) -> None:
        try:
            if self.api_url:
                remote = AskWorker(self.raw_text, self.context, trusted_text=self.trusted_text, storage_session_id=self.storage_session_id)
                result = remote._ask_remote()
                if result.answer:
                    self.delta.emit(result.answer)
                self.finished.emit(result)
                return
            if self.client is None:
                raise RuntimeError("Local Ollama client is not initialized")
            result = self.client.ask_stream(
                self.raw_text,
                self.context,
                trusted_text=self.trusted_text,
                on_recovery=self.recovered.emit,
                on_delta=self.delta.emit,
            )
            self.finished.emit(result)
        except RequestException as exc:
            self.failed.emit(_remote_request_error("Processing request failed", self.api_url, exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ExpandWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, question: str, previous_answer: str, mode: str, storage_session_id: int | None = None) -> None:
        super().__init__()
        self.question = question
        self.previous_answer = previous_answer
        self.mode = mode
        self.storage_session_id = storage_session_id
        self.api_url = STACKWIRE_API_URL
        self.client = None if self.api_url else OllamaClient(storage_session_id=storage_session_id)
        self.session = requests.Session()
        self.session.trust_env = False

    @Slot()
    def run(self) -> None:
        try:
            if self.api_url:
                self.finished.emit(self._expand_remote())
                return
            if self.client is None:
                raise RuntimeError("Local Ollama client is not initialized")
            self.finished.emit(self.client.expand(self.question, self.previous_answer, self.mode))
        except RequestException as exc:
            self.failed.emit(_remote_request_error("Expand request failed", self.api_url, exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _expand_remote(self) -> ExpandResult:
        payload = {
            "question": self.question,
            "previous_answer": self.previous_answer,
            "mode": self.mode,
        }
        response = self.session.post(
            f"{self.api_url}/expand",
            json=cast(Any, payload),
            headers=_auth_headers(),
            timeout=(STACKWIRE_API_CONNECT_TIMEOUT, STACKWIRE_API_TIMEOUT),
        )
        self._raise_for_status(response, "/expand")
        data = response.json()
        return ExpandResult(
            question=self.question,
            previous_answer=self.previous_answer,
            answer=str(data.get("answer", "")),
            mode=str(data.get("mode", self.mode)),
            latency=self._as_float(data.get("latency"), 0.0),
            question_id=self._as_int(data.get("question_id")),
            answer_id=self._as_int(data.get("answer_id")),
            plan_domain=str(data.get("plan_domain") or "") or None,
            plan_intent=str(data.get("plan_intent") or "") or None,
        )

    def _as_int(self, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, int | float | str):
            try:
                parsed = int(value)
            except ValueError:
                return None
            return parsed if parsed > 0 else None
        return None

    def _as_float(self, value: object, default: float) -> float:
        if isinstance(value, int | float | str):
            try:
                return float(value)
            except ValueError:
                return default
        return default

    def _raise_for_status(self, response: requests.Response, endpoint: str) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text
            try:
                payload = response.json()
                detail = str(payload.get("detail", payload))
            except ValueError:
                pass
            raise RuntimeError(
                f"Remote API {endpoint} returned {response.status_code}: {detail[:500]}"
            ) from exc


class ImageAnalysisWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, image_b64: str, prompt: str) -> None:
        super().__init__()
        self.image_b64 = image_b64
        self.prompt = prompt
        self.api_url = STACKWIRE_API_URL
        self.client = None if self.api_url else OllamaClient()
        self.session = requests.Session()
        self.session.trust_env = False

    @Slot()
    def run(self) -> None:
        try:
            if self.api_url:
                self.finished.emit(self._analyze_remote())
                return
            if self.client is None:
                raise RuntimeError("Local Ollama client is not initialized")
            self.finished.emit(self.client.analyze_image(self.image_b64, self.prompt))
        except RequestException as exc:
            self.failed.emit(_remote_request_error("Image analysis request failed", self.api_url, exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _analyze_remote(self) -> str:
        payload: dict[str, str] = {"image_b64": self.image_b64, "prompt": self.prompt}
        response = self.session.post(
            f"{self.api_url}/analyze-image",
            json=cast(Any, payload),
            headers=_auth_headers(),
            timeout=(STACKWIRE_API_CONNECT_TIMEOUT, STACKWIRE_API_TIMEOUT),
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text
            try:
                payload = response.json()
                detail = str(payload.get("detail", payload))
            except ValueError:
                pass
            raise RuntimeError(
                f"Remote API /analyze-image returned {response.status_code}: {detail[:500]}"
            ) from exc
        data = response.json()
        return str(data.get("answer", "")).strip()


class RegionSelector(QWidget):
    captured = Signal(str)
    canceled = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.origin: QPoint | None = None
        self.selection = QRect(0, 0, 0, 0)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setGeometry(self._virtual_geometry())

    def _rect_from_points(self, start: QPoint, end: QPoint) -> QRect:
        return QRect(
            start.x(),
            start.y(),
            end.x() - start.x(),
            end.y() - start.y(),
        ).normalized()

    def _virtual_geometry(self) -> QRect:
        screens = QApplication.screens()
        if not screens:
            return QRect(0, 0, 1, 1)
        geometry = screens[0].geometry()
        for screen in screens[1:]:
            geometry = geometry.united(screen.geometry())
        return geometry

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.canceled.emit()
            self.close()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = event.globalPosition().toPoint()
        self.origin = point
        self.selection = QRect(point.x(), point.y(), 1, 1)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self.origin is None:
            return
        self.selection = self._rect_from_points(self.origin, event.globalPosition().toPoint())
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.selection.width() < 12 or self.selection.height() < 12:
            self.canceled.emit()
            self.close()
            return
        self.hide()
        QApplication.processEvents()
        QTimer.singleShot(120, self._capture_selection)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 72))
        if not self.selection.isNull():
            local_selection = self._rect_from_points(
                self.mapFromGlobal(self.selection.topLeft()),
                self.mapFromGlobal(self.selection.bottomRight()),
            )
            painter.setPen(QPen(QColor(154, 214, 189, 210), 2))
            painter.setBrush(QColor(154, 214, 189, 26))
            painter.drawRect(local_selection)
        painter.end()

    def _capture_selection(self) -> None:
        rect = self.selection.normalized()
        screen = QApplication.screenAt(rect.center()) or QApplication.primaryScreen()
        if screen is None:
            self.canceled.emit()
            self.close()
            return

        screen_rect = screen.geometry()
        local_rect = QRect(
            rect.x() - screen_rect.x(),
            rect.y() - screen_rect.y(),
            rect.width(),
            rect.height(),
        ).intersected(QRect(0, 0, screen_rect.width(), screen_rect.height()))
        if local_rect.width() < 12 or local_rect.height() < 12:
            self.canceled.emit()
            self.close()
            return

        pixmap = screen.grabWindow(0, local_rect.x(), local_rect.y(), local_rect.width(), local_rect.height())
        payload = QByteArray()
        buffer = QBuffer(payload)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        pixmap.save(buffer, "PNG")
        self.captured.emit(base64.b64encode(payload.data()).decode("ascii"))
        self.close()


@dataclass
class AudioDevice:
    index: int | None
    name: str
    hostapi: str = ""
    loopback: bool = False
    samplerate: int = 16000
    channels: int = 1
    loopback_match: str = ""
    auto_loopback: bool = False


class SpeechWorker(QObject):
    partial = Signal(str)
    final = Signal(str)
    stt_latency = Signal(float)
    failed = Signal(str)
    info = Signal(str)
    stopped = Signal()

    def __init__(self, device: AudioDevice) -> None:
        super().__init__()
        self.device = device
        self._running = True
        self.last_silence_notice = 0.0
        self.language_lock: str | None = None
        self.remote_session = requests.Session()
        self.remote_session.trust_env = False

    @Slot()
    def stop(self) -> None:
        self._running = False
        try:
            self.remote_session.close()
        except RuntimeError:
            pass

    @Slot()
    def run(self) -> None:
        if STT_BACKEND == "whisper":
            try:
                self._run_whisper()
                return
            except Exception as exc:  # noqa: BLE001
                if not STT_ALLOW_VOSK_FALLBACK:
                    LOGGER.warning("Whisper STT failed and Vosk fallback is disabled: %s", exc)
                    self.failed.emit(
                        "Whisper STT failed. Select a valid audio device or set "
                        f"STT_ALLOW_VOSK_FALLBACK=1 to use low-quality Vosk fallback. Details: {exc}"
                    )
                    self.stopped.emit()
                    return
                LOGGER.warning("Whisper STT failed, falling back to Vosk: %s", exc)
                self.info.emit(f"Whisper STT failed, Vosk fallback active: {exc}")
        self._run_vosk()

    def _whisper_model_attempts(self) -> list[tuple[str, str]]:
        return whisper_model_attempts(STT_SETTINGS)

    def _load_whisper_model(self, whisper_model_class: Any) -> Any:
        attempts = self._whisper_model_attempts()
        last_exc: Exception | None = None

        for attempt_index, (device, compute_type) in enumerate(attempts):
            try:
                self.info.emit(f"Loading Whisper model {WHISPER_MODEL} on {device}/{compute_type}...")
                LOGGER.info(
                    "STT backend=whisper model=%s device=%s compute_type=%s",
                    WHISPER_MODEL,
                    device,
                    compute_type,
                )
                candidate_model = whisper_model_class(
                    WHISPER_MODEL,
                    device=device,
                    compute_type=compute_type,
                )
                self._warmup_whisper_model(candidate_model, device)
                return candidate_model
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                has_retry = attempt_index + 1 < len(attempts)
                if has_retry and device == "cuda" and _is_cuda_whisper_error(exc):
                    message = f"CUDA Whisper unavailable ({_short_error(exc)}). Falling back to CPU/int8."
                    LOGGER.warning(message)
                    self.info.emit(message)
                    continue
                raise

        raise RuntimeError(f"Unable to load Whisper model: {last_exc}") from last_exc

    def _warmup_whisper_model(self, model: Any, device: str) -> None:
        if device != "cuda":
            return
        try:
            import numpy as np
        except ImportError:
            return
        segments, _info = model.transcribe(
            np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32),
            language="en",
            task="transcribe",
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            vad_filter=False,
            no_speech_threshold=0.95,
        )
        list(segments)

    def _run_whisper(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("audio dependencies are not installed") from exc

        model = None
        if STACKWIRE_REMOTE_STT:
            if not STACKWIRE_API_URL:
                raise RuntimeError("STACKWIRE_API_URL is required for remote STT")
            self.info.emit(f"Listening with remote Whisper API: {STACKWIRE_API_URL}")
            LOGGER.info("STT backend=remote-whisper api=%s", STACKWIRE_API_URL)
        else:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("faster-whisper is not installed") from exc

            model = self._load_whisper_model(WhisperModel)

        if self.device.loopback:
            self._run_whisper_loopback(np, model)
            return

        audio_queue: queue.Queue[object] = queue.Queue()

        def callback(indata, frames, time, status):  # noqa: ANN001, ARG001
            if status:
                self.info.emit(str(status))
            audio_queue.put(indata.copy())

        self.info.emit(f"Listening with {'remote ' if STACKWIRE_REMOTE_STT else ''}Whisper: {self.device.name}")
        chunk_frames = max(int(WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SECONDS), WHISPER_SAMPLE_RATE)
        buffers: list[object] = []
        buffered_frames = 0

        with sd.InputStream(
            samplerate=WHISPER_SAMPLE_RATE,
            blocksize=4096,
            device=self.device.index,
            dtype="float32",
            channels=self.device.channels,
            callback=callback,
        ):
            while self._running:
                try:
                    data = cast(np.ndarray, audio_queue.get(timeout=0.2))
                except queue.Empty:
                    continue

                mono = self._to_mono_float32(data, np)
                buffers.append(mono)
                buffered_frames += len(mono)

                if buffered_frames >= chunk_frames:
                    self._transcribe_whisper_buffers(np, model, buffers)
                    buffers, buffered_frames = self._keep_overlap_buffers(np, buffers)

            if buffered_frames >= WHISPER_SAMPLE_RATE:
                self._transcribe_whisper_buffers(np, model, buffers)
        self.stopped.emit()

    def _loopback_candidates(self, pa) -> list[dict[str, Any]]:  # noqa: ANN001
        candidates: list[dict[str, Any]] = []
        try:
            generator = getattr(pa, "get_loopback_device_info_generator", None)
            if generator is not None:
                candidates = [dict(candidate) for candidate in generator()]
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("failed to enumerate WASAPI loopback devices: %s", exc)

        if candidates:
            return candidates

        try:
            return [dict(pa.get_default_wasapi_loopback())]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("No WASAPI loopback devices found") from exc

    def _normalize_audio_device_name(self, name: str) -> str:
        lowered = name.lower()
        lowered = lowered.replace("system audio:", " ")
        lowered = lowered.replace("[loopback]", " ")
        lowered = lowered.replace("loopback", " ")
        lowered = re.sub(r"[\[\]():,._-]+", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    def _audio_name_score(self, target: str, candidate: str) -> int:
        normalized_candidate = self._normalize_audio_device_name(candidate)
        if not target or not normalized_candidate:
            return 0
        if target in normalized_candidate or normalized_candidate in target:
            return 100
        target_tokens = set(target.split())
        candidate_tokens = set(normalized_candidate.split())
        return len(target_tokens & candidate_tokens)

    def _measure_loopback_rms(self, pyaudio, pa, np, candidate: dict[str, Any]) -> float:  # noqa: ANN001
        stream = None
        try:
            source_rate = int(candidate.get("defaultSampleRate") or WHISPER_SAMPLE_RATE)
            channels = max(1, int(candidate.get("maxInputChannels") or 1))
            device_index = int(candidate["index"])
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=source_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=4096,
            )
            raw = stream.read(4096, exception_on_overflow=False)
            data = np.frombuffer(raw, dtype=np.float32)
            if data.size == 0:
                return 0.0
            if channels > 1:
                usable_size = data.size - (data.size % channels)
                if usable_size <= 0:
                    return 0.0
                data = data[:usable_size].reshape(-1, channels)
            mono = self._to_mono_float32(data, np)
            return float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("loopback probe failed for %s: %s", candidate.get("name"), exc)
            return -1.0
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                finally:
                    stream.close()

    def _select_loopback_device(self, pyaudio, pa, np) -> dict[str, Any]:  # noqa: ANN001
        candidates = self._loopback_candidates(pa)
        if not candidates:
            raise RuntimeError("No WASAPI loopback devices found")

        if not self.device.auto_loopback and self.device.loopback_match:
            target = self._normalize_audio_device_name(self.device.loopback_match)
            scored = sorted(
                ((self._audio_name_score(target, str(candidate.get("name", ""))), candidate) for candidate in candidates),
                key=lambda item: item[0],
                reverse=True,
            )
            if scored and scored[0][0] > 0:
                return scored[0][1]

        if not self.device.auto_loopback:
            return candidates[0]

        if not STT_PROBE_LOOPBACK_DEVICES:
            try:
                default_loopback = dict(pa.get_default_wasapi_loopback())
                self.info.emit(f"Auto system audio selected default: {default_loopback.get('name', 'WASAPI loopback')}")
                return default_loopback
            except Exception:
                return candidates[0]

        measured: list[tuple[float, dict[str, Any]]] = []
        for candidate in candidates:
            rms = self._measure_loopback_rms(pyaudio, pa, np, candidate)
            if rms >= 0.0:
                measured.append((rms, candidate))
                LOGGER.info("loopback probe device=%s rms=%.6f", candidate.get("name"), rms)

        if measured:
            best_rms, best_candidate = max(measured, key=lambda item: item[0])
            if best_rms > 0.0:
                self.info.emit(f"Auto system audio selected: {best_candidate.get('name', 'WASAPI loopback')}")
                return best_candidate

        try:
            default_loopback = dict(pa.get_default_wasapi_loopback())
            self.info.emit(f"Auto system audio selected default: {default_loopback.get('name', 'WASAPI loopback')}")
            return default_loopback
        except Exception:
            return candidates[0]

    def _run_whisper_loopback(self, np, model) -> None:  # noqa: ANN001
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as exc:
            raise RuntimeError("pyaudiowpatch is required for Windows system audio capture") from exc

        pa = pyaudio.PyAudio()
        stream = None
        try:
            loopback_device = self._select_loopback_device(pyaudio, pa, np)
            device_name = str(loopback_device.get("name", "default WASAPI loopback"))
            device_index = int(loopback_device["index"])
            source_rate = int(loopback_device.get("defaultSampleRate") or WHISPER_SAMPLE_RATE)
            channels = max(1, int(loopback_device.get("maxInputChannels") or 1))
            self.info.emit(f"Listening system audio with {'remote ' if STACKWIRE_REMOTE_STT else ''}Whisper: {device_name}")
            LOGGER.info(
                "system audio loopback device=%s index=%s sample_rate=%s channels=%s",
                device_name,
                device_index,
                source_rate,
                channels,
            )
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=source_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=4096,
            )
        except Exception:
            pa.terminate()
            raise

        chunk_frames = max(int(WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SECONDS), WHISPER_SAMPLE_RATE)
        buffers: list[object] = []
        buffered_frames = 0

        try:
            while self._running:
                raw = stream.read(4096, exception_on_overflow=False)
                data = np.frombuffer(raw, dtype=np.float32)
                if data.size == 0:
                    continue
                if channels > 1:
                    data = data.reshape(-1, channels)
                mono = self._to_mono_float32(data, np)
                mono = self._resample_mono(mono, np, source_rate)
                buffers.append(mono)
                buffered_frames += len(mono)
                if buffered_frames >= chunk_frames:
                    self._transcribe_whisper_buffers(np, model, buffers)
                    buffers, buffered_frames = self._keep_overlap_buffers(np, buffers)

            if buffered_frames >= WHISPER_SAMPLE_RATE:
                self._transcribe_whisper_buffers(np, model, buffers)
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                finally:
                    stream.close()
            pa.terminate()
        self.stopped.emit()

    def _resample_mono(self, mono, np, source_rate: int):  # noqa: ANN001
        mono = mono.astype(np.float32, copy=False)
        if source_rate == WHISPER_SAMPLE_RATE or len(mono) == 0:
            return mono

        target_len = max(1, int(round(len(mono) * WHISPER_SAMPLE_RATE / source_rate)))
        source_positions = np.arange(len(mono), dtype=np.float32)
        target_positions = np.linspace(0, len(mono) - 1, target_len, dtype=np.float32)
        return np.interp(target_positions, source_positions, mono).astype(np.float32)

    def _keep_overlap_buffers(self, np, buffers: list[object]) -> tuple[list[object], int]:  # noqa: ANN001
        overlap_frames = int(max(0.0, WHISPER_CHUNK_OVERLAP_SECONDS) * WHISPER_SAMPLE_RATE)
        if overlap_frames <= 0 or not buffers:
            return [], 0

        audio = np.concatenate(buffers).astype(np.float32)
        if len(audio) <= overlap_frames:
            return [audio], len(audio)
        tail = audio[-overlap_frames:]
        return [tail], len(tail)

    def _whisper_vad_parameters(self) -> dict[str, int | float]:
        return whisper_vad_parameters(STT_SETTINGS)

    def _transcribe_local_whisper_audio(self, model, audio, *, vad_filter: bool) -> tuple[str, dict[str, float | str]]:  # noqa: ANN001
        segments, info = model.transcribe(
            audio,
            language=whisper_language(STT_SETTINGS, locked_language=self.language_lock),
            task="transcribe",
            beam_size=STT_SETTINGS.beam_size,
            best_of=STT_SETTINGS.best_of,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=WHISPER_INITIAL_PROMPT,
            vad_filter=vad_filter,
            vad_parameters=self._whisper_vad_parameters() if vad_filter else None,
            no_speech_threshold=STT_SETTINGS.no_speech_threshold,
            log_prob_threshold=STT_SETTINGS.log_prob_threshold,
            compression_ratio_threshold=STT_SETTINGS.compression_ratio_threshold,
            repetition_penalty=STT_SETTINGS.repetition_penalty,
            no_repeat_ngram_size=STT_SETTINGS.no_repeat_ngram_size,
            hallucination_silence_threshold=STT_SETTINGS.hallucination_silence_threshold,
            hotwords=WHISPER_HOTWORDS,
        )
        segment_list = list(segments)
        text = " ".join(segment.text.strip() for segment in segment_list).strip()
        diagnostics: dict[str, float | str] = {
            "language": str(getattr(info, "language", whisper_language(STT_SETTINGS, locked_language=self.language_lock) or "")),
            "avg_logprob": self._mean_segment_attr(segment_list, "avg_logprob"),
            "no_speech_prob": self._max_segment_attr(segment_list, "no_speech_prob"),
            "compression_ratio": self._max_segment_attr(segment_list, "compression_ratio"),
            "vad": "1" if vad_filter else "0",
        }
        return text, diagnostics

    def _mean_segment_attr(self, segments: list[object], attr: str) -> float:
        values = [float(value) for segment in segments if isinstance((value := getattr(segment, attr, None)), int | float)]
        return sum(values) / len(values) if values else 0.0

    def _max_segment_attr(self, segments: list[object], attr: str) -> float:
        values = [float(value) for segment in segments if isinstance((value := getattr(segment, attr, None)), int | float)]
        return max(values) if values else 0.0

    def _is_bad_whisper_text(self, text: str, diagnostics: dict[str, float | str], rms: float) -> bool:
        return is_probable_stt_hallucination(
            text,
            avg_logprob=float(diagnostics.get("avg_logprob") or 0.0),
            no_speech_prob=float(diagnostics.get("no_speech_prob") or 0.0),
            compression_ratio=float(diagnostics.get("compression_ratio") or 0.0),
            rms=rms,
        )

    def _transcribe_whisper_buffers(self, np, model, buffers: list[object]) -> None:  # noqa: ANN001
        if not buffers:
            return
        audio = np.concatenate(buffers).astype(np.float32)
        if len(audio) < WHISPER_SAMPLE_RATE:
            return
        signal_threshold = STT_LOOPBACK_SIGNAL_THRESHOLD if self.device.loopback else STT_MIC_SIGNAL_THRESHOLD
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < signal_threshold:
            now = time.monotonic()
            if now - self.last_silence_notice >= 8.0:
                self.last_silence_notice = now
                if self.device.loopback:
                    self.info.emit(
                        "System audio selected, but no signal is detected on the selected/active playback loopback."
                    )
                else:
                    self.info.emit("Microphone selected, but no input signal is detected.")
            return

        if STACKWIRE_REMOTE_STT:
            self._transcribe_remote_whisper_audio(np, audio)
            return
        if model is None:
            raise RuntimeError("Local Whisper model is not initialized")

        started = time.perf_counter()
        text, diagnostics = self._transcribe_local_whisper_audio(model, audio, vad_filter=WHISPER_VAD_FILTER)
        bad_text = self._is_bad_whisper_text(text, diagnostics, rms)
        if WHISPER_RETRY_WITHOUT_VAD and WHISPER_VAD_FILTER and (not text.strip() or bad_text):
            retry_text, retry_diagnostics = self._transcribe_local_whisper_audio(model, audio, vad_filter=False)
            retry_bad = self._is_bad_whisper_text(retry_text, retry_diagnostics, rms)
            if retry_text.strip() and not retry_bad:
                text = retry_text
                diagnostics = retry_diagnostics
                bad_text = False

        raw_text = text
        text = clean_stt_output(raw_text)
        bad_text = bad_text or self._is_bad_whisper_text(text, diagnostics, rms)
        self.language_lock = update_stt_language_lock(
            STT_SETTINGS,
            self.language_lock,
            str(diagnostics.get("language", "")),
            text,
            bad_text=bad_text,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        LOGGER.info(
            "stt_latency_ms=%.0f language=%s language_lock=%s vad=%s rms=%.6f avg_logprob=%.2f no_speech=%.2f compression=%.2f raw=%r cleaned=%r",
            latency_ms,
            diagnostics.get("language", ""),
            self.language_lock or "",
            diagnostics.get("vad", ""),
            rms,
            float(diagnostics.get("avg_logprob") or 0.0),
            float(diagnostics.get("no_speech_prob") or 0.0),
            float(diagnostics.get("compression_ratio") or 0.0),
            raw_text,
            text,
        )
        self.stt_latency.emit(latency_ms)
        if bad_text:
            LOGGER.info("drop probable whisper hallucination=%r", text)
            return
        if text:
            LOGGER.info("whisper raw_stt=%r", text)
            self.final.emit(text)

    def _transcribe_remote_whisper_audio(self, np, audio) -> None:  # noqa: ANN001
        started = time.perf_counter()
        payload = base64.b64encode(audio.astype(np.float32).tobytes()).decode("ascii")
        request_payload: dict[str, str | int] = {"audio_b64": payload, "sample_rate": WHISPER_SAMPLE_RATE}
        request_language = whisper_language(STT_SETTINGS, locked_language=self.language_lock)
        if request_language:
            request_payload["language"] = request_language
        response = self.remote_session.post(
            f"{STACKWIRE_API_URL}/transcribe",
            json=cast(Any, request_payload),
            timeout=(STACKWIRE_API_CONNECT_TIMEOUT, STACKWIRE_STT_TIMEOUT),
        )
        self._raise_remote_stt_for_status(response)
        data = response.json()
        raw_latency: Any = data.get("latency_ms")
        if isinstance(raw_latency, int | float | str):
            try:
                latency_ms = float(raw_latency)
            except ValueError:
                latency_ms = (time.perf_counter() - started) * 1000
        else:
            latency_ms = (time.perf_counter() - started) * 1000
        raw_text = str(data.get("raw_text") or data.get("text") or "").strip()
        text = str(data.get("cleaned_text") or data.get("text") or "").strip()
        text = clean_stt_output(text)
        detected_language = str(data.get("language", "")).strip()
        bad_text = is_probable_stt_hallucination(text)
        self.language_lock = update_stt_language_lock(
            STT_SETTINGS,
            self.language_lock,
            detected_language,
            text,
            bad_text=bad_text,
        )
        diagnostics = data.get("diagnostics") if isinstance(data.get("diagnostics"), dict) else {}
        LOGGER.info(
            "remote stt_latency_ms=%.0f language=%s language_lock=%s vad=%s raw=%r cleaned=%r",
            latency_ms,
            detected_language,
            self.language_lock or "",
            diagnostics.get("vad", ""),
            raw_text,
            text,
        )
        self.stt_latency.emit(latency_ms)
        if bad_text:
            LOGGER.info("drop probable remote whisper hallucination=%r", text)
            return
        if text:
            LOGGER.info("remote whisper raw_stt=%r", text)
            self.final.emit(text)

    def _raise_remote_stt_for_status(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text
            try:
                payload = response.json()
                detail = str(payload.get("detail", payload))
            except ValueError:
                pass
            raise RuntimeError(
                f"Remote API /transcribe returned {response.status_code}: {detail[:500]}"
            ) from exc

    def _run_vosk(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model
        except ImportError:
            self.failed.emit("Нет audio-зависимостей. Выполни: python -m pip install -r requirements.txt")
            self.stopped.emit()
            return

        model_path = self._ensure_model()
        if not model_path:
            self.stopped.emit()
            return

        LOGGER.info("STT backend=vosk model=%s", model_path)

        if self.device.loopback:
            self._run_loopback(np, KaldiRecognizer, Model, model_path)
            return

        audio_queue: queue.Queue[object] = queue.Queue()

        def callback(indata, frames, time, status):  # noqa: ANN001, ARG001
            if status:
                self.partial.emit(str(status))
            audio_queue.put(indata.copy())

        try:
            self.info.emit("Loading speech model...")
            model = self._load_vosk_model(Model, model_path)
            recognizer = KaldiRecognizer(model, float(self.device.samplerate))
            recognizer.SetWords(True)

            self.info.emit(f"Listening: {self.device.name}")

            with sd.InputStream(
                samplerate=self.device.samplerate,
                blocksize=4096,
                device=self.device.index,
                dtype="int16",
                channels=self.device.channels,
                callback=callback,
            ):
                while self._running:
                    try:
                        data = audio_queue.get(timeout=0.2)
                        data = cast(np.ndarray, data)
                    except queue.Empty:
                        continue

                    pcm = self._to_mono_pcm(data, np)
                    if recognizer.AcceptWaveform(pcm):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").strip()
                        if text:
                            self.final.emit(text)
                    else:
                        result = json.loads(recognizer.PartialResult())
                        text = result.get("partial", "").strip()
                        if text:
                            self.partial.emit(text)

                final_result = json.loads(recognizer.FinalResult())
                final_text = final_result.get("text", "").strip()
                if final_text:
                    self.final.emit(final_text)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(
                f"{exc}\n\nВыбери другое устройство. На Windows чаще всего работает WASAPI microphone или System audio WASAPI, а MME часто дает PaError -9999."
            )
        finally:
            self.stopped.emit()

    def _run_loopback(self, np, KaldiRecognizer, Model, model_path: Path) -> None:  # noqa: ANN001
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            self.failed.emit("Для системного звука нужен пакет pyaudiowpatch: python -m pip install pyaudiowpatch")
            self.stopped.emit()
            return

        pa = pyaudio.PyAudio()
        stream = None
        try:
            self.info.emit("Loading speech model...")
            model = self._load_vosk_model(Model, model_path)
            sample_rate = 16000
            recognizer = KaldiRecognizer(model, float(sample_rate))
            recognizer.SetWords(True)
            loopback_device = self._select_loopback_device(pyaudio, pa, np)
            device_name = str(loopback_device.get("name", "default WASAPI loopback"))
            source_rate = int(loopback_device.get("defaultSampleRate") or sample_rate)
            channels = max(1, int(loopback_device.get("maxInputChannels") or 1))
            self.info.emit(f"Listening system audio: {device_name}")

            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=source_rate,
                input=True,
                input_device_index=int(loopback_device["index"]),
                frames_per_buffer=4096,
            )
            while self._running:
                raw = stream.read(4096, exception_on_overflow=False)
                data = np.frombuffer(raw, dtype=np.float32)
                if data.size == 0:
                    continue
                if channels > 1:
                    data = data.reshape(-1, channels)
                mono = self._to_mono_float32(data, np)
                mono = self._resample_mono(mono, np, source_rate)
                pcm = np.clip(mono * 32767, -32768, 32767).astype(np.int16).tobytes()
                if recognizer.AcceptWaveform(pcm):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        self.final.emit(text)
                else:
                    result = json.loads(recognizer.PartialResult())
                    text = result.get("partial", "").strip()
                    if text:
                        self.partial.emit(text)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"System audio capture failed: {exc}")
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                finally:
                    stream.close()
            pa.terminate()
            self.stopped.emit()

    def _ensure_model(self) -> Path | None:
        env_path = os.getenv("VOSK_MODEL_PATH", "").strip()
        if env_path:
            path = Path(env_path)
            if path.is_dir():
                return path
            self.failed.emit(f"VOSK_MODEL_PATH не найден: {path}")
            return None

        if DEFAULT_VOSK_MODEL_DIR.is_dir():
            return DEFAULT_VOSK_MODEL_DIR

        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self.info.emit("Downloading Vosk RU model, first run only...")
            urllib.request.urlretrieve(DEFAULT_VOSK_MODEL_URL, DEFAULT_VOSK_MODEL_ZIP)
            self.info.emit("Unpacking Vosk model...")
            with ZipFile(DEFAULT_VOSK_MODEL_ZIP) as archive:
                archive.extractall(MODELS_DIR)
            return DEFAULT_VOSK_MODEL_DIR
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Не удалось скачать Vosk model: {exc}")
            return None

    def _ensure_fallback_model(self) -> Path | None:
        if FALLBACK_VOSK_MODEL_DIR.is_dir():
            return FALLBACK_VOSK_MODEL_DIR

        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self.info.emit("Downloading fallback Vosk model...")
            urllib.request.urlretrieve(FALLBACK_VOSK_MODEL_URL, FALLBACK_VOSK_MODEL_ZIP)
            self.info.emit("Unpacking fallback Vosk model...")
            with ZipFile(FALLBACK_VOSK_MODEL_ZIP) as archive:
                archive.extractall(MODELS_DIR)
            return FALLBACK_VOSK_MODEL_DIR
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Не удалось скачать fallback Vosk model: {exc}")
            return None

    def _load_vosk_model(self, model_class, model_path: Path):  # noqa: ANN001
        try:
            return model_class(str(model_path))
        except Exception as exc:  # noqa: BLE001
            fallback = self._ensure_fallback_model()
            if fallback is None or fallback == model_path:
                raise
            self.info.emit(
                f"Model {model_path.name} is incompatible with python-vosk, using {fallback.name}"
            )
            try:
                return model_class(str(fallback))
            except Exception:
                raise exc

    def _to_mono_float32(self, data, np):  # noqa: ANN001
        if len(data.shape) == 2 and data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data.flatten()
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / 32768.0
        else:
            data = data.astype(np.float32)
        return np.clip(data, -1.0, 1.0)

    def _to_mono_pcm(self, data, np) -> bytes:  # noqa: ANN001
        if len(data.shape) == 2 and data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data.flatten()

        if np.issubdtype(data.dtype, np.floating):
            data = np.clip(data, -1.0, 1.0) * 32767

        return np.clip(data, -32768, 32767).astype(np.int16).tobytes()

class OverlayWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ask_thread: QThread | None = None
        self.ask_worker: AskStreamWorker | None = None
        self.expand_thread: QThread | None = None
        self.expand_worker: ExpandWorker | None = None
        self._stream_active = False
        self._stream_buffer = ""
        self._stream_render_pending = False
        self._typing_phase = 0
        self._stream_anchor = 0
        self._stream_prefix_snippets = 0
        self.typing_timer = QTimer(self)
        self.typing_timer.setInterval(60)
        self.typing_timer.timeout.connect(self._tick_typing)
        self.image_thread: QThread | None = None
        self.image_worker: ImageAnalysisWorker | None = None
        self.region_selector: RegionSelector | None = None
        self.speech_thread: QThread | None = None
        self.speech_worker: SpeechWorker | None = None
        self.drag_position: QPoint | None = None
        self.last_final_speech = ""
        self.current_partial_speech = ""
        self.last_stt_latency_ms: float | None = None
        self.last_recovery_latency_ms: float | None = None
        self.last_answer_latency_ms: float | None = None
        self.last_total_latency_ms: float | None = None
        self.raw_transcript_lines: list[str] = []
        self.transcript_lines: list[str] = []
        self._closing = False
        self._close_retry_count = 0
        self.last_question_candidate = ""
        self.pending_capture_b64 = ""
        self.pending_attachment: dict[str, str] | None = None
        self.visibility_hotkey_down = False
        self.record_hotkey_down = False
        self.submit_after_speech_stop = False
        self.speech_input_locked = False
        self.debug_expanded = False
        self._first_show_done = False
        self._fade_animation: QPropertyAnimation | None = None
        # Authentication state (server account). Token cached locally between launches.
        self.auth_required = STACKWIRE_REQUIRE_AUTH
        self.auth_username: str = ""
        self.authenticated = not self.auth_required
        self._load_cached_auth()
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry() if screen is not None else None
        self.compact_screen = bool(
            screen_geometry is not None
            and (screen_geometry.width() <= 1920 or screen_geometry.height() <= 1080)
        )
        default_zoom = 0.86 if self.compact_screen else 1.0
        self.ui_zoom = self._clamp_zoom(_env_float("STACKWIRE_UI_SCALE", default_zoom))
        self.chat_messages: list[tuple[str, str]] = []
        self.last_answer_question = ""
        self.last_answer_text = ""
        self.last_main_answer_text = ""
        self.last_answer_id: int | None = None
        self.last_answer_domain: str | None = None
        self.last_answer_intent: str | None = None
        try:
            self.storage_session_id: int | None = None if STACKWIRE_API_URL else create_session("StackWire desktop")
        except Exception:
            LOGGER.debug("desktop storage session creation failed", exc_info=True)
            self.storage_session_id = None
        self.question_count = 0
        self.auto_ask_timer = QTimer(self)
        self.auto_ask_timer.setSingleShot(True)
        self.auto_ask_timer.setInterval(2200)
        self.auto_ask_timer.timeout.connect(self.refresh_question_candidate)
        self.global_hotkey_timer = QTimer(self)
        self.global_hotkey_timer.setInterval(120)
        self.global_hotkey_timer.timeout.connect(self.poll_global_hotkeys)

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(make_icon("mark", 32, ACCENT))
        flags = Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint
        flags |= Qt.WindowType.Tool if STACKWIRE_HIDE_TASKBAR else Qt.WindowType.Window
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        if self.compact_screen:
            self.resize(960, 620)
        else:
            self.resize(1180, 760)
        self._build_ui()
        self._load_audio_devices(os.getenv("STACKWIRE_AUDIO_DEVICE", "").strip())
        QTimer.singleShot(0, self.apply_capture_exclusion)
        # Warm up the local vector store (index knowledge once) without blocking the UI.
        QTimer.singleShot(0, self._warm_vector_store)

    def showEvent(self, event) -> None:  # noqa: ANN001
        super().showEvent(event)
        if self._first_show_done:
            return
        self._first_show_done = True
        self.setWindowOpacity(0.0)
        animation = QPropertyAnimation(self, b"windowOpacity", self)
        animation.setDuration(260)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.finished.connect(lambda: self.setWindowOpacity(1.0))
        self._fade_animation = animation
        animation.start()
        QTimer.singleShot(0, self._apply_acrylic)
        if self.auth_required and not self.authenticated:
            QTimer.singleShot(320, self.prompt_login)

    # --- Authentication -------------------------------------------------- #
    def _load_cached_auth(self) -> None:
        if not self.auth_required:
            return
        try:
            from app import auth_client

            cached = auth_client.load_credentials()
            if cached and auth_client.verify(cached.token):
                self.auth_username = cached.username
                self.authenticated = True
                set_auth_token(cached.token)
        except Exception:
            LOGGER.debug("cached auth load failed", exc_info=True)

    def prompt_login(self) -> bool:
        if self.authenticated:
            return True
        dialog = LoginDialog(self, default_username=self.auth_username)
        dialog.setStyleSheet(build_window_styles(self.ui_zoom))
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.token:
            self.auth_username = dialog.username
            self.authenticated = True
            set_auth_token(dialog.token)
            if hasattr(self, "status"):
                self.status.setText(f"Вы вошли как {dialog.username}")
            self.update_account_chip()
            self.render_chat()
            return True
        if hasattr(self, "status"):
            self.status.setText("Вход не выполнен — чат недоступен.")
        return False

    def _require_login(self) -> bool:
        if not self.auth_required or self.authenticated:
            return True
        return self.prompt_login()

    def logout(self) -> None:
        try:
            from app import auth_client

            auth_client.logout(CURRENT_AUTH_TOKEN)
        except Exception:
            LOGGER.debug("logout failed", exc_info=True)
        set_auth_token("")
        self.authenticated = not self.auth_required
        self.update_account_chip()
        if self.auth_required:
            self.status.setText("Вы вышли из аккаунта.")
            QTimer.singleShot(150, self.prompt_login)

    def update_account_chip(self) -> None:
        chip = getattr(self, "account_chip", None)
        if chip is None:
            return
        if self.authenticated and self.auth_username:
            chip.setText(self.auth_username)
        elif self.auth_required:
            chip.setText("гость")
        else:
            chip.setText("")

    def _warm_vector_store(self) -> None:
        try:
            from app import vectorstore

            if vectorstore.is_available():
                vectorstore.ensure_indexed()
        except Exception:
            LOGGER.debug("vector warm-up failed", exc_info=True)

    def closeEvent(self, event) -> None:
        if self._closing:
            event.ignore()
            return
        self._closing = True
        if self._close_retry_count == 0:
            self._log_client_event("client_close")

        self.auto_ask_timer.stop()
        self.global_hotkey_timer.stop()
        self.speech_input_locked = True

        if self.speech_worker:
            try:
                self.speech_worker.stop()
            except RuntimeError:
                pass
        if self.region_selector:
            try:
                self.region_selector.close()
            except RuntimeError:
                pass

        shutdown_pending = False

        if not self._shutdown_thread(self.speech_thread, "speech", timeout_ms=20_000):
            shutdown_pending = True
        else:
            self.speech_thread = None
            self.speech_worker = None

        ask_thread = self.ask_thread
        if self.ask_worker:
            try:
                self.ask_worker.session.close()
                if self.ask_worker.client is not None:
                    self.ask_worker.client.session.close()
            except RuntimeError:
                pass
        if not self._shutdown_thread(ask_thread, "ask", timeout_ms=5_000):
            shutdown_pending = True
        else:
            self.ask_thread = None
            self.ask_worker = None

        expand_thread = self.expand_thread
        if self.expand_worker:
            try:
                self.expand_worker.session.close()
                if self.expand_worker.client is not None:
                    self.expand_worker.client.session.close()
            except RuntimeError:
                pass
        if not self._shutdown_thread(expand_thread, "expand", timeout_ms=5_000):
            shutdown_pending = True
        else:
            self.expand_thread = None
            self.expand_worker = None

        image_thread = self.image_thread
        if self.image_worker:
            try:
                self.image_worker.session.close()
                if self.image_worker.client is not None:
                    self.image_worker.client.session.close()
            except RuntimeError:
                pass
        if not self._shutdown_thread(image_thread, "image", timeout_ms=5_000):
            shutdown_pending = True
        else:
            self.image_thread = None
            self.image_worker = None

        if shutdown_pending:
            self._close_retry_count += 1
            self._closing = False
            self.status.setText("Stopping background work...")
            LOGGER.warning("close delayed waiting for background threads retry=%s", self._close_retry_count)
            event.ignore()
            if self._close_retry_count <= 6:
                QTimer.singleShot(1000, self.close)
            return

        try:
            from app import vectorstore

            vectorstore.close()
        except Exception:
            LOGGER.debug("vector store close failed", exc_info=True)

        event.accept()
        app = QApplication.instance()
        if app is not None:
            QTimer.singleShot(0, lambda: app.exit(0))

    def _log_client_event(self, event_name: str) -> None:
        details = {
            "api_url": STACKWIRE_API_URL or "local",
            "answer_model": current_answer_model(),
            "recovery_model": current_recovery_model(),
            "question_count": self.question_count,
            "last_stt_latency_ms": self.last_stt_latency_ms,
            "last_recovery_latency_ms": self.last_recovery_latency_ms,
            "last_answer_latency_ms": self.last_answer_latency_ms,
            "last_total_latency_ms": self.last_total_latency_ms,
        }
        payload = {
            "event": event_name,
            "client_time": datetime.now().astimezone().replace(microsecond=0).isoformat(),
            "details": details,
        }

        if STACKWIRE_API_URL:
            try:
                response = requests.post(
                    f"{STACKWIRE_API_URL}/client-event",
                    json=payload,
                    timeout=(min(STACKWIRE_API_CONNECT_TIMEOUT, 1.0), 1.5),
                )
                response.raise_for_status()
            except RequestException:
                LOGGER.debug("client event log request failed", exc_info=True)
            return

        try:
            append_client_event(event_name, {**details, "client_time": payload["client_time"]})
        except Exception:
            LOGGER.debug("local client event log failed", exc_info=True)

    def _shutdown_thread(self, thread: QThread | None, name: str, timeout_ms: int = 900) -> bool:
        if thread is None:
            return True
        try:
            if not thread.isRunning():
                return True
            thread.requestInterruption()
            thread.quit()
            if thread.wait(timeout_ms):
                return True
            LOGGER.warning("%s thread did not stop in %sms; keeping it alive for graceful shutdown", name, timeout_ms)
            return False
        except RuntimeError:
            return True
        
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        shell = QFrame()
        shell.setObjectName("shell")
        self.shell = shell
        shadow = QGraphicsDropShadowEffect(shell)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 210))
        shell.setGraphicsEffect(shadow)
        shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        header = QHBoxLayout()
        header.setSpacing(10)

        self.title_mark = QLabel()
        self.title_mark.setObjectName("titleMark")

        self.title = QLabel(APP_NAME)
        self.title.setObjectName("title")
        self.subtitle = QLabel("overlay")
        self.subtitle.setObjectName("subtitle")
        self.subtitle.setVisible(True)

        self.device_combo = NoWheelComboBox()
        self.device_combo.setObjectName("deviceCombo")

        self.listen_button = QPushButton()
        self.listen_button.setObjectName("iconButton")
        self.listen_button.setToolTip("Listen")
        self.listen_button.clicked.connect(self.toggle_listening)

        self.clear_button = QPushButton()
        self.clear_button.setObjectName("iconButton")
        self.clear_button.setToolTip("Clear")
        self.clear_button.clicked.connect(self.clear_answer)

        self.capture_button = QPushButton()
        self.capture_button.setObjectName("iconButton")
        self.capture_button.setToolTip("Capture screen")
        self.capture_button.clicked.connect(self.start_region_capture)

        self.debug_button = QPushButton()
        self.debug_button.setObjectName("iconButton")
        self.debug_button.setToolTip("Debug")
        self.debug_button.setCheckable(True)
        self.debug_button.setChecked(False)
        self.debug_button.clicked.connect(self.toggle_debug_panel)

        self.settings_button = QPushButton()
        self.settings_button.setObjectName("iconButton")
        self.settings_button.setToolTip("Settings")
        self.settings_button.clicked.connect(self.show_settings_dialog)

        self.close_button = QPushButton()
        self.close_button.setObjectName("closeButton")
        self.close_button.setToolTip("Close")
        self.close_button.clicked.connect(self.close)

        rail = QFrame()
        rail.setObjectName("rail")
        self.rail = rail
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(12, 14, 12, 14)
        rail_layout.setSpacing(10)

        brand = QVBoxLayout()
        brand.setSpacing(2)
        brand.addWidget(self.title_mark, 0, Qt.AlignmentFlag.AlignHCenter)
        brand.addWidget(self.title, 0, Qt.AlignmentFlag.AlignHCenter)
        brand.addWidget(self.subtitle, 0, Qt.AlignmentFlag.AlignHCenter)
        rail_layout.addLayout(brand)
        rail_layout.addSpacing(14)
        for button in (self.capture_button, self.clear_button, self.debug_button):
            rail_layout.addWidget(button, 0, Qt.AlignmentFlag.AlignHCenter)
        rail_layout.addStretch(1)
        rail_layout.addWidget(self.settings_button, 0, Qt.AlignmentFlag.AlignHCenter)

        content = QFrame()
        content.setObjectName("content")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(18, 14, 18, 16)
        content_layout.setSpacing(12)

        self.status = QLabel("Ready")
        self.status.setObjectName("status")
        self.model_chip = QLabel(current_answer_model())
        self.model_chip.setObjectName("modelChip")
        self.api_chip = QLabel("remote" if STACKWIRE_API_URL else "local")
        self.api_chip.setObjectName("apiChip")
        self.account_chip = QLabel("")
        self.account_chip.setObjectName("accountChip")

        header.addWidget(self.model_chip)
        header.addWidget(self.api_chip)
        header.addWidget(self.account_chip)
        header.addStretch(1)
        header.addWidget(self.status)
        header.addWidget(self.close_button)
        self.update_account_chip()

        # Chat is a scrollable column of message-bubble widgets (GPT-style), with an
        # animated welcome backdrop shown when there is no conversation yet.
        self.neural_bg = NeuralBackground()
        self.chat_area = ChatArea(self.neural_bg)
        self.chat_area.setMinimumHeight(280)
        # Per-message assistant row currently being streamed into (None when idle).
        self._stream_row: AssistantRow | None = None
        self._assistant_rows: dict[int, AssistantRow] = {}

        composer = QFrame()
        composer.setObjectName("composer")
        self.composer = composer
        _soft_shadow(composer, blur=_px(26), dy=_px(6), alpha=120)
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(8, 6, 8, 6)
        composer_layout.setSpacing(6)

        # Pending attachment chip (hidden until a file is attached).
        self.attach_bar = QFrame()
        self.attach_bar.setObjectName("attachBar")
        self.attach_bar.setVisible(False)
        attach_bar_layout = QHBoxLayout(self.attach_bar)
        attach_bar_layout.setContentsMargins(10, 5, 6, 5)
        attach_bar_layout.setSpacing(8)
        self.attach_chip = QLabel("")
        self.attach_chip.setObjectName("attachChip")
        self.attach_remove = QPushButton("✕")
        self.attach_remove.setObjectName("attachRemove")
        self.attach_remove.setFixedSize(22, 22)
        self.attach_remove.clicked.connect(self.clear_attachment)
        attach_bar_layout.addWidget(self.attach_chip, 1)
        attach_bar_layout.addWidget(self.attach_remove, 0)
        composer_layout.addWidget(self.attach_bar)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)

        self.attach_button = QPushButton()
        self.attach_button.setObjectName("composerIcon")
        self.attach_button.setToolTip("Attach file")
        self.attach_button.clicked.connect(self.attach_file)

        self.input = PromptEdit()
        self.input.setObjectName("prompt")
        self.input.setPlaceholderText("Message StackWire...")
        self.input.setFixedHeight(_px(40))
        self.input.submitted.connect(self.submit_question)
        # Focus ring: highlight the composer pill while the input is focused.
        self.input.installEventFilter(self)

        # The mic (listen) button lives inside the composer pill, GPT-style.
        self.listen_button.setObjectName("composerIcon")

        self.ask_button = QPushButton()
        self.ask_button.setObjectName("composerSend")
        self.ask_button.setToolTip("Ask")
        self.ask_button.setFixedSize(_px(40), _px(40))
        self.ask_button.clicked.connect(self.submit_question)

        footer.addWidget(self.attach_button)
        footer.addWidget(self.input, 1)
        footer.addWidget(self.listen_button)
        footer.addWidget(self.ask_button)
        composer_layout.addLayout(footer)

        self.debug_panel = QLabel()
        self.debug_panel.setObjectName("debugPanel")
        self.debug_panel.setWordWrap(True)
        self.debug_panel.setMaximumHeight(140)
        self.debug_panel.setVisible(False)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(QSizeGrip(self), 0, Qt.AlignmentFlag.AlignRight)

        content_layout.addLayout(header)
        content_layout.addWidget(self.chat_area, 1)
        content_layout.addWidget(composer)
        content_layout.addWidget(self.debug_panel)
        content_layout.addLayout(bottom)
        shell_layout.addWidget(rail)
        shell_layout.addWidget(content, 1)
        layout.addWidget(shell)
        self.setCentralWidget(root)

        self.setStyleSheet(STYLES)
        self.install_zoom_shortcuts()
        self.install_global_hotkeys()
        self.apply_ui_zoom()
        self.update_answer_actions()
        self.update_debug_panel()
        self.render_chat()
        self.input.setFocus()

    def install_zoom_shortcuts(self) -> None:
        shortcuts = (
            ("Ctrl++", self.zoom_in),
            ("Ctrl+=", self.zoom_in),
            ("Ctrl+-", self.zoom_out),
            ("Ctrl+0", self.reset_zoom),
        )
        self.zoom_shortcuts: list[QShortcut] = []
        for sequence, handler in shortcuts:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(handler)
            self.zoom_shortcuts.append(shortcut)

    def install_global_hotkeys(self) -> None:
        if sys.platform == "win32":
            self.global_hotkey_timer.start()

    def poll_global_hotkeys(self) -> None:
        if sys.platform != "win32":
            return

        try:
            import ctypes

            user32 = ctypes.windll.user32
            visibility_pressed = bool(user32.GetAsyncKeyState(0x71) & 0x8000)
            record_pressed = bool(user32.GetAsyncKeyState(0x72) & 0x8000)
            if visibility_pressed and not self.visibility_hotkey_down:
                self.visibility_hotkey_down = True
                self.toggle_window_visibility()
            elif not visibility_pressed:
                self.visibility_hotkey_down = False

            if record_pressed and not self.record_hotkey_down:
                self.record_hotkey_down = True
                self.toggle_recording_hotkey()
            elif not record_pressed:
                self.record_hotkey_down = False
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("global hotkey poll failed: %s", exc)
            self.global_hotkey_timer.stop()

    def zoom_in(self) -> None:
        self.set_ui_zoom(self.ui_zoom + ZOOM_STEP)

    def zoom_out(self) -> None:
        self.set_ui_zoom(self.ui_zoom - ZOOM_STEP)

    def reset_zoom(self) -> None:
        self.set_ui_zoom(1.0)

    def set_ui_zoom(self, value: float) -> None:
        self.ui_zoom = self._clamp_zoom(value)
        self.apply_ui_zoom()
        self.render_chat()
        self.status.setText(f"Zoom: {self.ui_zoom:.0%}")

    def apply_ui_zoom(self) -> None:
        global UI_ZOOM
        UI_ZOOM = self.ui_zoom

        font = QFont("Space Grotesk")
        font.setPointSizeF(10 * self.ui_zoom)
        self.setFont(font)

        self.chat_area.setMinimumHeight(_px(300))
        self.input.setFixedHeight(_px(40))
        self.input.keep_arrow_cursor()
        self.ask_button.setFixedSize(_px(40), _px(40))
        for _btn in (self.attach_button, self.listen_button):
            _btn.setFixedSize(_px(38), _px(38))
        if hasattr(self, "rail"):
            self.rail.setFixedWidth(_px(116))
        self.debug_panel.setMaximumHeight(_px(140))
        self.setStyleSheet(build_window_styles(self.ui_zoom))
        self.apply_icons()

    def apply_icons(self) -> None:
        icon_size = _px(16)
        self.title_mark.setPixmap(icon_pixmap("mark", _px(24), ACCENT))
        listening = self._speech_is_running()
        self.listen_button.setIcon(make_icon("stop" if listening else "listen", icon_size, CORAL if listening else ACCENT))
        self.listen_button.setToolTip("Stop listening" if listening else "Listen")
        self.clear_button.setIcon(make_icon("clear", icon_size, "#83aeb8"))
        self.capture_button.setIcon(make_icon("capture", icon_size, "#83aeb8"))
        self.debug_button.setIcon(make_icon("debug", icon_size, "#83aeb8"))
        self.settings_button.setIcon(make_icon("settings", icon_size, "#83aeb8"))
        self.attach_button.setIcon(make_icon("attach", icon_size, "#83aeb8"))
        self.ask_button.setIcon(make_icon("ask", icon_size, "#10131a"))
        self.close_button.setIcon(make_icon("close", icon_size, "#6f8793"))
        for button in (
            self.listen_button,
            self.clear_button,
            self.capture_button,
            self.debug_button,
            self.settings_button,
            self.attach_button,
            self.ask_button,
            self.close_button,
        ):
            button.setIconSize(QSize(icon_size, icon_size))
        # Rail icon buttons stay 34px squares; composer icons (attach/listen) keep their
        # 38px pill sizing set in apply_ui_zoom — don't resize them here.
        for button in (
            self.clear_button,
            self.capture_button,
            self.debug_button,
            self.settings_button,
        ):
            button.setFixedSize(_px(34), _px(34))
        self.close_button.setFixedSize(_px(34), _px(34))

    def _clamp_zoom(self, value: float) -> float:
        return max(MIN_UI_ZOOM, min(MAX_UI_ZOOM, value))

    def apply_capture_exclusion(self) -> None:
        if not STACKWIRE_HIDE_FROM_CAPTURE or sys.platform != "win32":
            return

        try:
            import ctypes

            hwnd = int(self.winId())
            wda_exclude_from_capture = 0x00000011
            wda_monitor = 0x00000001
            user32 = ctypes.windll.user32
            ok = user32.SetWindowDisplayAffinity(hwnd, wda_exclude_from_capture)
            if not ok:
                user32.SetWindowDisplayAffinity(hwnd, wda_monitor)
            LOGGER.info("capture exclusion applied ok=%s", bool(ok))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("failed to apply capture exclusion: %s", exc)

    def _apply_acrylic(self) -> None:
        """Best-effort Windows 11 'matte glass' backdrop (blurs the desktop behind the
        window). No-op on non-Windows or older builds — the app still looks fine."""
        if not STACKWIRE_ACRYLIC or sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes

            hwnd = int(self.winId())
            dwm = ctypes.windll.dwmapi
            # 1) Round the window corners (DWMWA_WINDOW_CORNER_PREFERENCE=33, ROUND=2).
            corner = ctypes.c_int(2)
            dwm.DwmSetWindowAttribute(wintypes.HWND(hwnd), 33, ctypes.byref(corner), ctypes.sizeof(corner))
            # 2) Acrylic system backdrop (DWMWA_SYSTEMBACKDROP_TYPE=38, ACRYLIC=3). Win11 22H2+.
            backdrop = ctypes.c_int(3)
            result = dwm.DwmSetWindowAttribute(wintypes.HWND(hwnd), 38, ctypes.byref(backdrop), ctypes.sizeof(backdrop))
            LOGGER.info("acrylic backdrop applied hr=%s", result)
        except Exception as exc:  # noqa: BLE001
            LOGGER.info("acrylic backdrop unavailable: %s", exc)

    def toggle_window_visibility(self) -> None:
        if self.isVisible():
            self.hide()
            return

        self.show()
        self.raise_()
        self.activateWindow()
        self.apply_capture_exclusion()
        self.status.setText("Ready")

    def start_region_capture(self) -> None:
        if self.image_thread is not None or self.region_selector is not None:
            return
        self.status.setText("Select screen area...")
        self.capture_button.setEnabled(False)
        self.hide()
        QTimer.singleShot(180, self.show_region_selector)

    def show_region_selector(self) -> None:
        selector = RegionSelector()
        self.region_selector = selector
        selector.captured.connect(self.on_region_captured)
        selector.canceled.connect(self.on_region_capture_canceled)
        selector.destroyed.connect(lambda *_: setattr(self, "region_selector", None))
        selector.show()
        selector.raise_()
        selector.activateWindow()

    def on_region_capture_canceled(self) -> None:
        self.region_selector = None
        self.show()
        self.raise_()
        self.activateWindow()
        self.capture_button.setEnabled(True)
        self.status.setText("Capture canceled")

    def on_region_captured(self, image_b64: str) -> None:
        self.region_selector = None
        self.show()
        self.raise_()
        self.activateWindow()
        if not image_b64:
            self.capture_button.setEnabled(True)
            self.status.setText("Capture failed")
            return

        # Analyze the capture immediately — no labels, no forced prompt; let the model decide.
        self.speech_input_locked = True
        self.auto_ask_timer.stop()
        self.current_partial_speech = ""
        self.pending_capture_b64 = ""
        self.ask_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.chat_messages.append(("user", f"[[screenshot:{image_b64}]]"))
        self.chat_messages.append(("assistant", "[[thinking:0]]"))
        self._begin_streaming()
        self.render_chat()
        self._scroll_to_message(len(self.chat_messages) - 2)
        self.status.setText("Анализирую…")

        self.image_thread = QThread()
        self.image_worker = ImageAnalysisWorker(image_b64, "")
        self.image_worker.moveToThread(self.image_thread)
        self.image_thread.started.connect(self.image_worker.run)
        self.image_worker.finished.connect(self.show_image_answer)
        self.image_worker.failed.connect(self.show_error)
        self.image_worker.finished.connect(self.image_thread.quit)
        self.image_worker.failed.connect(self.image_thread.quit)
        self.image_worker.finished.connect(self.image_worker.deleteLater)
        self.image_worker.failed.connect(self.image_worker.deleteLater)
        self.image_thread.finished.connect(self.on_image_thread_finished)
        self.image_thread.finished.connect(self.image_thread.deleteLater)
        self.image_thread.start()

    def submit_capture_question(self, prompt: str) -> None:
        image_b64 = self.pending_capture_b64
        if not image_b64:
            return
        if not self._require_login():
            return

        prompt = prompt.strip() or "Что изображено на скриншоте и что важно?"
        self.pending_capture_b64 = ""
        self.ask_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.chat_messages.append(("user", f"Запрос к скриншоту\n\n{prompt}"))
        self.chat_messages.append(("assistant", "Анализирую область экрана..."))
        self.input.clear()
        self.render_chat(focus_latest_assistant=True)
        self.status.setText("Analyzing captured area...")

        self.image_thread = QThread()
        self.image_worker = ImageAnalysisWorker(image_b64, prompt)
        self.image_worker.moveToThread(self.image_thread)
        self.image_thread.started.connect(self.image_worker.run)
        self.image_worker.finished.connect(self.show_image_answer)
        self.image_worker.failed.connect(self.show_error)
        self.image_worker.finished.connect(self.image_thread.quit)
        self.image_worker.failed.connect(self.image_thread.quit)
        self.image_worker.finished.connect(self.image_worker.deleteLater)
        self.image_worker.failed.connect(self.image_worker.deleteLater)
        self.image_thread.finished.connect(self.on_image_thread_finished)
        self.image_thread.finished.connect(self.image_thread.deleteLater)
        self.image_thread.start()

    def _load_audio_devices(self, preferred_name: str = "") -> None:
        self.device_combo.clear()

        try:
            import sounddevice as sd
        except ImportError:
            self.device_combo.addItem("Install sounddevice for audio", None)
            return

        hostapis = cast(list[dict[str, Any]], sd.query_hostapis())
        devices: list[AudioDevice] = []

        def hostapi_name(raw_device: dict) -> str:
            hostapi_index = int(raw_device.get("hostapi", -1))
            if 0 <= hostapi_index < len(hostapis):
                return str(hostapis[hostapi_index].get("name", ""))
            return ""

        for index, raw_device in enumerate(cast(list[dict[str, Any]], sd.query_devices())):
            samplerate = int(raw_device.get("default_samplerate") or 16000)
            api_name = hostapi_name(raw_device)

            if int(raw_device.get("max_input_channels", 0)) > 0:
                name = str(raw_device.get("name", f"Device {index}"))
                devices.append(
                    AudioDevice(
                        index=index,
                        name=f"{api_name}: {name}" if api_name else name,
                        hostapi=api_name,
                        loopback=False,
                        samplerate=samplerate,
                        channels=1,
                    )
                )

            if int(raw_device.get("max_output_channels", 0)) > 0 and "WASAPI" in api_name.upper():
                name = str(raw_device.get("name", f"Device {index}"))
                devices.append(
                    AudioDevice(
                        index=index,
                        name=f"System audio: {name}",
                        hostapi=api_name,
                        loopback=True,
                        samplerate=samplerate,
                        channels=min(2, int(raw_device.get("max_output_channels", 1))),
                    )
                )

        devices.sort(key=self._device_sort_key)
        system_devices = [device for device in devices if device.loopback]
        input_devices = [device for device in devices if not device.loopback]
        default_mic = next((device for device in input_devices), None)
        stereo_mix = next(
            (
                device
                for device in devices
                if "stereo mix" in device.name.lower() or "стерео микшер" in device.name.lower()
            ),
            None,
        )
        compact_devices: list[AudioDevice] = []
        if system_devices or stereo_mix:
            first_system = system_devices[0] if system_devices else stereo_mix
            compact_devices.append(
                AudioDevice(
                    index=None,
                    name="Auto system audio",
                    hostapi=first_system.hostapi if first_system else "WASAPI",
                    loopback=True,
                    samplerate=first_system.samplerate if first_system else WHISPER_SAMPLE_RATE,
                    channels=2,
                    auto_loopback=True,
                )
            )
        seen_loopbacks: set[str] = set()
        for system_device in system_devices:
            key = system_device.name.lower()
            if key in seen_loopbacks:
                continue
            seen_loopbacks.add(key)
            compact_devices.append(
                AudioDevice(
                    index=system_device.index,
                    name=system_device.name,
                    hostapi=system_device.hostapi,
                    loopback=True,
                    samplerate=system_device.samplerate,
                    channels=system_device.channels,
                    loopback_match=system_device.name.removeprefix("System audio:").strip(),
                )
            )
        if stereo_mix and not system_devices:
            compact_devices.append(stereo_mix)
        if default_mic:
            compact_devices.append(
                AudioDevice(
                    index=default_mic.index,
                    name="Default microphone",
                    hostapi=default_mic.hostapi,
                    loopback=False,
                    samplerate=default_mic.samplerate,
                    channels=1,
                )
            )
        seen_inputs = {"default microphone"} if default_mic else set()
        default_mic_key = default_mic.name.lower() if default_mic else ""
        for input_device in input_devices:
            key = input_device.name.lower()
            if key in seen_inputs or key == default_mic_key:
                continue
            seen_inputs.add(key)
            compact_devices.append(input_device)
            if len(compact_devices) >= 10:
                break
        devices = compact_devices
        if not devices:
            self.device_combo.addItem("No input devices found", None)
            return

        for device in devices:
            self.device_combo.addItem(device.name, device)

        if preferred_name:
            if preferred_name == "Default system audio":
                preferred_name = "Auto system audio"
            for index in range(self.device_combo.count()):
                if self.device_combo.itemText(index) == preferred_name:
                    self.device_combo.setCurrentIndex(index)
                    break

    def _device_sort_key(self, device: AudioDevice) -> tuple[int, str]:
        name = device.name.lower()
        api = device.hostapi.upper()
        if not device.loopback and "WASAPI" in api:
            return (0, name)
        if device.loopback:
            if "realtek" in name or "динамики" in name or "speakers" in name:
                return (1, name)
            return (2, name)
        if "DIRECTSOUND" in api:
            return (3, name)
        if "стерео микшер" in name or "stereo mix" in name:
            return (4, name)
        if not device.loopback and ("микрофон" in name or "microphone" in name or "mic " in name):
            return (5, name)
        if not device.loopback and ("линейный" in name or "лин. вход" in name or "line in" in name):
            return (7, name)
        if "MME" in api:
            return (9, name)
        return (6, name)

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if event.buttons() & Qt.MouseButton.LeftButton and self.drag_position is not None:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Alt:
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Alt:
            event.accept()
            return
        super().keyReleaseEvent(event)

    def eventFilter(self, obj, event) -> bool:  # noqa: ANN001
        if obj is getattr(self, "input", None):
            if event.type() == QEvent.Type.FocusIn:
                self._set_composer_focused(True)
            elif event.type() == QEvent.Type.FocusOut:
                self._set_composer_focused(False)
        return super().eventFilter(obj, event)

    def _set_composer_focused(self, focused: bool) -> None:
        self.composer.setProperty("focused", "true" if focused else "false")
        style = self.composer.style()
        style.unpolish(self.composer)
        style.polish(self.composer)

    def submit_question(self) -> None:
        if not self.ask_button.isEnabled():
            return
        if not self._require_login():
            return

        question = self.input.toPlainText().strip()

        if self.pending_attachment is not None:
            self.submit_with_attachment(question)
            return

        if not question:
            return

        if self.pending_capture_b64:
            self.submit_capture_question(question)
            return

        trusted_text = self.is_manual_input(question)
        self.stop_speech_capture_for_submit()
        speech_context = (self.raw_transcript_lines or self.transcript_lines)[-STT_CONTEXT_LINES:]
        chat_context = [
            content.split("\n\n", 1)[-1].strip()
            for role, content in self.chat_messages
            if role == "user" and content.strip()
        ][-8:]
        context = [*speech_context, *chat_context][-(STT_CONTEXT_LINES + len(chat_context)) :]
        self.ask_button.setEnabled(False)
        self.update_answer_actions(force_disabled=True)
        self.status.setText("Генерация…")
        self.question_count += 1
        self.chat_messages.append(("user", question))
        self.chat_messages.append(("assistant", "[[thinking:0]]"))
        self.input.clear()
        self.last_final_speech = ""
        self.current_partial_speech = ""
        self.last_question_candidate = ""
        self._begin_streaming()  # renders history once + inserts the streaming block
        self._launch_ask_stream(question, context, trusted_text)

    def _launch_ask_stream(self, question: str, context: list[str], trusted_text: bool) -> None:
        self.ask_thread = QThread()
        self.ask_worker = AskStreamWorker(question, context, trusted_text=trusted_text, storage_session_id=self.storage_session_id)
        self.ask_worker.moveToThread(self.ask_thread)
        self.ask_thread.started.connect(self.ask_worker.run)
        self.ask_worker.delta.connect(self.on_answer_delta)
        self.ask_worker.finished.connect(self.on_stream_finished)
        self.ask_worker.failed.connect(self.show_error)
        self.ask_worker.finished.connect(self.ask_thread.quit)
        self.ask_worker.failed.connect(self.ask_thread.quit)
        self.ask_worker.finished.connect(self.ask_worker.deleteLater)
        self.ask_worker.failed.connect(self.ask_worker.deleteLater)
        self.ask_thread.finished.connect(self.ask_thread.deleteLater)
        self.ask_thread.start()

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
    MAX_TEXT_FILE_BYTES = 80_000

    def attach_file(self) -> None:
        if not self.ask_button.isEnabled():
            return
        path_str, _ = QFileDialog.getOpenFileName(self, "Прикрепить файл", "", "Все файлы (*.*)")
        if not path_str:
            return
        path = Path(path_str)
        try:
            raw = path.read_bytes()
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Не удалось прочитать файл: {_short_error(exc)}")
            return
        if path.suffix.lower() in self.IMAGE_EXTENSIONS:
            self.pending_attachment = {"kind": "image", "name": path.name, "data": base64.b64encode(raw).decode("ascii")}
        else:
            try:
                text = raw[: self.MAX_TEXT_FILE_BYTES + 1].decode("utf-8")
                if len(raw) > self.MAX_TEXT_FILE_BYTES:
                    text = text[: self.MAX_TEXT_FILE_BYTES] + "\n… (файл обрезан)"
                self.pending_attachment = {"kind": "text", "name": path.name, "content": text}
            except UnicodeDecodeError:
                self.pending_attachment = {"kind": "binary", "name": path.name}
        self._refresh_attachment_bar()
        self.input.setFocus()
        self.status.setText("Файл прикреплён. Добавьте текст (по желанию) и нажмите Enter.")

    def clear_attachment(self) -> None:
        self.pending_attachment = None
        self._refresh_attachment_bar()

    def _refresh_attachment_bar(self) -> None:
        attachment = self.pending_attachment
        if not attachment:
            self.attach_bar.setVisible(False)
            self.attach_chip.setText("")
            return
        kind_label = {"image": "изображение", "text": "текст", "binary": "файл"}.get(attachment["kind"], "файл")
        self.attach_chip.setText(f"📎  {attachment['name']}  ·  {kind_label}")
        self.attach_bar.setVisible(True)

    def submit_with_attachment(self, text: str) -> None:
        attachment = self.pending_attachment
        if attachment is None:
            return
        self.pending_attachment = None
        self._refresh_attachment_bar()
        self.stop_speech_capture_for_submit()
        self.ask_button.setEnabled(False)
        self.input.clear()

        if attachment["kind"] == "image":
            user_md = f"[[screenshot:{attachment['data']}]]" + (f"\n\n{text}" if text else "")
            self.chat_messages.append(("user", user_md))
            self.chat_messages.append(("assistant", "[[thinking:0]]"))
            self._begin_streaming()
            self.status.setText("Анализирую…")
            self.image_thread = QThread()
            self.image_worker = ImageAnalysisWorker(attachment["data"], text)
            self.image_worker.moveToThread(self.image_thread)
            self.image_thread.started.connect(self.image_worker.run)
            self.image_worker.finished.connect(self.show_image_answer)
            self.image_worker.failed.connect(self.show_error)
            self.image_worker.finished.connect(self.image_thread.quit)
            self.image_worker.failed.connect(self.image_thread.quit)
            self.image_worker.finished.connect(self.image_worker.deleteLater)
            self.image_worker.failed.connect(self.image_worker.deleteLater)
            self.image_thread.finished.connect(self.on_image_thread_finished)
            self.image_thread.finished.connect(self.image_thread.deleteLater)
            self.image_thread.start()
            return

        if attachment["kind"] == "text":
            instruction = text.strip() or "Проанализируй этот файл и объясни, что он делает."
            question = f"{instruction}\n\nФайл «{attachment['name']}»:\n```\n{attachment['content']}\n```"
        else:
            instruction = text.strip() or "Что это за файл и для чего он?"
            question = f"{instruction}\n\n(Приложен файл «{attachment['name']}», его содержимое не текстовое и не прочитано.)"
        file_md = f"[[file:{attachment['name']}]]" + (f"\n\n{text}" if text else "")
        self.chat_messages.append(("user", file_md))
        self.chat_messages.append(("assistant", "[[thinking:0]]"))
        self._begin_streaming()
        self.status.setText("Генерация…")
        self._launch_ask_stream(question, [], True)

    def _last_assistant_index(self) -> int:
        for index in range(len(self.chat_messages) - 1, -1, -1):
            if self.chat_messages[index][0] == "assistant":
                return index
        return -1

    def _begin_streaming(self) -> None:
        # Rebuild the message widgets; the trailing assistant message starts as
        # animated thinking dots, then we stream rich text into that one row only.
        self._stream_active = True
        self._stream_buffer = ""
        self._stream_render_pending = False
        _DIAGRAM_RENDER["enabled"] = False  # skip diagram rendering on partial source
        # Animate just the freshly added user message + assistant row into view.
        self.render_chat(animate_from=max(0, len(self.chat_messages) - 2))
        self._stream_prefix_snippets = len(CODE_SNIPPETS)
        self._stream_row = self._assistant_rows.get(self._last_assistant_index())
        QTimer.singleShot(0, self.chat_area.scroll_to_bottom)

    def _stop_streaming(self) -> None:
        self._stream_active = False
        _DIAGRAM_RENDER["enabled"] = True
        self.typing_timer.stop()

    def _scroll_answer_to_bottom(self) -> None:
        self.chat_area.scroll_to_bottom()

    def _scroll_to_message(self, index: int) -> None:
        self.chat_area.scroll_to_bottom()

    def _tick_typing(self) -> None:
        # Thinking dots animate themselves now; nothing to drive here.
        return

    def on_answer_delta(self, chunk: str) -> None:
        if not self._stream_active:
            return
        self._stream_buffer += chunk
        index = self._last_assistant_index()
        if index >= 0:
            self.chat_messages[index] = ("assistant", balance_streaming_markdown(self._stream_buffer))
        if not self._stream_render_pending:
            self._stream_render_pending = True
            QTimer.singleShot(70, self._flush_stream_render)

    def _flush_stream_render(self) -> None:
        self._stream_render_pending = False
        if not self._stream_active or self._stream_row is None:
            return
        bar = self.chat_area.scroll.verticalScrollBar()
        follow = bar.value() >= bar.maximum() - 8
        # Keep snippet ids stable/bounded: drop anything from the previous frame.
        del CODE_SNIPPETS[self._stream_prefix_snippets:]
        balanced = balance_streaming_markdown(self._stream_buffer)
        markup = markdown_to_html(balanced)
        # Mint caret at the end while generating (skip when inside a code fence).
        if not balanced.rstrip().endswith("```"):
            markup = _with_stream_caret(markup)
        self._stream_row.show_html(markup, final=False)
        if follow:
            QTimer.singleShot(0, self.chat_area.scroll_to_bottom)

    def on_stream_finished(self, result: object) -> None:
        self._stop_streaming()
        self.show_answer(result)

    def _action_icon_link(self, kind: str, href: str, size: int) -> str:
        png = pixmap_to_base64_png(icon_pixmap(kind, size, "#8290a0"))
        return f"<a href='{href}'><img src='data:image/png;base64,{png}' width='{size}' height='{size}' /></a>"

    def update_answer_actions(self, force_disabled: bool = False) -> None:
        # Expand / answer-action buttons were removed; kept as a no-op so existing callers stay valid.
        return

    def submit_expand(self, mode: str) -> None:
        if self.expand_thread is not None:
            return
        base_answer = (self.last_main_answer_text or self.last_answer_text).strip()
        if not self.last_answer_question.strip() or not base_answer:
            self.status.setText("No answer to expand.")
            return

        mode = mode if mode in EXPAND_LABELS else "details"
        header = EXPAND_LABELS[mode]
        self.chat_messages.append(("assistant", f"{header}\n\nГенерирую расширение..."))
        self.render_chat(focus_latest_assistant=True)
        self.status.setText("Expanding answer...")
        self.update_answer_actions(force_disabled=True)

        self.expand_thread = QThread()
        self.expand_worker = ExpandWorker(self.last_answer_question, base_answer, mode, self.storage_session_id)
        self.expand_worker.moveToThread(self.expand_thread)
        self.expand_thread.started.connect(self.expand_worker.run)
        self.expand_worker.finished.connect(self.show_expand_answer)
        self.expand_worker.failed.connect(self.show_error)
        self.expand_worker.finished.connect(self.expand_thread.quit)
        self.expand_worker.failed.connect(self.expand_thread.quit)
        self.expand_worker.finished.connect(self.expand_worker.deleteLater)
        self.expand_worker.failed.connect(self.expand_worker.deleteLater)
        self.expand_thread.finished.connect(self.on_expand_thread_finished)
        self.expand_thread.finished.connect(self.expand_thread.deleteLater)
        self.expand_thread.start()

    def show_expand_answer(self, result: object) -> None:
        if isinstance(result, ExpandResult):
            mode = result.mode if result.mode in EXPAND_LABELS else "details"
            cleaned = result.answer.strip()
            self.last_answer_text = cleaned
            self.last_answer_id = result.answer_id
            self.last_answer_domain = result.plan_domain
            self.last_answer_intent = result.plan_intent
        else:
            mode = "details"
            cleaned = str(result).strip()
            self.last_answer_text = cleaned
            self.last_answer_id = None
        if not cleaned:
            cleaned = "Модель вернула пустое расширение."
        self.replace_last_assistant(f"{EXPAND_LABELS[mode]}\n\n{cleaned}")
        self.ask_button.setEnabled(True)
        self.listen_button.setEnabled(True)
        self.status.setText("Ready")
        self.update_answer_actions()

    def on_expand_thread_finished(self) -> None:
        self.expand_worker = None
        self.expand_thread = None
        self.update_answer_actions()

    def submit_feedback(self, label: str) -> None:
        if self.last_answer_id is None:
            self.status.setText("No stored answer id for feedback.")
            return
        try:
            if STACKWIRE_API_URL:
                response = requests.post(
                    f"{STACKWIRE_API_URL}/feedback",
                    json={"answer_id": self.last_answer_id, "label": label},
                    timeout=(STACKWIRE_API_CONNECT_TIMEOUT, 10),
                )
                response.raise_for_status()
            else:
                log_feedback(self.last_answer_id, label)
            self.status.setText("Feedback saved.")
            LOGGER.info("feedback saved answer_id=%s label=%s", self.last_answer_id, label)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("feedback failed: %s", exc)
            self.status.setText(f"Feedback failed: {exc}")

    def save_current_good_answer(self) -> None:
        question = self.last_answer_question.strip()
        answer = self.last_answer_text.strip()
        if not question or not answer:
            self.status.setText("No answer to save.")
            return
        try:
            payload = {
                "question": question,
                "answer": answer,
                "domain": self.last_answer_domain,
                "intent": self.last_answer_intent,
                "tags": [],
                "rating": 5,
            }
            if STACKWIRE_API_URL:
                response = requests.post(
                    f"{STACKWIRE_API_URL}/good-answer",
                    json=payload,
                    timeout=(STACKWIRE_API_CONNECT_TIMEOUT, 15),
                )
                response.raise_for_status()
            else:
                save_good_answer(
                    question=question,
                    answer=answer,
                    domain=self.last_answer_domain,
                    intent=self.last_answer_intent,
                    tags=[],
                    rating=5,
                )
            self.status.setText("Good answer saved.")
            LOGGER.info("good answer saved domain=%s intent=%s", self.last_answer_domain, self.last_answer_intent)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("save good answer failed: %s", exc)
            self.status.setText(f"Save good failed: {exc}")

    def toggle_recording_hotkey(self) -> None:
        if not self.ask_button.isEnabled():
            return

        if self._speech_is_running():
            self.submit_after_speech_stop = True
            if self.speech_worker:
                self.speech_worker.stop()
            self.listen_button.setEnabled(False)
            self.apply_icons()
            self.status.setText("Stopping audio...")
            return

        self.submit_after_speech_stop = False
        self.toggle_listening()

    def submit_recorded_question(self) -> None:
        self.submit_after_speech_stop = False
        if not self.ask_button.isEnabled():
            return
        if not self._require_login():
            return

        candidate = (
            self.input.toPlainText().strip()
            or self.last_question_candidate
            or self.last_final_speech
            or " ".join(self.transcript_lines[-STT_CONTEXT_LINES:]).strip()
        )
        if not candidate:
            self.status.setText("No speech captured.")
            return

        if not self.is_manual_input(candidate):
            candidate = condense_spoken_question(candidate) or candidate
        self.input.setPlainText(candidate)
        self.input.moveCursor(QTextCursor.MoveOperation.End)
        self.submit_question()

    def stop_speech_capture_for_submit(self) -> None:
        self.speech_input_locked = True
        self.auto_ask_timer.stop()
        self.current_partial_speech = ""
        if self.speech_worker:
            try:
                self.speech_worker.stop()
            except RuntimeError:
                pass
        if self.speech_thread:
            self.listen_button.setEnabled(False)
            self.apply_icons()

    def is_manual_input(self, question: str) -> bool:
        normalized_question = clean_live_transcript(question).strip().lower()
        speech_sources = [
            self.last_final_speech,
            self.last_question_candidate,
            " ".join(self.transcript_lines[-STT_CONTEXT_LINES:]),
        ]
        normalized_sources = {
            clean_live_transcript(source).strip().lower()
            for source in speech_sources
            if source.strip()
        }
        if not normalized_sources:
            return True
        return normalized_question not in normalized_sources

    def show_answer(self, result: object) -> None:
        if isinstance(result, AskResult):
            recovery = result.recovery
            recovered_question = recovery.recovered_question.strip()
            self.update_debug_panel(
                raw_stt=result.raw_text,
                recovered_question=recovered_question,
                confidence=recovery.confidence,
                detected_topic=recovery.detected_topic,
                stt_latency_ms=self.last_stt_latency_ms,
                recovery_latency_ms=result.recovery_latency * 1000,
                answer_latency_ms=result.answer_latency * 1000,
                total_latency_ms=result.total_latency * 1000,
            )
            cleaned = result.answer.strip() or "Модель вернула пустой ответ. Попробуй нажать Enter еще раз."
            if not result.answered:
                self.input.setPlainText(recovered_question or result.raw_text)
                self.input.moveCursor(QTextCursor.MoveOperation.End)
                self.last_answer_question = ""
                self.last_answer_text = ""
                self.last_main_answer_text = ""
                self.last_answer_id = None
                self.last_answer_domain = None
                self.last_answer_intent = None
            else:
                self.last_answer_question = recovered_question or result.raw_text
                self.last_answer_text = cleaned
                self.last_main_answer_text = cleaned
                self.last_answer_id = result.answer_id
                self.last_answer_domain = result.plan_domain
                self.last_answer_intent = result.plan_intent
        else:
            cleaned = str(result).strip()
            if not cleaned:
                cleaned = "Модель вернула пустой ответ. Попробуй нажать Enter еще раз."
            elif len(cleaned) < 80:
                cleaned = f"{cleaned}\n\nНюанс:\nОтвет выглядит неполным. Модель могла остановиться раньше времени; повтори запрос или уменьши контекст."
        if not isinstance(result, AskResult):
            self.last_answer_text = cleaned
            self.last_answer_id = None
        self.replace_last_assistant(cleaned)
        self.ask_button.setEnabled(True)
        self.listen_button.setEnabled(True)
        self.status.setText("Ready")
        self.update_answer_actions()

    def show_image_answer(self, text: str) -> None:
        self._stop_streaming()
        cleaned = text.strip() or "Не удалось распознать содержимое области."
        self.replace_last_assistant(cleaned)
        self.last_answer_question = ""
        self.last_answer_text = cleaned
        self.last_answer_id = None
        self.ask_button.setEnabled(True)
        self.capture_button.setEnabled(True)
        self.status.setText("Ready")
        self.update_answer_actions()

    def show_error(self, message: str) -> None:
        self._stop_streaming()
        self.replace_last_assistant(f"Ошибка: {message}")
        self.ask_button.setEnabled(True)
        self.listen_button.setEnabled(True)
        self.capture_button.setEnabled(True)
        if self.pending_capture_b64:
            self.input.setFocus()
        self.status.setText("Error")
        self.update_answer_actions()

    def on_anchor_clicked(self, url: QUrl) -> None:
        target = url.toString()
        for prefix, handler in (("edit:", self.start_edit_message), ("copy:", self.copy_message), ("copycode:", self.copy_code_snippet)):
            if target.startswith(prefix):
                try:
                    index = int(target[len(prefix) :])
                except ValueError:
                    return
                handler(index)
                return
        if url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)

    def copy_code_snippet(self, index: int) -> None:
        if 0 <= index < len(CODE_SNIPPETS):
            QApplication.clipboard().setText(CODE_SNIPPETS[index])
            self.status.setText("Код скопирован")

    def _message_plain_text(self, index: int) -> str:
        if not (0 <= index < len(self.chat_messages)):
            return ""
        content = self.chat_messages[index][1]
        match = re.match(r"^Вопрос\s+\d+\s*\n\n(.+)$", content.strip(), flags=re.DOTALL)
        return (match.group(1).strip() if match else content).strip()

    def copy_message(self, index: int) -> None:
        text = self._message_plain_text(index)
        if not text:
            return
        QApplication.clipboard().setText(text)
        self.status.setText("Скопировано")

    def start_edit_message(self, index: int) -> None:
        if not self.ask_button.isEnabled():
            return  # busy generating
        if index < 0 or index >= len(self.chat_messages):
            return
        if self.chat_messages[index][0] != "user":
            return
        text = self._message_plain_text(index)
        # ChatGPT-style: drop this message and everything after it, then re-ask on submit.
        self.chat_messages = self.chat_messages[:index]
        self.last_answer_question = ""
        self.last_answer_text = ""
        self.last_main_answer_text = ""
        self.last_answer_id = None
        self.last_answer_domain = None
        self.last_answer_intent = None
        self.input.setPlainText(text)
        self.input.moveCursor(QTextCursor.MoveOperation.End)
        self.input.setFocus()
        self.render_chat()
        self.update_answer_actions()
        self.status.setText("Правка — измените текст и нажмите Enter")

    def toggle_debug_panel(self) -> None:
        self.debug_expanded = self.debug_button.isChecked()
        self.debug_panel.setVisible(self.debug_expanded)
        self.debug_button.setToolTip("Hide debug" if self.debug_expanded else "Debug")

    def audio_device_names(self) -> list[str]:
        return [self.device_combo.itemText(index) for index in range(self.device_combo.count()) if self.device_combo.itemText(index).strip()]

    def current_audio_device_name(self) -> str:
        return self.device_combo.currentText().strip()

    def set_audio_device_name(self, name: str) -> None:
        if not name:
            return
        self._load_audio_devices(name)
        for index in range(self.device_combo.count()):
            if self.device_combo.itemText(index) == name:
                self.device_combo.setCurrentIndex(index)
                return

    def show_settings_dialog(self) -> None:
        if self.device_combo.count() == 0:
            self._load_audio_devices(os.getenv("STACKWIRE_AUDIO_DEVICE", "").strip())
        dialog = SettingsDialog(self, _model_choices(), self.audio_device_names(), self.current_audio_device_name())
        dialog.setStyleSheet(build_window_styles(self.ui_zoom))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        values = dialog.values()
        # Only model fields are mandatory; toggles/urls may be empty by design.
        required_keys = ("ANSWER_MODEL", "RECOVERY_MODEL", "VISION_MODEL")
        missing = [key for key in required_keys if not values.get(key, "").strip()]
        if missing:
            self.status.setText(f"Не сохранено: пустое поле {missing[0]}")
            return

        try:
            _save_local_env_values(values)
            for key, value in values.items():
                os.environ[key] = value
            self.set_audio_device_name(values.get("STACKWIRE_AUDIO_DEVICE", ""))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("settings save failed: %s", exc)
            self.status.setText(f"Не удалось сохранить настройки: {_short_error(exc)}")
            return

        self.model_chip.setText(current_answer_model())
        self.update_account_chip()
        self.update_debug_panel()
        self.render_chat()
        if STACKWIRE_API_URL:
            self.status.setText("Настройки сохранены. Удалённый API использует свою конфигурацию.")
        else:
            self.status.setText("Настройки сохранены.")

    def update_debug_panel(
        self,
        *,
        raw_stt: str = "-",
        recovered_question: str = "-",
        confidence: float | None = None,
        detected_topic: str = "-",
        stt_latency_ms: float | None = None,
        recovery_latency_ms: float | None = None,
        answer_latency_ms: float | None = None,
        total_latency_ms: float | None = None,
    ) -> None:
        confidence_text = "-" if confidence is None else f"{confidence:.2f}"
        stt_text = "-" if stt_latency_ms is None else f"{stt_latency_ms:.0f}"
        recovery_text = "-" if recovery_latency_ms is None else f"{recovery_latency_ms:.0f}"
        answer_text = "-" if answer_latency_ms is None else f"{answer_latency_ms:.0f}"
        total_text = "-" if total_latency_ms is None else f"{total_latency_ms:.0f}"
        self.debug_panel.setText(
            "debug\n"
            f"mode: {STACKWIRE_MODE}\n"
            f"answer_mode: {ANSWER_MODE}\n"
            f"api: {STACKWIRE_API_URL or 'local'}\n"
            f"answer_model: {current_answer_model()}\n"
            f"recovery_model: {current_recovery_model()}\n"
            f"vision_model: {current_vision_model()}\n"
            f"raw_stt: {raw_stt or '-'}\n"
            f"recovered_question: {recovered_question or '-'}\n"
            f"confidence: {confidence_text}\n"
            f"detected_topic: {detected_topic or '-'}\n"
            f"stt_latency_ms: {stt_text}\n"
            f"recovery_latency_ms: {recovery_text}\n"
            f"answer_latency_ms: {answer_text}\n"
            f"total_latency_ms: {total_text}"
        )

    def clear_answer(self) -> None:
        self._stop_streaming()
        self.input.clear()
        self.chat_messages.clear()
        self.raw_transcript_lines.clear()
        self.transcript_lines.clear()
        self.last_question_candidate = ""
        self.current_partial_speech = ""
        self.pending_capture_b64 = ""
        self.pending_attachment = None
        self._refresh_attachment_bar()
        self.speech_input_locked = False
        self.submit_after_speech_stop = False
        self.last_stt_latency_ms = None
        self.last_recovery_latency_ms = None
        self.last_answer_latency_ms = None
        self.last_total_latency_ms = None
        self.last_answer_question = ""
        self.last_answer_text = ""
        self.last_main_answer_text = ""
        self.last_answer_id = None
        self.last_answer_domain = None
        self.last_answer_intent = None
        self.question_count = 0
        self.render_chat()
        self.update_answer_actions()
        self.update_debug_panel()
        self.status.setText("Ready")
        self.last_final_speech = ""

    def build_llm_input(self, question: str) -> str:
        context_lines = self.transcript_lines[-STT_CONTEXT_LINES:]
        context = "\n".join(context_lines)
        if not context:
            return question
        return (
            "Контекст распознанный из записи:\n"
            f"{context}\n\n"
            "Задача: найди последний технический вопрос в этом контексте, отбрось лишние фразы, исправь ошибки речевого распознавания и ответь.\n\n"
            "Если поле ниже уже похоже на вопрос, отвечай на него:\n"
            f"{question}"
        )

    def replace_last_assistant(self, text: str) -> None:
        for index in range(len(self.chat_messages) - 1, -1, -1):
            if self.chat_messages[index][0] == "assistant":
                self.chat_messages[index] = ("assistant", text)
                self.render_chat(focus_latest_assistant=True)
                return
        self.chat_messages.append(("assistant", text))
        self.render_chat(focus_latest_assistant=True)

    @staticmethod
    def _is_thinking(content: str) -> bool:
        return re.fullmatch(r"\[\[thinking:\d+\]\]", content.strip()) is not None

    def _screenshot_pixmap(self, content: str) -> QPixmap | None:
        match = re.search(r"\[\[screenshot:([A-Za-z0-9+/=]+)\]\]", content)
        if not match:
            return None
        try:
            pixmap = QPixmap()
            pixmap.loadFromData(base64.b64decode(match.group(1)))
            return pixmap if not pixmap.isNull() else None
        except Exception:
            return None

    def _make_user_row(self, index: int, content: str) -> QWidget:
        row = QWidget()
        outer = QHBoxLayout(row)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addStretch(1)  # push the bubble group to the right

        group = QWidget()
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(_px(4))

        is_screenshot = "[[screenshot:" in content
        is_file = "[[file:" in content
        caption = self._message_plain_text(index) if (is_screenshot or is_file) else ""
        caption = re.sub(r"\[\[(?:screenshot|file):[^\]]*\]\]", "", caption).strip()

        if is_screenshot:
            pixmap = self._screenshot_pixmap(content)
            if pixmap is not None:
                shot = QLabel()
                shot.setObjectName("shotLabel")
                scaled = pixmap.scaledToWidth(_px(240), Qt.TransformationMode.SmoothTransformation)
                shot.setPixmap(_rounded_pixmap(scaled, _px(12)))
                group_layout.addWidget(shot, 0, Qt.AlignmentFlag.AlignRight)
            if caption:
                group_layout.addWidget(self._user_bubble_label(caption), 0, Qt.AlignmentFlag.AlignRight)
        elif is_file:
            file_match = re.search(r"\[\[file:([^\]]+)\]\]", content)
            name = file_match.group(1) if file_match else "файл"
            group_layout.addWidget(self._user_bubble_label(f"📎  {name}" + (f"\n{caption}" if caption else "")), 0, Qt.AlignmentFlag.AlignRight)
        else:
            group_layout.addWidget(self._user_bubble_label(self._message_plain_text(index)), 0, Qt.AlignmentFlag.AlignRight)

        actions = QWidget()
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(_px(2))
        actions_layout.addStretch(1)
        actions_layout.addWidget(_flat_icon_button("copy", "Скопировать", lambda i=index: self.copy_message(i)))
        if not is_screenshot:
            actions_layout.addWidget(_flat_icon_button("edit", "Изменить", lambda i=index: self.start_edit_message(i)))
        group_layout.addWidget(actions)

        outer.addWidget(group)
        return row

    def _user_bubble_label(self, text: str) -> QLabel:
        bubble = QLabel(text)
        bubble.setObjectName("userBubble")
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # Set the font explicitly (not only via QSS) so fontMetrics is correct and the
        # bubble hugs the text: one line when it fits, wrapping only past the cap.
        font = QFont("Space Grotesk")
        font.setPixelSize(_px(15))
        bubble.setFont(font)
        available = max(_px(320), self.chat_area.width() - _px(96))
        cap = min(_px(680), int(available * 0.72))
        metrics = bubble.fontMetrics()
        natural = max((metrics.horizontalAdvance(line) for line in (text.splitlines() or [""])), default=0)
        # Qt can underestimate Cyrillic text when the requested UI font falls
        # back to another face, so keep enough room for the real painted glyphs.
        width = min(cap, natural + _px(64))
        bubble.setMinimumWidth(min(width, cap))
        bubble.setMaximumWidth(cap)
        return bubble

    def _add_message_row(self, index: int, role: str, content: str, *, animate: bool = False) -> None:
        if role == "user":
            row: QWidget = self._make_user_row(index, content)
            self.chat_area.add_row(row)
        else:
            row = AssistantRow(index, self.on_anchor_clicked, self.copy_message)
            self.chat_area.add_row(row)
            self._assistant_rows[index] = row
            if self._is_thinking(content):
                row.show_thinking()
            else:
                row.show_html(markdown_to_html(content), final=True)
        if animate:
            _animate_in(row)

    def render_chat(self, focus_latest_assistant: bool = False, animate_from: int = -1) -> None:
        CODE_SNIPPETS.clear()
        self._stream_row = None
        self._assistant_rows = {}
        self.chat_area.clear_rows()
        if not self.chat_messages:
            self.chat_area.show_welcome()
            return
        self.chat_area.show_list()
        for index, (role, content) in enumerate(self.chat_messages):
            self._add_message_row(index, role, content, animate=(animate_from >= 0 and index >= animate_from))
        if focus_latest_assistant:
            QTimer.singleShot(0, self.chat_area.scroll_to_bottom)

    def toggle_listening(self) -> None:
        if self._speech_is_running():
            self.submit_after_speech_stop = False
            if self.speech_worker:
                self.speech_worker.stop()
            self.listen_button.setEnabled(False)
            self.apply_icons()
            self.status.setText("Stopping audio...")
            return

        preferred_device = self.device_combo.currentText()
        self._load_audio_devices(preferred_device)
        device = self.device_combo.currentData()
        if device is None:
            self.show_error("Не выбрано аудио-устройство.")
            return

        self.speech_input_locked = False
        self.speech_thread = QThread()
        self.speech_worker = SpeechWorker(device)
        self.speech_worker.moveToThread(self.speech_thread)
        self.speech_thread.started.connect(self.speech_worker.run)
        self.speech_worker.partial.connect(self.on_partial_speech)
        self.speech_worker.final.connect(self.on_final_speech)
        self.speech_worker.stt_latency.connect(self.on_stt_latency)
        self.speech_worker.failed.connect(self.show_error)
        self.speech_worker.info.connect(self.status.setText)
        self.speech_worker.stopped.connect(self.on_speech_stopped)
        self.speech_worker.stopped.connect(self.speech_thread.quit)
        self.speech_thread.finished.connect(self.on_speech_thread_finished)
        self.speech_thread.finished.connect(self.speech_thread.deleteLater)
        self.speech_thread.start()

        self.apply_icons()
        self.status.setText("Listening...")

    def on_partial_speech(self, text: str) -> None:
        if self.speech_input_locked:
            return
        if text:
            self.current_partial_speech = clean_live_transcript(text)
            visible = clean_live_transcript(append_transcript_segment(self.last_final_speech, self.current_partial_speech))
            if visible and not self.ask_button.isEnabled():
                self.status.setText(f"Hearing: {visible[:90]}")
                return
            if visible:
                self.input.setPlainText(visible)
                self.input.moveCursor(QTextCursor.MoveOperation.End)
            self.status.setText(f"Hearing: {self.current_partial_speech[:90]}")

    def on_stt_latency(self, latency_ms: float) -> None:
        self.last_stt_latency_ms = latency_ms
        LOGGER.info("stt_latency_ms=%.0f", latency_ms)

    def on_final_speech(self, text: str) -> None:
        if self.speech_input_locked:
            return
        if not text:
            return
        raw_text = text.strip()
        normalized = clean_live_transcript(text)
        if not normalized:
            return
        self.raw_transcript_lines.append(raw_text)
        self.raw_transcript_lines = self.raw_transcript_lines[-60:]
        self.transcript_lines.append(normalized)
        self.transcript_lines = self.transcript_lines[-60:]
        LOGGER.info("raw transcript=%r", raw_text)
        LOGGER.info("normalized transcript=%r", normalized)
        self.update_debug_panel(raw_stt=raw_text, stt_latency_ms=self.last_stt_latency_ms)
        starts_new_question = looks_like_question(normalized) and (
            not looks_like_question(self.last_final_speech) or len(self.last_final_speech.split()) > STT_LIVE_MAX_WORDS
        )
        if starts_new_question:
            self.last_final_speech = normalized
        else:
            self.last_final_speech = clean_live_transcript(append_transcript_segment(self.last_final_speech, normalized))
        self.current_partial_speech = ""
        if looks_like_question(normalized):
            self.last_question_candidate = self.last_final_speech
        elif not self.last_question_candidate:
            recent = " ".join(self.transcript_lines[-2:])
            if looks_like_question(recent):
                self.last_question_candidate = recent
        if not self.last_question_candidate:
            self.last_question_candidate = " ".join(self.transcript_lines[-STT_CONTEXT_LINES:]).strip()
        if self.last_final_speech and self.ask_button.isEnabled():
            self.input.setPlainText(self.last_final_speech)
            self.input.moveCursor(QTextCursor.MoveOperation.End)
        self.status.setText("Speech captured. Press Enter when question is ready.")
        self.auto_ask_timer.start()

    def refresh_question_candidate(self) -> None:
        if self.speech_input_locked:
            return
        if self.input.toPlainText().strip():
            return
        candidate = self.last_question_candidate or self.last_final_speech or " ".join(self.transcript_lines[-STT_CONTEXT_LINES:]).strip()
        if not candidate:
            return
        self.input.setPlainText(candidate)
        self.input.moveCursor(QTextCursor.MoveOperation.End)

    def on_speech_stopped(self) -> None:
        self.listen_button.setEnabled(True)
        self.apply_icons()
        if self.status.text().startswith(("Listening", "Stopping")):
            self.status.setText("Ready")

    def on_speech_thread_finished(self) -> None:
        self.speech_worker = None
        self.speech_thread = None
        self.listen_button.setEnabled(True)
        self.apply_icons()
        if self.submit_after_speech_stop:
            QTimer.singleShot(150, self.submit_recorded_question)

    def on_image_thread_finished(self) -> None:
        self.image_worker = None
        self.image_thread = None
        self.capture_button.setEnabled(True)

    def _speech_is_running(self) -> bool:
        if self.speech_thread is None:
            return False
        try:
            return self.speech_thread.isRunning()
        except RuntimeError:
            self.speech_thread = None
            self.speech_worker = None
            return False


def build_window_styles(scale: float | None = None) -> str:
    return f"""
QWidget#root {{
    background: transparent;
}}

QFrame#shell {{
    background: transparent;
    border: none;
    border-radius: {_px(22, scale)}px;
}}

QFrame#rail {{
    background: {RAIL};
    border-top-left-radius: {_px(22, scale)}px;
    border-bottom-left-radius: {_px(22, scale)}px;
    border-right: none;
}}

QFrame#content {{
    background: {SURFACE};
    border-top-right-radius: {_px(22, scale)}px;
    border-bottom-right-radius: {_px(22, scale)}px;
}}

QLabel#title {{
    color: #d8d5db;
    font-size: {_px(14, scale)}px;
    font-weight: 850;
}}

QLabel#titleMark {{
    min-width: {_px(20, scale)}px;
    min-height: {_px(20, scale)}px;
}}

QLabel#subtitle,
QLabel#status {{
    color: {MUTED};
    font-size: {_px(11, scale)}px;
}}

QLabel#modelChip,
QLabel#apiChip,
QLabel#accountChip,
QLabel#status {{
    min-height: {_px(28, scale)}px;
    padding: 0 {_px(10, scale)}px;
    border-radius: {_px(9, scale)}px;
    font-family: {FONT_STACK};
    font-size: {_px(11, scale)}px;
}}

QLabel#accountChip {{
    color: #c7d1db;
    background: rgba(4, 8, 11, 0.52);
    border: 1px solid rgba(154, 214, 189, 0.13);
}}

QLabel#modelChip {{
    color: #b5d7c8;
    background: rgba(4, 8, 11, 0.52);
    border: 1px solid rgba(154, 214, 189, 0.13);
}}

QLabel#apiChip {{
    color: #9ad6bd;
    background: rgba(154, 214, 189, 0.08);
    border: 1px solid rgba(154, 214, 189, 0.16);
}}

QLabel#status {{
    color: #6f8793;
    background: rgba(4, 8, 11, 0.35);
    border: 1px solid rgba(154, 214, 189, 0.08);
}}

QLabel#debugPanel {{
    color: #a9cdbd;
    background: rgba(5, 7, 10, 170);
    border: 1px solid rgba(154, 214, 189, 0.11);
    border-radius: {_px(14, scale)}px;
    padding: {_px(6, scale)}px {_px(8, scale)}px;
    font-family: Consolas, Courier New, monospace;
    font-size: {_px(10, scale)}px;
}}

QWidget#chatArea {{
    background: transparent;
    border: none;
}}

QScrollArea#chatScroll {{
    background: transparent;
    border: none;
}}

QWidget#chatContainer {{
    background: transparent;
}}

QScrollBar:vertical {{
    background: transparent;
    width: {_px(10, scale)}px;
    margin: {_px(4, scale)}px {_px(2, scale)}px;
}}

QScrollBar::handle:vertical {{
    background: rgba(154, 214, 189, 0.26);
    border-radius: {_px(5, scale)}px;
    min-height: {_px(32, scale)}px;
}}

QScrollBar::handle:vertical:hover {{
    background: rgba(154, 214, 189, 0.46);
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}

QLabel#userBubble {{
    background: {ELEVATED};
    border: none;
    border-radius: {_px(18, scale)}px;
    padding: {_px(10, scale)}px {_px(15, scale)}px;
    color: #eef3f9;
    font-family: {FONT_STACK};
    font-size: {_px(15, scale)}px;
}}

QLabel#shotLabel {{
    background: transparent;
    border-radius: {_px(12, scale)}px;
}}

QLabel#roleLabel {{
    color: #7d8a99;
    font-size: {_px(11, scale)}px;
    font-weight: 700;
}}

QTextBrowser#msgBrowser {{
    background: transparent;
    border: none;
    selection-background-color: #263139;
}}

QPushButton#msgActionButton {{
    background: transparent;
    border: none;
    border-radius: {_px(7, scale)}px;
    padding: 0;
    min-height: 0;
}}

QPushButton#msgActionButton:hover {{
    background: rgba(154, 214, 189, 0.14);
}}

QLabel#welcomeTitle {{
    font-family: {FONT_DISPLAY};
    color: #e6f4ee;
    font-size: {_px(40, scale)}px;
    font-weight: 700;
}}

QLabel#welcomeSub {{
    color: #88a096;
    font-size: {_px(14, scale)}px;
}}

QFrame#composer {{
    background: {ELEVATED};
    border: none;
    border-radius: {_px(24, scale)}px;
}}

QFrame#composer[focused="true"] {{
    border: 1px solid rgba(154, 214, 189, 0.16);
}}

QPushButton#composerIcon {{
    min-width: {_px(38, scale)}px;
    max-width: {_px(38, scale)}px;
    min-height: {_px(38, scale)}px;
    max-height: {_px(38, scale)}px;
    padding: 0;
    background: transparent;
    border: none;
    border-radius: {_px(19, scale)}px;
}}

QPushButton#composerIcon:hover {{
    background: rgba(154, 214, 189, 0.14);
}}

QPushButton#composerIcon:pressed {{
    background: rgba(154, 214, 189, 0.22);
}}

QPushButton#composerSend {{
    min-width: {_px(40, scale)}px;
    max-width: {_px(40, scale)}px;
    min-height: {_px(40, scale)}px;
    max-height: {_px(40, scale)}px;
    padding: 0;
    color: #0d1411;
    background: {ACCENT};
    border: none;
    border-radius: {_px(20, scale)}px;
}}

QPushButton#composerSend:hover {{
    background: #abe2cb;
}}

QPushButton#composerSend:pressed {{
    background: #84c4ad;
}}

QPushButton#composerSend:disabled {{
    color: rgba(13, 20, 17, 0.45);
    background: rgba(154, 214, 189, 0.22);
}}

QFrame#attachBar {{
    background: rgba(154, 214, 189, 0.07);
    border: 1px solid rgba(154, 214, 189, 0.16);
    border-radius: {_px(10, scale)}px;
}}

QLabel#attachChip {{
    color: #b9c6d2;
    font-size: {_px(12, scale)}px;
}}

QPushButton#attachRemove {{
    color: #8fa0ad;
    background: transparent;
    border: none;
    border-radius: {_px(6, scale)}px;
    font-size: {_px(13, scale)}px;
    font-weight: 800;
}}

QPushButton#attachRemove:hover {{
    color: #f06b6b;
    background: rgba(246, 102, 102, 0.12);
}}

QTextEdit#prompt {{
    background: transparent;
    border: none;
    color: {TEXT};
    padding: {_px(8, scale)}px {_px(6, scale)}px;
    font-family: {FONT_STACK};
    font-size: {_px(15, scale)}px;
    selection-background-color: #263139;
}}

QPushButton {{
    min-height: {_px(30, scale)}px;
    border-radius: {_px(9, scale)}px;
    padding: 0 {_px(11, scale)}px;
    font-size: {_px(12, scale)}px;
    font-weight: 760;
}}

QPushButton#askButton {{
    min-width: {_px(44, scale)}px;
    max-width: {_px(44, scale)}px;
    min-height: {_px(44, scale)}px;
    max-height: {_px(44, scale)}px;
    padding: 0;
    color: #10131a;
    background: {ACCENT};
    border: 1px solid rgba(154, 214, 189, 0.30);
    border-radius: {_px(13, scale)}px;
}}

QPushButton#askButton:disabled {{
    color: rgba(201, 237, 244, 0.4);
    background: rgba(154, 214, 189, 0.18);
}}

QPushButton#ghostButton {{
    color: {TEXT};
    background: rgba(31, 38, 54, 126);
    border: 1px solid rgba(154, 214, 189, 0.11);
}}

QPushButton#ghostButton:hover {{
    border: 1px solid rgba(154, 214, 189, 0.22);
}}

QDialog#settingsDialog {{
    background: #101722;
    color: {TEXT};
}}

QLabel#dialogTitle {{
    color: #ffffff;
    font-size: {_px(18, scale)}px;
    font-weight: 800;
}}

QLabel#dialogNote {{
    color: {MUTED};
    font-size: {_px(12, scale)}px;
}}

QPushButton#dialogPrimaryButton {{
    min-width: {_px(92, scale)}px;
    min-height: {_px(32, scale)}px;
    color: #07130d;
    background: {ACCENT};
    border: 0;
    border-radius: {_px(8, scale)}px;
    font-weight: 760;
}}

QPushButton#iconButton {{
    min-width: {_px(34, scale)}px;
    max-width: {_px(34, scale)}px;
    min-height: {_px(34, scale)}px;
    max-height: {_px(34, scale)}px;
    padding: 0;
    color: {TEXT};
    background: rgba(20, 28, 34, 0.55);
    border: 1px solid rgba(154, 214, 189, 0.10);
    border-radius: {_px(11, scale)}px;
}}

QPushButton#iconButton:hover {{
    background: rgba(37, 53, 60, 0.72);
    border: 1px solid rgba(154, 214, 189, 0.28);
}}

QPushButton#iconButton:checked {{
    background: rgba(154, 214, 189, 0.16);
    border: 1px solid rgba(154, 214, 189, 0.34);
}}

QFrame#actionPopup {{
    background: rgba(18, 24, 38, 245);
    border: 1px solid rgba(154, 214, 189, 0.18);
    border-radius: {_px(10, scale)}px;
}}

QPushButton#popupButton {{
    min-width: {_px(190, scale)}px;
    min-height: {_px(30, scale)}px;
    padding: 0 {_px(10, scale)}px;
    color: {TEXT};
    background: transparent;
    border: 1px solid transparent;
    border-radius: {_px(7, scale)}px;
    text-align: left;
}}

QPushButton#popupButton:hover {{
    background: rgba(154, 214, 189, 0.14);
    border: 1px solid rgba(154, 214, 189, 0.22);
}}

QPushButton#closeButton {{
    min-width: {_px(34, scale)}px;
    max-width: {_px(34, scale)}px;
    min-height: {_px(34, scale)}px;
    max-height: {_px(34, scale)}px;
    padding: 0;
    color: #6f8793;
    background: rgba(20, 22, 27, 0.68);
    border: 1px solid rgba(154, 214, 189, 0.08);
    border-radius: {_px(11, scale)}px;
}}

QPushButton#closeButton:hover {{
    background: rgba(246, 102, 102, 0.16);
    border: 1px solid rgba(246, 102, 102, 0.30);
}}

QComboBox#deviceCombo {{
    min-width: {_px(220, scale)}px;
    max-width: {_px(280, scale)}px;
    min-height: {_px(28, scale)}px;
    border-radius: {_px(9, scale)}px;
    padding: 0 {_px(10, scale)}px;
    color: #a9cdbd;
    background: rgba(5, 7, 10, 0.44);
    border: 1px solid rgba(154, 214, 189, 0.11);
}}

QComboBox#settingsCombo {{
    min-width: {_px(280, scale)}px;
    min-height: {_px(32, scale)}px;
    border-radius: {_px(8, scale)}px;
    padding: 0 {_px(10, scale)}px;
    color: {TEXT};
    background: rgba(31, 38, 54, 176);
    border: 1px solid rgba(154, 214, 189, 0.16);
}}

QComboBox QAbstractItemView {{
    color: {TEXT};
    background: #111820;
    selection-background-color: #263139;
}}

QLineEdit#settingsCombo {{
    min-width: {_px(280, scale)}px;
    min-height: {_px(32, scale)}px;
    border-radius: {_px(8, scale)}px;
    padding: 0 {_px(10, scale)}px;
    color: {TEXT};
    background: rgba(31, 38, 54, 176);
    border: 1px solid rgba(154, 214, 189, 0.16);
}}

QLineEdit#settingsCombo:focus {{
    border: 1px solid rgba(154, 214, 189, 0.42);
}}

QLabel#dialogError {{
    color: #f08a8a;
    font-size: {_px(12, scale)}px;
}}

QCheckBox {{
    color: {TEXT};
    font-size: {_px(13, scale)}px;
    spacing: {_px(8, scale)}px;
}}

QCheckBox::indicator {{
    width: {_px(16, scale)}px;
    height: {_px(16, scale)}px;
    border-radius: {_px(4, scale)}px;
    border: 1px solid rgba(154, 214, 189, 0.34);
    background: rgba(5, 7, 10, 0.44);
}}

QCheckBox::indicator:checked {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
}}

QTabWidget#settingsTabs::pane {{
    border: 1px solid rgba(154, 214, 189, 0.12);
    border-radius: {_px(10, scale)}px;
    top: -1px;
}}

QTabBar::tab {{
    color: {MUTED};
    background: transparent;
    padding: {_px(7, scale)}px {_px(14, scale)}px;
    margin-right: {_px(2, scale)}px;
    border-top-left-radius: {_px(8, scale)}px;
    border-top-right-radius: {_px(8, scale)}px;
}}

QTabBar::tab:selected {{
    color: #07130d;
    background: {ACCENT};
    font-weight: 700;
}}

QTabBar::tab:hover:!selected {{
    color: {TEXT};
    background: rgba(154, 214, 189, 0.12);
}}
"""


STYLES = build_window_styles()


def _load_app_fonts() -> None:
    """Register bundled fonts so the UI does not depend on system-installed faces."""
    for filename in ("SpaceGrotesk.ttf", "Manrope.ttf"):
        path = FONTS_DIR / filename
        if path.exists():
            if QFontDatabase.addApplicationFont(str(path)) == -1:
                LOGGER.warning("failed to load bundled font %s", path)
        else:
            LOGGER.warning("bundled font missing: %s", path)


def main() -> int:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    _load_app_fonts()
    app.setFont(QFont("Space Grotesk", 10))
    app.setWindowIcon(make_icon("mark", 32, ACCENT))
    window = OverlayWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
