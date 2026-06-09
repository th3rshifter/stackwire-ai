import base64
import hashlib
import html
import json
import logging
import os
import queue
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
import shiboken6
from PySide6.QtCore import QBuffer, QEasingCurve, QEvent, QIODevice, QMimeData, QObject, QPoint, QRect, QSize, QPropertyAnimation, QThread, QTimer, QUrl, Qt, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QFont, QFontDatabase, QIcon, QKeyEvent, QKeySequence, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QShortcut, QTextCursor, QTextOption, QWheelEvent
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
    QAbstractButton,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QProgressBar,
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
from app import chat_sessions  # noqa: E402

load_local_env()

LOGGER = logging.getLogger(__name__)

from app.llm import ANSWER_MODE, DEFAULT_STACKWIRE_MODEL, AskResult, ExpandResult, current_answer_model, current_vision_model  # noqa: E402
from app.modelhub import (  # noqa: E402
    MODELHUB_RECOMMENDED,
    ModelHubRefreshWorker,
    ModelPullWorker,
    ModelRecommendation,
    ModelTestWorker,
    dedupe_models as _dedupe_models,
    hardware_recommendation_note as _hardware_recommendation_note,
    hardware_summary as _hardware_summary,
    installed_ollama_models as _installed_ollama_models,
    is_vision_model as _is_vision_model,
    llm_provider as _llm_provider,
    ollama_base_url as _ollama_base_url,
    ollama_chat_url as _ollama_chat_url,
    openai_base_url as _openai_base_url,
)
from app.notes import notes_path  # noqa: E402
from app.question_recovery import STACKWIRE_MODE, current_recovery_model  # noqa: E402
from app.storage import create_session, log_feedback, save_good_answer  # noqa: E402
from app.tech_terms import WHISPER_TECHNICAL_PROMPT, normalize_spoken_technical_terms  # noqa: E402
from app.transcript_repair import clean_stt_output, collapse_repeated_phrases, condense_spoken_question, is_probable_stt_hallucination, repair_live_transcript  # noqa: E402
from app.ui.actions import MAIN_RAIL_ACTIONS, RailActionSpec  # noqa: E402
from app.ui.styles import build_window_styles as _build_window_styles  # noqa: E402
from app.widgets.chat import AssistantRow, ChatArea, NeuralBackground, configure_chat_widgets  # noqa: E402
from app.widgets.dialogs import ClickableImageLabel, FullImageDialog, NotesDialog, configure_dialog_widgets  # noqa: E402
from app.workers.chat import AskStreamWorker, ExpandWorker, ImageAnalysisWorker, ImageGenWorker, SuggestionsWorker, configure_chat_workers  # noqa: E402


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
ACCENT2 = "#8ab4f0"         # secondary cool вЂ" role, links, selection
CORAL = "#e8896b"           # warm вЂ" active recording indicator
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

FONT_STACK = '"Manrope", "Segoe UI", Arial, sans-serif'
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
    "details": "Р Р°СЃС€РёСЂРµРЅРёРµ: РџРѕРґСЂРѕР±РЅРµРµ",
    "components": "Р Р°СЃС€РёСЂРµРЅРёРµ: РЎ РєРѕРјРїРѕРЅРµРЅС‚Р°РјРё",
    "example": "Р Р°СЃС€РёСЂРµРЅРёРµ: РџСЂРёРјРµСЂ",
    "compare": "Р Р°СЃС€РёСЂРµРЅРёРµ: РЎСЂР°РІРЅРµРЅРёРµ",
    "troubleshoot": "Р Р°СЃС€РёСЂРµРЅРёРµ: Troubleshooting",
}

EXPAND_MENU_ITEMS: tuple[tuple[str, str], ...] = (
    ("details", "РџРѕРґСЂРѕР±РЅРµРµ"),
    ("components", "РЎ РєРѕРјРїРѕРЅРµРЅС‚Р°РјРё"),
    ("example", "РЎ РїСЂРёРјРµСЂРѕРј РєРѕРґР°/РєРѕРЅС„РёРіР°"),
    ("compare", "РЎСЂР°РІРЅРёС‚СЊ СЃ Р°РЅР°Р»РѕРіР°РјРё"),
    ("troubleshoot", "Troubleshooting"),
)

# Slash commands typed into the composer. (command, short hint shown in the popup).
SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/image", "Сгенерировать изображение по описанию"),
    ("/clear", "Очистить текущий чат"),
    ("/explain", "Подробно объяснить тему"),
    ("/code", "Показать рабочий пример кода"),
    ("/translate", "Перевести текст"),
)

LIGHTWEIGHT_STT_CORRECTIONS: tuple[tuple[str, str], ...] = (
    (r"\bРґРµРІ\s+Рё\s+РїСЂРѕРє\b", "/dev Рё /proc"),
    (r"\bРґРµРІ\b", "/dev"),
    (r"\bРїСЂРѕРє\b", "/proc"),
    (r"\bРµС‚СЃ\b", "/etc"),
    (r"\bРІР°СЂ\s+Р»РѕРі\b", "/var/log"),
)

LIVE_FILLER_WORDS = (
    "РѕРєРµР№",
    "С…Рј",
    "С…РјРј",
    "РјРј",
    "РјРјРј",
    "Рј",
    "Р°Р°",
    "Р°Р°Р°",
    "СЌСЌ",
    "СЌСЌСЌ",
    "Р»Р°РґРЅРѕ",
    "СЃР»СѓС€Р°Р№",
    "СЃРјРѕС‚СЂРё",
    "Р·РЅР°С‡РёС‚",
    "РєРѕСЂРѕС‡Рµ",
    "С‚РёРїР°",
    "РЅСѓ",
    "РІРѕС‚",
    "РІРѕРѕР±С‰Рµ",
    "РїРѕР¶Р°Р»СѓР№СЃС‚Р°",
    "РґР°РІР°Р№",
    "РґР°РІР°Р№С‚Рµ",
)


def _model_choices() -> list[str]:
    # Selection dropdowns must reflect what can actually run now. Recommendations
    # live in ModelHub cards, not in the active model selectors.
    return _dedupe_models(_installed_ollama_models())


def _save_local_env_values(values: dict[str, str]) -> None:
    existing = LOCAL_ENV_FILE.read_text(encoding="utf-8-sig").splitlines() if LOCAL_ENV_FILE.exists() else []
    remaining = {key: value.strip() for key, value in values.items()}
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
    "РІ РѕР±С‰РµРј",
    "РЅР° СЃР°РјРѕРј РґРµР»Рµ",
    "РјРѕР¶РµС€СЊ СЂР°СЃСЃРєР°Р·Р°С‚СЊ",
    "РјРѕР¶РµС€СЊ РѕР±СЉСЏСЃРЅРёС‚СЊ",
    "СЂР°СЃСЃРєР°Р¶Рё РїРѕР¶Р°Р»СѓР№СЃС‚Р°",
    "РѕР±СЉСЏСЃРЅРё РїРѕР¶Р°Р»СѓР№СЃС‚Р°",
)

LIVE_TRAILING_NOISE = (
    "Р·РЅР°РµС€СЊ",
    "Р·РЅР°РµС€СЊ РЅРµС‚",
    "РїРѕРЅРёРјР°РµС€СЊ",
    "РґР°",
    "РЅРµС‚",
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
    cleaned = re.sub(r"\b(?:Р°|Рё)\s+(?=(С‡С‚Рѕ|РєР°Рє|С‡РµРј|РєРѕРіРґР°|РїРѕС‡РµРјСѓ|Р·Р°С‡РµРј|СЂР°СЃСЃРєР°Р¶Рё|РѕР±СЉСЏСЃРЅРё)\b)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:СЂР°СЃСЃРєР°Р¶Рё|СЂР°СЃСЃРєР°Р·Р°С‚СЊ|РѕР±СЉСЏСЃРЅРё|РѕР±СЉСЏСЃРЅРёС‚СЊ)\s+(?=С‡С‚Рѕ С‚Р°РєРѕРµ\b)", "", cleaned, flags=re.IGNORECASE)
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
    parts = [part.strip(" ,.?") for part in re.split(r"\bС‡С‚Рѕ С‚Р°РєРѕРµ\b", text, flags=re.IGNORECASE)]
    if len(parts) <= 2:
        return text

    terms: list[str] = []
    for part in parts[1:]:
        part = re.sub(r"\b(?:Р·РЅР°РµС€СЊ|РЅРµС‚|РґР°|РїРѕР¶Р°Р»СѓР№СЃС‚Р°)\b", "", part, flags=re.IGNORECASE)
        part = re.sub(r"\s+", " ", part).strip(" ,.?")
        if part:
            terms.append(part)

    if not terms:
        return text
    if len(terms) == 1:
        return f"С‡С‚Рѕ С‚Р°РєРѕРµ {terms[0]}"
    return "С‡С‚Рѕ С‚Р°РєРѕРµ " + ", ".join(terms[:-1]) + " Рё " + terms[-1]


QUESTION_MARKERS = (
    "С‡С‚Рѕ",
    "РєР°Рє",
    "РїРѕС‡РµРјСѓ",
    "Р·Р°С‡РµРј",
    "РєРѕРіРґР°",
    "РіРґРµ",
    "С‡РµРј",
    "РєР°РєРѕР№",
    "РєР°РєР°СЏ",
    "РєР°РєРёРµ",
    "РѕР±СЉСЏСЃРЅРё",
    "РѕР±СЉСЏСЃРЅРёС‚СЊ",
    "СЂР°СЃСЃРєР°Р¶Рё",
    "СЂР°СЃСЃРєР°Р·Р°С‚СЊ",
    "РѕРїРёС€Рё",
    "СЃСЂР°РІРЅРё",
    "СЂР°Р·РЅРёС†Р°",
    "РѕС‚Р»РёС‡Р°РµС‚СЃСЏ",
    "РґРёР°РіРЅРѕСЃС‚РёСЂРѕРІР°С‚СЊ",
    "РїРѕС‡РёРЅРёС‚СЊ",
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
    elif kind == "diff":
        painter.drawRoundedRect(int(s * 0.18), int(s * 0.18), int(s * 0.28), int(s * 0.58), int(s * 0.05), int(s * 0.05))
        painter.drawRoundedRect(int(s * 0.54), int(s * 0.24), int(s * 0.28), int(s * 0.58), int(s * 0.05), int(s * 0.05))
        painter.drawLine(int(s * 0.26), int(s * 0.36), int(s * 0.38), int(s * 0.36))
        painter.drawLine(int(s * 0.26), int(s * 0.50), int(s * 0.38), int(s * 0.50))
        painter.drawLine(int(s * 0.62), int(s * 0.42), int(s * 0.74), int(s * 0.42))
        painter.drawLine(int(s * 0.62), int(s * 0.56), int(s * 0.74), int(s * 0.56))
    elif kind == "research":
        painter.drawEllipse(int(s * 0.20), int(s * 0.20), int(s * 0.44), int(s * 0.44))
        painter.drawLine(int(s * 0.56), int(s * 0.56), int(s * 0.80), int(s * 0.80))
        painter.drawLine(int(s * 0.30), int(s * 0.42), int(s * 0.54), int(s * 0.42))
        painter.drawLine(int(s * 0.42), int(s * 0.30), int(s * 0.42), int(s * 0.54))
    elif kind == "search":
        painter.drawEllipse(int(s * 0.20), int(s * 0.20), int(s * 0.44), int(s * 0.44))
        painter.drawLine(int(s * 0.56), int(s * 0.56), int(s * 0.80), int(s * 0.80))
        painter.drawLine(int(s * 0.32), int(s * 0.42), int(s * 0.52), int(s * 0.42))
        painter.drawLine(int(s * 0.42), int(s * 0.32), int(s * 0.42), int(s * 0.52))
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
    elif kind == "menu":
        painter.drawLine(int(s * 0.22), int(s * 0.30), int(s * 0.78), int(s * 0.30))
        painter.drawLine(int(s * 0.22), int(s * 0.50), int(s * 0.78), int(s * 0.50))
        painter.drawLine(int(s * 0.22), int(s * 0.70), int(s * 0.78), int(s * 0.70))
    elif kind == "collapse":
        painter.drawLine(int(s * 0.26), int(s * 0.50), int(s * 0.76), int(s * 0.50))
        painter.drawLine(int(s * 0.26), int(s * 0.50), int(s * 0.48), int(s * 0.30))
        painter.drawLine(int(s * 0.26), int(s * 0.50), int(s * 0.48), int(s * 0.70))
    elif kind == "chats":
        painter.drawRoundedRect(int(s * 0.18), int(s * 0.22), int(s * 0.60), int(s * 0.44), int(s * 0.09), int(s * 0.09))
        painter.drawLine(int(s * 0.34), int(s * 0.66), int(s * 0.24), int(s * 0.82))
        painter.drawLine(int(s * 0.34), int(s * 0.66), int(s * 0.48), int(s * 0.66))
    elif kind == "notes":
        painter.drawRoundedRect(int(s * 0.24), int(s * 0.16), int(s * 0.52), int(s * 0.68), int(s * 0.06), int(s * 0.06))
        painter.drawLine(int(s * 0.34), int(s * 0.16), int(s * 0.34), int(s * 0.84))
        painter.drawLine(int(s * 0.42), int(s * 0.36), int(s * 0.66), int(s * 0.36))
        painter.drawLine(int(s * 0.42), int(s * 0.50), int(s * 0.66), int(s * 0.50))
        painter.drawLine(int(s * 0.42), int(s * 0.64), int(s * 0.58), int(s * 0.64))
    elif kind == "plus":
        painter.drawLine(int(s * 0.50), int(s * 0.24), int(s * 0.50), int(s * 0.76))
        painter.drawLine(int(s * 0.24), int(s * 0.50), int(s * 0.76), int(s * 0.50))
    elif kind == "trash":
        painter.drawLine(int(s * 0.28), int(s * 0.30), int(s * 0.72), int(s * 0.30))
        painter.drawLine(int(s * 0.38), int(s * 0.22), int(s * 0.62), int(s * 0.22))
        painter.drawRoundedRect(int(s * 0.34), int(s * 0.36), int(s * 0.32), int(s * 0.42), int(s * 0.05), int(s * 0.05))
        painter.drawLine(int(s * 0.44), int(s * 0.44), int(s * 0.44), int(s * 0.70))
        painter.drawLine(int(s * 0.56), int(s * 0.44), int(s * 0.56), int(s * 0.70))
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
    elif kind == "live":
        # Broadcast / radar: filled centre dot + two concentric arcs
        painter.setBrush(QColor(color))
        painter.drawEllipse(int(s * 0.41), int(s * 0.41), int(s * 0.18), int(s * 0.18))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(int(s * 0.29), int(s * 0.29), int(s * 0.42), int(s * 0.42))
        painter.drawEllipse(int(s * 0.16), int(s * 0.16), int(s * 0.68), int(s * 0.68))
    elif kind == "image":
        # Picture frame with mountains + sun
        painter.drawRoundedRect(int(s * 0.16), int(s * 0.16), int(s * 0.68), int(s * 0.68), int(s * 0.08), int(s * 0.08))
        # small sun circle top-right
        painter.drawEllipse(int(s * 0.60), int(s * 0.24), int(s * 0.14), int(s * 0.14))
        # mountain peak left
        painter.drawLine(int(s * 0.22), int(s * 0.72), int(s * 0.44), int(s * 0.46))
        painter.drawLine(int(s * 0.44), int(s * 0.46), int(s * 0.62), int(s * 0.66))
        # mountain peak right
        painter.drawLine(int(s * 0.54), int(s * 0.56), int(s * 0.72), int(s * 0.72))
    elif kind == "vision":
        painter.drawArc(int(s * 0.18), int(s * 0.30), int(s * 0.64), int(s * 0.40), 25 * 16, 130 * 16)
        painter.drawArc(int(s * 0.18), int(s * 0.30), int(s * 0.64), int(s * 0.40), 205 * 16, 130 * 16)
        painter.drawEllipse(int(s * 0.42), int(s * 0.42), int(s * 0.16), int(s * 0.16))
    elif kind == "stop_gen":
        # Filled rounded square — stop/interrupt generation
        painter.setBrush(QColor(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(int(s * 0.28), int(s * 0.28), int(s * 0.44), int(s * 0.44), int(s * 0.08), int(s * 0.08))
    elif kind == "regen":
        # Circular refresh arrow — regenerate / new variant
        rect = QRect(int(s * 0.24), int(s * 0.24), int(s * 0.52), int(s * 0.52))
        painter.drawArc(rect, 60 * 16, 280 * 16)
        # arrowhead at the arc's open end (top-right, ~60°)
        ax, ay = int(s * 0.70), int(s * 0.30)
        painter.drawLine(ax, ay, int(s * 0.70), int(s * 0.46))
        painter.drawLine(ax, ay, int(s * 0.54), int(s * 0.32))
    elif kind == "mini":
        # Picture-in-picture: outer frame + small filled inner box (compact chat)
        painter.drawRoundedRect(int(s * 0.18), int(s * 0.22), int(s * 0.64), int(s * 0.50), int(s * 0.06), int(s * 0.06))
        painter.setBrush(QColor(color))
        painter.drawRoundedRect(int(s * 0.48), int(s * 0.44), int(s * 0.30), int(s * 0.24), int(s * 0.04), int(s * 0.04))
    elif kind == "mini_exit":
        # Expand arrows pointing outward (leave mini mode)
        painter.drawLine(int(s * 0.30), int(s * 0.30), int(s * 0.46), int(s * 0.46))
        painter.drawLine(int(s * 0.30), int(s * 0.30), int(s * 0.30), int(s * 0.44))
        painter.drawLine(int(s * 0.30), int(s * 0.30), int(s * 0.44), int(s * 0.30))
        painter.drawLine(int(s * 0.70), int(s * 0.70), int(s * 0.54), int(s * 0.54))
        painter.drawLine(int(s * 0.70), int(s * 0.70), int(s * 0.70), int(s * 0.56))
        painter.drawLine(int(s * 0.70), int(s * 0.70), int(s * 0.56), int(s * 0.70))

    painter.end()
    return pixmap


def make_icon(kind: str, size: int, color: str = TEXT) -> QIcon:
    return QIcon(icon_pixmap(kind, size, color))


def pixmap_to_base64_png(pixmap: QPixmap) -> str:
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    pixmap.save(buffer, "PNG")
    return base64.b64encode(bytes(buffer.data().data())).decode("ascii")


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
    inside_open_fence = len(segments) % 2 == 0  # an odd number of ``` в‡’ inside a block
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
.code-foot-cell {{
  padding: {_px(6)}px {_px(11)}px {_px(8)}px;
  background: #14171d;
  text-align: right;
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
.code-diagram {{
  font-size: {_px(12)}px;
  line-height: 1.35;
  white-space: pre;
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
.md-table {{
  border-collapse: collapse;
  width: 100%;
  margin: {_px(10)}px 0 {_px(14)}px;
  font-size: {_px(14)}px;
}}
.md-table th {{
  background: rgba(154, 214, 189, 0.10);
  color: #e4e9f0;
  font-weight: 700;
  padding: {_px(6)}px {_px(10)}px;
  border: 1px solid rgba(154, 214, 189, 0.14);
  text-align: left;
}}
.md-table td {{
  padding: {_px(5)}px {_px(10)}px;
  border: 1px solid rgba(154, 214, 189, 0.09);
  color: #c7d1db;
  vertical-align: top;
}}
.md-table tr:nth-child(even) td {{
  background: rgba(255, 255, 255, 0.025);
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
CODE_BLOCK_KEYS: list[str] = []
EXPANDED_CODE_BLOCKS: set[str] = set()
CODE_PREVIEW_LINES = 14
# Generated images cache: id → base64 PNG string (cleared on each render_chat).
_GENERATED_IMAGES: dict[int, str] = {}


def _document_css() -> str:
    """All chat CSS as a raw stylesheet (no <style> tags) for setDefaultStyleSheet."""
    raw = f"{build_html_style()}{build_chat_style()}"
    return raw.replace("<style>", "").replace("</style>", "")

# Diagram rendering is skipped while a message is streaming (incomplete source) and
# enabled for the final render, so we never shell out to a renderer per token.
_DIAGRAM_RENDER = {"enabled": True}


def _code_block_key(language: str, raw_code: str) -> str:
    payload = f"{language.strip().lower()}\0{raw_code}".encode("utf-8", errors="replace")
    return hashlib.sha1(payload).hexdigest()


def _looks_like_ascii_diagram(raw_code: str) -> bool:
    lines = [line for line in raw_code.splitlines() if line.strip()]
    if len(lines) < 6:
        return False
    diagram_chars = sum(raw_code.count(char) for char in "+-|<>^v")
    box_lines = sum(1 for line in lines if re.search(r"[+|][-+| ]{4,}", line))
    return diagram_chars >= 24 and box_lines >= 2


def _is_collapsible_code_block(language: str, raw_code: str) -> bool:
    lines = raw_code.splitlines()
    return (
        len(lines) > CODE_PREVIEW_LINES + 4
        or len(raw_code) > 1600
        or _looks_like_ascii_diagram(raw_code)
        or language.strip().lower() in {"text", "txt"} and len(lines) > 10
    )


def _code_preview(raw_code: str) -> str:
    lines = raw_code.splitlines()
    if len(lines) <= CODE_PREVIEW_LINES:
        return raw_code
    return "\n".join(lines[:CODE_PREVIEW_LINES])


def _code_action_link(kind: str, href: str, *, size: int = 13, color: str = "#8b97a6") -> str:
    icon = pixmap_to_base64_png(icon_pixmap(kind, _px(size), color))
    px_size = _px(size)
    return f"<a href='{href}'><img src='data:image/png;base64,{icon}' width='{px_size}' height='{px_size}' /></a>"


def _code_block_html(raw_language: str, raw_code: str) -> str:
    language = html.escape(raw_language or "code")
    block_key = _code_block_key(raw_language, raw_code)
    collapsible = _is_collapsible_code_block(raw_language, raw_code)
    expanded = not collapsible or block_key in EXPANDED_CODE_BLOCKS
    display_code = raw_code if expanded else _code_preview(raw_code)
    code = highlight_code(raw_language, display_code)

    snippet_id = len(CODE_SNIPPETS)
    CODE_SNIPPETS.append(raw_code)
    CODE_BLOCK_KEYS.append(block_key)

    copy_link = _code_action_link("copy", f"copycode:{snippet_id}")
    footer = ""
    pre_class = "code-diagram" if _looks_like_ascii_diagram(raw_code) else ""
    if collapsible:
        icon_kind = "collapse" if expanded else "expand"
        toggle_link = _code_action_link(icon_kind, f"togglecode:{snippet_id}", size=15)
        footer = f'<tr><td class="code-foot-cell">{toggle_link}</td></tr>'

    return (
        f'<table class="code-card" width="100%" cellspacing="0" cellpadding="0">'
        f'<tr><td class="code-head-cell">'
        f'<table width="100%" cellspacing="0" cellpadding="0"><tr>'
        f'<td class="code-lang">{language}</td>'
        f'<td align="right">{copy_link}</td>'
        f"</tr></table>"
        f"</td></tr>"
        f'<tr><td class="code-body"><pre class="{pre_class}">{code}</pre></td></tr>'
        f"{footer}"
        f"</table>"
    )


def markdown_to_html(markdown: str) -> str:
    markdown = normalize_unfenced_code_blocks(markdown)
    parts: list[str] = []
    # Combined pattern: code blocks OR generated_image tags
    pattern = re.compile(
        r"```([a-zA-Z0-9_.+-]*)\n(.*?)```"
        r"|\[\[generated_image:([A-Za-z0-9+/=]+)\]\]",
        re.DOTALL,
    )
    cursor = 0

    for match in pattern.finditer(markdown):
        parts.append(text_to_html(markdown[cursor : match.start()]))

        if match.group(3) is not None:
            # Generated image: render as clickable thumbnail
            b64 = match.group(3)
            gen_id = len(_GENERATED_IMAGES)
            _GENERATED_IMAGES[gen_id] = b64
            img_w = _px(340)
            safe_src = html.escape(b64, quote=True)
            parts.append(
                f"<p><a href='viewimage:{gen_id}'>"
                f"<img src='data:image/png;base64,{safe_src}' width='{img_w}' /></a></p>"
            )
        else:
            raw_language = match.group(1) or "code"
            raw_code = match.group(2).strip("\n")

            if _DIAGRAM_RENDER["enabled"] and is_diagram_language(raw_language):
                diagram_png = render_diagram(raw_language, raw_code)
                if diagram_png:
                    parts.append(
                        f'<div class="diagram-card"><img src="data:image/png;base64,{diagram_png}" /></div>'
                    )
                    cursor = match.end()
                    continue

            parts.append(_code_block_html(raw_language, raw_code))
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
    "РєРѕСЂРѕС‚РєРѕ:",
    "РєР°Рє СЂР°Р±РѕС‚Р°РµС‚:",
    "РїСЂР°РєС‚РёРєР°:",
    "РїСЂРёРјРµСЂ:",
    "РЅСЋР°РЅСЃ:",
    "best practices:",
    "РѕСЃРЅРѕРІРЅРѕР№ РѕС‚РІРµС‚:",
    "РїРѕРґСЂРѕР±РЅС‹Р№ РѕС‚РІРµС‚:",
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
    return bool(re.search(r"[Р°-СЏРђ-РЇ]", text)) and not bool(re.search(r"[{}:=#;/\\[\\]$]", text))


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
            "<div class='thinking-label'>Thinking</div>"
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
        # paragraph вЂ" no table/card wrapper, so Qt does not add big block spacing
        # around it. A modest fixed width keeps it a neat preview.
        img_w = _px(240)
        fragments.append(
            f"<p class='shot'><img src='data:image/png;base64,{safe_src}' width='{img_w}' /></p>"
        )
    if text:
        fragments.append(markdown_fragment(text))
    return "".join(fragments)


def render_user_message_fragment(markdown: str) -> str:
    # Older sessions stored a "Р’РѕРїСЂРѕСЃ N" prefix; strip it so no label/stripe is shown.
    match = re.match(r"^Р’РѕРїСЂРѕСЃ\s+\d+\s*\n\n(.+)$", markdown.strip(), flags=re.DOTALL)
    body = match.group(1).strip() if match else markdown
    return render_message_fragment(body)


def _table_lines_to_html(raw_lines: list[str]) -> str:
    """Convert a Markdown table block (already HTML-escaped) to an HTML table."""
    if len(raw_lines) < 2:
        return "".join(f"<p>{line}</p>" for line in raw_lines)

    def split_row(line: str) -> list[str]:
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]
        return [cell.strip() for cell in stripped.split("|")]

    # Second row must look like a separator (---, :---, etc.)
    sep = raw_lines[1].strip()
    if not re.fullmatch(r"[\s|:\-]+", sep):
        return "".join(f"<p>{line}</p>" for line in raw_lines)

    headers = split_row(raw_lines[0])
    th = "".join(f"<th>{_inline_format(c)}</th>" for c in headers)
    rows_html = f"<thead><tr>{th}</tr></thead>"

    body_rows = []
    for line in raw_lines[2:]:
        cells = split_row(line)
        # Pad or trim to match header count
        while len(cells) < len(headers):
            cells.append("")
        cells = cells[: len(headers)]
        td = "".join(f"<td>{_inline_format(c)}</td>" for c in cells)
        body_rows.append(f"<tr>{td}</tr>")
    if body_rows:
        rows_html += f"<tbody>{''.join(body_rows)}</tbody>"

    return f'<table class="md-table">{rows_html}</table>'


def _inline_format(text: str) -> str:
    """Apply inline markdown: backtick code, bold, links."""
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(
        r"(https?://[^\s<]+)",
        lambda m: f'<a href="{m.group(1)}" style="color:{BLUE};text-decoration:none">{m.group(1)}</a>',
        text,
    )
    return text


def text_to_html(text: str) -> str:
    escaped = html.escape(text.strip())
    if not escaped:
        return ""

    lines = escaped.splitlines()
    out: list[str] = []
    in_list = False
    in_table = False
    table_lines: list[str] = []

    def flush_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    def flush_table() -> None:
        nonlocal in_table, table_lines
        if in_table:
            out.append(_table_lines_to_html(table_lines))
            table_lines = []
            in_table = False

    for line in lines:
        stripped = line.strip()

        # Blank line — flush open blocks
        if not stripped:
            flush_list()
            flush_table()
            continue

        # Table row detection: starts with | and has at least one more |
        if stripped.startswith("|") and stripped.count("|") >= 2:
            flush_list()
            in_table = True
            table_lines.append(stripped)
            continue

        # Non-table line while in table → flush table first
        if in_table:
            flush_table()

        formatted = _inline_format(stripped)

        if stripped.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline_format(stripped[2:])}</li>")
            continue

        numbered = re.match(r"^\d+\.\s+(.+)$", stripped)
        if numbered:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline_format(numbered.group(1))}</li>")
            continue

        flush_list()

        if stripped.startswith("## "):
            out.append(f"<h2>{_inline_format(stripped[3:])}</h2>")
        elif stripped.startswith("# "):
            out.append(f"<h2>{_inline_format(stripped[2:])}</h2>")
        elif stripped.endswith(":") and len(stripped) < 80 and "|" not in stripped:
            out.append(f"<h2>{formatted}</h2>")
        else:
            out.append(f"<p>{formatted}</p>")

    flush_list()
    flush_table()

    return "".join(out)


class PromptEdit(QTextEdit):
    submitted = Signal()
    image_pasted = Signal(str, str)   # emits (base64 PNG, filename)

    slash_accepted = Signal(str)   # keyboard-accepted slash command, e.g. "/image"

    def __init__(self) -> None:
        super().__init__()
        self._min_height = _px(40)
        self._max_height = _px(132)
        self.slash_popup: "SlashPopup | None" = None
        self.setAcceptRichText(False)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.document().setDocumentMargin(0)
        self.textChanged.connect(self.sync_height)
        self.keep_arrow_cursor()
        self.sync_height()

    def set_height_limits(self, minimum: int, maximum: int) -> None:
        self._min_height = minimum
        self._max_height = max(minimum, maximum)
        self.sync_height()

    def sync_height(self) -> None:
        self.document().setTextWidth(max(1, self.viewport().width()))
        doc_height = int(self.document().documentLayout().documentSize().height())
        wanted = max(self._min_height, min(self._max_height, doc_height + _px(18)))
        if self.height() != wanted:
            self.setFixedHeight(wanted)
        policy = Qt.ScrollBarPolicy.ScrollBarAsNeeded if wanted >= self._max_height else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        if self.verticalScrollBarPolicy() != policy:
            self.setVerticalScrollBarPolicy(policy)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        QTimer.singleShot(0, self.sync_height)

    def keep_arrow_cursor(self) -> None:
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.viewport().setCursor(Qt.CursorShape.ArrowCursor)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        # While the slash-command popup is open, arrows/Tab/Enter drive it.
        popup = self.slash_popup
        if popup is not None and popup.isVisible():
            key = event.key()
            if key == Qt.Key.Key_Down:
                popup.move_active(1); event.accept(); return
            if key == Qt.Key.Key_Up:
                popup.move_active(-1); event.accept(); return
            if key == Qt.Key.Key_Escape:
                popup.hide(); event.accept(); return
            if key in (Qt.Key.Key_Tab, Qt.Key.Key_Return, Qt.Key.Key_Enter):
                cmd = popup.current_command()
                if cmd:
                    self.slash_accepted.emit(cmd)
                    event.accept(); return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.submitted.emit()
            return
        super().keyPressEvent(event)

    def insertFromMimeData(self, source: QMimeData) -> None:
        """Intercept Ctrl+V: if clipboard has an image, emit image_pasted instead of inserting."""
        from PySide6.QtGui import QImage
        # Check file URLs FIRST — Windows copies a file to clipboard with both a URL and image
        # data; checking URLs first lets us preserve the original filename.
        if source.hasUrls():
            for url in source.urls():
                path = url.toLocalFile()
                if path and Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
                    try:
                        raw = Path(path).read_bytes()
                        b64 = base64.b64encode(raw).decode("ascii")
                        self.image_pasted.emit(b64, Path(path).name)
                        return
                    except Exception:
                        pass
        # Fall back to raw image data (screenshot from Win+Shift+S, PrtSc, etc.)
        img = source.imageData()
        if isinstance(img, QImage) and not img.isNull():
            b64 = pixmap_to_base64_png(QPixmap.fromImage(img))
            self.image_pasted.emit(b64, "clipboard.png")
            return
        super().insertFromMimeData(source)

    def enterEvent(self, event) -> None:  # noqa: ANN001
        self.keep_arrow_cursor()
        super().enterEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        self.keep_arrow_cursor()
        super().mouseMoveEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.accept()


class SlashPopup(QFrame):
    """Suggestion popup that lists /slash commands while the user types '/'."""

    command_chosen = Signal(str)  # mouse-clicked command, e.g. "/image"

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("slashPopup")
        self.setVisible(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # never steal focus from the input
        lay = QVBoxLayout(self)
        lay.setContentsMargins(_px(6), _px(6), _px(6), _px(6))
        lay.setSpacing(_px(2))
        self._rows: list[QPushButton] = []
        for cmd, hint in SLASH_COMMANDS:
            btn = QPushButton(f"{cmd}   {hint}")
            btn.setObjectName("slashRow")
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("cmd", cmd)
            btn.clicked.connect(lambda _=False, c=cmd: self.command_chosen.emit(c))
            lay.addWidget(btn)
            self._rows.append(btn)
        self._visible_rows: list[QPushButton] = []
        self._active = 0

    def update_for(self, text: str) -> bool:
        """Show rows whose command starts with the typed token. Returns True if shown."""
        # Only while typing the command token itself (before any space/argument).
        if not text.startswith("/") or " " in text or "\n" in text:
            self.hide()
            return False
        prefix = text.lower()
        self._visible_rows = []
        for btn in self._rows:
            cmd = str(btn.property("cmd"))
            match = cmd.startswith(prefix)
            btn.setVisible(match)
            if match:
                self._visible_rows.append(btn)
        if not self._visible_rows:
            self.hide()
            return False
        self._active = 0
        self._refresh_highlight()
        self.adjustSize()
        self.setVisible(True)
        self.raise_()
        return True

    def _refresh_highlight(self) -> None:
        for i, btn in enumerate(self._visible_rows):
            btn.setProperty("active", i == self._active)
            style = btn.style()
            style.unpolish(btn)
            style.polish(btn)

    def move_active(self, delta: int) -> None:
        if not self._visible_rows:
            return
        self._active = (self._active + delta) % len(self._visible_rows)
        self._refresh_highlight()

    def current_command(self) -> str | None:
        if self._visible_rows and 0 <= self._active < len(self._visible_rows):
            return str(self._visible_rows[self._active].property("cmd"))
        return None


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

        self.setWindowTitle("Stackwire - sign in")
        self.setModal(True)
        self.setObjectName("settingsDialog")
        self.setMinimumWidth(_px(420))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(_px(22), _px(20), _px(22), _px(18))
        layout.setSpacing(_px(12))

        title = QLabel("Sign in to Stackwire")
        title.setObjectName("dialogTitle")
        self._subtitle = QLabel("Sign in to use chat.")
        self._subtitle.setObjectName("dialogNote")
        self._subtitle.setWordWrap(True)

        form = QFormLayout()
        form.setHorizontalSpacing(_px(12))
        form.setVerticalSpacing(_px(10))

        self.username_edit = QLineEdit()
        self.username_edit.setObjectName("settingsCombo")
        self.username_edit.setPlaceholderText("username")
        self.username_edit.setText(default_username)
        self.password_edit = QLineEdit()
        self.password_edit.setObjectName("settingsCombo")
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText("password")
        self.password_edit.returnPressed.connect(self._submit)

        server_label = self._auth_client.auth_base_url()
        self._server_hint = QLabel(f"Server: {server_label}")
        self._server_hint.setObjectName("dialogNote")
        self._server_hint.setWordWrap(True)

        form.addRow("Username", self.username_edit)
        form.addRow("Password", self.password_edit)

        self._error = QLabel("")
        self._error.setObjectName("dialogError")
        self._error.setWordWrap(True)
        self._error.setVisible(False)

        actions = QHBoxLayout()
        self._toggle_button = QPushButton("Create account")
        self._toggle_button.setObjectName("ghostButton")
        self._toggle_button.clicked.connect(self._toggle_mode)
        actions.addWidget(self._toggle_button)
        actions.addStretch(1)
        self._primary = QPushButton("Sign in")
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
            self._primary.setText("Register")
            self._toggle_button.setText("I already have an account")
            self._subtitle.setText("Create a local account. Password must be at least 6 characters.")
        else:
            self._primary.setText("Sign in")
            self._toggle_button.setText("Create account")
            self._subtitle.setText("Sign in to use chat.")
        self._error.setVisible(False)

    def _show_error(self, message: str) -> None:
        self._error.setText(message)
        self._error.setVisible(True)

    def _submit(self) -> None:
        username = self.username_edit.text().strip()
        password = self.password_edit.text()
        if not username or not password:
            self._show_error("Enter username and password.")
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
    """Detailed, tabbed settings вЂ" everything editable from the UI, no env hand-editing."""

    def __init__(self, parent: QWidget, models: list[str], audio_devices: list[str], current_audio_device: str) -> None:
        super().__init__(parent)
        # The parent is the main StackWire window, which exposes custom attributes
        # (authenticated, logout, prompt_login, ...) not on QWidget. Access is guarded
        # with hasattr/getattr at runtime; type as Any so the checker allows it.
        self._window: Any = parent
        self.setWindowTitle("Stackwire settings")
        self.setModal(True)
        self.setObjectName("settingsDialog")
        self.setMinimumWidth(_px(940))
        self.setMinimumHeight(_px(690))
        self._modelhub_threads: list[QThread] = []
        self._installed_models: set[str] = set(_installed_ollama_models())
        self._model_cards: dict[str, QFrame] = {}
        self._modelhub_pull_active = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(_px(18), _px(16), _px(18), _px(16))
        layout.setSpacing(_px(12))

        heading = QLabel("Settings")
        heading.setObjectName("dialogTitle")

        tabs = QTabWidget()
        tabs.setObjectName("settingsTabs")
        tabs.addTab(self._build_account_tab(), "Account")
        tabs.addTab(self._build_models_tab(models), "ModelHub")
        tabs.addTab(self._build_speech_tab(audio_devices, current_audio_device), "Speech")
        tabs.addTab(self._build_knowledge_tab(), "Knowledge")
        tabs.addTab(self._build_diagnostics_tab(), "Diagnostics")

        actions = QHBoxLayout()
        actions.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("ghostButton")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save")
        save.setObjectName("dialogPrimaryButton")
        save.clicked.connect(self.accept)
        actions.addWidget(cancel)
        actions.addWidget(save)

        layout.addWidget(heading)
        layout.addWidget(tabs)
        layout.addLayout(actions)
        QTimer.singleShot(0, self.refresh_modelhub)

    def done(self, result: int) -> None:
        if any(thread.isRunning() for thread in getattr(self, "_modelhub_threads", [])):
            action = "download" if getattr(self, "_modelhub_pull_active", False) else "check/test"
            self.modelhub_status.setText(f"ModelHub {action} is still running. Wait until it finishes before closing settings.")
            return
        super().done(result)

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

        status = "not required" if not required else (f"signed in as {username}" if authed else "not signed in")
        self._account_status = QLabel(status)
        self._account_status.setObjectName("dialogNote")
        form.addRow("Status", self._account_status)

        self.auth_url_edit = QLineEdit()
        self.auth_url_edit.setObjectName("settingsCombo")
        from app import auth_client

        self.auth_url_edit.setText(os.getenv("STACKWIRE_AUTH_URL", "").strip() or auth_client.auth_base_url())
        self.auth_url_edit.setPlaceholderText("http://127.0.0.1:8000")
        form.addRow("Auth server", self.auth_url_edit)

        buttons = QHBoxLayout()
        self._login_button = QPushButton("Sign out" if authed else "Sign in")
        self._login_button.setObjectName("ghostButton")
        self._login_button.clicked.connect(self._account_action)
        buttons.addWidget(self._login_button)
        buttons.addStretch(1)
        holder = QWidget()
        holder.setLayout(buttons)
        form.addRow("", holder)

        note = QLabel("Account state is managed here; the token is cached locally after sign in.")
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
        self._login_button.setText("Sign out" if authed else "Sign in")
        self._account_status.setText(f"signed in as {username}" if authed else "not signed in")

    def _build_models_tab(self, models: list[str]) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(_px(6), _px(10), _px(6), _px(6))
        layout.setSpacing(_px(12))

        model_box = QFrame()
        model_box.setObjectName("modelHubPanel")
        model_form = QFormLayout(model_box)
        model_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        model_form.setHorizontalSpacing(_px(12))
        model_form.setVerticalSpacing(_px(8))
        model_form.setContentsMargins(_px(12), _px(10), _px(12), _px(10))
        installed_models = _dedupe_models(models)
        installed_vision_models = [model for model in installed_models if _is_vision_model(model)]
        self.answer_model = self._model_combo(installed_models, current_answer_model())
        self.recovery_model = self._model_combo(installed_models, current_recovery_model())
        self.vision_model = self._model_combo(installed_vision_models or installed_models, current_vision_model())
        self.answer_mode = NoWheelComboBox()
        self.answer_mode.setObjectName("settingsCombo")
        self.answer_mode.addItems(["normal", "deep"])
        self.answer_mode.setCurrentText(os.getenv("ANSWER_MODE", os.getenv("STACKWIRE_ANSWER_MODE", "normal")).strip().lower() or "normal")
        model_form.addRow("Answer model", self.answer_model)
        model_form.addRow("Question recovery", self.recovery_model)
        model_form.addRow("Vision model", self.vision_model)
        model_form.addRow("Answer mode", self.answer_mode)

        provider_box = QFrame()
        provider_box.setObjectName("modelHubPanel")
        provider_form = QFormLayout(provider_box)
        provider_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        provider_form.setHorizontalSpacing(_px(12))
        provider_form.setVerticalSpacing(_px(8))
        provider_form.setContentsMargins(_px(12), _px(10), _px(12), _px(10))

        self.provider_combo = NoWheelComboBox()
        self.provider_combo.setObjectName("settingsCombo")
        self.provider_combo.addItems(["ollama", "openai_compatible"])
        self.provider_combo.setCurrentText(_llm_provider())
        self.provider_combo.currentTextChanged.connect(lambda _text: self._on_provider_changed(refresh=True))

        self.ollama_endpoint_edit = QLineEdit()
        self.ollama_endpoint_edit.setObjectName("settingsCombo")
        self.ollama_endpoint_edit.setText(_ollama_base_url())
        self.ollama_endpoint_edit.setPlaceholderText("http://127.0.0.1:11434")
        self.ollama_endpoint_edit.editingFinished.connect(self.refresh_modelhub)

        self.openai_endpoint_edit = QLineEdit()
        self.openai_endpoint_edit.setObjectName("settingsCombo")
        self.openai_endpoint_edit.setText(_openai_base_url())
        self.openai_endpoint_edit.setPlaceholderText("http://127.0.0.1:11434/v1")
        self.openai_endpoint_edit.editingFinished.connect(self.refresh_modelhub)

        self.openai_key_edit = QLineEdit()
        self.openai_key_edit.setObjectName("settingsCombo")
        self.openai_key_edit.setText(os.getenv("STACKWIRE_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip())
        self.openai_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.openai_key_edit.setPlaceholderText("optional API key")
        self.openai_key_edit.editingFinished.connect(self.refresh_modelhub)

        provider_form.addRow("Provider", self.provider_combo)
        provider_form.addRow("Ollama endpoint", self.ollama_endpoint_edit)
        provider_form.addRow("OpenAI-compatible", self.openai_endpoint_edit)
        provider_form.addRow("API key", self.openai_key_edit)

        health_row = QHBoxLayout()
        self.modelhub_status = QLabel("ModelHub: checking...")
        self.modelhub_status.setObjectName("dialogNote")
        self.modelhub_status.setWordWrap(True)
        self.modelhub_refresh_button = QPushButton("Refresh")
        self.modelhub_refresh_button.setObjectName("ghostButton")
        self.modelhub_refresh_button.clicked.connect(self.refresh_modelhub)
        self.modelhub_test_button = QPushButton("Test prompt")
        self.modelhub_test_button.setObjectName("ghostButton")
        self.modelhub_test_button.clicked.connect(lambda: self.test_model(self.answer_model.currentText().strip()))
        health_row.addWidget(self.modelhub_status, 1)
        health_row.addWidget(self.modelhub_refresh_button)
        health_row.addWidget(self.modelhub_test_button)
        health_holder = QWidget()
        health_holder.setLayout(health_row)

        self.hardware_label = QLabel(_hardware_summary()[0])
        self.hardware_label.setObjectName("dialogNote")
        self.hardware_label.setWordWrap(True)

        self.modelhub_progress = QProgressBar()
        self.modelhub_progress.setObjectName("modelHubProgress")
        self.modelhub_progress.setRange(0, 100)
        self.modelhub_progress.setValue(0)
        self.modelhub_progress.setVisible(False)
        self.modelhub_progress_label = QLabel("")
        self.modelhub_progress_label.setObjectName("dialogNote")
        self.modelhub_progress_label.setVisible(False)

        self.model_cards_container = QWidget()
        self.model_cards_container.setObjectName("modelCardsContainer")
        self.model_cards_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.model_cards_layout = QVBoxLayout(self.model_cards_container)
        self.model_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.model_cards_layout.setSpacing(_px(8))
        self.model_cards_layout.addStretch(1)
        cards_scroll = QScrollArea()
        cards_scroll.setObjectName("modelHubScroll")
        cards_scroll.setFrameShape(QFrame.Shape.NoFrame)
        cards_scroll.setWidgetResizable(True)
        cards_scroll.setMinimumHeight(_px(310))
        cards_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        cards_scroll.setWidget(self.model_cards_container)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(_px(10))
        top_row.addWidget(model_box, 1)
        top_row.addWidget(provider_box, 1)
        top_holder = QWidget()
        top_holder.setLayout(top_row)

        layout.addWidget(top_holder)
        layout.addWidget(health_holder)
        layout.addWidget(self.hardware_label)
        layout.addWidget(cards_scroll, 1)
        layout.addWidget(self.modelhub_progress)
        layout.addWidget(self.modelhub_progress_label)
        self._on_provider_changed()
        self.render_model_cards()
        return page

    def _on_provider_changed(self, *, refresh: bool = False) -> None:
        provider = self.provider_combo.currentText().strip()
        is_openai = provider == "openai_compatible"
        self.ollama_endpoint_edit.setEnabled(not is_openai)
        self.openai_endpoint_edit.setEnabled(is_openai)
        self.openai_key_edit.setEnabled(is_openai)
        if hasattr(self, "modelhub_test_button"):
            self.modelhub_test_button.setEnabled(True)
        if hasattr(self, "model_cards_layout"):
            self.render_model_cards()
        if refresh and hasattr(self, "modelhub_refresh_button"):
            QTimer.singleShot(0, self.refresh_modelhub)

    def _run_modelhub_worker(self, worker: QObject) -> None:
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)  # type: ignore[attr-defined]
        finished_signal = getattr(worker, "finished", None)
        failed_signal = getattr(worker, "failed", None)
        if finished_signal is not None:
            finished_signal.connect(lambda *args, t=thread: t.quit())
            finished_signal.connect(lambda *args, w=worker: w.deleteLater())
        if failed_signal is not None:
            failed_signal.connect(lambda *args, t=thread: t.quit())
            failed_signal.connect(lambda *args, w=worker: w.deleteLater())
        thread.finished.connect(lambda t=thread: self._modelhub_threads.remove(t) if t in self._modelhub_threads else None)
        thread.finished.connect(thread.deleteLater)
        self._modelhub_threads.append(thread)
        thread.start()

    def refresh_modelhub(self) -> None:
        if not hasattr(self, "provider_combo"):
            return
        if getattr(self, "_modelhub_pull_active", False):
            return
        if hasattr(self, "modelhub_refresh_button") and not self.modelhub_refresh_button.isEnabled():
            return
        self.modelhub_refresh_button.setEnabled(False)
        self.modelhub_status.setText("ModelHub: checking provider and installed models...")
        worker = ModelHubRefreshWorker(
            provider=self.provider_combo.currentText().strip(),
            ollama_endpoint=self.ollama_endpoint_edit.text().strip(),
            openai_endpoint=self.openai_endpoint_edit.text().strip(),
            api_key=self.openai_key_edit.text().strip(),
        )
        worker.finished.connect(self._on_modelhub_refreshed)
        worker.failed.connect(self._on_modelhub_failed)
        self._run_modelhub_worker(worker)

    @Slot(dict)
    def _on_modelhub_refreshed(self, data: dict) -> None:
        self.modelhub_refresh_button.setEnabled(True)
        installed = {model for model in data.get("installed", []) if isinstance(model, str) and model.strip()}
        self._installed_models = installed
        version = str(data.get("version", "unknown"))
        provider = str(data.get("provider", "ollama"))
        count = len(installed)
        if provider == "openai_compatible":
            self.modelhub_status.setText(f"OpenAI-compatible endpoint OK - {count} models")
        else:
            self.modelhub_status.setText(f"Ollama OK - version {version} - {count} installed models")
        self._sync_model_combos(installed)
        self.render_model_cards()

    @Slot(str)
    def _on_modelhub_failed(self, message: str) -> None:
        self.modelhub_refresh_button.setEnabled(True)
        self.modelhub_status.setText(
            "ModelHub: provider is unreachable. For local Ollama start it with `ollama serve` "
            f"or check endpoint. Details: {message}"
        )
        self._installed_models = set()
        self._sync_model_combos(set())
        self.render_model_cards()

    def _sync_model_combos(self, models: set[str]) -> None:
        current_values = [self.answer_model.currentText(), self.recovery_model.currentText(), self.vision_model.currentText()]
        choices = _dedupe_models(sorted(model.strip() for model in models if model.strip()))
        vision_choices = [model for model in choices if _is_vision_model(model)]

        def selected(current: str, available: list[str]) -> str:
            current = current.strip()
            if current in available:
                return current
            return available[0] if available else ""

        for combo, current in ((self.answer_model, current_values[0]), (self.recovery_model, current_values[1]), (self.vision_model, current_values[2])):
            available = vision_choices or choices if combo is self.vision_model else choices
            next_value = selected(current, available)
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(available)
            combo.setEnabled(bool(available))
            if next_value:
                combo.setCurrentText(next_value)
            combo.blockSignals(False)

    def render_model_cards(self) -> None:
        if not hasattr(self, "model_cards_layout"):
            return
        while self.model_cards_layout.count() > 1:
            item = self.model_cards_layout.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._model_cards = {}
        hardware, ram_gb, vram_gb = _hardware_summary()
        self.hardware_label.setText(hardware)
        installed_only = [
            model
            for model in sorted(self._installed_models)
            if model not in {recommendation.name for recommendation in MODELHUB_RECOMMENDED}
        ]
        for recommendation in MODELHUB_RECOMMENDED:
            self.model_cards_layout.insertWidget(self.model_cards_layout.count() - 1, self._make_model_card(recommendation, ram_gb, vram_gb))
        for model in installed_only:
            recommendation = ModelRecommendation(model, model, "installed", "custom", "Already installed in provider.", 0)
            self.model_cards_layout.insertWidget(self.model_cards_layout.count() - 1, self._make_model_card(recommendation, ram_gb, vram_gb))
        self.model_cards_layout.activate()
        self.model_cards_container.adjustSize()
        self.model_cards_container.updateGeometry()
        self.model_cards_container.update()

    def _make_model_card(self, recommendation: ModelRecommendation, ram_gb: int, vram_gb: int) -> QFrame:
        installed = recommendation.name in self._installed_models
        suitability = _hardware_recommendation_note(recommendation, ram_gb, vram_gb)
        card = QFrame()
        card.setObjectName("modelCard")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(_px(12), _px(10), _px(12), _px(10))
        layout.setSpacing(_px(10))

        icon = QLabel()
        icon.setObjectName("modelIcon")
        icon.setPixmap(icon_pixmap("mark", _px(24), ACCENT if installed else MUTED))

        text_box = QVBoxLayout()
        title = QLabel(recommendation.title)
        title.setObjectName("modelCardTitle")
        meta = QLabel(f"{recommendation.name} - {recommendation.kind} - {recommendation.size} - {suitability}")
        meta.setObjectName("dialogNote")
        meta.setWordWrap(True)
        note = QLabel(recommendation.note)
        note.setObjectName("dialogNote")
        note.setWordWrap(True)
        text_box.addWidget(title)
        text_box.addWidget(meta)
        text_box.addWidget(note)

        status_text = "available" if installed else ("not listed" if self.provider_combo.currentText().strip() == "openai_compatible" else "not installed")
        status = QLabel(status_text)
        status.setObjectName("modelState")

        use = QPushButton("Use")
        use.setObjectName("ghostButton")
        use.setEnabled(installed)
        use.clicked.connect(lambda _checked=False, model=recommendation.name: self.use_model(model))

        pull = QPushButton("Pull")
        pull.setObjectName("ghostButton")
        pull.setEnabled(not installed and self.provider_combo.currentText().strip() == "ollama")
        pull.clicked.connect(lambda _checked=False, model=recommendation.name: self.pull_model(model))

        test = QPushButton("Test")
        test.setObjectName("ghostButton")
        test.setEnabled(installed)
        test.clicked.connect(lambda _checked=False, model=recommendation.name: self.test_model(model))

        buttons = QHBoxLayout()
        buttons.addWidget(status)
        buttons.addWidget(use)
        buttons.addWidget(pull)
        buttons.addWidget(test)

        layout.addWidget(icon)
        layout.addLayout(text_box, 1)
        layout.addLayout(buttons)
        self._model_cards[recommendation.name] = card
        return card

    def use_model(self, model: str) -> None:
        if not model:
            return
        if _is_vision_model(model):
            self.vision_model.setCurrentText(model)
            self.modelhub_status.setText(f"Selected {model} for screenshot/image analysis. Save settings to apply.")
            return
        self.answer_model.setCurrentText(model)
        self.recovery_model.setCurrentText(model)
        self.modelhub_status.setText(f"Selected {model} for answer and recovery. Save settings to apply.")

    def pull_model(self, model: str) -> None:
        if not model:
            return
        self.modelhub_progress.setVisible(True)
        self.modelhub_progress_label.setVisible(True)
        self.modelhub_progress.setRange(0, 100)
        self.modelhub_progress.setValue(0)
        self.modelhub_progress_label.setText(f"Pulling {model}...")
        self._modelhub_pull_active = True
        self.modelhub_refresh_button.setEnabled(False)
        worker = ModelPullWorker(model, self.ollama_endpoint_edit.text().strip())
        worker.progress.connect(self._on_pull_progress)
        worker.finished.connect(self._on_pull_finished)
        worker.failed.connect(self._on_pull_failed)
        self._run_modelhub_worker(worker)

    @Slot(str, str, int, int, str)
    def _on_pull_progress(self, model: str, status: str, completed: int, total: int, speed: str) -> None:
        if total > 0:
            percent = max(0, min(100, int(completed * 100 / total)))
            self.modelhub_progress.setValue(percent)
            size = f"{completed / (1024**3):.2f}/{total / (1024**3):.2f} GB"
            self.modelhub_progress_label.setText(f"{model}: {percent}% - {size}" + (f" - {speed}" if speed else f" - {status}"))
        else:
            self.modelhub_progress.setRange(0, 0)
            self.modelhub_progress_label.setText(f"{model}: {status}")

    @Slot(str)
    def _on_pull_finished(self, model: str) -> None:
        self.modelhub_progress.setRange(0, 100)
        self.modelhub_progress.setValue(100)
        self.modelhub_progress_label.setText(f"{model}: downloaded. Refreshing installed models...")
        self._modelhub_pull_active = False
        self._installed_models.add(model)
        self.modelhub_refresh_button.setEnabled(True)
        self.render_model_cards()
        self.refresh_modelhub()

    @Slot(str, str)
    def _on_pull_failed(self, model: str, message: str) -> None:
        self.modelhub_progress.setRange(0, 100)
        self.modelhub_progress.setValue(0)
        self.modelhub_progress_label.setText(f"{model}: pull failed - {message}")
        self._modelhub_pull_active = False
        self.modelhub_refresh_button.setEnabled(True)

    def test_model(self, model: str) -> None:
        model = model.strip()
        if not model:
            self.modelhub_status.setText("Choose a model before test.")
            return
        self.modelhub_test_button.setEnabled(False)
        self.modelhub_status.setText(f"Testing {model}...")
        worker = ModelTestWorker(
            provider=self.provider_combo.currentText().strip(),
            model=model,
            ollama_endpoint=self.ollama_endpoint_edit.text().strip(),
            openai_endpoint=self.openai_endpoint_edit.text().strip(),
            api_key=self.openai_key_edit.text().strip(),
        )
        worker.finished.connect(self._on_test_finished)
        worker.failed.connect(self._on_test_failed)
        self._run_modelhub_worker(worker)

    @Slot(str, float, str)
    def _on_test_finished(self, model: str, latency: float, text: str) -> None:
        self.modelhub_test_button.setEnabled(True)
        self.modelhub_status.setText(f"{model}: test OK - {latency * 1000:.0f} ms - output: {text or '-'}")

    @Slot(str, str)
    def _on_test_failed(self, model: str, message: str) -> None:
        self.modelhub_test_button.setEnabled(True)
        self.modelhub_status.setText(f"{model}: test failed - {message}")

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
        self.vad_filter = QCheckBox("Silence filter (VAD)")
        self.vad_filter.setChecked(os.getenv("WHISPER_VAD_FILTER", "1").strip().lower() in {"1", "true", "yes", "on"})
        form.addRow("Audio device", self.audio_device)
        form.addRow("Speech language", self.language_mode)
        form.addRow("Beam size", self.beam_size)
        form.addRow("", self.vad_filter)
        hint = QLabel("Use a fixed language (ru/en) when speech is mostly one language for better accuracy.")
        hint.setObjectName("dialogNote")
        hint.setWordWrap(True)
        form.addRow("", hint)
        return page

    def _build_knowledge_tab(self) -> QWidget:
        page, form = self._form_page()
        self.web_search = QCheckBox("Use web search when the model is unsure (DuckDuckGo)")
        self.web_search.setChecked(os.getenv("STACKWIRE_WEB_SEARCH", "1").strip().lower() not in {"0", "false", "no", "off"})
        self.remember_answers = QCheckBox("Save useful answers to the local knowledge base")
        self.remember_answers.setChecked(os.getenv("STACKWIRE_REMEMBER_ANSWERS", "1").strip().lower() not in {"0", "false", "no", "off"})
        form.addRow("", self.web_search)
        form.addRow("", self.remember_answers)

        try:
            from app import vectorstore

            info = vectorstore.stats()
            if info.get("available"):
                state = f"Qdrant: {info.get('points', 0)} records - model {str(info.get('model','')).split('/')[-1]}"
            else:
                state = "Vector store is disabled (qdrant-client/fastembed is not installed)."
        except Exception:
            state = "Vector store is unavailable."
        self._vector_state = QLabel(state)
        self._vector_state.setObjectName("dialogNote")
        self._vector_state.setWordWrap(True)
        form.addRow("Local store", self._vector_state)

        reindex = QPushButton("Reindex knowledge")
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
            self._vector_state.setText(f"Ready - {info.get('points', 0)} records in store.")
        except Exception as exc:  # noqa: BLE001
            self._vector_state.setText(f"Reindex failed: {_short_error(exc)}")

    def _build_diagnostics_tab(self) -> QWidget:
        page, form = self._form_page()
        self.runtime_debug_panel = QCheckBox("Show runtime diagnostics panel")
        self.runtime_debug_panel.setChecked(bool(getattr(self._window, "debug_expanded", False)))
        form.addRow("", self.runtime_debug_panel)

        note = QLabel("This is StackWire's internal panel with STT/recovery/latency diagnostics. The rail Debug button is reserved for answer troubleshooting.")
        note.setObjectName("dialogNote")
        note.setWordWrap(True)
        form.addRow("", note)
        return page

    # -- helpers -------------------------------------------------------- #
    def _model_combo(self, models: list[str], current: str) -> NoWheelComboBox:
        combo = NoWheelComboBox()
        combo.setEditable(False)
        combo.setObjectName("settingsCombo")
        choices = _dedupe_models(models)
        combo.addItems(choices)
        combo.setEnabled(bool(choices))
        if current in choices:
            combo.setCurrentText(current)
        elif choices:
            combo.setCurrentIndex(0)
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
            "STACKWIRE_LLM_PROVIDER": self.provider_combo.currentText().strip(),
            "OLLAMA_URL": _ollama_chat_url(self.ollama_endpoint_edit.text().strip()),
            "STACKWIRE_OPENAI_BASE_URL": _openai_base_url(self.openai_endpoint_edit.text().strip()),
            "STACKWIRE_OPENAI_API_KEY": self.openai_key_edit.text().strip(),
            "STACKWIRE_AUDIO_DEVICE": self.audio_device.currentText().strip(),
            "STT_LANGUAGE_MODE": self.language_mode.currentText().strip(),
            "WHISPER_BEAM_SIZE": self.beam_size.currentText().strip(),
            "WHISPER_VAD_FILTER": "1" if self.vad_filter.isChecked() else "0",
            "STACKWIRE_WEB_SEARCH": "1" if self.web_search.isChecked() else "0",
            "STACKWIRE_REMEMBER_ANSWERS": "1" if self.remember_answers.isChecked() else "0",
            "STACKWIRE_AUTH_URL": self.auth_url_edit.text().strip(),
            "STACKWIRE_SHOW_DEBUG_PANEL": "1" if self.runtime_debug_panel.isChecked() else "0",
        }


class AnswerBrowser(QTextBrowser):
    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            event.accept()
            return
        super().wheelEvent(event)


def _flat_icon_button(kind: str, tooltip: str, on_click) -> QPushButton:  # noqa: ANN001
    button = QPushButton()
    button.setObjectName("msgActionButton")
    button.setToolTip(tooltip)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setFixedSize(_px(26), _px(26))
    button.setIcon(make_icon(kind, _px(15), "#8290a0"))
    button.clicked.connect(lambda: on_click())
    return button


configure_chat_widgets(px=_px, icon_pixmap=icon_pixmap, flat_icon_button=_flat_icon_button, accent=ACCENT)
configure_dialog_widgets(px=_px)


def _soft_shadow(widget: QWidget, *, blur: int = 28, dy: int = 8, alpha: int = 130) -> None:
    """Attach a soft drop shadow. NOTE: a widget can hold only one QGraphicsEffect,
    so never combine this with a fade-in opacity effect on the same widget."""
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setOffset(0, dy)
    effect.setColor(QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(effect)


def _animate_in(widget: QWidget, *, duration: int = 130) -> None:
    """Fade a freshly added row in (opacity 0в†’1)."""
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(0.45)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    # Drop the effect when done so it doesn't keep intercepting paints.
    # Qt accepts None to clear the effect; cast quiets the over-strict stub.
    anim.finished.connect(lambda: widget.setGraphicsEffect(cast(QGraphicsOpacityEffect, None)))
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
        if status_code is not None:
            detail = (response_text or str(exc)).strip()
            if prefix.lower().startswith("image"):
                return (
                    f"{prefix}: local Ollama is running, but rejected the image request "
                    f"(HTTP {status_code}). Current VISION_MODEL={current_vision_model()}. "
                    "Use a vision-capable model for screenshots, for example gemma3:4b or qwen2.5vl:7b "
                    "in Settings > ModelHub. If the model is still loading or stuck, run `ollama ps` "
                    "and restart/stop the stuck model. "
                    f"Details: {detail}"
                )
            return (
                f"{prefix}: local Ollama is running, but the chat request failed "
                f"(HTTP {status_code}). The selected model may be still loading, stuck, too heavy, "
                "or rejected the request payload. Run `ollama ps`, restart Ollama, or choose a smaller model. "
                f"Details: {detail}"
            )
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


configure_chat_workers(
    api_url=STACKWIRE_API_URL,
    api_connect_timeout=STACKWIRE_API_CONNECT_TIMEOUT,
    api_timeout=STACKWIRE_API_TIMEOUT,
    auth_headers=_auth_headers,
    remote_request_error=_remote_request_error,
)


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
        try:
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

            # On Windows, prefer the Win32 BitBlt path: Qt's grabWindow(0) can trigger
            # a C++ crash inside GPU driver code that Python try/except cannot catch.
            if sys.platform == "win32":
                pixmap = self._capture_win32_fallback(local_rect, screen_rect)
                if pixmap is None or pixmap.isNull():
                    full = screen.grabWindow(0)
                    pixmap = full.copy(local_rect) if not full.isNull() else None
            else:
                full = screen.grabWindow(0)
                pixmap = full.copy(local_rect) if not full.isNull() else None

            if pixmap is None or pixmap.isNull():
                LOGGER.warning("screen capture returned null pixmap rect=%s", local_rect)
                self.canceled.emit()
                self.close()
                return

            buffer = QBuffer()
            if not buffer.open(QIODevice.OpenModeFlag.WriteOnly):
                LOGGER.warning("screen capture buffer open failed")
                self.canceled.emit()
                self.close()
                return
            if not pixmap.save(buffer, "PNG"):
                LOGGER.warning("screen capture pixmap save failed rect=%s", local_rect)
                self.canceled.emit()
                self.close()
                return
            self.captured.emit(base64.b64encode(bytes(buffer.data().data())).decode("ascii"))
            self.close()
        except Exception:
            LOGGER.exception("screen capture failed")
            self.canceled.emit()
            self.close()

    def _capture_win32_fallback(self, local_rect: QRect, screen_rect: QRect) -> QPixmap | None:
        """Win32 BitBlt fallback when QScreen.grabWindow returns null."""
        if sys.platform != "win32":
            return None
        try:
            import ctypes
            import ctypes.wintypes as wt

            x = screen_rect.x() + local_rect.x()
            y = screen_rect.y() + local_rect.y()
            w = local_rect.width()
            h = local_rect.height()

            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32

            # Must set BOTH restype AND argtypes for handle-bearing functions.
            # Without argtypes, ctypes cannot marshal the 64-bit c_void_p int back
            # into the next call's argument slot, causing OverflowError on Win64.
            _HDC  = ctypes.c_void_p
            _HWND = ctypes.c_void_p
            _HBMP = ctypes.c_void_p
            _INT  = ctypes.c_int
            _UINT = ctypes.c_uint
            _LONG = ctypes.c_long

            user32.GetDC.restype = _HDC
            user32.GetDC.argtypes = [_HWND]
            user32.ReleaseDC.restype = _INT
            user32.ReleaseDC.argtypes = [_HWND, _HDC]

            gdi32.CreateCompatibleDC.restype = _HDC
            gdi32.CreateCompatibleDC.argtypes = [_HDC]
            gdi32.CreateCompatibleBitmap.restype = _HBMP
            gdi32.CreateCompatibleBitmap.argtypes = [_HDC, _INT, _INT]
            gdi32.SelectObject.restype = ctypes.c_void_p
            gdi32.SelectObject.argtypes = [_HDC, ctypes.c_void_p]
            gdi32.BitBlt.restype = _INT
            gdi32.BitBlt.argtypes = [_HDC, _INT, _INT, _INT, _INT, _HDC, _INT, _INT, ctypes.c_uint32]
            gdi32.GetDIBits.restype = _INT
            gdi32.GetDIBits.argtypes = [_HDC, _HBMP, _UINT, _UINT, ctypes.c_void_p, ctypes.c_void_p, _UINT]
            gdi32.DeleteObject.restype = _INT
            gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
            gdi32.DeleteDC.restype = _INT
            gdi32.DeleteDC.argtypes = [_HDC]

            hdc_screen = user32.GetDC(0)
            hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
            hbitmap = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
            gdi32.SelectObject(hdc_mem, hbitmap)
            SRCCOPY = 0x00CC0020
            gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, x, y, SRCCOPY)

            # Read bitmap bits into bytes
            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [
                    ("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG),
                    ("biPlanes", wt.WORD), ("biBitCount", wt.WORD), ("biCompression", wt.DWORD),
                    ("biSizeImage", wt.DWORD), ("biXPelsPerMeter", wt.LONG),
                    ("biYPelsPerMeter", wt.LONG), ("biClrUsed", wt.DWORD),
                    ("biClrImportant", wt.DWORD),
                ]

            bmi = BITMAPINFOHEADER()
            bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.biWidth = w
            bmi.biHeight = -h  # top-down
            bmi.biPlanes = 1
            bmi.biBitCount = 32
            bmi.biCompression = 0  # BI_RGB

            buf = (ctypes.c_char * (w * h * 4))()
            gdi32.GetDIBits(hdc_mem, hbitmap, 0, h, buf, ctypes.byref(bmi), 0)

            gdi32.DeleteObject(hbitmap)
            gdi32.DeleteDC(hdc_mem)
            user32.ReleaseDC(0, hdc_screen)

            from PySide6.QtGui import QImage
            # GetDIBits BI_RGB 32bpp = bytes B,G,R,0x00 — Format_RGB32 matches this layout
            # and Qt ignores the high byte, treating the image as fully opaque.
            # IMPORTANT: keep raw_bytes in a named variable — QImage(bytes(...), ...) creates a
            # shallow view; the anonymous temporary would be freed immediately after the ctor
            # returns, leaving the QImage with a dangling pointer → crash in fromImage().
            raw_bytes = bytes(buf)
            img = QImage(raw_bytes, w, h, w * 4, QImage.Format.Format_RGB32)
            pm = QPixmap.fromImage(img)  # deep-copies the pixel data into the QPixmap
            del img, raw_bytes  # safe to release now
            return pm if not pm.isNull() else None
        except Exception:
            LOGGER.exception("win32 capture fallback failed")
            return None


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
            self.failed.emit("Audio dependencies are missing. Run: python -m pip install -r requirements.txt")
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
                f"{exc}\n\nSelect another device. On Windows, WASAPI microphone or System audio WASAPI usually works better than MME."
            )
        finally:
            self.stopped.emit()

    def _run_loopback(self, np, KaldiRecognizer, Model, model_path: Path) -> None:  # noqa: ANN001
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            self.failed.emit("System audio capture requires pyaudiowpatch: python -m pip install pyaudiowpatch")
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
            self.failed.emit(f"VOSK_MODEL_PATH not found: {path}")
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
            self.failed.emit(f"Could not download Vosk model: {exc}")
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
            self.failed.emit(f"Could not download fallback Vosk model: {exc}")
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
    chats_button: QPushButton
    notes_button: QPushButton
    capture_button: QPushButton
    diff_button: QPushButton
    search_button: QPushButton
    debug_button: QPushButton

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
        self._stream_generation = 0
        self._active_stream_generation = 0
        self._ask_threads: list[QThread] = []
        self._generating = False          # True while LLM is producing tokens
        self.typing_timer = QTimer(self)
        self.typing_timer.setInterval(60)
        self.typing_timer.timeout.connect(self._tick_typing)
        # Follow-up suggestions state
        self._suggestions_thread: QThread | None = None
        self._suggestions_worker: "SuggestionsWorker | None" = None
        self._suggestions_generation = 0
        self.image_thread: QThread | None = None
        self.image_worker: ImageAnalysisWorker | None = None
        self._image_generation = 0
        self._active_image_generation = 0
        self.imagegen_thread: QThread | None = None
        self.imagegen_worker: ImageGenWorker | None = None
        self._imagegen_generation = 0
        self._active_imagegen_generation = 0
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
        # Multi-turn vision: the last analyzed screenshot stays pinned so follow-up
        # text questions can ask about the SAME image without re-capturing.
        self._vision_context_b64 = ""
        self._last_vision_b64 = ""
        self.visibility_hotkey_down = False
        self.record_hotkey_down = False
        self.submit_after_speech_stop = False
        self.speech_input_locked = False
        self.live_mode = False          # continuous listen→answer→listen loop
        self.debug_expanded = os.getenv("STACKWIRE_SHOW_DEBUG_PANEL", "0").strip().lower() in {"1", "true", "yes", "on"}
        self._first_show_done = False
        self._fade_animation: QPropertyAnimation | None = None
        self._modal_overlay: QWidget | None = None
        self._modal_dialog: QDialog | None = None
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
        self.rail_expanded = os.getenv("STACKWIRE_RAIL_EXPANDED", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.chats_panel_visible = False
        self.chat_session_id = chat_sessions.current_session_id()
        self.chat_messages: list[tuple[str, str]] = chat_sessions.load_session(self.chat_session_id) if self.chat_session_id else []
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
        # Live-mode timer: fires ~1.2 s after the last speech fragment to auto-submit
        self.live_submit_timer = QTimer(self)
        self.live_submit_timer.setSingleShot(True)
        self.live_submit_timer.setInterval(1200)
        self.live_submit_timer.timeout.connect(self._live_auto_submit)
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
                self.status.setText(f"Signed in as {dialog.username}")
            self.update_account_chip()
            self.render_chat()
            return True
        if hasattr(self, "status"):
            self.status.setText("Sign in canceled")
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
            self.status.setText("Signed out")
            QTimer.singleShot(150, self.prompt_login)

    def update_account_chip(self) -> None:
        chip = getattr(self, "account_chip", None)
        if chip is None:
            return
        if self.authenticated and self.auth_username:
            chip.setText(self.auth_username)
        elif self.auth_required:
            chip.setText("guest")
        else:
            chip.setText("")
        chip.setVisible(False)

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

        suggestions_thread = self._suggestions_thread
        if not self._shutdown_thread(suggestions_thread, "suggestions", timeout_ms=1_000):
            shutdown_pending = True
        else:
            self._suggestions_thread = None
            self._suggestions_worker = None

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

        imagegen_thread = self.imagegen_thread
        if self.imagegen_worker:
            try:
                self.imagegen_worker.session.close()
            except RuntimeError:
                pass
        if not self._shutdown_thread(imagegen_thread, "imagegen", timeout_ms=5_000):
            shutdown_pending = True
        else:
            self.imagegen_thread = None
            self.imagegen_worker = None

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

    def _make_rail_button(self, kind: str, label: str, tooltip: str, handler, *, checkable: bool = False) -> QPushButton:  # noqa: ANN001
        button = QPushButton(label if self.rail_expanded else "")
        button.setObjectName("railButton")
        button.setProperty("kind", kind)
        button.setProperty("label", label)
        button.setToolTip(tooltip)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setCheckable(checkable)
        button.clicked.connect(handler)
        return button

    def _rail_action_handler(self, spec: RailActionSpec):  # noqa: ANN001
        handler = getattr(self, spec.handler)
        if spec.expand_mode:
            return lambda _checked=False, mode=spec.expand_mode: handler(mode)
        return handler

    def _schedule_chat_sessions_refresh(self) -> None:
        if hasattr(self, "chat_sessions_layout"):
            QTimer.singleShot(0, self.refresh_chat_sessions)

    def _persist_current_chat(self, *, refresh: bool = False) -> None:
        try:
            if not self.chat_session_id and not self.chat_messages:
                return
            was_draft = not self.chat_session_id
            self.chat_session_id = chat_sessions.save_session(self.chat_session_id, self.chat_messages)
            if refresh or was_draft:
                self._schedule_chat_sessions_refresh()
        except Exception:
            LOGGER.debug("chat session save failed", exc_info=True)

    def new_chat_session(self) -> None:
        self._stop_streaming(discard_generation=True)
        self._hide_suggestions()
        if not self.chat_messages and not self.input.toPlainText().strip() and not self.chat_session_id:
            self.input.setFocus()
            self.status.setText("New chat draft")
            return
        self.chat_session_id = ""
        chat_sessions.set_current_session("")
        self.chat_messages = []
        self.input.clear()
        self.clear_attachment()  # drop any staged screenshot / pinned vision context
        self.last_answer_question = ""
        self.last_answer_text = ""
        self.last_main_answer_text = ""
        self.last_answer_id = None
        self.render_chat()
        self._schedule_chat_sessions_refresh()
        self.input.setFocus()
        self.status.setText("New chat draft")

    def open_chat_session(self, session_id: str) -> None:
        if not session_id or session_id == self.chat_session_id:
            return
        self._stop_streaming(discard_generation=True)
        self.chat_session_id = session_id
        chat_sessions.set_current_session(session_id)
        self.chat_messages = chat_sessions.load_session(session_id)
        self.input.clear()
        self.clear_attachment()  # drop any staged screenshot / pinned vision context
        self.render_chat(focus_latest_assistant=True)
        self._schedule_chat_sessions_refresh()
        self.status.setText("Chat opened")

    def show_delete_chat_confirm(self, session_id: str, title: str) -> None:
        dialog = QDialog(self)
        dialog.setObjectName("deleteChatDialog")
        dialog.setWindowTitle("Delete chat")
        dialog.setMinimumWidth(_px(420))
        dialog.setMaximumWidth(_px(500))

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(_px(18), _px(16), _px(18), _px(16))
        layout.setSpacing(_px(11))

        heading = QLabel("Delete chat?")
        heading.setObjectName("deleteChatTitle")
        message = QLabel(f"This will delete: <b>{html.escape(title or 'New chat')}</b>")
        message.setObjectName("deleteChatText")
        message.setTextFormat(Qt.TextFormat.RichText)
        message.setWordWrap(True)
        note = QLabel("Cannot be undone.")
        note.setObjectName("deleteChatNote")
        note.setWordWrap(True)

        actions = QHBoxLayout()
        actions.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("deleteCancelButton")
        delete = QPushButton("Delete")
        delete.setObjectName("deleteDangerButton")
        cancel.clicked.connect(dialog.reject)
        delete.clicked.connect(dialog.accept)
        actions.addWidget(cancel)
        actions.addWidget(delete)

        layout.addWidget(heading)
        layout.addWidget(message)
        layout.addWidget(note)
        layout.addLayout(actions)
        self._show_embedded_dialog(dialog, on_accept=lambda _dialog: self._delete_chat_session_confirmed(session_id))

    def delete_chat_session(self, session_id: str, title: str = "") -> None:
        if not session_id:
            return
        resolved_title = title or next((item.title for item in chat_sessions.list_sessions() if item.id == session_id), "New chat")
        self.show_delete_chat_confirm(session_id, resolved_title)

    def _delete_chat_session_confirmed(self, session_id: str) -> None:
        self._stop_streaming(discard_generation=True)
        next_id = chat_sessions.delete_session(session_id)
        if session_id == self.chat_session_id:
            self.chat_session_id = next_id
            self.chat_messages = chat_sessions.load_session(self.chat_session_id) if self.chat_session_id else []
            self.render_chat(focus_latest_assistant=True)
        self._schedule_chat_sessions_refresh()
        self.status.setText("Chat deleted")

    def refresh_chat_sessions(self) -> None:
        if not hasattr(self, "chat_sessions_layout"):
            return
        while self.chat_sessions_layout.count() > 1:
            item = self.chat_sessions_layout.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        summaries = chat_sessions.list_sessions()
        for summary in summaries:
            self.chat_sessions_layout.insertWidget(self.chat_sessions_layout.count() - 1, self._make_chat_session_row(summary))
        self.chat_sessions_layout.activate()
        self.chat_sessions_container.updateGeometry()
        self.chat_sessions_container.update()

    def rename_chat_session(self, session_id: str, current_title: str) -> None:
        dialog = QDialog(self)
        dialog.setObjectName("inlineFormDialog")
        dialog.setWindowTitle("Rename chat")
        dialog.setMinimumWidth(_px(420))
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(_px(20), _px(18), _px(20), _px(18))
        layout.setSpacing(_px(12))
        title = QLabel("Rename chat")
        title.setObjectName("dialogTitle")
        field = QLineEdit()
        field.setObjectName("settingsCombo")
        field.setText(current_title)
        field.selectAll()
        field.returnPressed.connect(dialog.accept)
        actions = QHBoxLayout()
        actions.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("ghostButton")
        save = QPushButton("Save")
        save.setObjectName("dialogPrimaryButton")
        cancel.clicked.connect(dialog.reject)
        save.clicked.connect(dialog.accept)
        actions.addWidget(cancel)
        actions.addWidget(save)
        layout.addWidget(title)
        layout.addWidget(field)
        layout.addLayout(actions)

        def apply_rename(_dialog: QDialog) -> None:
            clean_title = " ".join(field.text().strip().split())
            if clean_title and chat_sessions.rename_session(session_id, clean_title):
                self.refresh_chat_sessions()
                self.status.setText("Chat renamed")

        self._show_embedded_dialog(dialog, on_accept=apply_rename)
        field.setFocus()

    def _make_chat_session_row(self, summary: chat_sessions.ChatSessionSummary) -> QFrame:
        row = QFrame()
        row.setObjectName("chatSessionItem")
        row.setProperty("active", summary.id == self.chat_session_id)
        row.setMinimumHeight(_px(46))
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(8, 7, 6, 7)
        layout.setSpacing(6)
        title = summary.title.strip() or "New chat"
        open_button = QPushButton()
        open_button.setObjectName("chatSessionButton")
        open_button.setMinimumWidth(0)
        open_button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        open_button.setText(open_button.fontMetrics().elidedText(title, Qt.TextElideMode.ElideRight, _px(160)))
        open_button.setToolTip(title)
        open_button.clicked.connect(lambda _checked=False, session_id=summary.id: self.open_chat_session(session_id))
        rename_button = QPushButton()
        rename_button.setObjectName("chatRenameButton")
        rename_button.setToolTip("Rename chat")
        rename_button.setIcon(make_icon("edit", _px(13), "#8290a0"))
        rename_button.setIconSize(QSize(_px(13), _px(13)))
        rename_button.setFixedSize(_px(28), _px(28))
        rename_button.clicked.connect(lambda _checked=False, session_id=summary.id, title=summary.title: self.rename_chat_session(session_id, title))
        delete_button = QPushButton()
        delete_button.setObjectName("chatDeleteButton")
        delete_button.setToolTip("Delete chat")
        delete_button.setIcon(make_icon("trash", _px(13), "#8290a0"))
        delete_button.setIconSize(QSize(_px(13), _px(13)))
        delete_button.setFixedSize(_px(28), _px(28))
        delete_button.clicked.connect(lambda _checked=False, session_id=summary.id, title=summary.title: self.delete_chat_session(session_id, title))
        layout.addWidget(open_button, 1)
        layout.addWidget(rename_button, 0)
        layout.addWidget(delete_button, 0)
        return row

    def toggle_chats_panel(self) -> None:
        self.chats_panel_visible = not self.chats_panel_visible
        self.chat_panel.setVisible(self.chats_panel_visible)
        self.chats_button.setChecked(self.chats_panel_visible)
        if self.chats_panel_visible:
            self.refresh_chat_sessions()
        self.apply_icons()

    def toggle_rail_expanded(self) -> None:
        self.rail_expanded = not self.rail_expanded
        os.environ["STACKWIRE_RAIL_EXPANDED"] = "1" if self.rail_expanded else "0"
        try:
            _save_local_env_values({"STACKWIRE_RAIL_EXPANDED": os.environ["STACKWIRE_RAIL_EXPANDED"]})
        except Exception:
            LOGGER.debug("rail state save failed", exc_info=True)
        self.apply_ui_zoom()

    def toggle_mini_mode(self) -> None:
        """Compact mode: hide the sidebar + chats panel, leaving just the chat + composer
        (type or use voice). Toggle again to restore the full layout."""
        self._mini_mode = not getattr(self, "_mini_mode", False)
        mini = self._mini_mode
        self.rail.setVisible(not mini)
        # Round all four corners of the content panel — in full mode the left corners are
        # squared because the rail/chat-panel sit against them; in mini mode content is alone.
        self.content.setProperty("mini", "true" if mini else "false")
        style = self.content.style()
        style.unpolish(self.content)
        style.polish(self.content)
        # The shell's dark drop-shadow gets clipped at the window edge and pools as black
        # blobs in the rounded corners when the window is small — turn it off in mini mode.
        if getattr(self, "_shell_shadow", None) is not None:
            self._shell_shadow.setEnabled(not mini)
        if mini:
            self._normal_geometry = self.geometry()
            self.chat_panel.setVisible(False)
            # Hard-cap the width so the window genuinely shrinks (children just wrap),
            # regardless of any wide message bubble's minimum-size hint.
            compact_w, compact_h = _px(1230), _px(600)
            self.setMinimumSize(0, 0)
            self.setMaximumWidth(compact_w)
            self.resize(compact_w, compact_h)
        else:
            self.setMaximumWidth(16_777_215)  # QWIDGETSIZE_MAX — lift the cap
            self.chat_panel.setVisible(self.chats_panel_visible)
            geo = getattr(self, "_normal_geometry", None)
            if geo is not None:
                self.setGeometry(geo)
        self.mini_button.setToolTip("Выйти из mini mode" if mini else "Mini mode (компактный чат)")
        self.apply_icons()
        self.status.setText("Mini mode" if mini else "Ready")

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
        self._shell_shadow = shadow  # toggled off in mini mode (its dark halo pools in the rounded corners)
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

        self.device_combo = NoWheelComboBox()
        self.device_combo.setObjectName("deviceCombo")

        self.listen_button = QPushButton()
        self.listen_button.setObjectName("iconButton")
        self.listen_button.setToolTip("Listen")
        self.listen_button.clicked.connect(self.toggle_listening)

        self.rail_toggle_button = self._make_rail_button("menu", "Compact", "Expand/collapse sidebar", self.toggle_rail_expanded)
        for spec in MAIN_RAIL_ACTIONS:
            button = self._make_rail_button(
                spec.kind,
                spec.label,
                spec.tooltip,
                self._rail_action_handler(spec),
                checkable=spec.checkable,
            )
            setattr(self, spec.attr, button)
        self.settings_button = self._make_rail_button("settings", "Settings", "Settings", self.show_settings_dialog)

        self.mini_button = QPushButton()
        self.mini_button.setObjectName("closeButton")  # share the title-bar button styling
        self.mini_button.setToolTip("Mini mode (компактный чат)")
        self.mini_button.clicked.connect(self.toggle_mini_mode)

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
        rail_layout.addWidget(self.rail_toggle_button, 0, Qt.AlignmentFlag.AlignHCenter)
        rail_layout.addStretch(1)

        rail_nav = QWidget()
        rail_nav.setObjectName("railNav")
        rail_nav_layout = QVBoxLayout(rail_nav)
        rail_nav_layout.setContentsMargins(0, 0, 0, 0)
        rail_nav_layout.setSpacing(10)
        for spec in MAIN_RAIL_ACTIONS:
            rail_nav_layout.addWidget(getattr(self, spec.attr))
        rail_layout.addWidget(rail_nav, 0, Qt.AlignmentFlag.AlignHCenter)
        rail_layout.addStretch(2)
        rail_layout.addWidget(self.settings_button, 0, Qt.AlignmentFlag.AlignHCenter)

        self.chat_panel = QFrame()
        self.chat_panel.setObjectName("chatPanel")
        self.chat_panel.setVisible(self.chats_panel_visible)
        chat_panel_layout = QVBoxLayout(self.chat_panel)
        chat_panel_layout.setContentsMargins(12, 14, 12, 14)
        chat_panel_layout.setSpacing(10)
        chat_header = QHBoxLayout()
        chat_title = QLabel("Chats")
        chat_title.setObjectName("chatPanelTitle")
        self.new_chat_button = QPushButton("New")
        self.new_chat_button.setObjectName("ghostButton")
        self.new_chat_button.setIcon(make_icon("plus", _px(14), ACCENT))
        self.new_chat_button.clicked.connect(self.new_chat_session)
        chat_header.addWidget(chat_title)
        chat_header.addStretch(1)
        chat_header.addWidget(self.new_chat_button)
        self.chat_sessions_container = QWidget()
        self.chat_sessions_container.setObjectName("chatSessionsContainer")
        self.chat_sessions_layout = QVBoxLayout(self.chat_sessions_container)
        self.chat_sessions_layout.setContentsMargins(0, 0, 0, 0)
        self.chat_sessions_layout.setSpacing(6)
        self.chat_sessions_layout.addStretch(1)
        self.chat_sessions_scroll = QScrollArea()
        self.chat_sessions_scroll.setObjectName("chatSessionsScroll")
        self.chat_sessions_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.chat_sessions_scroll.setWidgetResizable(True)
        self.chat_sessions_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_sessions_scroll.setWidget(self.chat_sessions_container)
        chat_panel_layout.addLayout(chat_header)
        chat_panel_layout.addWidget(self.chat_sessions_scroll, 1)

        content = QFrame()
        content.setObjectName("content")
        self.content = content
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(18, 14, 18, 16)
        content_layout.setSpacing(12)

        self.status = QLabel("Ready")
        self.status.setObjectName("status")
        self.model_chip = QLabel(current_answer_model())
        self.model_chip.setObjectName("modelChip")
        self.model_chip.setVisible(False)
        self.api_chip = QLabel("remote" if STACKWIRE_API_URL else "local")
        self.api_chip.setObjectName("apiChip")
        self.api_chip.setVisible(False)
        self.account_chip = QLabel("")
        self.account_chip.setObjectName("accountChip")
        self.account_chip.setVisible(False)

        header.addStretch(1)
        header.addWidget(self.status)
        header.addWidget(self.mini_button)
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
        self.attach_remove = QPushButton("вњ•")
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
        self.input.set_height_limits(_px(40), _px(132))
        self.input.submitted.connect(self.submit_question)
        self.input.image_pasted.connect(self._on_clipboard_image)
        # Focus ring: highlight the composer pill while the input is focused.
        self.input.installEventFilter(self)

        # The mic (listen) button lives inside the composer pill, GPT-style.
        self.listen_button.setObjectName("composerIcon")

        self.ask_button = QPushButton()
        self.ask_button.setObjectName("composerSend")
        self.ask_button.setToolTip("Ask")
        self.ask_button.setFixedSize(_px(40), _px(40))
        self.ask_button.clicked.connect(self.submit_question)

        footer.addWidget(self.attach_button, 0, Qt.AlignmentFlag.AlignBottom)
        footer.addWidget(self.input, 1)
        footer.addWidget(self.listen_button, 0, Qt.AlignmentFlag.AlignBottom)
        footer.addWidget(self.ask_button, 0, Qt.AlignmentFlag.AlignBottom)
        composer_layout.addLayout(footer)

        self.debug_panel = QLabel()
        self.debug_panel.setObjectName("debugPanel")
        self.debug_panel.setWordWrap(True)
        self.debug_panel.setMaximumHeight(140)
        self.debug_panel.setVisible(self.debug_expanded)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        bottom.addWidget(QSizeGrip(self), 0, Qt.AlignmentFlag.AlignRight)

        # Follow-up suggestion chips (hidden until after an answer is received).
        self.suggestions_bar = QWidget()
        self.suggestions_bar.setObjectName("suggestionsBar")
        self.suggestions_bar.setVisible(False)
        sugg_layout = QHBoxLayout(self.suggestions_bar)
        sugg_layout.setContentsMargins(_px(2), _px(4), _px(2), _px(2))
        sugg_layout.setSpacing(_px(6))
        self._suggestion_buttons: list[QPushButton] = []
        for _ in range(3):
            btn = QPushButton()
            btn.setObjectName("suggestionChip")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, b=btn: self._on_suggestion_clicked(b.text()))
            sugg_layout.addWidget(btn, 1)
            self._suggestion_buttons.append(btn)

        content_layout.addLayout(header)
        content_layout.addWidget(self.chat_area, 1)
        content_layout.addWidget(self.suggestions_bar)
        content_layout.addWidget(composer)
        content_layout.addWidget(self.debug_panel)
        content_layout.addLayout(bottom)
        shell_layout.addWidget(rail)
        shell_layout.addWidget(self.chat_panel)
        shell_layout.addWidget(content, 1)
        layout.addWidget(shell)
        self.setCentralWidget(root)

        # Slash-command suggestion popup, floating above the composer.
        self.slash_popup = SlashPopup(root)
        self.input.slash_popup = self.slash_popup
        self.input.textChanged.connect(self._update_slash_popup)
        self.input.slash_accepted.connect(self._insert_slash_command)
        self.slash_popup.command_chosen.connect(self._insert_slash_command)

        self.setStyleSheet(STYLES)
        self.install_zoom_shortcuts()
        self.install_global_hotkeys()
        self.apply_ui_zoom()
        self.update_answer_actions()
        self.update_debug_panel()
        self.render_chat()
        self.refresh_chat_sessions()
        self.input.setFocus()

    def _ensure_modal_overlay(self) -> QWidget:
        root = self.centralWidget()
        if self._modal_overlay is None or self._modal_overlay.parentWidget() is not root:
            overlay = QWidget(root)
            overlay.setObjectName("modalOverlay")
            overlay.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            overlay.setVisible(False)
            overlay_layout = QVBoxLayout(overlay)
            overlay_layout.setContentsMargins(_px(24), _px(24), _px(24), _px(24))
            overlay_layout.setSpacing(0)
            overlay_layout.addStretch(1)
            center = QHBoxLayout()
            center.setContentsMargins(0, 0, 0, 0)
            center.addStretch(1)
            center.addStretch(1)
            overlay_layout.addLayout(center)
            overlay_layout.addStretch(1)
            self._modal_overlay = overlay
            self._modal_center_layout = center
        if root is not None:
            self._modal_overlay.setGeometry(root.rect())
        return self._modal_overlay

    def _show_embedded_dialog(self, dialog: QDialog, on_accept=None, on_reject=None) -> None:  # noqa: ANN001
        overlay = self._ensure_modal_overlay()
        if self._modal_dialog is not None:
            self._finish_embedded_dialog(self._modal_dialog, accepted=False, callback=None)

        dialog.setParent(overlay)
        dialog.setWindowFlags(Qt.WindowType.Widget)
        dialog.setModal(False)
        dialog.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        dialog.setStyleSheet(build_window_styles(self.ui_zoom))
        available = overlay.size() - QSize(_px(48), _px(48))
        max_width = max(dialog.minimumWidth(), available.width())
        max_height = max(dialog.minimumHeight(), available.height())
        dialog.setMaximumSize(max_width, max_height)
        self._modal_dialog = dialog

        center = self._modal_center_layout
        center.insertWidget(center.count() - 1, dialog, 0, Qt.AlignmentFlag.AlignCenter)
        dialog.accepted.connect(lambda d=dialog: self._finish_embedded_dialog(d, accepted=True, callback=on_accept))
        dialog.rejected.connect(lambda d=dialog: self._finish_embedded_dialog(d, accepted=False, callback=on_reject))
        overlay.show()
        overlay.raise_()
        dialog.show()
        dialog.raise_()

    def _finish_embedded_dialog(self, dialog: QDialog, *, accepted: bool, callback) -> None:  # noqa: ANN001
        if dialog is not self._modal_dialog:
            return
        if callback is not None:
            callback(dialog)
        if hasattr(self, "_modal_center_layout"):
            self._modal_center_layout.removeWidget(dialog)
        dialog.setParent(None)
        dialog.deleteLater()
        self._modal_dialog = None
        if self._modal_overlay is not None:
            self._modal_overlay.hide()
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

        font = QFont("Manrope")
        font.setPointSizeF(10 * self.ui_zoom)
        self.setFont(font)

        self.chat_area.setMinimumHeight(_px(300))
        self.input.set_height_limits(_px(40), _px(132))
        self.input.keep_arrow_cursor()
        self.ask_button.setFixedSize(_px(40), _px(40))
        for _btn in (self.attach_button, self.listen_button):
            _btn.setFixedSize(_px(38), _px(38))
        if hasattr(self, "rail"):
            self.rail.setFixedWidth(_px(190 if self.rail_expanded else 116))
        if hasattr(self, "chat_panel"):
            self.chat_panel.setFixedWidth(_px(270))
        self.debug_panel.setMaximumHeight(_px(140))
        self.setStyleSheet(build_window_styles(self.ui_zoom))
        self.apply_icons()

    def apply_icons(self) -> None:
        icon_size = _px(16)
        self.title_mark.setPixmap(icon_pixmap("mark", _px(24), ACCENT))
        listening = self._speech_is_running()
        self.rail_toggle_button.setIcon(make_icon("collapse" if self.rail_expanded else "menu", icon_size, "#83aeb8"))
        self.rail_toggle_button.setToolTip("Collapse sidebar" if self.rail_expanded else "Expand sidebar")
        self.chats_button.setChecked(self.chats_panel_visible)
        self.chats_button.setIcon(make_icon("chats", icon_size, ACCENT if self.chats_panel_visible else "#83aeb8"))
        self.notes_button.setIcon(make_icon("notes", icon_size, "#83aeb8"))
        self.listen_button.setIcon(make_icon("stop" if listening else "listen", icon_size, CORAL if listening else ACCENT))
        self.listen_button.setToolTip("Stop listening" if listening else "Listen")
        self.capture_button.setIcon(make_icon("capture", icon_size, "#83aeb8"))
        self.diff_button.setIcon(make_icon("diff", icon_size, "#83aeb8"))
        self.search_button.setIcon(make_icon("search", icon_size, "#83aeb8"))
        self.debug_button.setIcon(make_icon("debug", icon_size, "#83aeb8"))
        self.settings_button.setIcon(make_icon("settings", icon_size, "#83aeb8"))
        self.attach_button.setIcon(make_icon("attach", icon_size, "#83aeb8"))
        if self._generating:
            self.ask_button.setIcon(make_icon("stop_gen", icon_size, "#10131a"))
            self.ask_button.setToolTip("Stop generation (ESC)")
        else:
            self.ask_button.setIcon(make_icon("ask", icon_size, "#10131a"))
            self.ask_button.setToolTip("Ask")
        self.close_button.setIcon(make_icon("close", icon_size, "#6f8793"))
        self.mini_button.setIcon(make_icon("mini_exit" if getattr(self, "_mini_mode", False) else "mini", icon_size, "#83aeb8"))
        for button in (
            self.rail_toggle_button,
            self.chats_button,
            self.notes_button,
            self.listen_button,
            self.capture_button,
            self.diff_button,
            self.search_button,
            self.debug_button,
            self.settings_button,
            self.attach_button,
            self.ask_button,
            self.mini_button,
            self.close_button,
        ):
            button.setIconSize(QSize(icon_size, icon_size))
        # Rail icon buttons stay 34px squares; composer icons (attach/listen) keep their
        # 38px pill sizing set in apply_ui_zoom — don't resize them here.
        for button in (
            self.rail_toggle_button,
            self.chats_button,
            self.notes_button,
            self.capture_button,
            self.diff_button,
            self.search_button,
            self.debug_button,
            self.settings_button,
        ):
            label = str(button.property("label") or "")
            button.setText(label if self.rail_expanded else "")
            if button is self.rail_toggle_button and self.rail_expanded:
                button.setText("Compact")
            button.setFixedSize(_px(166 if self.rail_expanded else 34), _px(36 if self.rail_expanded else 34))
        self.close_button.setFixedSize(_px(34), _px(34))
        self.mini_button.setFixedSize(_px(34), _px(34))

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
        window). No-op on non-Windows or older builds вЂ" the app still looks fine."""
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

    def _launch_image_analysis(self, image_b64: str, prompt: str) -> None:
        if self.image_thread is not None:
            return
        self._last_vision_b64 = image_b64  # pinned as follow-up context once the answer lands
        self._image_generation += 1
        image_generation = self._image_generation
        self._active_image_generation = image_generation
        self.image_thread = QThread()
        # Stream the vision answer through the SAME pipeline as text answers: deltas are
        # tagged with the active stream generation so on_answer_delta renders them live.
        self.image_worker = ImageAnalysisWorker(
            image_b64, prompt,
            image_generation=image_generation,
            stream_generation=self._active_stream_generation,
        )
        self.image_worker.moveToThread(self.image_thread)
        self.image_thread.started.connect(self.image_worker.run)
        self.image_worker.delta.connect(self.on_answer_delta)
        self.image_worker.finished.connect(self.on_image_answer)
        self.image_worker.failed.connect(self.on_image_error)
        self.image_worker.done.connect(self.image_thread.quit)
        self.image_worker.done.connect(self.image_worker.deleteLater)
        self.image_thread.finished.connect(self.on_image_thread_finished)
        self.image_thread.finished.connect(self.image_thread.deleteLater)
        self.image_thread.start()

    # ------------------------------------------------------------------ image gen
    def start_image_generation(self, prompt: str | None = None) -> None:
        """Generate an image. Invoked via the /image slash command with the prompt text."""
        if not self._require_login():
            return
        if self.imagegen_thread is not None:
            self.status.setText("Генерация уже идёт...")
            return
        prompt = (prompt if prompt is not None else self.input.toPlainText()).strip()
        if not prompt:
            self.status.setText("Использование: /image <описание картинки>")
            return
        self.input.clear()
        self.chat_messages.append(("user", prompt))
        self.chat_messages.append(("assistant", "[[thinking:0]]"))
        self._begin_streaming()
        self.ask_button.setEnabled(False)
        self.status.setText("Генерирую изображение…")

        self._imagegen_generation += 1
        gen = self._imagegen_generation
        self._active_imagegen_generation = gen
        self.imagegen_thread = QThread()
        self.imagegen_worker = ImageGenWorker(prompt, generation=gen)
        self.imagegen_worker.moveToThread(self.imagegen_thread)
        self.imagegen_thread.started.connect(self.imagegen_worker.run)
        self.imagegen_worker.finished.connect(self.on_imagegen_finished)
        self.imagegen_worker.failed.connect(self.on_imagegen_error)
        self.imagegen_worker.done.connect(self.imagegen_thread.quit)
        self.imagegen_worker.done.connect(self.imagegen_worker.deleteLater)
        self.imagegen_thread.finished.connect(self.on_imagegen_thread_finished)
        self.imagegen_thread.finished.connect(self.imagegen_thread.deleteLater)
        self.imagegen_thread.start()

    @Slot(int, str, str)
    def on_imagegen_finished(self, generation: int, image_b64: str, prompt: str) -> None:
        if generation != self._active_imagegen_generation:
            return
        self._active_imagegen_generation = 0
        self._stop_streaming(discard_generation=True)
        answer = f"[[generated_image:{image_b64}]]\n\n*Сгенерировано: {prompt}*"
        self.replace_last_assistant(answer)
        self.last_answer_text = answer
        self.ask_button.setEnabled(True)
        self.status.setText("Готово")
        self.update_answer_actions()

    @Slot(int, str)
    def on_imagegen_error(self, generation: int, message: str) -> None:
        if generation != self._active_imagegen_generation:
            return
        self._active_imagegen_generation = 0
        self.show_error(message)
        self.ask_button.setEnabled(True)

    def on_imagegen_thread_finished(self) -> None:
        self.imagegen_worker = None
        self.imagegen_thread = None
        self._active_imagegen_generation = 0
        self.ask_button.setEnabled(True)

    def _show_image_viewer(self, pixmap_or_b64: QPixmap | str) -> None:
        """Open FullImageDialog for a screenshot or generated image."""
        if isinstance(pixmap_or_b64, str):
            pixmap = QPixmap()
            pixmap.loadFromData(base64.b64decode(pixmap_or_b64))
        else:
            pixmap = pixmap_or_b64
        if pixmap.isNull():
            return
        dialog = FullImageDialog(pixmap, self)
        dialog.exec()

    # ------------------------------------------------------------------ region capture
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
        self.capture_button.setEnabled(True)
        if not image_b64:
            self.status.setText("Capture failed")
            return

        # Stage the screenshot as a pending attachment so the user can type a question
        # (e.g. "почему тут ошибка?") and press Enter — or just press Enter with no text
        # for a full automatic analysis. Submission flows through submit_with_attachment().
        self._vision_context_b64 = ""  # fresh capture: drop any prior follow-up context
        self.pending_attachment = {"kind": "image", "name": "screenshot.png", "data": image_b64}
        self._refresh_attachment_bar()
        self.status.setText("Скриншот готов — задайте вопрос и Enter, или просто Enter для разбора")
        self.input.setFocus()

    def submit_capture_question(self, prompt: str) -> None:
        image_b64 = self.pending_capture_b64
        if not image_b64:
            return
        if not self._require_login():
            return

        prompt = prompt.strip() or "What is shown in the screenshot and what is important?"
        self.pending_capture_b64 = ""
        self.ask_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.chat_messages.append(("user", f"Screenshot request\n\n{prompt}"))
        self.chat_messages.append(("assistant", "Analyzing selected screen area..."))
        self.input.clear()
        self.render_chat(focus_latest_assistant=True)
        self.status.setText("Analyzing captured area...")

        self._launch_image_analysis(image_b64, prompt)

    def submit_vision_followup(self, question: str) -> None:
        """Answer a follow-up question about the screenshot pinned in context.

        The image is re-sent to the vision model silently — only the user's question
        text appears as a new bubble (the screenshot is already shown earlier in the
        chat, so we don't duplicate it)."""
        image_b64 = self._vision_context_b64
        if not image_b64:
            return
        if not self._require_login():
            return
        self._hide_suggestions()
        self.stop_speech_capture_for_submit()
        self.ask_button.setEnabled(False)
        self.update_answer_actions(force_disabled=True)
        self.input.clear()
        self.question_count += 1
        self.chat_messages.append(("user", question))
        self.chat_messages.append(("assistant", "[[thinking:0]]"))
        self._begin_streaming()
        self.status.setText("Анализирую скриншот...")
        self._launch_image_analysis(image_b64, question)

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
                if "stereo mix" in device.name.lower() or "СЃС‚РµСЂРµРѕ РјРёРєС€РµСЂ" in device.name.lower()
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
            if "realtek" in name or "РґРёРЅР°РјРёРєРё" in name or "speakers" in name:
                return (1, name)
            return (2, name)
        if "DIRECTSOUND" in api:
            return (3, name)
        if "СЃС‚РµСЂРµРѕ РјРёРєС€РµСЂ" in name or "stereo mix" in name:
            return (4, name)
        if not device.loopback and ("РјРёРєСЂРѕС„РѕРЅ" in name or "microphone" in name or "mic " in name):
            return (5, name)
        if not device.loopback and ("Р»РёРЅРµР№РЅС‹Р№" in name or "Р»РёРЅ. РІС…РѕРґ" in name or "line in" in name):
            return (7, name)
        if "MME" in api:
            return (9, name)
        return (6, name)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        if self._modal_overlay is not None and self.centralWidget() is not None:
            self._modal_overlay.setGeometry(self.centralWidget().rect())

    def _drag_blocked_by_widget(self, widget: QWidget | None) -> bool:
        blocked_types = (QAbstractButton, QTextEdit, QTextBrowser, QComboBox, QLineEdit, QScrollArea, QSizeGrip)
        while widget is not None:
            if isinstance(widget, blocked_types):
                return True
            widget = widget.parentWidget()
        return False

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag_blocked_by_widget(self.childAt(event.position().toPoint())):
                self.drag_position = None
                super().mousePressEvent(event)
                return
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if event.buttons() & Qt.MouseButton.LeftButton and self.drag_position is not None:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()
            return
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            self.drag_position = None
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        self.drag_position = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and self._generating:
            self.stop_generation()
            event.accept()
            return
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

    def _update_slash_popup(self) -> None:
        popup = getattr(self, "slash_popup", None)
        if popup is None:
            return
        if popup.update_for(self.input.toPlainText()):
            self._position_slash_popup()

    def _position_slash_popup(self) -> None:
        popup = self.slash_popup
        popup.adjustSize()
        parent = popup.parentWidget()
        if parent is None:
            return
        top_left = self.input.mapTo(parent, QPoint(0, 0))
        x = max(_px(6), top_left.x())
        y = max(_px(6), top_left.y() - popup.height() - _px(6))
        popup.move(x, y)

    def _insert_slash_command(self, cmd: str) -> None:
        self.input.setPlainText(f"{cmd} ")
        cursor = self.input.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.input.setTextCursor(cursor)
        if getattr(self, "slash_popup", None) is not None:
            self.slash_popup.hide()
        self.input.setFocus()

    def _apply_slash_command(self, text: str) -> tuple[bool, str]:
        """Handle a /command typed in the composer.

        Returns (handled, question):
          handled=True  → command performed its own action; caller should stop.
          handled=False → `question` is the (possibly rewritten) text to submit normally.
        """
        parts = text.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/image":
            if not arg:
                self.status.setText("Использование: /image <описание картинки>")
                return True, text
            self.input.clear()
            self.start_image_generation(arg)
            return True, text

        if cmd == "/clear":
            self.clear_answer()
            return True, text

        # Text-prefix commands: rewrite into a normal prompt and let submit_question continue.
        rewrites = {
            "/explain": "Подробно и структурированно объясни простыми словами: ",
            "/code": "Покажи рабочий пример кода с пояснениями по теме: ",
            "/translate": "Определи язык и переведи (русский↔английский) этот текст: ",
        }
        if cmd in rewrites:
            if not arg:
                self.status.setText(f"Использование: {cmd} <текст>")
                return True, text
            return False, rewrites[cmd] + arg

        # Unknown slash token → treat as ordinary text.
        return False, text

    def submit_question(self) -> None:
        # If currently generating, the ask button acts as stop.
        if self._generating:
            self.stop_generation()
            return
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

        # Slash commands (/image, /clear, /explain, ...). Returns (handled, rewritten_question):
        # handled → the command did its own action; otherwise question may be rewritten for normal flow.
        if question.startswith("/"):
            handled, question = self._apply_slash_command(question)
            if handled:
                return

        if self.pending_capture_b64:
            self.submit_capture_question(question)
            return

        # A screenshot is pinned in context → answer this question about that image.
        if self._vision_context_b64:
            self.submit_vision_followup(question)
            return

        self._hide_suggestions()
        trusted_text = self.is_manual_input(question)
        self.stop_speech_capture_for_submit()
        speech_context = (self.raw_transcript_lines or self.transcript_lines)[-STT_CONTEXT_LINES:]
        # Build Q&A history context: last 4 exchanges (user + assistant pairs).
        history_messages = [
            (role, content)
            for role, content in self.chat_messages
            if role in ("user", "assistant")
            and content.strip()
            and not self._is_thinking(content)
            and "[[screenshot:" not in content
            and "[[file:" not in content
        ][-8:]  # last 4 Q&A pairs
        chat_context: list[str] = []
        for role, content in history_messages:
            prefix = "User" if role == "user" else "Assistant"
            text = content.split("\n\n", 1)[-1].strip()
            text = text[:350] + "..." if len(text) > 350 else text
            chat_context.append(f"{prefix}: {text}")
        context = [*speech_context, *chat_context][-(STT_CONTEXT_LINES + len(chat_context)) :]
        self.ask_button.setEnabled(False)
        self.update_answer_actions(force_disabled=True)
        self.status.setText("Generating...")
        self.question_count += 1
        self.chat_messages.append(("user", question))
        self.chat_messages.append(("assistant", "[[thinking:0]]"))
        self.input.clear()
        self.last_final_speech = ""
        self.current_partial_speech = ""
        self.last_question_candidate = ""
        self._begin_streaming()  # renders history once + inserts the streaming block
        self._launch_ask_stream(question, context, trusted_text)

    def _launch_ask_stream(self, question: str, context: list[str], trusted_text: bool, *, creative: bool = False) -> None:
        self.ask_thread = QThread()
        stream_generation = self._active_stream_generation
        self.ask_worker = AskStreamWorker(
            question,
            context,
            trusted_text=trusted_text,
            storage_session_id=self.storage_session_id,
            stream_generation=stream_generation,
            creative=creative,
        )
        self.ask_worker.moveToThread(self.ask_thread)
        self.ask_thread.started.connect(self.ask_worker.run)
        self.ask_worker.delta.connect(self.on_answer_delta)
        self.ask_worker.finished.connect(self.on_stream_finished)
        self.ask_worker.failed.connect(self.on_stream_failed)
        self.ask_worker.done.connect(self.ask_thread.quit)
        self.ask_worker.done.connect(self.ask_worker.deleteLater)
        self._ask_threads.append(self.ask_thread)
        self.ask_thread.finished.connect(lambda thread=self.ask_thread: self._ask_threads.remove(thread) if thread in self._ask_threads else None)
        self.ask_thread.finished.connect(self.ask_thread.deleteLater)
        self.ask_thread.start()

    def regenerate_message(self, index: int) -> None:
        """Re-run the question that produced the answer at `index`, replacing it with a fresh
        variant. Uses elevated temperature + a random seed so each click differs."""
        if self._generating or self.image_thread is not None or getattr(self, "imagegen_thread", None) is not None:
            self.status.setText("Подождите, идёт генерация…")
            return
        if not self._require_login():
            return
        if index >= len(self.chat_messages) or self.chat_messages[index][0] != "assistant":
            return
        # Find the user message that prompted this answer.
        user_idx = -1
        for i in range(index - 1, -1, -1):
            if self.chat_messages[i][0] == "user":
                user_idx = i
                break
        if user_idx < 0:
            self.status.setText("Нет вопроса для перегенерации")
            return

        user_content = self.chat_messages[user_idx][1]
        # ChatGPT-style: drop the old answer (and anything after the question).
        prior_messages = self.chat_messages[:user_idx]
        self.chat_messages = self.chat_messages[: user_idx + 1]
        self._hide_suggestions()
        self.ask_button.setEnabled(False)
        self.update_answer_actions(force_disabled=True)
        self.chat_messages.append(("assistant", "[[thinking:0]]"))
        self._begin_streaming()
        self.status.setText("Перегенерация…")

        # Screenshot question → regenerate via the vision model (with its caption, if any).
        screenshot_match = re.search(r"\[\[screenshot:([A-Za-z0-9+/=]+)\]\]", user_content)
        if screenshot_match:
            caption = re.sub(r"\[\[(?:screenshot|file):[^\]]*\]\]", "", user_content).strip()
            self._launch_image_analysis(screenshot_match.group(1), caption)
            return

        # Build the same kind of short Q&A history context submit_question uses.
        history_messages = [
            (role, content)
            for role, content in prior_messages
            if role in ("user", "assistant")
            and content.strip()
            and not self._is_thinking(content)
            and "[[screenshot:" not in content
            and "[[file:" not in content
        ][-8:]
        chat_context: list[str] = []
        for role, content in history_messages:
            prefix = "User" if role == "user" else "Assistant"
            text = content.split("\n\n", 1)[-1].strip()
            text = text[:350] + "..." if len(text) > 350 else text
            chat_context.append(f"{prefix}: {text}")
        self._launch_ask_stream(user_content.strip(), chat_context, trusted_text=True, creative=True)

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
    MAX_TEXT_FILE_BYTES = 80_000

    def attach_file(self) -> None:
        if not self.ask_button.isEnabled():
            return
        path_str, _ = QFileDialog.getOpenFileName(self, "Attach file", "", "All files (*.*)")
        if not path_str:
            return
        path = Path(path_str)
        try:
            raw = path.read_bytes()
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Could not read file: {_short_error(exc)}")
            return
        if path.suffix.lower() in self.IMAGE_EXTENSIONS:
            self.pending_attachment = {"kind": "image", "name": path.name, "data": base64.b64encode(raw).decode("ascii")}
        else:
            try:
                text = raw[: self.MAX_TEXT_FILE_BYTES + 1].decode("utf-8")
                if len(raw) > self.MAX_TEXT_FILE_BYTES:
                    text = text[: self.MAX_TEXT_FILE_BYTES] + "\n... (file truncated)"
                self.pending_attachment = {"kind": "text", "name": path.name, "content": text}
            except UnicodeDecodeError:
                self.pending_attachment = {"kind": "binary", "name": path.name}
        self._refresh_attachment_bar()
        self.input.setFocus()
        self.status.setText("File attached. Add text if needed, then press Enter.")

    def clear_attachment(self) -> None:
        self.pending_attachment = None
        self._vision_context_b64 = ""  # also drop pinned screenshot follow-up context
        self._refresh_attachment_bar()

    def _refresh_attachment_bar(self) -> None:
        attachment = self.pending_attachment
        if attachment:
            kind_label = {"image": "image", "text": "text", "binary": "file"}.get(attachment["kind"], "file")
            self.attach_chip.setText(f"{attachment['name']} - {kind_label}")
            self.attach_bar.setVisible(True)
            return
        if self._vision_context_b64:
            # A screenshot is pinned in context — follow-up questions ask about it.
            self.attach_chip.setText("🖼 Скриншот активен — спросите ещё")
            self.attach_bar.setVisible(True)
            return
        self.attach_bar.setVisible(False)
        self.attach_chip.setText("")

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
            self.status.setText("Analyzing...")
            self._launch_image_analysis(attachment["data"], text)
            return

        if attachment["kind"] == "text":
            instruction = text.strip() or "Analyze this file and explain what it does."
            question = f"{instruction}\n\nFile \"{attachment['name']}\":\n```\n{attachment['content']}\n```"
        else:
            instruction = text.strip() or "What is this file and what is it used for?"
            question = f"{instruction}\n\n(File \"{attachment['name']}\" is attached, but its contents are not readable as text.)"
        file_md = f"[[file:{attachment['name']}]]" + (f"\n\n{text}" if text else "")
        self.chat_messages.append(("user", file_md))
        self.chat_messages.append(("assistant", "[[thinking:0]]"))
        self._begin_streaming()
        self.status.setText("Generating...")
        self._launch_ask_stream(question, [], True)

    def _last_assistant_index(self) -> int:
        for index in range(len(self.chat_messages) - 1, -1, -1):
            if self.chat_messages[index][0] == "assistant":
                return index
        return -1

    def _begin_streaming(self) -> None:
        # Rebuild the message widgets; the trailing assistant message starts as
        # animated thinking dots, then we stream rich text into that one row only.
        self._stream_generation += 1
        self._active_stream_generation = self._stream_generation
        self._stream_active = True
        self._stream_buffer = ""
        self._stream_render_pending = False
        _DIAGRAM_RENDER["enabled"] = False  # skip diagram rendering on partial source
        # Animate just the freshly added user message + assistant row into view.
        self.render_chat(animate_from=max(0, len(self.chat_messages) - 2))
        self._stream_prefix_snippets = len(CODE_SNIPPETS)
        self._stream_row = self._assistant_rows.get(self._last_assistant_index())
        # Show ask_button as stop button while generating
        self._generating = True
        self.ask_button.setEnabled(True)
        self.apply_icons()

    def _stop_streaming(self, *, discard_generation: bool = False) -> None:
        self._stream_active = False
        if discard_generation:
            self._active_stream_generation = 0
            self._generating = False
            if hasattr(self, "ask_button"):
                self.ask_button.setEnabled(True)
            if hasattr(self, "listen_button"):
                self.listen_button.setEnabled(True)
            if hasattr(self, "capture_button"):
                self.capture_button.setEnabled(True)
            if hasattr(self, "update_answer_actions"):
                self.update_answer_actions()
        _DIAGRAM_RENDER["enabled"] = True
        self.typing_timer.stop()

    def _scroll_answer_to_bottom(self) -> None:
        self.chat_area.scroll_to_bottom()

    def _scroll_to_message(self, index: int) -> None:
        self.chat_area.scroll_to_bottom()

    def _tick_typing(self) -> None:
        # Thinking dots animate themselves now; nothing to drive here.
        return

    @Slot(int, str)
    def on_answer_delta(self, stream_generation: int, chunk: str) -> None:
        if stream_generation != self._active_stream_generation:
            return
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
        bar = self.chat_area.scroll_area.verticalScrollBar()
        follow = bar.value() >= bar.maximum() - 8
        # Keep snippet ids stable/bounded: drop anything from the previous frame.
        del CODE_SNIPPETS[self._stream_prefix_snippets:]
        del CODE_BLOCK_KEYS[self._stream_prefix_snippets:]
        balanced = balance_streaming_markdown(self._stream_buffer)
        markup = markdown_to_html(balanced)
        # Mint caret at the end while generating (skip when inside a code fence).
        if not balanced.rstrip().endswith("```"):
            markup = _with_stream_caret(markup)
        self._stream_row.show_html(markup, final=False)
        if follow:
            QTimer.singleShot(0, lambda: self.chat_area.scroll_to_bottom(animated=False))

    @Slot(int, object)
    def on_stream_finished(self, stream_generation: int, result: object) -> None:
        if stream_generation != self._active_stream_generation:
            return
        self._stop_streaming(discard_generation=True)
        self.show_answer(result)

    @Slot(int, str)
    def on_stream_failed(self, stream_generation: int, message: str) -> None:
        if stream_generation != self._active_stream_generation:
            return
        self.show_error(message)

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
        self.chat_messages.append(("assistant", f"{header}\n\nGenerating expansion..."))
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
            cleaned = "The model returned an empty expansion."
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
            cleaned = result.answer.strip() or "The model returned an empty answer. Try sending the question again."
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
                cleaned = "The model returned an empty answer. Try sending the question again."
            elif len(cleaned) < 80:
                cleaned = f"{cleaned}\n\nNote:\nThe answer looks incomplete. Try the request again or reduce the context."
        if not isinstance(result, AskResult):
            self.last_answer_text = cleaned
            self.last_answer_id = None
        self.replace_last_assistant(cleaned)
        self._generating = False
        self.ask_button.setEnabled(True)
        self.listen_button.setEnabled(True)
        self.update_answer_actions()
        self.apply_icons()

        if self.live_mode:
            # Unlock STT so the next question can be captured
            self.speech_input_locked = False
            # Restart the worker if it somehow stopped (e.g. VAD timeout)
            if not self._speech_is_running():
                self.toggle_listening()
            self.status.setText("Live — говорите следующий вопрос...")
        else:
            self.status.setText("Ready")
            # Launch follow-up suggestions (only in regular mode, not live)
            q = self.last_answer_question or ""
            a = cleaned
            if q and a and len(a) > 30:
                self._launch_suggestions(q, a)

    @Slot(int, str)
    def on_image_answer(self, image_generation: int, text: str) -> None:
        if image_generation != self._active_image_generation:
            return
        self._active_image_generation = 0
        self.show_image_answer(text)

    @Slot(int, str)
    def on_image_error(self, image_generation: int, message: str) -> None:
        if image_generation != self._active_image_generation:
            return
        self._active_image_generation = 0
        self.show_error(message)

    @Slot(str)
    def show_image_answer(self, text: str) -> None:
        self._stop_streaming(discard_generation=True)
        cleaned = text.strip() or "Could not recognize the selected area."
        self.replace_last_assistant(cleaned)
        self.last_answer_question = ""
        self.last_answer_text = cleaned
        self.last_answer_id = None
        self.ask_button.setEnabled(True)
        self.capture_button.setEnabled(True)
        # Pin the analyzed screenshot so the user can ask follow-up questions about the
        # SAME image (re-sent silently, without a duplicate image bubble). ✕ clears it.
        if self._last_vision_b64:
            self._vision_context_b64 = self._last_vision_b64
            self._refresh_attachment_bar()
        self.status.setText("Ready")
        self.update_answer_actions()

    @Slot(str)
    def show_error(self, message: str) -> None:
        self._stop_streaming(discard_generation=True)
        self.replace_last_assistant(f"Error: {message}")
        self._generating = False
        self.ask_button.setEnabled(True)
        self.listen_button.setEnabled(True)
        self.capture_button.setEnabled(True)
        if self.pending_capture_b64:
            self.input.setFocus()
        self.update_answer_actions()
        self.apply_icons()
        if self.live_mode:
            # On error in live mode: unlock so the next question can be captured
            self.speech_input_locked = False
            if not self._speech_is_running():
                self.toggle_listening()
            self.status.setText("Live — ошибка, слушаю дальше...")
        else:
            self.status.setText("Error")

    # ------------------------------------------------------------------ stop generation
    def stop_generation(self) -> None:
        """Interrupt the current LLM generation immediately."""
        if not self._generating:
            return
        self._generating = False
        self._stop_streaming(discard_generation=True)
        self.replace_last_assistant("Остановлено")
        self.ask_button.setEnabled(True)
        self.listen_button.setEnabled(True)
        self.capture_button.setEnabled(True)
        self.apply_icons()
        if self.live_mode:
            self.speech_input_locked = False
        self.status.setText("Stopped.")

    # ------------------------------------------------------------------ clipboard paste
    def _on_clipboard_image(self, b64: str, name: str) -> None:
        """Handle an image pasted from clipboard (Ctrl+V) into the input field."""
        self.pending_attachment = {"kind": "image", "name": name, "data": b64}
        self._refresh_attachment_bar()
        self.status.setText("Image pasted. Add text if needed, then press Enter.")
        self.input.setFocus()

    # ------------------------------------------------------------------ follow-up suggestions
    def _suggestions_thread_is_running(self) -> bool:
        thread = self._suggestions_thread
        if thread is None:
            return False
        try:
            return bool(shiboken6.isValid(thread) and thread.isRunning())
        except RuntimeError:
            self._suggestions_thread = None
            self._suggestions_worker = None
            return False

    def _on_suggestions_thread_finished(self, thread: QThread) -> None:
        if self._suggestions_thread is thread:
            self._suggestions_thread = None
            self._suggestions_worker = None

    def _launch_suggestions(self, question: str, answer: str) -> None:
        """Asynchronously generate follow-up suggestion chips."""
        self._suggestions_generation += 1
        generation = self._suggestions_generation

        if self._suggestions_thread_is_running():
            assert self._suggestions_thread is not None
            self._suggestions_thread.quit()
            self._suggestions_thread.wait(100)

        thread = QThread(self)
        worker = SuggestionsWorker(question, answer)
        self._suggestions_thread = thread
        self._suggestions_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(lambda items, gen=generation: self._show_suggestions_for_generation(gen, items))
        worker.finished.connect(thread.quit)
        worker.done.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.done.connect(worker.deleteLater)
        thread.finished.connect(lambda t=thread: self._on_suggestions_thread_finished(t))
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _show_suggestions_for_generation(self, generation: int, items: list) -> None:
        if generation != self._suggestions_generation:
            return
        self._show_suggestions(items)

    @Slot(list)
    def _show_suggestions(self, items: list) -> None:
        for i, btn in enumerate(self._suggestion_buttons):
            if i < len(items) and items[i].strip():
                btn.setText(items[i].strip())
                btn.setVisible(True)
            else:
                btn.setVisible(False)
        self.suggestions_bar.setVisible(any(b.isVisible() for b in self._suggestion_buttons))

    def _hide_suggestions(self) -> None:
        self._suggestions_generation += 1
        self.suggestions_bar.setVisible(False)
        for btn in self._suggestion_buttons:
            btn.setVisible(False)

    def _on_suggestion_clicked(self, text: str) -> None:
        """Fill input with the suggestion text and submit immediately."""
        self._hide_suggestions()
        self.input.setPlainText(text)
        self.input.moveCursor(QTextCursor.MoveOperation.End)
        self.submit_question()

    def on_anchor_clicked(self, url: QUrl) -> None:
        target = url.toString()
        for prefix, handler in (
            ("edit:", self.start_edit_message),
            ("copy:", self.copy_message),
            ("copycode:", self.copy_code_snippet),
            ("togglecode:", self.toggle_code_block),
        ):
            if target.startswith(prefix):
                try:
                    index = int(target[len(prefix) :])
                except ValueError:
                    return
                handler(index)
                return
        if target.startswith("viewimage:"):
            try:
                gen_id = int(target[len("viewimage:"):])
                b64 = _GENERATED_IMAGES.get(gen_id)
                if b64:
                    self._show_image_viewer(b64)
            except ValueError:
                pass
            return
        if url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)

    def copy_code_snippet(self, index: int) -> None:
        if 0 <= index < len(CODE_SNIPPETS):
            QApplication.clipboard().setText(CODE_SNIPPETS[index])
            self.status.setText("Code copied")

    def toggle_code_block(self, index: int) -> None:
        if not (0 <= index < len(CODE_BLOCK_KEYS)):
            return
        block_key = CODE_BLOCK_KEYS[index]
        if block_key in EXPANDED_CODE_BLOCKS:
            EXPANDED_CODE_BLOCKS.remove(block_key)
        else:
            EXPANDED_CODE_BLOCKS.add(block_key)
        self.render_chat()

    def _message_plain_text(self, index: int) -> str:
        if not (0 <= index < len(self.chat_messages)):
            return ""
        content = self.chat_messages[index][1]
        match = re.match(r"^Р’РѕРїСЂРѕСЃ\s+\d+\s*\n\n(.+)$", content.strip(), flags=re.DOTALL)
        return (match.group(1).strip() if match else content).strip()

    def copy_message(self, index: int) -> None:
        text = self._message_plain_text(index)
        if not text:
            return
        QApplication.clipboard().setText(text)
        self.status.setText("Copied")

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
        self.status.setText("Editing - update the text and press Enter")

    def set_runtime_debug_panel(self, visible: bool) -> None:
        self.debug_expanded = visible
        if hasattr(self, "debug_panel"):
            self.debug_panel.setVisible(visible)

    def show_notes_dialog(self) -> None:
        dialog = NotesDialog(notes_path(), self)

        def finish_notes(_dialog: NotesDialog) -> None:
            self.status.setText("Notes saved")

        self._show_embedded_dialog(dialog, on_accept=finish_notes, on_reject=finish_notes)

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

        def apply_settings(settings_dialog: SettingsDialog) -> None:
            values = settings_dialog.values()
            required_keys = ("ANSWER_MODEL", "RECOVERY_MODEL", "VISION_MODEL")
            missing = [key for key in required_keys if not values.get(key, "").strip()]
            if missing:
                self.status.setText(f"Settings not saved: empty {missing[0]}")
                return

            try:
                _save_local_env_values(values)
                for key, value in values.items():
                    os.environ[key] = value
                self.set_audio_device_name(values.get("STACKWIRE_AUDIO_DEVICE", ""))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("settings save failed: %s", exc)
                self.status.setText(f"Settings save failed: {_short_error(exc)}")
                return

            self.model_chip.setText(current_answer_model())
            self.update_account_chip()
            self.update_debug_panel()
            self.render_chat()
            self.status.setText("Settings saved")

        self._show_embedded_dialog(dialog, on_accept=apply_settings)

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
        self._stop_streaming(discard_generation=True)
        self._hide_suggestions()
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
            "РљРѕРЅС‚РµРєСЃС‚ СЂР°СЃРїРѕР·РЅР°РЅРЅС‹Р№ РёР· Р·Р°РїРёСЃРё:\n"
            f"{context}\n\n"
            "Р—Р°РґР°С‡Р°: РЅР°Р№РґРё РїРѕСЃР»РµРґРЅРёР№ С‚РµС…РЅРёС‡РµСЃРєРёР№ РІРѕРїСЂРѕСЃ РІ СЌС‚РѕРј РєРѕРЅС‚РµРєСЃС‚Рµ, РѕС‚Р±СЂРѕСЃСЊ Р»РёС€РЅРёРµ С„СЂР°Р·С‹, РёСЃРїСЂР°РІСЊ РѕС€РёР±РєРё СЂРµС‡РµРІРѕРіРѕ СЂР°СЃРїРѕР·РЅР°РІР°РЅРёСЏ Рё РѕС‚РІРµС‚СЊ.\n\n"
            "Р•СЃР»Рё РїРѕР»Рµ РЅРёР¶Рµ СѓР¶Рµ РїРѕС…РѕР¶Рµ РЅР° РІРѕРїСЂРѕСЃ, РѕС‚РІРµС‡Р°Р№ РЅР° РЅРµРіРѕ:\n"
            f"{question}"
        )

    def replace_last_assistant(self, text: str, *, focus_latest: bool = False) -> None:
        for index in range(len(self.chat_messages) - 1, -1, -1):
            if self.chat_messages[index][0] == "assistant":
                self.chat_messages[index] = ("assistant", text)
                row = self._assistant_rows.get(index)
                if row is not None and shiboken6.isValid(row):
                    del CODE_SNIPPETS[self._stream_prefix_snippets:]
                    del CODE_BLOCK_KEYS[self._stream_prefix_snippets:]
                    row.show_html(markdown_to_html(text), final=True)
                    self._persist_current_chat()
                    if focus_latest:
                        QTimer.singleShot(0, lambda: self.chat_area.scroll_to_bottom(animated=True))
                    return
                self.render_chat(focus_latest_assistant=focus_latest)
                return
        self.chat_messages.append(("assistant", text))
        self.render_chat(focus_latest_assistant=focus_latest)

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
                full_pixmap = pixmap
                scaled = pixmap.scaledToWidth(_px(240), Qt.TransformationMode.SmoothTransformation)
                shot = ClickableImageLabel(lambda px=full_pixmap: self._show_image_viewer(px))
                shot.setObjectName("shotLabel")
                shot.setPixmap(_rounded_pixmap(scaled, _px(12)))
                shot.setCursor(Qt.CursorShape.PointingHandCursor)
                shot.setToolTip("Нажмите для просмотра")
                group_layout.addWidget(shot, 0, Qt.AlignmentFlag.AlignRight)
            if caption:
                group_layout.addWidget(self._user_bubble_label(caption), 0, Qt.AlignmentFlag.AlignRight)
        elif is_file:
            file_match = re.search(r"\[\[file:([^\]]+)\]\]", content)
            name = file_match.group(1) if file_match else "file"
            group_layout.addWidget(self._user_bubble_label(f"{name}" + (f"\n{caption}" if caption else "")), 0, Qt.AlignmentFlag.AlignRight)
        else:
            group_layout.addWidget(self._user_bubble_label(self._message_plain_text(index)), 0, Qt.AlignmentFlag.AlignRight)

        actions = QWidget()
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(_px(2))
        actions_layout.addStretch(1)
        actions_layout.addWidget(_flat_icon_button("copy", "Copy", lambda i=index: self.copy_message(i)))
        if not is_screenshot:
            actions_layout.addWidget(_flat_icon_button("edit", "Edit", lambda i=index: self.start_edit_message(i)))
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
        font = QFont("Manrope")
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
            previous_content = self.chat_messages[index - 1][1] if index > 0 and self.chat_messages[index - 1][0] == "user" else ""
            model_name = current_vision_model() if "[[screenshot:" in previous_content else current_answer_model()
            row = AssistantRow(
                index, self.on_anchor_clicked, self.copy_message, model_name,
                on_regenerate=self.regenerate_message,
            )
            self.chat_area.add_row(row)
            self._assistant_rows[index] = row
            if self._is_thinking(content):
                row.show_thinking()
            else:
                row.show_html(markdown_to_html(content), final=True)
        if animate:
            _animate_in(row)
        self._message_rows[index] = row

    def render_chat(self, focus_latest_assistant: bool = False, animate_from: int = -1) -> None:
        CODE_SNIPPETS.clear()
        CODE_BLOCK_KEYS.clear()
        _GENERATED_IMAGES.clear()
        previous_bar = self.chat_area.scroll_area.verticalScrollBar()
        previous_value = previous_bar.value()
        self._stream_row = None
        self._assistant_rows = {}
        self._message_rows = {}
        self.chat_area.clear_rows()
        if not self.chat_messages:
            self.chat_area.show_welcome()
            self._persist_current_chat()
            return
        self.chat_area.show_list()
        for index, (role, content) in enumerate(self.chat_messages):
            self._add_message_row(index, role, content, animate=(animate_from >= 0 and index >= animate_from))
        if animate_from >= 0:
            target_row = self._message_rows.get(animate_from)
            if target_row is not None:
                QTimer.singleShot(0, lambda row=target_row, start=previous_value: self.chat_area.scroll_to_message_start(row, animated=True, start_value=start))
        elif focus_latest_assistant:
            QTimer.singleShot(0, lambda: self.chat_area.scroll_to_bottom(animated=True))
        else:
            QTimer.singleShot(0, lambda value=previous_value: self.chat_area.restore_scroll_position(value))
        self._persist_current_chat()

    # ------------------------------------------------------------------ live mode
    def _live_auto_submit(self) -> None:
        """Called by live_submit_timer when speech has settled for ~1.2 s."""
        if not self.live_mode:
            return
        if self.speech_input_locked:
            # LLM is still generating; the lock will be cleared in show_answer
            return
        if not self.ask_button.isEnabled():
            return

        candidate = self.last_final_speech or self.last_question_candidate
        if not candidate or len(candidate.split()) < 2:
            # Too short / noise — wait for more speech
            return

        # ---- lock STT input (but do NOT stop the worker) ----
        self.speech_input_locked = True
        self.auto_ask_timer.stop()
        self.live_submit_timer.stop()
        self.current_partial_speech = ""

        question = condense_spoken_question(candidate) or candidate
        context = list((self.raw_transcript_lines or self.transcript_lines)[-STT_CONTEXT_LINES:])

        self.ask_button.setEnabled(False)
        self.update_answer_actions(force_disabled=True)
        self.status.setText("Live — генерирую ответ...")
        self.question_count += 1
        self.chat_messages.append(("user", question))
        self.chat_messages.append(("assistant", "[[thinking:0]]"))
        self.input.clear()
        self.last_final_speech = ""
        self.current_partial_speech = ""
        self.last_question_candidate = ""
        self._begin_streaming()
        self._launch_ask_stream(question, context, trusted_text=False)

    def toggle_listening(self) -> None:
        if self._speech_is_running():
            self.submit_after_speech_stop = False
            # If the user manually stops listening while live mode is on, turn live off too
            if self.live_mode:
                self.live_mode = False
                self.live_submit_timer.stop()
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
            self.show_error("No audio device selected.")
            return

        # Every listening session is live interview mode
        self.live_mode = True
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
        self.status.setText("Live — говорите вопрос...")

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

        if self.live_mode and self.ask_button.isEnabled():
            # Live mode: restart silence-timer on every new fragment.
            # When it fires without new speech arriving, we auto-submit.
            self.status.setText(f"Live — слышу: {self.last_final_speech[:80]}")
            self.live_submit_timer.start()
        else:
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
        self._active_image_generation = 0
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
    return _build_window_styles(
        scale=scale,
        px=_px,
        accent=ACCENT,
        coral=CORAL,
        elevated=ELEVATED,
        font_display=FONT_DISPLAY,
        font_stack=FONT_STACK,
        muted=MUTED,
        rail=RAIL,
        surface=SURFACE,
        text=TEXT,
        ui_zoom=UI_ZOOM,
    )

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
    app.setFont(QFont("Manrope", 10))
    app.setWindowIcon(make_icon("mark", 32, ACCENT))
    window = OverlayWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
