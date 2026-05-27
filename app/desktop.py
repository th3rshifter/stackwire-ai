import base64
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
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QObject, QPoint, QRect, QSize, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QIcon, QKeyEvent, QKeySequence, QPainter, QPen, QPixmap, QShortcut, QTextCursor, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizeGrip,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from requests import RequestException

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import APP_NAME, load_local_env  # noqa: E402
from app.event_log import append_client_event  # noqa: E402

load_local_env()

LOGGER = logging.getLogger(__name__)

from app.llm import ANSWER_MODE, MODEL, VISION_MODEL, AskResult, ExpandResult, OllamaClient  # noqa: E402
from app.question_recovery import DEFAULT_MODEL as RECOVERY_MODEL, RecoveryResult, STACKWIRE_MODE  # noqa: E402
from app.storage import create_session, log_feedback, save_good_answer  # noqa: E402
from app.tech_terms import WHISPER_TECHNICAL_PROMPT, normalize_spoken_technical_terms  # noqa: E402
from app.transcript_repair import clean_stt_output, collapse_repeated_phrases, is_probable_stt_hallucination, repair_live_transcript  # noqa: E402

ACCENT = "#72d6a3"
BLUE = "#7aa2ff"
PANEL = "rgba(10, 13, 20, 214)"
PANEL_LIGHT = "rgba(20, 25, 36, 200)"
TEXT = "#eef3ff"
MUTED = "#9aa8bf"
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
STT_BACKEND = os.getenv("STT_BACKEND", "whisper").strip().lower()
STT_ALLOW_VOSK_FALLBACK = os.getenv("STT_ALLOW_VOSK_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}
STT_ALLOW_CPU_WHISPER_FALLBACK = os.getenv("STT_ALLOW_CPU_WHISPER_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "on"}
STT_MIC_SIGNAL_THRESHOLD = float(os.getenv("STT_MIC_SIGNAL_THRESHOLD", "0.003"))
STT_LOOPBACK_SIGNAL_THRESHOLD = float(os.getenv("STT_LOOPBACK_SIGNAL_THRESHOLD", "0.00025"))
STT_PROBE_LOOPBACK_DEVICES = os.getenv("STT_PROBE_LOOPBACK_DEVICES", "0").strip().lower() in {"1", "true", "yes", "on"}
STT_LIVE_MAX_WORDS = int(os.getenv("STT_LIVE_MAX_WORDS", "900"))
STT_CONTEXT_LINES = int(os.getenv("STT_CONTEXT_LINES", "60"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_CHUNK_SECONDS = float(os.getenv("WHISPER_CHUNK_SECONDS", "3.5"))
WHISPER_CHUNK_OVERLAP_SECONDS = float(os.getenv("WHISPER_CHUNK_OVERLAP_SECONDS", "1.0"))
WHISPER_SAMPLE_RATE = int(os.getenv("WHISPER_SAMPLE_RATE", "16000"))
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ru").strip() or None
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_BEST_OF = int(os.getenv("WHISPER_BEST_OF", "5"))
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "1").strip().lower() not in {"0", "false", "no", "off"}
WHISPER_RETRY_WITHOUT_VAD = os.getenv("WHISPER_RETRY_WITHOUT_VAD", "1").strip().lower() in {"1", "true", "yes", "on"}
WHISPER_VAD_THRESHOLD = float(os.getenv("WHISPER_VAD_THRESHOLD", "0.20"))
WHISPER_VAD_MIN_SPEECH_MS = int(os.getenv("WHISPER_VAD_MIN_SPEECH_MS", "100"))
WHISPER_VAD_MIN_SILENCE_MS = int(os.getenv("WHISPER_VAD_MIN_SILENCE_MS", "650"))
WHISPER_VAD_SPEECH_PAD_MS = int(os.getenv("WHISPER_VAD_SPEECH_PAD_MS", "450"))
WHISPER_NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.75"))
WHISPER_LOG_PROB_THRESHOLD = float(os.getenv("WHISPER_LOG_PROB_THRESHOLD", "-2.0"))
WHISPER_COMPRESSION_RATIO_THRESHOLD = float(os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "3.0"))
WHISPER_REPETITION_PENALTY = float(os.getenv("WHISPER_REPETITION_PENALTY", "1.08"))
WHISPER_NO_REPEAT_NGRAM_SIZE = int(os.getenv("WHISPER_NO_REPEAT_NGRAM_SIZE", "3"))
WHISPER_HALLUCINATION_SILENCE_THRESHOLD = float(os.getenv("WHISPER_HALLUCINATION_SILENCE_THRESHOLD", "1.0"))
WHISPER_HOTWORDS = os.getenv(
    "WHISPER_HOTWORDS",
    "Kubernetes kubectl kubelet Deployment StatefulSet DaemonSet Pod Service Ingress ConfigMap Secret PVC "
    "Docker Dockerfile docker-compose GitLab CI Jenkins Terraform Ansible Prometheus Grafana Linux TCP UDP DNS TLS mTLS HTTPS",
).strip() or None
WHISPER_INITIAL_PROMPT = WHISPER_TECHNICAL_PROMPT
CUDA_WHISPER_ERROR_MARKERS = (
    "cuda",
    "cublas",
    "cublas64",
    "cudnn",
    "nvrtc",
    "ctranslate2",
)
STACKWIRE_API_URL = os.getenv("STACKWIRE_API_URL", "").strip().rstrip("/")
STACKWIRE_API_CONNECT_TIMEOUT = float(os.getenv("STACKWIRE_API_CONNECT_TIMEOUT", "5"))
STACKWIRE_API_TIMEOUT = float(os.getenv("STACKWIRE_API_TIMEOUT", "300"))
STACKWIRE_REMOTE_STT = os.getenv("STACKWIRE_REMOTE_STT", "1" if STACKWIRE_API_URL else "0").strip() == "1"
STACKWIRE_STT_TIMEOUT = float(os.getenv("STACKWIRE_STT_TIMEOUT", "120"))
STACKWIRE_HIDE_FROM_CAPTURE = os.getenv("STACKWIRE_HIDE_FROM_CAPTURE", "1").strip() == "1"
STACKWIRE_HIDE_TASKBAR = os.getenv("STACKWIRE_HIDE_TASKBAR", "1").strip() == "1"
MIN_UI_ZOOM = 0.75
MAX_UI_ZOOM = 1.55
ZOOM_STEP = 0.1
UI_ZOOM = 1.0

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
    text = str(exc).lower()
    return any(marker in text for marker in CUDA_WHISPER_ERROR_MARKERS)


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
        painter.setBrush(QColor(114, 214, 163, 34))
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
    elif kind == "close":
        painter.drawLine(int(s * 0.30), int(s * 0.30), int(s * 0.70), int(s * 0.70))
        painter.drawLine(int(s * 0.70), int(s * 0.30), int(s * 0.30), int(s * 0.70))

    painter.end()
    return pixmap


def make_icon(kind: str, size: int, color: str = TEXT) -> QIcon:
    return QIcon(icon_pixmap(kind, size, color))


def build_html_style() -> str:
    return f"""
<style>
body {{
  margin: 0;
  color: #eef3ff;
  font-family: Inter, Segoe UI, Arial, sans-serif;
  font-size: {_px(17)}px;
  line-height: 1.45;
}}
h2 {{
  margin: {_px(12)}px 0 {_px(7)}px;
  color: #ffffff;
  font-size: {_px(19)}px;
  font-weight: 800;
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
  background: rgba(122, 162, 255, 0.16);
  color: #dce7ff;
  font-family: JetBrains Mono, Consolas, monospace;
}}
strong {{
  color: #ffffff;
  font-weight: 800;
}}
.code-card {{
  margin: {_px(12)}px 0 {_px(14)}px;
  border: 1px solid rgba(122, 162, 255, 0.20);
  border-radius: {_px(8)}px;
  background: #111827;
  overflow: hidden;
}}
.code-head {{
  padding: {_px(7)}px {_px(14)}px;
  border-bottom: 1px solid rgba(122, 162, 255, 0.16);
  background: rgba(122, 162, 255, 0.10);
  color: #b8c7ef;
  font-size: {_px(11)}px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}}
pre {{
  margin: 0;
  padding: {_px(18)}px {_px(18)}px;
  color: #edf4ff;
  font-family: JetBrains Mono, Consolas, monospace;
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
  color: #d5bfff;
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
.message {{
  margin: 0 0 {_px(14)}px;
  padding: {_px(8)}px {_px(10)}px;
  border-radius: {_px(8)}px;
}}
.message.user {{
  margin-left: 8%;
  background: rgba(114, 214, 163, 0.10);
  border: 1px solid rgba(114, 214, 163, 0.18);
}}
.message.assistant {{
  margin-right: 0;
  background: rgba(122, 162, 255, 0.055);
  border: 1px solid rgba(185, 203, 255, 0.09);
}}
.role {{
  margin-bottom: {_px(8)}px;
  color: #9aa8bf;
  font-size: {_px(12)}px;
  font-weight: 800;
  text-transform: uppercase;
}}
.question-tag {{
  display: inline-block;
  margin: 0 0 {_px(10)}px;
  padding: {_px(4)}px {_px(9)}px;
  border-radius: {_px(7)}px;
  background: rgba(114, 214, 163, 0.18);
  border: 1px solid rgba(114, 214, 163, 0.32);
  color: #cbf7dd;
  font-size: {_px(12)}px;
  font-weight: 850;
}}
.screenshot-card {{
  margin-top: {_px(10)}px;
  border: 1px solid rgba(185, 203, 255, 0.16);
  border-radius: {_px(10)}px;
  background: rgba(5, 8, 14, 0.65);
  overflow: hidden;
}}
.screenshot-card img {{
  display: block;
  max-width: 100%;
  max-height: {_px(300)}px;
  object-fit: contain;
}}
</style>
"""


def markdown_to_html(markdown: str) -> str:
    markdown = normalize_unfenced_code_blocks(markdown)
    parts: list[str] = []
    pattern = re.compile(r"```([a-zA-Z0-9_.+-]*)\n(.*?)```", re.DOTALL)
    cursor = 0

    for match in pattern.finditer(markdown):
        parts.append(text_to_html(markdown[cursor : match.start()]))
        raw_language = match.group(1) or "code"
        language = html.escape(raw_language)
        code = highlight_code(raw_language, match.group(2).strip("\n"))
        parts.append(
            f"""
            <div class="code-card">
              <div class="code-head">{language}</div>
              <pre>{code}</pre>
            </div>
            """
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
    screenshot_pattern = re.compile(r"\[\[screenshot:([A-Za-z0-9+/=]+)\]\]")
    screenshots = screenshot_pattern.findall(markdown)
    text = screenshot_pattern.sub("", markdown).strip()
    fragments = [markdown_fragment(text)] if text else []
    for screenshot in screenshots:
        safe_src = html.escape(screenshot, quote=True)
        fragments.append(
            f"""
            <div class="screenshot-card">
              <img src="data:image/png;base64,{safe_src}" />
            </div>
            """
        )
    return "".join(fragments)


def render_user_message_fragment(markdown: str) -> str:
    match = re.match(r"^(Вопрос\s+\d+)\s*\n\n(.+)$", markdown.strip(), flags=re.DOTALL)
    if not match:
        return render_message_fragment(markdown)
    tag = html.escape(match.group(1))
    body = match.group(2).strip()
    return f'<div class="question-tag">{tag}</div>' + render_message_fragment(body)


def text_to_html(text: str) -> str:
    escaped = html.escape(text.strip())
    if not escaped:
        return ""

    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
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


class AnswerBrowser(QTextBrowser):
    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            event.accept()
            return
        super().wheelEvent(event)


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


def _remote_request_error(prefix: str, api_url: str, exc: RequestException) -> str:
    if not api_url:
        return (
            f"{prefix}: local Ollama is not available at 127.0.0.1:11434. "
            "Start Ollama, or run start_client.bat <SERVER_IP> <PORT> to use remote StackWire API. "
            "If Ollama is installed on this PC, run: ollama serve. "
            f"Details: {exc}"
        )

    return (
        f"{prefix}: remote StackWire API is unavailable at {api_url}. "
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
            headers={"Content-Type": "application/json; charset=utf-8"},
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
            painter.setPen(QPen(QColor(114, 214, 163, 210), 2))
            painter.setBrush(QColor(114, 214, 163, 26))
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
        configured_device = (WHISPER_DEVICE or "auto").strip().lower()
        configured_compute = (WHISPER_COMPUTE_TYPE or "float16").strip().lower()

        if configured_device == "auto":
            cuda_compute = "float16" if configured_compute in {"", "auto", "int8"} else configured_compute
            return [("cpu", "int8"), ("cuda", cuda_compute)]

        if configured_device in {"cuda", "gpu"}:
            attempts = [("cuda", configured_compute or "float16")]
            if STT_ALLOW_CPU_WHISPER_FALLBACK:
                attempts.append(("cpu", "int8"))
            return attempts

        if configured_device == "cpu":
            cpu_compute = "int8" if configured_compute in {"", "auto", "float16"} else configured_compute
            return [("cpu", cpu_compute)]

        return [(configured_device, configured_compute or "float16")]

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
                return whisper_model_class(
                    WHISPER_MODEL,
                    device=device,
                    compute_type=compute_type,
                )
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
        return {
            "threshold": WHISPER_VAD_THRESHOLD,
            "min_speech_duration_ms": WHISPER_VAD_MIN_SPEECH_MS,
            "min_silence_duration_ms": WHISPER_VAD_MIN_SILENCE_MS,
            "speech_pad_ms": WHISPER_VAD_SPEECH_PAD_MS,
        }

    def _transcribe_local_whisper_audio(self, model, audio, *, vad_filter: bool) -> tuple[str, dict[str, float | str]]:  # noqa: ANN001
        segments, info = model.transcribe(
            audio,
            language=WHISPER_LANGUAGE,
            task="transcribe",
            beam_size=WHISPER_BEAM_SIZE,
            best_of=WHISPER_BEST_OF,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=WHISPER_INITIAL_PROMPT,
            vad_filter=vad_filter,
            vad_parameters=self._whisper_vad_parameters() if vad_filter else None,
            no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
            log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
            compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
            repetition_penalty=WHISPER_REPETITION_PENALTY,
            no_repeat_ngram_size=WHISPER_NO_REPEAT_NGRAM_SIZE,
            hallucination_silence_threshold=WHISPER_HALLUCINATION_SILENCE_THRESHOLD,
            hotwords=WHISPER_HOTWORDS,
        )
        segment_list = list(segments)
        text = " ".join(segment.text.strip() for segment in segment_list).strip()
        diagnostics: dict[str, float | str] = {
            "language": str(getattr(info, "language", WHISPER_LANGUAGE or "")),
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

        text = clean_stt_output(text)
        bad_text = bad_text or self._is_bad_whisper_text(text, diagnostics, rms)
        latency_ms = (time.perf_counter() - started) * 1000
        LOGGER.info(
            "stt_latency_ms=%.0f language=%s vad=%s rms=%.6f avg_logprob=%.2f no_speech=%.2f compression=%.2f",
            latency_ms,
            diagnostics.get("language", ""),
            diagnostics.get("vad", ""),
            rms,
            float(diagnostics.get("avg_logprob") or 0.0),
            float(diagnostics.get("no_speech_prob") or 0.0),
            float(diagnostics.get("compression_ratio") or 0.0),
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
        text = str(data.get("text", "")).strip()
        text = clean_stt_output(text)
        LOGGER.info("remote stt_latency_ms=%.0f", latency_ms)
        self.stt_latency.emit(latency_ms)
        if is_probable_stt_hallucination(text):
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
        self.ask_worker: AskWorker | None = None
        self.expand_thread: QThread | None = None
        self.expand_worker: ExpandWorker | None = None
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
        self.last_question_candidate = ""
        self.pending_capture_b64 = ""
        self.visibility_hotkey_down = False
        self.record_hotkey_down = False
        self.submit_after_speech_stop = False
        self.speech_input_locked = False
        self.debug_expanded = False
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
            self.resize(860, 520)
        else:
            self.resize(1040, 680)
        self._build_ui()
        self._load_audio_devices()
        QTimer.singleShot(0, self.apply_capture_exclusion)
        
    def closeEvent(self, event) -> None:
        if self._closing:
            event.accept()
            return
        self._closing = True
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

        speech_thread = self.speech_thread
        self.speech_thread = None
        self.speech_worker = None

        self._shutdown_thread(speech_thread, "speech")

        ask_thread = self.ask_thread
        if self.ask_worker:
            try:
                self.ask_worker.session.close()
                if self.ask_worker.client is not None:
                    self.ask_worker.client.session.close()
            except RuntimeError:
                pass
        self.ask_thread = None
        self.ask_worker = None

        self._shutdown_thread(ask_thread, "ask")

        expand_thread = self.expand_thread
        if self.expand_worker:
            try:
                self.expand_worker.session.close()
                if self.expand_worker.client is not None:
                    self.expand_worker.client.session.close()
            except RuntimeError:
                pass
        self.expand_thread = None
        self.expand_worker = None

        self._shutdown_thread(expand_thread, "expand")

        image_thread = self.image_thread
        if self.image_worker:
            try:
                self.image_worker.session.close()
                if self.image_worker.client is not None:
                    self.image_worker.client.session.close()
            except RuntimeError:
                pass
        self.image_thread = None
        self.image_worker = None

        self._shutdown_thread(image_thread, "image")

        event.accept()
        app = QApplication.instance()
        if app is not None:
            QTimer.singleShot(0, lambda: app.exit(0))

    def _log_client_event(self, event_name: str) -> None:
        details = {
            "api_url": STACKWIRE_API_URL or "local",
            "answer_model": MODEL,
            "recovery_model": RECOVERY_MODEL,
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

    def _shutdown_thread(self, thread: QThread | None, name: str, timeout_ms: int = 900) -> None:
        if thread is None:
            return
        try:
            if not thread.isRunning():
                return
            thread.requestInterruption()
            thread.quit()
            if thread.wait(timeout_ms):
                return
            LOGGER.warning("%s thread did not stop in %sms; terminating", name, timeout_ms)
            thread.terminate()
            thread.wait(1000)
        except RuntimeError:
            pass
        
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        shell = QFrame()
        shell.setObjectName("shell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(14, 12, 14, 12)
        shell_layout.setSpacing(9)

        header = QHBoxLayout()
        header.setSpacing(10)

        self.title_mark = QLabel()
        self.title_mark.setObjectName("titleMark")

        self.title = QLabel(APP_NAME)
        self.title.setObjectName("title")
        self.subtitle = QLabel("")
        self.subtitle.setObjectName("subtitle")
        self.subtitle.setVisible(False)

        title_line = QHBoxLayout()
        title_line.setSpacing(8)
        title_line.addWidget(self.title_mark)
        title_line.addWidget(self.title)
        title_line.addStretch(1)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title_box.addLayout(title_line)
        title_box.addWidget(self.subtitle)

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

        self.close_button = QPushButton()
        self.close_button.setObjectName("closeButton")
        self.close_button.clicked.connect(self.close)

        header.addLayout(title_box, 1)
        header.addWidget(self.device_combo)
        header.addWidget(self.listen_button)
        header.addWidget(self.clear_button)
        header.addWidget(self.capture_button)
        header.addWidget(self.debug_button)
        header.addWidget(self.close_button)

        self.answer = AnswerBrowser()
        self.answer.setObjectName("answer")
        self.answer.setOpenExternalLinks(True)
        self.answer.setMinimumHeight(280)

        answer_actions = QHBoxLayout()
        answer_actions.setSpacing(8)

        self.expand_button = QPushButton()
        self.expand_button.setObjectName("iconButton")
        self.expand_button.setToolTip("Expand answer")
        self.expand_button.clicked.connect(self.show_expand_popup)
        self.expand_popup = ActionPopup(self, EXPAND_MENU_ITEMS, self.submit_expand)

        self.actions_button = QPushButton()
        self.actions_button.setObjectName("iconButton")
        self.actions_button.setToolTip("Answer actions")
        self.actions_button.clicked.connect(self.show_answer_actions_popup)
        self.answer_actions_popup = ActionPopup(
            self,
            (
                ("good", "Good"),
                ("bad", "Bad"),
                ("save", "Save"),
            ),
            self.handle_answer_action,
        )

        answer_actions.addWidget(self.expand_button)
        answer_actions.addWidget(self.actions_button)
        answer_actions.addStretch(1)

        footer = QHBoxLayout()
        footer.setSpacing(10)

        self.input = PromptEdit()
        self.input.setObjectName("prompt")
        self.input.setPlaceholderText("Введи вопрос и нажми Enter...")
        self.input.setFixedHeight(54)
        self.input.submitted.connect(self.submit_question)

        self.ask_button = QPushButton()
        self.ask_button.setObjectName("askButton")
        self.ask_button.setToolTip("Ask")
        self.ask_button.setFixedSize(54, 54)
        self.ask_button.clicked.connect(self.submit_question)

        footer.addWidget(self.input, 1)
        footer.addWidget(self.ask_button)

        self.status = QLabel("Ready")
        self.status.setObjectName("status")

        self.debug_panel = QLabel()
        self.debug_panel.setObjectName("debugPanel")
        self.debug_panel.setWordWrap(True)
        self.debug_panel.setMaximumHeight(140)
        self.debug_panel.setVisible(False)

        bottom = QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(QSizeGrip(self), 0, Qt.AlignmentFlag.AlignRight)

        shell_layout.addLayout(header)
        shell_layout.addWidget(self.answer, 1)
        shell_layout.addLayout(answer_actions)
        shell_layout.addLayout(footer)
        shell_layout.addWidget(self.debug_panel)
        shell_layout.addLayout(bottom)
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

        font = QFont("Segoe UI")
        font.setPointSizeF(10 * self.ui_zoom)
        self.setFont(font)

        self.answer.setMinimumHeight(_px(280))
        self.input.setFixedHeight(_px(54))
        self.input.keep_arrow_cursor()
        self.ask_button.setFixedSize(_px(54), _px(54))
        self.debug_panel.setMaximumHeight(_px(140))
        self.setStyleSheet(build_window_styles(self.ui_zoom))
        self.apply_icons()

    def apply_icons(self) -> None:
        icon_size = _px(16)
        self.title_mark.setPixmap(icon_pixmap("mark", _px(20), ACCENT))
        listening = self._speech_is_running()
        self.listen_button.setIcon(make_icon("stop" if listening else "listen", icon_size, "#ff8f8f" if listening else "#c4cee4"))
        self.listen_button.setToolTip("Stop listening" if listening else "Listen")
        self.clear_button.setIcon(make_icon("clear", icon_size, "#aeb8cc"))
        self.capture_button.setIcon(make_icon("capture", icon_size, "#aeb8cc"))
        self.debug_button.setIcon(make_icon("debug", icon_size, "#aeb8cc"))
        self.expand_button.setIcon(make_icon("expand", icon_size, "#c4cee4"))
        self.actions_button.setIcon(make_icon("actions", icon_size, "#c4cee4"))
        self.ask_button.setIcon(make_icon("ask", icon_size, "#07130d"))
        self.close_button.setIcon(make_icon("close", icon_size, "#7f8797"))
        for button in (
            self.listen_button,
            self.clear_button,
            self.capture_button,
            self.debug_button,
            self.expand_button,
            self.actions_button,
            self.ask_button,
            self.close_button,
        ):
            button.setIconSize(QSize(icon_size, icon_size))
        for button in (
            self.listen_button,
            self.clear_button,
            self.capture_button,
            self.debug_button,
            self.expand_button,
            self.actions_button,
        ):
            button.setFixedSize(_px(30), _px(30))
        self.close_button.setFixedSize(_px(30), _px(30))

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

        self.pending_capture_b64 = image_b64
        self.speech_input_locked = True
        self.auto_ask_timer.stop()
        self.current_partial_speech = ""
        self.question_count += 1
        self.chat_messages.append(
            (
                "user",
                f"Скриншот {self.question_count}\n\n[[screenshot:{image_b64}]]\n\n"
                "Добавь запрос к скриншоту в поле ниже и нажми Enter.",
            )
        )
        self.render_chat(focus_latest_assistant=True)
        self.input.setPlainText("Что изображено на скриншоте и что важно?")
        self.input.setFocus()
        self.input.selectAll()
        self.capture_button.setEnabled(True)
        self.status.setText("Screenshot captured. Add prompt and press Enter.")

    def submit_capture_question(self, prompt: str) -> None:
        image_b64 = self.pending_capture_b64
        if not image_b64:
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

    def submit_question(self) -> None:
        if not self.ask_button.isEnabled():
            return

        question = self.input.toPlainText().strip()
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
        self.status.setText("Recovering question...")
        self.question_count += 1
        self.chat_messages.append(("user", f"Вопрос {self.question_count}\n\n{question}"))
        self.chat_messages.append(("assistant", "Генерирую ответ..."))
        self.input.clear()
        self.last_final_speech = ""
        self.current_partial_speech = ""
        self.last_question_candidate = ""
        self.render_chat(focus_latest_assistant=True)

        self.ask_thread = QThread()
        self.ask_worker = AskWorker(question, context, trusted_text=trusted_text, storage_session_id=self.storage_session_id)
        self.ask_worker.moveToThread(self.ask_thread)
        self.ask_thread.started.connect(self.ask_worker.run)
        self.ask_worker.finished.connect(self.show_answer)
        self.ask_worker.failed.connect(self.show_error)
        self.ask_worker.finished.connect(self.ask_thread.quit)
        self.ask_worker.failed.connect(self.ask_thread.quit)
        self.ask_worker.finished.connect(self.ask_worker.deleteLater)
        self.ask_worker.failed.connect(self.ask_worker.deleteLater)
        self.ask_thread.finished.connect(self.ask_thread.deleteLater)
        self.ask_thread.start()

    def update_answer_actions(self, force_disabled: bool = False) -> None:
        has_answer = bool(self.last_answer_text.strip())
        has_question = bool(self.last_answer_question.strip())
        busy = force_disabled or self.expand_thread is not None or not self.ask_button.isEnabled()
        self.expand_button.setEnabled(has_answer and has_question and not busy)
        self.actions_button.setEnabled(has_answer and has_question and not busy)

    def show_expand_popup(self) -> None:
        if self.expand_button.isEnabled():
            self.expand_popup.show_below(self.expand_button)

    def show_answer_actions_popup(self) -> None:
        if self.actions_button.isEnabled():
            self.answer_actions_popup.show_below(self.actions_button)

    def handle_answer_action(self, action: str) -> None:
        if action == "save":
            self.save_current_good_answer()
        elif action in {"good", "bad"}:
            self.submit_feedback(action)

    def submit_expand(self, mode: str) -> None:
        if self.expand_thread is not None:
            return
        if not self.last_answer_question.strip() or not self.last_answer_text.strip():
            self.status.setText("No answer to expand.")
            return

        mode = mode if mode in EXPAND_LABELS else "details"
        header = EXPAND_LABELS[mode]
        self.chat_messages.append(("assistant", f"{header}\n\nГенерирую расширение..."))
        self.render_chat(focus_latest_assistant=True)
        self.status.setText("Expanding answer...")
        self.update_answer_actions(force_disabled=True)

        self.expand_thread = QThread()
        self.expand_worker = ExpandWorker(self.last_answer_question, self.last_answer_text, mode, self.storage_session_id)
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

        candidate = (
            self.input.toPlainText().strip()
            or self.last_question_candidate
            or self.last_final_speech
            or " ".join(self.transcript_lines[-STT_CONTEXT_LINES:]).strip()
        )
        if not candidate:
            self.status.setText("No speech captured.")
            return

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
                self.last_answer_id = None
                self.last_answer_domain = None
                self.last_answer_intent = None
            else:
                self.last_answer_question = recovered_question or result.raw_text
                self.last_answer_text = cleaned
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
        self.replace_last_assistant(f"Ошибка: {message}")
        self.ask_button.setEnabled(True)
        self.listen_button.setEnabled(True)
        self.capture_button.setEnabled(True)
        if self.pending_capture_b64:
            self.input.setFocus()
        self.status.setText("Error")
        self.update_answer_actions()

    def toggle_debug_panel(self) -> None:
        self.debug_expanded = self.debug_button.isChecked()
        self.debug_panel.setVisible(self.debug_expanded)
        self.debug_button.setToolTip("Hide debug" if self.debug_expanded else "Debug")

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
            f"answer_model: {MODEL}\n"
            f"recovery_model: {RECOVERY_MODEL}\n"
            f"vision_model: {VISION_MODEL}\n"
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
        self.input.clear()
        self.chat_messages.clear()
        self.raw_transcript_lines.clear()
        self.transcript_lines.clear()
        self.last_question_candidate = ""
        self.current_partial_speech = ""
        self.pending_capture_b64 = ""
        self.speech_input_locked = False
        self.submit_after_speech_stop = False
        self.last_stt_latency_ms = None
        self.last_recovery_latency_ms = None
        self.last_answer_latency_ms = None
        self.last_total_latency_ms = None
        self.last_answer_question = ""
        self.last_answer_text = ""
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

    def render_chat(self, focus_latest_assistant: bool = False) -> None:
        if not self.chat_messages:
            self.answer.setHtml(
                markdown_to_html(
            (
                "Введите вопрос и нажмите Enter.\n\n"
                "Гайд:\n"
                f"- Модель: {VISION_MODEL}\n"
                "- Помогает разбирать технические темы, команды, конфиги, ошибки и практические сценарии.\n"
                "- Может показать короткий пример, если явно попросить код, конфиг или команду.\n"
                "- Для диагностики формирует возможные причины, проверки и варианты исправления.\n"
            )
                    )
            )
            return

        chunks: list[str] = ["<html><head>", build_html_style(), build_chat_style(), "</head><body>"]
        latest_assistant_anchor = ""
        for index, (role, content) in enumerate(self.chat_messages):
            label = "Вы" if role == "user" else "Ассистент"
            anchor = f"msg-{index}"
            if role == "assistant":
                latest_assistant_anchor = anchor
            chunks.append(f'<a name="{anchor}"></a><section class="message {role}"><div class="role">{label}</div>')
            chunks.append(render_user_message_fragment(content) if role == "user" else render_message_fragment(content))
            chunks.append("</section>")
        chunks.append("</body></html>")
        self.answer.setHtml("".join(chunks))
        if focus_latest_assistant and latest_assistant_anchor:
            self.answer.scrollToAnchor(latest_assistant_anchor)

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
    background: {PANEL};
    border: 1px solid rgba(196, 211, 255, 0.14);
    border-radius: {_px(12, scale)}px;
}}

QLabel#title {{
    color: {TEXT};
    font-size: {_px(16, scale)}px;
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

QLabel#debugPanel {{
    color: #aebbe0;
    background: rgba(8, 11, 18, 120);
    border: 1px solid rgba(188, 204, 255, 0.10);
    border-radius: {_px(10, scale)}px;
    padding: {_px(6, scale)}px {_px(8, scale)}px;
    font-family: JetBrains Mono, Consolas, monospace;
    font-size: {_px(10, scale)}px;
}}

QTextBrowser#answer {{
    background: rgba(7, 10, 16, 104);
    border: 1px solid rgba(188, 204, 255, 0.09);
    border-radius: {_px(10, scale)}px;
    padding: {_px(12, scale)}px;
    selection-background-color: {BLUE};
}}

QTextEdit#prompt {{
    background: {PANEL_LIGHT};
    border: 1px solid rgba(188, 204, 255, 0.12);
    border-radius: {_px(10, scale)}px;
    color: {TEXT};
    padding: {_px(8, scale)}px {_px(10, scale)}px;
    font-size: {_px(15, scale)}px;
    selection-background-color: {BLUE};
}}

QTextEdit#prompt:focus {{
    border: 1px solid rgba(114, 214, 163, 0.75);
}}

QPushButton {{
    min-height: {_px(30, scale)}px;
    border-radius: {_px(9, scale)}px;
    padding: 0 {_px(11, scale)}px;
    font-size: {_px(12, scale)}px;
    font-weight: 760;
}}

QPushButton#askButton {{
    min-width: {_px(54, scale)}px;
    max-width: {_px(54, scale)}px;
    min-height: {_px(54, scale)}px;
    max-height: {_px(54, scale)}px;
    padding: 0;
    color: #07130d;
    background: {ACCENT};
    border: 0;
}}

QPushButton#askButton:disabled {{
    color: rgba(238, 243, 255, 0.5);
    background: rgba(114, 214, 163, 0.28);
}}

QPushButton#ghostButton {{
    color: {TEXT};
    background: rgba(31, 38, 54, 126);
    border: 1px solid rgba(188, 204, 255, 0.11);
}}

QPushButton#ghostButton:hover {{
    border: 1px solid rgba(114, 214, 163, 0.58);
}}

QPushButton#iconButton {{
    min-width: {_px(30, scale)}px;
    max-width: {_px(30, scale)}px;
    min-height: {_px(30, scale)}px;
    max-height: {_px(30, scale)}px;
    padding: 0;
    color: {TEXT};
    background: rgba(31, 38, 54, 112);
    border: 1px solid rgba(188, 204, 255, 0.10);
    border-radius: {_px(9, scale)}px;
}}

QPushButton#iconButton:hover {{
    background: rgba(43, 52, 72, 150);
    border: 1px solid rgba(114, 214, 163, 0.45);
}}

QPushButton#iconButton:checked {{
    background: rgba(122, 162, 255, 0.16);
    border: 1px solid rgba(122, 162, 255, 0.30);
}}

QFrame#actionPopup {{
    background: rgba(18, 24, 38, 245);
    border: 1px solid rgba(188, 204, 255, 0.18);
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
    background: rgba(122, 162, 255, 0.14);
    border: 1px solid rgba(122, 162, 255, 0.22);
}}

QPushButton#closeButton {{
    min-width: {_px(30, scale)}px;
    max-width: {_px(30, scale)}px;
    min-height: {_px(30, scale)}px;
    max-height: {_px(30, scale)}px;
    padding: 0;
    color: #7f8797;
    background: rgba(22, 26, 36, 145);
    border: 1px solid rgba(188, 204, 255, 0.10);
    border-radius: {_px(9, scale)}px;
}}

QPushButton#closeButton:hover {{
    background: rgba(52, 58, 72, 180);
    border: 1px solid rgba(188, 204, 255, 0.18);
}}

QComboBox#deviceCombo {{
    min-width: {_px(230, scale)}px;
    min-height: {_px(30, scale)}px;
    border-radius: {_px(9, scale)}px;
    padding: 0 {_px(10, scale)}px;
    color: {TEXT};
    background: rgba(31, 38, 54, 126);
    border: 1px solid rgba(188, 204, 255, 0.11);
}}

QComboBox QAbstractItemView {{
    color: {TEXT};
    background: #151b28;
    selection-background-color: #2e4c7d;
}}
"""


STYLES = build_window_styles()


def main() -> int:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    app.setFont(QFont("Segoe UI", 10))
    app.setWindowIcon(make_icon("mark", 32, ACCENT))
    window = OverlayWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
