import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

APP_NAME = "StackWire"
ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_ENV_FILE = ROOT_DIR / "stackwire.local.env"


def load_local_env() -> None:
    if LOCAL_ENV_FILE.exists():
        load_dotenv(LOCAL_ENV_FILE, override=False)


DEFAULT_WHISPER_HOTWORDS = ""


@dataclass(frozen=True)
class STTSettings:
    backend: str
    allow_vosk_fallback: bool
    allow_cpu_whisper_fallback: bool
    mic_signal_threshold: float
    loopback_signal_threshold: float
    probe_loopback_devices: bool
    live_max_words: int
    context_lines: int
    model: str
    device: str
    compute_type: str
    chunk_seconds: float
    chunk_overlap_seconds: float
    sample_rate: int
    language_mode: str
    beam_size: int
    best_of: int
    vad_filter: bool
    retry_without_vad: bool
    vad_threshold: float
    vad_min_speech_ms: int
    vad_min_silence_ms: int
    vad_speech_pad_ms: int
    no_speech_threshold: float
    log_prob_threshold: float
    compression_ratio_threshold: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    hallucination_silence_threshold: float
    hotwords: str | None


def get_stt_settings() -> STTSettings:
    # Recognition language: an explicit STT override wins; otherwise follow the UI
    # Language setting (View tab) so picking Russian/English there also pins Whisper's
    # decoding language — this kills the auto-detect flips that used to mis-hear short
    # Russian phrases as English. Falls back to "auto" when nothing is set.
    language_mode = _language_mode(
        os.getenv("STT_LANGUAGE_MODE")
        or os.getenv("WHISPER_LANGUAGE")
        or os.getenv("STACKWIRE_UI_LANGUAGE")
        or "auto"
    )
    return STTSettings(
        backend=os.getenv("STT_BACKEND", "whisper").strip().lower(),
        allow_vosk_fallback=_env_bool("STT_ALLOW_VOSK_FALLBACK", False),
        allow_cpu_whisper_fallback=_env_bool("STT_ALLOW_CPU_WHISPER_FALLBACK", True),
        mic_signal_threshold=_env_float("STT_MIC_SIGNAL_THRESHOLD", 0.003),
        loopback_signal_threshold=_env_float("STT_LOOPBACK_SIGNAL_THRESHOLD", 0.00025),
        probe_loopback_devices=_env_bool("STT_PROBE_LOOPBACK_DEVICES", False),
        live_max_words=_env_int("STT_LIVE_MAX_WORDS", 900),
        context_lines=_env_int("STT_CONTEXT_LINES", 60),
        model=os.getenv("WHISPER_MODEL", "large-v3-turbo").strip() or "large-v3-turbo",
        device=os.getenv("WHISPER_DEVICE", "auto").strip() or "auto",
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "auto").strip() or "auto",
        # 2.5s chunks: GhostGPT-class reactivity. large-v3-turbo on GPU transcribes
        # this far faster than real time; raise via env if running CPU-only.
        chunk_seconds=_env_float("WHISPER_CHUNK_SECONDS", 2.5),
        chunk_overlap_seconds=_env_float("WHISPER_CHUNK_OVERLAP_SECONDS", 0.8),
        sample_rate=_env_int("WHISPER_SAMPLE_RATE", 16000),
        language_mode=language_mode,
        beam_size=_env_int("WHISPER_BEAM_SIZE", 5),
        best_of=_env_int("WHISPER_BEST_OF", 5),
        vad_filter=_env_bool("WHISPER_VAD_FILTER", True),
        retry_without_vad=_env_bool("WHISPER_RETRY_WITHOUT_VAD", True),
        vad_threshold=_env_float("WHISPER_VAD_THRESHOLD", 0.20),
        vad_min_speech_ms=_env_int("WHISPER_VAD_MIN_SPEECH_MS", 100),
        vad_min_silence_ms=_env_int("WHISPER_VAD_MIN_SILENCE_MS", 650),
        vad_speech_pad_ms=_env_int("WHISPER_VAD_SPEECH_PAD_MS", 450),
        no_speech_threshold=_env_float("WHISPER_NO_SPEECH_THRESHOLD", 0.75),
        log_prob_threshold=_env_float("WHISPER_LOG_PROB_THRESHOLD", -2.0),
        compression_ratio_threshold=_env_float("WHISPER_COMPRESSION_RATIO_THRESHOLD", 3.0),
        repetition_penalty=_env_float("WHISPER_REPETITION_PENALTY", 1.08),
        no_repeat_ngram_size=_env_int("WHISPER_NO_REPEAT_NGRAM_SIZE", 3),
        hallucination_silence_threshold=_env_float("WHISPER_HALLUCINATION_SILENCE_THRESHOLD", 1.0),
        hotwords=os.getenv("WHISPER_HOTWORDS", DEFAULT_WHISPER_HOTWORDS).strip() or None,
    )


def whisper_language(settings: STTSettings, requested_language: str | None = None, locked_language: str | None = None) -> str | None:
    requested = normalize_stt_language(requested_language)
    if requested:
        return requested
    if settings.language_mode == "auto":
        return normalize_stt_language(locked_language)
    return settings.language_mode


def update_stt_language_lock(
    settings: STTSettings,
    current: str | None,
    detected_language: str | None,
    text: str,
    *,
    bad_text: bool,
) -> str | None:
    if settings.language_mode != "auto" or current:
        return current
    detected = normalize_stt_language(detected_language)
    if detected and text.strip() and not bad_text:
        return detected
    return current


def whisper_model_attempts(settings: STTSettings) -> list[tuple[str, str]]:
    configured_device = (settings.device or "auto").strip().lower()
    configured_compute = (settings.compute_type or "float16").strip().lower()

    if configured_device == "auto":
        cuda_compute = "float16" if configured_compute in {"", "auto", "int8"} else configured_compute
        return [("cuda", cuda_compute), ("cpu", "int8")]

    if configured_device in {"cuda", "gpu"}:
        attempts = [("cuda", configured_compute or "float16")]
        if settings.allow_cpu_whisper_fallback:
            attempts.append(("cpu", "int8"))
        return attempts

    if configured_device == "cpu":
        cpu_compute = "int8" if configured_compute in {"", "auto", "float16"} else configured_compute
        return [("cpu", cpu_compute)]

    return [(configured_device, configured_compute or "float16")]


def whisper_vad_parameters(settings: STTSettings) -> dict[str, int | float]:
    return {
        "threshold": settings.vad_threshold,
        "min_speech_duration_ms": settings.vad_min_speech_ms,
        "min_silence_duration_ms": settings.vad_min_silence_ms,
        "speech_pad_ms": settings.vad_speech_pad_ms,
    }


def is_cuda_whisper_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in ("cuda", "cublas", "cublas64", "cudnn", "nvrtc", "ctranslate2"))


def normalize_stt_language(language: str | None) -> str | None:
    value = (language or "").strip().lower()
    if value in {"ru", "russian", "русский"}:
        return "ru"
    if value in {"en", "english", "английский"}:
        return "en"
    return None


def _language_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"auto", "", "mixed", "ru-en", "ru_en"}:
        return "auto"
    return normalize_stt_language(normalized) or "auto"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default
