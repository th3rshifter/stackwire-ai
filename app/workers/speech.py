from __future__ import annotations

import base64
import json
import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import requests
from PySide6.QtCore import QObject, Signal, Slot

from app.config import (
    get_stt_settings,
    is_cuda_whisper_error,
    update_stt_language_lock,
    whisper_language,
    whisper_model_attempts,
    whisper_vad_parameters,
)
from app.tech_terms import WHISPER_TECHNICAL_PROMPT
from app.transcript_repair import clean_stt_output, is_probable_stt_hallucination

LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
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
WHISPER_MODEL = STT_SETTINGS.model
WHISPER_CHUNK_SECONDS = STT_SETTINGS.chunk_seconds
WHISPER_CHUNK_OVERLAP_SECONDS = STT_SETTINGS.chunk_overlap_seconds
WHISPER_SAMPLE_RATE = STT_SETTINGS.sample_rate
WHISPER_VAD_FILTER = STT_SETTINGS.vad_filter
WHISPER_RETRY_WITHOUT_VAD = STT_SETTINGS.retry_without_vad
WHISPER_HOTWORDS = STT_SETTINGS.hotwords
# The technical initial prompt biases Whisper toward DevOps terms — in silence/
# unclear audio it HALLUCINATES "nginx/tls/web server" and hurts English. Off by
# default; set STT_TECHNICAL_PROMPT=1 to re-enable tech-term biasing.
STT_USE_TECH_PROMPT = os.getenv("STT_TECHNICAL_PROMPT", "0").strip().lower() in {"1", "true", "yes", "on"}
WHISPER_INITIAL_PROMPT = WHISPER_TECHNICAL_PROMPT if STT_USE_TECH_PROMPT else ""
# Live (real-time) partials: re-transcribe the in-progress buffer every ~0.7s so words
# appear as they are spoken (GhostGPT-style) instead of only after a full ~3.5s chunk.
WHISPER_LIVE_PARTIALS = os.getenv("WHISPER_LIVE_PARTIALS", "1").strip().lower() not in {"0", "false", "no", "off"}
WHISPER_PARTIAL_SECONDS = float(os.getenv("WHISPER_PARTIAL_SECONDS", "0.7") or "0.7")

# Runtime values injected from desktop.py via configure_speech_worker().
STACKWIRE_API_URL = ""
STACKWIRE_API_CONNECT_TIMEOUT = 5.0
STACKWIRE_REMOTE_STT = False
STACKWIRE_STT_TIMEOUT = 120.0


def configure_speech_worker(*, api_url, api_connect_timeout, stt_timeout, remote_stt) -> None:  # noqa: ANN001
    global STACKWIRE_API_URL, STACKWIRE_API_CONNECT_TIMEOUT, STACKWIRE_STT_TIMEOUT, STACKWIRE_REMOTE_STT
    STACKWIRE_API_URL = api_url
    STACKWIRE_API_CONNECT_TIMEOUT = api_connect_timeout
    STACKWIRE_STT_TIMEOUT = stt_timeout
    STACKWIRE_REMOTE_STT = remote_stt


def _is_cuda_whisper_error(exc: BaseException) -> bool:
    return is_cuda_whisper_error(exc)


def _short_error(exc: BaseException, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(exc)).strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


# Whisper model is cached at module level so it loads ONCE per process (not on every
# listen session, which reloaded multiple GB into VRAM each time) and can be preloaded
# in the background at startup. Inference across the shared model is serialized below.
STACKWIRE_PRELOAD_STT = os.getenv("STACKWIRE_PRELOAD_STT", "0").strip().lower() not in {"0", "false", "no", "off"}
_SHARED_WHISPER_MODEL: "Any" = None
_SHARED_WHISPER_LOAD_LOCK = threading.Lock()
_SHARED_WHISPER_INFER_LOCK = threading.Lock()


def _warmup_model(model: "Any", device: str) -> None:
    if device != "cuda":
        return
    try:
        import numpy as np
    except ImportError:
        return
    try:
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
    except Exception:  # noqa: BLE001
        LOGGER.debug("whisper warmup skipped", exc_info=True)


def _build_whisper_model(on_status=None) -> "Any":  # noqa: ANN001
    """Load + warm the WhisperModel per the configured device/compute attempts, caching
    it at module level so subsequent listen sessions reuse it instantly."""
    global _SHARED_WHISPER_MODEL
    if _SHARED_WHISPER_MODEL is not None:
        return _SHARED_WHISPER_MODEL
    with _SHARED_WHISPER_LOAD_LOCK:
        if _SHARED_WHISPER_MODEL is not None:
            return _SHARED_WHISPER_MODEL
        from faster_whisper import WhisperModel

        attempts = whisper_model_attempts(STT_SETTINGS)
        last_exc: "Exception | None" = None
        for attempt_index, (device, compute_type) in enumerate(attempts):
            try:
                if on_status is not None:
                    on_status("Загружаю модель распознавания…")
                LOGGER.info("STT backend=whisper model=%s device=%s compute_type=%s", WHISPER_MODEL, device, compute_type)
                model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
                _warmup_model(model, device)
                _SHARED_WHISPER_MODEL = model
                LOGGER.info("whisper model loaded and cached (device=%s compute=%s)", device, compute_type)
                return model
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt_index + 1 < len(attempts) and device == "cuda" and _is_cuda_whisper_error(exc):
                    LOGGER.warning("CUDA Whisper unavailable (%s). Falling back to CPU/int8.", _short_error(exc))
                    if on_status is not None:
                        on_status("CUDA недоступна, переключаюсь на CPU…")
                    continue
                raise
        raise RuntimeError(f"Unable to load Whisper model: {last_exc}") from last_exc


def preload_whisper_model() -> None:
    """Warm the Whisper model in a background thread at startup (gated by
    STACKWIRE_PRELOAD_STT) so the first listen is instant instead of stalling on a
    multi-GB model load mid-use."""
    if not STACKWIRE_PRELOAD_STT or STACKWIRE_REMOTE_STT or STT_BACKEND != "whisper":
        return
    if _SHARED_WHISPER_MODEL is not None:
        return

    def _worker() -> None:
        try:
            _build_whisper_model()
        except Exception:  # noqa: BLE001
            LOGGER.debug("whisper preload failed", exc_info=True)

    threading.Thread(target=_worker, daemon=True, name="whisper-preload").start()


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
    dual: bool = False  # interview mode: capture system audio AND the microphone together


class SpeechWorker(QObject):
    partial = Signal(str)
    final = Signal(str, str)  # text, source ("" = single device, "interviewer"/"me" = dual mode)
    stt_latency = Signal(float)
    level = Signal(float)  # 0..1 audio level for the live waveform
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
        # Dual (interview) mode runs two capture loops sharing one Whisper model;
        # transcribe calls are serialized so GPU/CPU inference never interleaves.
        self._transcribe_lock = _SHARED_WHISPER_INFER_LOCK  # shared so the cached model is serialized across sessions

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

    def _load_whisper_model(self, whisper_model_class: Any) -> Any:  # noqa: ARG002
        return _build_whisper_model(on_status=self.info.emit)

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
            self.info.emit("Слушаю (удалённо)…")
            LOGGER.info("STT backend=remote-whisper api=%s", STACKWIRE_API_URL)
        else:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("faster-whisper is not installed") from exc

            model = self._load_whisper_model(WhisperModel)

        if self.device.dual:
            self._run_whisper_dual(np, sd, model)
            return

        if self.device.loopback:
            self._run_whisper_loopback(np, model)
            return

        self.info.emit("Слушаю микрофон")
        self._run_mic_capture(np, sd, model, device_index=self.device.index, channels=self.device.channels, source="")
        self.stopped.emit()

    def _run_whisper_dual(self, np, sd, model) -> None:  # noqa: ANN001
        """Smart Capture: transcribe system audio and the microphone together
        through one shared model (two capture loops, serialized inference)."""
        self.info.emit("Smart Capture: системный звук + микрофон")

        # The companion gets its OWN stop signal so a loopback failure (below) can
        # shut it down without touching self._running — otherwise the mic thread
        # would leak and keep capturing while Vosk fallback / the failed state runs.
        companion_stop = threading.Event()

        def mic_companion() -> None:
            try:
                self._run_mic_capture(np, sd, model, device_index=None, channels=1, source="me", extra_stop=companion_stop)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("dual-mode mic capture failed: %s", exc)
                self.info.emit(f"Микрофон в Smart Capture недоступен: {exc}")

        companion = threading.Thread(target=mic_companion, name="stt-mic-companion", daemon=True)
        companion.start()
        try:
            self._run_whisper_loopback(np, model, source="interviewer")
        finally:
            companion_stop.set()
            companion.join(timeout=5.0)
            if companion.is_alive():
                LOGGER.warning("dual-mode mic companion did not stop within timeout")

    def _emit_level(self, np, mono) -> None:  # noqa: ANN001
        try:
            if mono is None or len(mono) == 0:
                return
            rms = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2)))
            self.level.emit(min(1.0, rms * 10.0))
        except Exception:
            pass

    def _run_mic_capture(self, np, sd, model, *, device_index: int | None, channels: int, source: str, extra_stop: "threading.Event | None" = None) -> None:  # noqa: ANN001
        audio_queue: queue.Queue[object] = queue.Queue()

        def callback(indata, frames, time, status):  # noqa: ANN001, ARG001
            if status:
                self.info.emit(str(status))
            audio_queue.put(indata.copy())

        chunk_frames = max(int(WHISPER_SAMPLE_RATE * WHISPER_CHUNK_SECONDS), WHISPER_SAMPLE_RATE)
        buffers: list[object] = []
        buffered_frames = 0
        last_partial_frames = 0
        partial_frames = max(int(WHISPER_SAMPLE_RATE * WHISPER_PARTIAL_SECONDS), WHISPER_SAMPLE_RATE // 3)

        with sd.InputStream(
            samplerate=WHISPER_SAMPLE_RATE,
            blocksize=4096,
            device=device_index,
            dtype="float32",
            channels=channels,
            callback=callback,
        ):
            while self._running and not (extra_stop is not None and extra_stop.is_set()):
                try:
                    data = cast(np.ndarray, audio_queue.get(timeout=0.2))
                except queue.Empty:
                    continue

                mono = self._to_mono_float32(data, np)
                self._emit_level(np, mono)
                buffers.append(mono)
                buffered_frames += len(mono)

                if buffered_frames >= chunk_frames:
                    self._transcribe_whisper_buffers(np, model, buffers, source=source)
                    buffers, buffered_frames = self._keep_overlap_buffers(np, buffers)
                    last_partial_frames = buffered_frames
                elif WHISPER_LIVE_PARTIALS and source != "me" and buffered_frames - last_partial_frames >= partial_frames:
                    self._transcribe_whisper_partial(np, model, buffers, source=source)
                    last_partial_frames = buffered_frames

            if buffered_frames >= WHISPER_SAMPLE_RATE:
                self._transcribe_whisper_buffers(np, model, buffers, source=source)

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
                self.info.emit("Системный звук выбран автоматически")
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
                self.info.emit("Системный звук выбран автоматически")
                return best_candidate

        try:
            default_loopback = dict(pa.get_default_wasapi_loopback())
            self.info.emit("Системный звук выбран автоматически")
            return default_loopback
        except Exception:
            return candidates[0]

    def _run_whisper_loopback(self, np, model, *, source: str = "") -> None:  # noqa: ANN001
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
            self.info.emit("Слушаю системный звук")
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
        last_partial_frames = 0
        partial_frames = max(int(WHISPER_SAMPLE_RATE * WHISPER_PARTIAL_SECONDS), WHISPER_SAMPLE_RATE // 3)

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
                self._emit_level(np, mono)
                buffers.append(mono)
                buffered_frames += len(mono)
                if buffered_frames >= chunk_frames:
                    self._transcribe_whisper_buffers(np, model, buffers, source=source)
                    buffers, buffered_frames = self._keep_overlap_buffers(np, buffers)
                    last_partial_frames = buffered_frames
                elif WHISPER_LIVE_PARTIALS and source != "me" and buffered_frames - last_partial_frames >= partial_frames:
                    self._transcribe_whisper_partial(np, model, buffers, source=source)
                    last_partial_frames = buffered_frames

            if buffered_frames >= WHISPER_SAMPLE_RATE:
                self._transcribe_whisper_buffers(np, model, buffers, source=source)
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

        # Anti-alias before downsampling: linear interpolation alone (np.interp) folds
        # frequencies above the 8 kHz target Nyquist back into the speech band as noise,
        # hurting Whisper accuracy on system audio (loopback is usually 44.1/48 kHz).
        # A short Hann low-pass (cutoff ~ source_rate / taps, below the new Nyquist)
        # smooths them out first. Upsampling needs no pre-filter.
        if source_rate > WHISPER_SAMPLE_RATE:
            taps = 2 * round(source_rate / WHISPER_SAMPLE_RATE) + 1
            if 3 <= taps < len(mono):
                window = np.hanning(taps).astype(np.float32)
                window /= window.sum()
                mono = np.convolve(mono, window, mode="same").astype(np.float32)

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
        with self._transcribe_lock:
            return self._transcribe_local_whisper_audio_locked(model, audio, vad_filter=vad_filter)

    def _transcribe_local_whisper_audio_locked(self, model, audio, *, vad_filter: bool) -> tuple[str, dict[str, float | str]]:  # noqa: ANN001
        segments, info = model.transcribe(
            audio,
            language=whisper_language(STT_SETTINGS, locked_language=self.language_lock),
            task="transcribe",
            beam_size=STT_SETTINGS.beam_size,
            best_of=STT_SETTINGS.best_of,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=(WHISPER_INITIAL_PROMPT or None),
            vad_filter=vad_filter,
            vad_parameters=self._whisper_vad_parameters() if vad_filter else None,
            no_speech_threshold=STT_SETTINGS.no_speech_threshold,
            log_prob_threshold=STT_SETTINGS.log_prob_threshold,
            compression_ratio_threshold=STT_SETTINGS.compression_ratio_threshold,
            repetition_penalty=STT_SETTINGS.repetition_penalty,
            no_repeat_ngram_size=STT_SETTINGS.no_repeat_ngram_size,
            hallucination_silence_threshold=STT_SETTINGS.hallucination_silence_threshold,
            hotwords=(WHISPER_HOTWORDS if STT_USE_TECH_PROMPT else None),
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

    def _transcribe_whisper_partial(self, np, model, buffers: list[object], *, source: str = "") -> None:  # noqa: ANN001
        """Emit an interim live transcript of the in-progress buffer so words show up as
        they are spoken. Fast greedy decode; the authoritative text still comes from the
        full final pass in _transcribe_whisper_buffers."""
        if STACKWIRE_REMOTE_STT or model is None or not buffers:
            return
        audio = np.concatenate(buffers).astype(np.float32)
        if len(audio) < WHISPER_SAMPLE_RATE // 2:
            return
        signal_threshold = STT_LOOPBACK_SIGNAL_THRESHOLD if self.device.loopback else STT_MIC_SIGNAL_THRESHOLD
        if float(np.sqrt(np.mean(audio**2))) < signal_threshold:
            return
        try:
            with self._transcribe_lock:
                segments, _info = model.transcribe(
                    audio,
                    language=whisper_language(STT_SETTINGS, locked_language=self.language_lock),
                    task="transcribe",
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=False,
                    no_speech_threshold=STT_SETTINGS.no_speech_threshold,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception:  # noqa: BLE001
            return
        text = clean_stt_output(text)
        if text:
            self.partial.emit(text)

    def _transcribe_whisper_buffers(self, np, model, buffers: list[object], *, source: str = "") -> None:  # noqa: ANN001
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
            self._transcribe_remote_whisper_audio(np, audio, source=source)
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
            LOGGER.info("whisper raw_stt=%r source=%s", text, source or "-")
            self.final.emit(text, source)

    def _transcribe_remote_whisper_audio(self, np, audio, *, source: str = "") -> None:  # noqa: ANN001
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
            LOGGER.info("remote whisper raw_stt=%r source=%s", text, source or "-")
            self.final.emit(text, source)

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

            self.info.emit("Слушаю микрофон")

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
                            self.final.emit(text, "")
                    else:
                        result = json.loads(recognizer.PartialResult())
                        text = result.get("partial", "").strip()
                        if text:
                            self.partial.emit(text)

                final_result = json.loads(recognizer.FinalResult())
                final_text = final_result.get("text", "").strip()
                if final_text:
                    self.final.emit(final_text, "")
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
            self.info.emit("Слушаю системный звук")

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
                        self.final.emit(text, "")
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
