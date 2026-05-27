import base64
import binascii
import logging
import os
import time
from threading import Lock
from typing import Any

from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from requests import RequestException

from app.config import APP_NAME, load_local_env
from app.event_log import append_client_event

load_local_env()

from app.answer_planner import build_answer_plan
from app.llm import ANSWER_MODE, ANSWER_PROMPT_PROFILE, ARTIFACT_ANSWER_NUM_PREDICT, DEFAULT_ANSWER_NUM_PREDICT, EXPAND_ANSWER_NUM_PREDICT, MODEL, OLLAMA_KEEP_ALIVE, OLLAMA_URL, VISION_MODEL, OllamaClient
from app.question_recovery import CONFIDENCE_THRESHOLD, DEFAULT_MODEL as RECOVERY_MODEL, RECOVERY_LOCAL_FAST_PATH
from app.question_recovery import STACKWIRE_MODE
from app.storage import init_db, log_feedback, save_good_answer
from app.tech_terms import WHISPER_TECHNICAL_PROMPT
from app.transcript_repair import clean_stt_output, is_probable_stt_hallucination


app = FastAPI(title=APP_NAME)
LOGGER = logging.getLogger(__name__)
client = OllamaClient()
init_db()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
STT_ALLOW_CPU_WHISPER_FALLBACK = os.getenv("STT_ALLOW_CPU_WHISPER_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "on"}
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ru").strip() or None
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_BEST_OF = int(os.getenv("WHISPER_BEST_OF", "5"))
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "1").strip().lower() not in {"0", "false", "no", "off"}
WHISPER_RETRY_WITHOUT_VAD = os.getenv("WHISPER_RETRY_WITHOUT_VAD", "1").strip().lower() in {"1", "true", "yes", "on"}
WHISPER_VAD_THRESHOLD = float(os.getenv("WHISPER_VAD_THRESHOLD", "0.20"))
WHISPER_VAD_MIN_SPEECH_MS = int(os.getenv("WHISPER_VAD_MIN_SPEECH_MS", "100"))
WHISPER_VAD_MIN_SILENCE_MS = int(os.getenv("WHISPER_VAD_MIN_SILENCE_MS", "650"))
WHISPER_VAD_SPEECH_PAD_MS = int(os.getenv("WHISPER_VAD_SPEECH_PAD_MS", "450"))
WHISPER_NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.60"))
WHISPER_LOG_PROB_THRESHOLD = float(os.getenv("WHISPER_LOG_PROB_THRESHOLD", "-1.15"))
WHISPER_COMPRESSION_RATIO_THRESHOLD = float(os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.6"))
WHISPER_REPETITION_PENALTY = float(os.getenv("WHISPER_REPETITION_PENALTY", "1.08"))
WHISPER_NO_REPEAT_NGRAM_SIZE = int(os.getenv("WHISPER_NO_REPEAT_NGRAM_SIZE", "3"))
WHISPER_HALLUCINATION_SILENCE_THRESHOLD = float(os.getenv("WHISPER_HALLUCINATION_SILENCE_THRESHOLD", "1.0"))
WHISPER_HOTWORDS = os.getenv(
    "WHISPER_HOTWORDS",
    "Kubernetes kubectl kubelet Deployment StatefulSet DaemonSet Pod Service Ingress ConfigMap Secret PVC "
    "Docker Dockerfile docker-compose GitLab CI Jenkins Terraform Ansible Prometheus Grafana Linux TCP UDP DNS TLS mTLS HTTPS",
).strip() or None
WHISPER_INITIAL_PROMPT = WHISPER_TECHNICAL_PROMPT
_whisper_model: Any | None = None
_whisper_model_lock = Lock()
_whisper_transcribe_lock = Lock()
CUDA_WHISPER_ERROR_MARKERS = (
    "cuda",
    "cublas",
    "cublas64",
    "cudnn",
    "nvrtc",
    "ctranslate2",
)


class Question(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    context: list[str] = Field(default_factory=list, max_length=30)
    trusted_text: bool = False


class TranscribeRequest(BaseModel):
    audio_b64: str = Field(..., min_length=1, max_length=20_000_000)
    sample_rate: int = Field(default=16000, ge=8000, le=48000)


class ImageAnalysisRequest(BaseModel):
    image_b64: str = Field(..., min_length=1, max_length=20_000_000)
    prompt: str = Field(default="", max_length=3000)


class ExpandRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    previous_answer: str = Field(..., min_length=1, max_length=30000)
    mode: str = Field(..., pattern="^(details|components|example|compare|troubleshoot)$")


class FeedbackRequest(BaseModel):
    answer_id: int = Field(..., ge=1)
    label: str = Field(..., pattern="^(good|bad|wrong_domain|too_short|no_code|bad_format|hallucination|other)$")
    note: str | None = Field(default=None, max_length=2000)


class SaveGoodRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    answer: str = Field(..., min_length=1, max_length=30000)
    domain: str | None = Field(default=None, max_length=80)
    intent: str | None = Field(default=None, max_length=80)
    tags: list[str] = Field(default_factory=list, max_length=20)
    rating: int = Field(default=5, ge=1, le=10)


class ClientEventRequest(BaseModel):
    event: str = Field(default="client_event", max_length=80)
    client_time: str = Field(default="", max_length=80)
    details: dict[str, Any] = Field(default_factory=dict)


def _is_cuda_whisper_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in CUDA_WHISPER_ERROR_MARKERS)


def _whisper_model_attempts() -> list[tuple[str, str]]:
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


def _whisper_vad_parameters() -> dict[str, int | float]:
    return {
        "threshold": WHISPER_VAD_THRESHOLD,
        "min_speech_duration_ms": WHISPER_VAD_MIN_SPEECH_MS,
        "min_silence_duration_ms": WHISPER_VAD_MIN_SILENCE_MS,
        "speech_pad_ms": WHISPER_VAD_SPEECH_PAD_MS,
    }


def _mean_segment_attr(segments: list[object], attr: str) -> float:
    values = [float(value) for segment in segments if isinstance((value := getattr(segment, attr, None)), int | float)]
    return sum(values) / len(values) if values else 0.0


def _max_segment_attr(segments: list[object], attr: str) -> float:
    values = [float(value) for segment in segments if isinstance((value := getattr(segment, attr, None)), int | float)]
    return max(values) if values else 0.0


def _transcribe_with_quality(model: Any, audio: Any, *, vad_filter: bool) -> tuple[str, dict[str, float | str]]:
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
        vad_parameters=_whisper_vad_parameters() if vad_filter else None,
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
        "avg_logprob": _mean_segment_attr(segment_list, "avg_logprob"),
        "no_speech_prob": _max_segment_attr(segment_list, "no_speech_prob"),
        "compression_ratio": _max_segment_attr(segment_list, "compression_ratio"),
        "vad": "1" if vad_filter else "0",
    }
    return text, diagnostics


def _is_bad_transcript(text: str, diagnostics: dict[str, float | str], rms: float) -> bool:
    return is_probable_stt_hallucination(
        text,
        avg_logprob=float(diagnostics.get("avg_logprob") or 0.0),
        no_speech_prob=float(diagnostics.get("no_speech_prob") or 0.0),
        compression_ratio=float(diagnostics.get("compression_ratio") or 0.0),
        rms=rms,
    )


def _get_whisper_model() -> Any:
    global _whisper_model
    with _whisper_model_lock:
        if _whisper_model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise HTTPException(status_code=500, detail="faster-whisper is not installed") from exc
            last_exc: Exception | None = None
            attempts = _whisper_model_attempts()
            for attempt_index, (device, compute_type) in enumerate(attempts):
                try:
                    LOGGER.info(
                        "loading whisper model=%s device=%s compute_type=%s",
                        WHISPER_MODEL,
                        device,
                        compute_type,
                    )
                    _whisper_model = WhisperModel(
                        WHISPER_MODEL,
                        device=device,
                        compute_type=compute_type,
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    has_retry = attempt_index + 1 < len(attempts)
                    if has_retry and device == "cuda" and _is_cuda_whisper_error(exc):
                        LOGGER.warning("CUDA Whisper unavailable, falling back to CPU/int8: %s", exc)
                        continue
                    raise HTTPException(status_code=500, detail=f"Whisper model failed to load: {exc}") from exc
            if _whisper_model is None:
                raise HTTPException(status_code=500, detail=f"Whisper model failed to load: {last_exc}")
        return _whisper_model


@app.post("/transcribe")
def transcribe(request: TranscribeRequest):
    try:
        import numpy as np
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="numpy is not installed") from exc

    try:
        audio_bytes = base64.b64decode(request.audio_b64)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="audio_b64 is not valid base64") from exc

    audio = np.frombuffer(audio_bytes, dtype=np.float32)
    if audio.size < request.sample_rate:
        return {"text": "", "latency_ms": 0.0}

    started = time.perf_counter()
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    model = _get_whisper_model()
    with _whisper_transcribe_lock:
        text, diagnostics = _transcribe_with_quality(model, audio, vad_filter=WHISPER_VAD_FILTER)
        bad_text = _is_bad_transcript(text, diagnostics, rms)
        if WHISPER_RETRY_WITHOUT_VAD and WHISPER_VAD_FILTER and (not text.strip() or bad_text):
            retry_text, retry_diagnostics = _transcribe_with_quality(model, audio, vad_filter=False)
            if retry_text.strip() and not _is_bad_transcript(retry_text, retry_diagnostics, rms):
                text = retry_text
                diagnostics = retry_diagnostics
                bad_text = False
        text = clean_stt_output(text)
        bad_text = bad_text or _is_bad_transcript(text, diagnostics, rms)
    latency_ms = (time.perf_counter() - started) * 1000
    LOGGER.info(
        "remote_transcribe latency_ms=%.0f language=%s vad=%s rms=%.6f avg_logprob=%.2f no_speech=%.2f compression=%.2f bad=%s text=%r",
        latency_ms,
        diagnostics.get("language", ""),
        diagnostics.get("vad", ""),
        rms,
        float(diagnostics.get("avg_logprob") or 0.0),
        float(diagnostics.get("no_speech_prob") or 0.0),
        float(diagnostics.get("compression_ratio") or 0.0),
        bad_text,
        text,
    )
    if bad_text:
        return {"text": "", "latency_ms": latency_ms}
    return {"text": text, "latency_ms": latency_ms}


@app.post("/ask")
def ask(question: Question):
    try:
        result = client.ask(question.text, question.context, trusted_text=question.trusted_text)
        payload = {
            "answer": result.answer,
            "answered": result.answered,
            "raw_text": result.raw_text,
            "recovery": {
                "confidence": result.recovery.confidence,
                "recovered_question": result.recovery.recovered_question,
                "detected_topic": result.recovery.detected_topic,
                "technical_entities": result.recovery.technical_entities,
                "ambiguities": result.recovery.ambiguities,
                "needs_manual_fix": result.recovery.needs_manual_fix,
                "candidate_questions": result.recovery.candidate_questions,
                "candidate_quality": result.recovery.candidate_quality,
                "candidate_details": result.recovery.candidate_details,
                "reason": result.recovery.reason,
            },
            "recovery_latency": result.recovery_latency,
            "answer_latency": result.answer_latency,
            "total_latency": result.total_latency,
            "question_id": result.question_id,
            "answer_id": result.answer_id,
            "plan_domain": result.plan_domain,
            "plan_intent": result.plan_intent,
        }
        return JSONResponse(
            content=payload,
            media_type="application/json; charset=utf-8",
        )
    except RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {exc}",
        ) from exc


@app.post("/expand")
def expand(request: ExpandRequest):
    try:
        result = client.expand(request.question, request.previous_answer, request.mode)
        return JSONResponse(
            content={
                "answer": result.answer,
                "mode": result.mode,
                "latency": result.latency,
                "question_id": result.question_id,
                "answer_id": result.answer_id,
                "plan_domain": result.plan_domain,
                "plan_intent": result.plan_intent,
            },
            media_type="application/json; charset=utf-8",
        )
    except RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {exc}",
        ) from exc


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    feedback_id = log_feedback(request.answer_id, request.label, request.note)
    return {"ok": True, "feedback_id": feedback_id}


@app.post("/good-answer")
def good_answer(request: SaveGoodRequest):
    domain = request.domain
    intent = request.intent
    if not domain or not intent:
        plan = build_answer_plan(request.question)
        domain = domain or plan.domain
        intent = intent or plan.intent
    good_answer_id = save_good_answer(
        question=request.question,
        answer=request.answer,
        domain=domain,
        intent=intent,
        tags=request.tags,
        rating=request.rating,
    )
    return {"ok": True, "good_answer_id": good_answer_id}


@app.post("/analyze-image")
def analyze_image(request: ImageAnalysisRequest):
    try:
        started = time.perf_counter()
        answer = client.analyze_image(request.image_b64, request.prompt)
        return JSONResponse(
            content={
                "answer": answer,
                "latency": time.perf_counter() - started,
            },
            media_type="application/json; charset=utf-8",
        )
    except RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama vision request failed: {exc}",
        ) from exc


@app.post("/client-event")
def client_event(request: ClientEventRequest):
    details = dict(request.details)
    if request.client_time:
        details["client_time"] = request.client_time
    logged_at = append_client_event(request.event, details)
    return {"ok": True, "logged_at": logged_at}
        

@app.get("/status")
def status():
    return {
        "status": "working",
        "answer_model": MODEL,
        "answer_mode": ANSWER_MODE,
        "answer_prompt_profile": ANSWER_PROMPT_PROFILE,
        "answer_num_predict": DEFAULT_ANSWER_NUM_PREDICT,
        "artifact_num_predict": ARTIFACT_ANSWER_NUM_PREDICT,
        "expand_num_predict": EXPAND_ANSWER_NUM_PREDICT,
        "vision_model": VISION_MODEL,
        "recovery_model": RECOVERY_MODEL,
        "recovery_local_fast_path": RECOVERY_LOCAL_FAST_PATH,
        "mode": STACKWIRE_MODE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "ollama_url": OLLAMA_URL,
        "ollama_keep_alive": OLLAMA_KEEP_ALIVE or "default",
        "whisper_model": WHISPER_MODEL,
        "whisper_device": WHISPER_DEVICE,
        "whisper_compute_type": WHISPER_COMPUTE_TYPE,
        "whisper_language": WHISPER_LANGUAGE or "auto",
        "whisper_beam_size": WHISPER_BEAM_SIZE,
        "whisper_best_of": WHISPER_BEST_OF,
        "whisper_vad_filter": WHISPER_VAD_FILTER,
        "whisper_retry_without_vad": WHISPER_RETRY_WITHOUT_VAD,
        "whisper_vad_threshold": WHISPER_VAD_THRESHOLD,
        "whisper_vad_min_silence_ms": WHISPER_VAD_MIN_SILENCE_MS,
        "whisper_no_speech_threshold": WHISPER_NO_SPEECH_THRESHOLD,
        "whisper_repetition_penalty": WHISPER_REPETITION_PENALTY,
        "whisper_no_repeat_ngram_size": WHISPER_NO_REPEAT_NGRAM_SIZE,
    }


@app.get("/")
def root():
    return {
        "name": APP_NAME,
        "ui": "Run the desktop app: python -m app.desktop",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("STACKWIRE_HOST", "127.0.0.1"),
        port=int(os.getenv("STACKWIRE_PORT", os.getenv("SERVER_PORT", "8000"))),
    )
