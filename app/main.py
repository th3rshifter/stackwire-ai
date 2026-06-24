import base64
import binascii
import json
import logging
import os
import queue
import threading
import time
from threading import Lock
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import Depends, FastAPI, Header, HTTPException
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from requests import RequestException

from app.config import APP_NAME, get_stt_settings, is_cuda_whisper_error, load_local_env, whisper_language, whisper_model_attempts, whisper_vad_parameters
from app.event_log import append_client_event

load_local_env()

from app import auth
from app.answer_planner import build_answer_plan
from app.llm import ANSWER_MODE, ANSWER_PROMPT_PROFILE, ARTIFACT_ANSWER_NUM_PREDICT, AskResult, DEFAULT_ANSWER_NUM_PREDICT, EXPAND_ANSWER_NUM_PREDICT, OLLAMA_KEEP_ALIVE, OLLAMA_URL, OllamaClient, current_answer_model, current_vision_model
from app.question_recovery import CONFIDENCE_THRESHOLD, RECOVERY_LOCAL_FAST_PATH, current_recovery_model
from app.question_recovery import STACKWIRE_MODE
from app.storage import init_db, log_feedback, save_good_answer
from app.tech_terms import WHISPER_TECHNICAL_PROMPT
from app.transcript_repair import clean_stt_output, is_probable_stt_hallucination


app = FastAPI(title=APP_NAME)
app.add_middleware(GZipMiddleware, minimum_size=1000)
LOGGER = logging.getLogger(__name__)
client = OllamaClient()
init_db()
auth.init_auth_db()

# When enabled, /ask, /expand and /analyze-image require a valid bearer token.
REQUIRE_AUTH = os.getenv("STACKWIRE_REQUIRE_AUTH", "1").strip().lower() in {"1", "true", "yes", "on"}


def require_user(authorization: str | None = Header(default=None)) -> auth.AuthUser:
    if not REQUIRE_AUTH:
        return auth.AuthUser(id=0, username="anonymous")
    token = ""
    if authorization:
        parts = authorization.split(" ", 1)
        token = parts[1].strip() if len(parts) == 2 and parts[0].lower() == "bearer" else authorization.strip()
    user = auth.verify_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    return user

STT_SETTINGS = get_stt_settings()
WHISPER_MODEL = STT_SETTINGS.model
WHISPER_DEVICE = STT_SETTINGS.device
WHISPER_COMPUTE_TYPE = STT_SETTINGS.compute_type
WHISPER_VAD_FILTER = STT_SETTINGS.vad_filter
WHISPER_RETRY_WITHOUT_VAD = STT_SETTINGS.retry_without_vad
WHISPER_HOTWORDS = STT_SETTINGS.hotwords
WHISPER_INITIAL_PROMPT = WHISPER_TECHNICAL_PROMPT
_whisper_model: Any | None = None
_whisper_model_lock = Lock()
_whisper_transcribe_lock = Lock()


def _request_error_detail(exc: RequestException, *, prefix: str = "Ollama request failed") -> str:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    response_text = ""
    if response is not None:
        try:
            response_text = str(response.text or "")
        except Exception:
            response_text = ""

    if status_code == 403 and "requires a subscription" in response_text.lower():
        return (
            "Ollama cloud model access denied: this model requires an Ollama account/subscription. "
            "Run `ollama signin`, check https://ollama.com/upgrade, or switch ANSWER_MODEL/RECOVERY_MODEL to local models. "
            f"Details: {response_text}"
        )

    detail = response_text.strip() if response_text.strip() else str(exc)
    if "remote end closed connection without response" in detail.lower() or "connection aborted" in detail.lower():
        return (
            "Local Ollama closed the chat request without a response. "
            "The selected model may be too heavy, still loading, or the Ollama runner crashed. "
            "Retry after the model loads, restart Ollama, or switch ANSWER_MODEL/RECOVERY_MODEL to a smaller local model. "
            f"Details: {detail}"
        )
    return f"{prefix}: {detail}"


class Question(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)
    context: list[str] = Field(default_factory=list, max_length=30)
    trusted_text: bool = False


class TranscribeRequest(BaseModel):
    audio_b64: str = Field(..., min_length=1, max_length=20_000_000)
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    language: str | None = Field(default=None, max_length=16)


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


class AuthRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=1, max_length=200)


def _is_cuda_whisper_error(exc: BaseException) -> bool:
    return is_cuda_whisper_error(exc)


def _whisper_model_attempts() -> list[tuple[str, str]]:
    return whisper_model_attempts(STT_SETTINGS)


def _whisper_vad_parameters() -> dict[str, int | float]:
    return whisper_vad_parameters(STT_SETTINGS)


def _mean_segment_attr(segments: list[object], attr: str) -> float:
    values = [float(value) for segment in segments if isinstance((value := getattr(segment, attr, None)), int | float)]
    return sum(values) / len(values) if values else 0.0


def _max_segment_attr(segments: list[object], attr: str) -> float:
    values = [float(value) for segment in segments if isinstance((value := getattr(segment, attr, None)), int | float)]
    return max(values) if values else 0.0


def _transcribe_with_quality(model: Any, audio: Any, *, vad_filter: bool, language: str | None = None) -> tuple[str, dict[str, float | str]]:
    segments, info = model.transcribe(
        audio,
        language=whisper_language(STT_SETTINGS, requested_language=language),
        task="transcribe",
        beam_size=STT_SETTINGS.beam_size,
        best_of=STT_SETTINGS.best_of,
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt=WHISPER_INITIAL_PROMPT,
        vad_filter=vad_filter,
        vad_parameters=_whisper_vad_parameters() if vad_filter else None,
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
        "language": str(getattr(info, "language", whisper_language(STT_SETTINGS, requested_language=language) or "")),
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


def _warmup_whisper_model(model: Any, device: str) -> None:
    if device != "cuda":
        return
    try:
        import numpy as np
    except ImportError:
        return
    segments, _info = model.transcribe(
        np.zeros(STT_SETTINGS.sample_rate, dtype=np.float32),
        language="en",
        task="transcribe",
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
        vad_filter=False,
        no_speech_threshold=0.95,
    )
    list(segments)


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
                    candidate_model = WhisperModel(
                        WHISPER_MODEL,
                        device=device,
                        compute_type=compute_type,
                    )
                    _warmup_whisper_model(candidate_model, device)
                    _whisper_model = candidate_model
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
        text, diagnostics = _transcribe_with_quality(model, audio, vad_filter=WHISPER_VAD_FILTER, language=request.language)
        bad_text = _is_bad_transcript(text, diagnostics, rms)
        if WHISPER_RETRY_WITHOUT_VAD and WHISPER_VAD_FILTER and (not text.strip() or bad_text):
            retry_text, retry_diagnostics = _transcribe_with_quality(model, audio, vad_filter=False, language=request.language)
            if retry_text.strip() and not _is_bad_transcript(retry_text, retry_diagnostics, rms):
                text = retry_text
                diagnostics = retry_diagnostics
                bad_text = False
        raw_text = text
        cleaned_text = clean_stt_output(raw_text)
        bad_text = bad_text or _is_bad_transcript(cleaned_text, diagnostics, rms)
    latency_ms = (time.perf_counter() - started) * 1000
    stt_diagnostics = {
        "language": diagnostics.get("language", ""),
        "vad": diagnostics.get("vad", ""),
        "rms": rms,
        "avg_logprob": float(diagnostics.get("avg_logprob") or 0.0),
        "no_speech_prob": float(diagnostics.get("no_speech_prob") or 0.0),
        "compression_ratio": float(diagnostics.get("compression_ratio") or 0.0),
        "bad_text": bad_text,
    }
    LOGGER.info(
        "remote_transcribe latency_ms=%.0f language=%s vad=%s rms=%.6f avg_logprob=%.2f no_speech=%.2f compression=%.2f bad=%s raw=%r cleaned=%r",
        latency_ms,
        stt_diagnostics["language"],
        stt_diagnostics["vad"],
        rms,
        stt_diagnostics["avg_logprob"],
        stt_diagnostics["no_speech_prob"],
        stt_diagnostics["compression_ratio"],
        bad_text,
        raw_text,
        cleaned_text,
    )
    if bad_text:
        return {
            "text": "",
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "latency_ms": latency_ms,
            "language": diagnostics.get("language", ""),
            "diagnostics": stt_diagnostics,
        }
    return {
        "text": cleaned_text,
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "latency_ms": latency_ms,
        "language": diagnostics.get("language", ""),
        "diagnostics": stt_diagnostics,
    }


@app.post("/auth/register")
def auth_register(request: AuthRequest):
    try:
        token = auth.register(request.username, request.password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"token": token, "username": request.username.strip()}


@app.post("/auth/login")
def auth_login(request: AuthRequest):
    try:
        token = auth.login(request.username, request.password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return {"token": token, "username": request.username.strip()}


@app.get("/auth/me")
def auth_me(user: auth.AuthUser = Depends(require_user)):
    return {"username": user.username, "id": user.id}


@app.post("/auth/logout")
def auth_logout(authorization: str | None = Header(default=None)):
    if authorization:
        token = authorization.split(" ", 1)[-1].strip()
        auth.logout(token)
    return {"ok": True}


@app.post("/ask")
def ask(question: Question, user: auth.AuthUser = Depends(require_user)):
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
            "answer_model": result.answer_model or current_answer_model(),
        }
        return JSONResponse(
            content=payload,
            media_type="application/json; charset=utf-8",
        )
    except RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=_request_error_detail(exc),
        ) from exc


@app.post("/ask/stream")
def ask_stream(question: Question, user: auth.AuthUser = Depends(require_user)):
    try:
        def generate():
            chunk_queue: queue.Queue[str | None] = queue.Queue()
            done_event = threading.Event()
            result_holder: list[AskResult | None] = [None]

            def on_recovery(text: str) -> None:
                chunk_queue.put(json.dumps({'type': 'recovery', 'content': text}, ensure_ascii=False))

            def on_delta(chunk: str) -> None:
                # Stream the whole chunk token-by-token (json.dumps escapes newlines, so
                # SSE framing stays valid). Splitting per line made the client render in
                # jerky line-blocks instead of the smooth token flow the desktop shows.
                chunk_queue.put(json.dumps({'type': 'delta', 'content': chunk}, ensure_ascii=False))

            def on_thinking(chunk: str) -> None:
                chunk_queue.put(json.dumps({'type': 'thinking', 'content': chunk}, ensure_ascii=False))

            def worker() -> None:
                try:
                    result = client.ask_stream(
                        question.text,
                        question.context,
                        trusted_text=question.trusted_text,
                        on_recovery=on_recovery,
                        on_delta=on_delta,
                        on_thinking=on_thinking,
                    )
                    result_holder[0] = result
                finally:
                    chunk_queue.put(None)
                    done_event.set()

            threading.Thread(target=worker, daemon=True).start()

            on_recovery(question.text)

            while True:
                item = chunk_queue.get()
                if item is None:
                    break
                yield f"data: {item}\n\n"

            done_event.wait(timeout=30)
            result = result_holder[0]
            if result is not None:
                yield f"data: {json.dumps({
                    'type': 'done',
                    'answered': result.answered,
                    'raw_text': result.raw_text,
                    'recovery': {
                        'confidence': result.recovery.confidence,
                        'recovered_question': result.recovery.recovered_question,
                        'detected_topic': result.recovery.detected_topic,
                        'technical_entities': result.recovery.technical_entities,
                        'ambiguities': result.recovery.ambiguities,
                        'needs_manual_fix': result.recovery.needs_manual_fix,
                        'candidate_questions': result.recovery.candidate_questions,
                        'candidate_quality': result.recovery.candidate_quality,
                        'candidate_details': result.recovery.candidate_details,
                        'reason': result.recovery.reason,
                    },
                    'recovery_latency': result.recovery_latency,
                    'answer_latency': result.answer_latency,
                    'total_latency': result.total_latency,
                    'question_id': result.question_id,
                    'answer_id': result.answer_id,
                    'plan_domain': result.plan_domain,
                    'plan_intent': result.plan_intent,
                    'answer_model': result.answer_model or current_answer_model(),
                }, ensure_ascii=False)}\n\n"""

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            # Defeat proxy/uvicorn response buffering so each token reaches the client
            # immediately instead of arriving in one batch at the end.
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )
    except RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=_request_error_detail(exc),
        ) from exc


@app.post("/expand")
def expand(request: ExpandRequest, user: auth.AuthUser = Depends(require_user)):
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
                "answer_model": result.answer_model or current_answer_model(),
            },
            media_type="application/json; charset=utf-8",
        )
    except RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=_request_error_detail(exc),
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
def analyze_image(request: ImageAnalysisRequest, user: auth.AuthUser = Depends(require_user)):
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
            detail=_request_error_detail(exc, prefix="Ollama vision request failed"),
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
        "answer_model": current_answer_model(),
        "answer_mode": ANSWER_MODE,
        "answer_prompt_profile": ANSWER_PROMPT_PROFILE,
        "answer_num_predict": DEFAULT_ANSWER_NUM_PREDICT,
        "artifact_num_predict": ARTIFACT_ANSWER_NUM_PREDICT,
        "expand_num_predict": EXPAND_ANSWER_NUM_PREDICT,
        "vision_model": current_vision_model(),
        "recovery_model": current_recovery_model(),
        "recovery_local_fast_path": RECOVERY_LOCAL_FAST_PATH,
        "mode": STACKWIRE_MODE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "ollama_url": OLLAMA_URL,
        "ollama_keep_alive": OLLAMA_KEEP_ALIVE or "default",
        "whisper_model": WHISPER_MODEL,
        "whisper_device": WHISPER_DEVICE,
        "whisper_compute_type": WHISPER_COMPUTE_TYPE,
        "stt_language_mode": STT_SETTINGS.language_mode,
        "whisper_language": whisper_language(STT_SETTINGS) or "auto",
        "whisper_beam_size": STT_SETTINGS.beam_size,
        "whisper_best_of": STT_SETTINGS.best_of,
        "whisper_vad_filter": WHISPER_VAD_FILTER,
        "whisper_retry_without_vad": WHISPER_RETRY_WITHOUT_VAD,
        "whisper_vad_threshold": STT_SETTINGS.vad_threshold,
        "whisper_vad_min_silence_ms": STT_SETTINGS.vad_min_silence_ms,
        "whisper_no_speech_threshold": STT_SETTINGS.no_speech_threshold,
        "whisper_repetition_penalty": STT_SETTINGS.repetition_penalty,
        "whisper_no_repeat_ngram_size": STT_SETTINGS.no_repeat_ngram_size,
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
