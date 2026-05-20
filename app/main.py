import base64
import binascii
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

from app.llm import ANSWER_MODE, ANSWER_PROMPT_PROFILE, ARTIFACT_ANSWER_NUM_PREDICT, DEFAULT_ANSWER_NUM_PREDICT, MODEL, OLLAMA_KEEP_ALIVE, OLLAMA_URL, VISION_MODEL, OllamaClient
from app.question_recovery import CONFIDENCE_THRESHOLD, DEFAULT_MODEL as RECOVERY_MODEL, RECOVERY_LOCAL_FAST_PATH
from app.question_recovery import STACKWIRE_MODE
from app.tech_terms import WHISPER_TECHNICAL_PROMPT


app = FastAPI(title=APP_NAME)
client = OllamaClient()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "").strip() or None
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
WHISPER_BEST_OF = int(os.getenv("WHISPER_BEST_OF", "1"))
WHISPER_VAD_MIN_SILENCE_MS = int(os.getenv("WHISPER_VAD_MIN_SILENCE_MS", "300"))
WHISPER_NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.65"))
WHISPER_INITIAL_PROMPT = WHISPER_TECHNICAL_PROMPT
_whisper_model: Any | None = None
_whisper_model_lock = Lock()
_whisper_transcribe_lock = Lock()


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


class ClientEventRequest(BaseModel):
    event: str = Field(default="client_event", max_length=80)
    client_time: str = Field(default="", max_length=80)
    details: dict[str, Any] = Field(default_factory=dict)


def _get_whisper_model() -> Any:
    global _whisper_model
    with _whisper_model_lock:
        if _whisper_model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise HTTPException(status_code=500, detail="faster-whisper is not installed") from exc
            _whisper_model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
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
    model = _get_whisper_model()
    with _whisper_transcribe_lock:
        segments, _info = model.transcribe(
            audio,
            language=WHISPER_LANGUAGE,
            task="transcribe",
            beam_size=WHISPER_BEAM_SIZE,
            best_of=WHISPER_BEST_OF,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=WHISPER_INITIAL_PROMPT,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": WHISPER_VAD_MIN_SILENCE_MS},
            no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    return {"text": text, "latency_ms": (time.perf_counter() - started) * 1000}


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
        "whisper_vad_min_silence_ms": WHISPER_VAD_MIN_SILENCE_MS,
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
