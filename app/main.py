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

from app.llm import MODEL, OLLAMA_URL, VISION_MODEL, OllamaClient
from app.question_recovery import CONFIDENCE_THRESHOLD, DEFAULT_MODEL as RECOVERY_MODEL
from app.question_recovery import STEALTHWIRE_MODE


app = FastAPI(title="Interview Assistant")
client = OllamaClient()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
WHISPER_INITIAL_PROMPT = (
    "This is a Russian DevOps/SRE technical interview with mixed Russian and English terminology. "
    
    "Preserve English product names, commands, file paths, acronyms, protocols, config keys, "
    "CLI tools, cloud services and technology names in English. "
    
    "Linux/system: systemd, journald, Bash, permissions, users, groups, sudo, SSH, cron, "
    "logs, processes, signals, namespaces, cgroups, /dev, /proc, /etc, /var/log. "
    
    "Networking: DNS, TCP, UDP, HTTP, HTTPS, TLS, mTLS, ICMP, ports, routing, NAT, "
    "load balancing, proxy, ingress, firewall, certificates. "
    
    "Containers: Docker, Dockerfile, image, container, volume, network, registry, "
    "layer cache, multi-stage build, Compose, containerd, OCI. "
    
    "Kubernetes: Pod, Deployment, ReplicaSet, StatefulSet, DaemonSet, Job, CronJob, "
    "Service, Ingress, ConfigMap, Secret, Volume, PVC, PV, StorageClass, Namespace, "
    "RBAC, ServiceAccount, probes, requests, limits, HPA, rolling update. "
    
    "Helm and GitOps: Helm, chart, values.yaml, templates, release, Argo CD, GitOps, "
    "sync, drift, rollback. "
    
    "CI/CD: GitLab CI, GitHub Actions, Jenkins, pipeline, declarative pipeline, "
    "scripted pipeline, runner, artifact, cache, stages, jobs, variables, environment, registry. "
    
    "Infrastructure as Code: Ansible, playbook, role, task, handler, template, inventory, "
    "collection, Terraform, provider, resource, module, state, plan, apply, workspace. "
    
    "Observability: Prometheus, Grafana, Alertmanager, metrics, logs, traces, dashboards, "
    "alerts, SLI, SLO, OpenTelemetry, Loki, ELK, OpenSearch, Jaeger, Tempo. "
    
    "Security: Vault, secrets, RBAC, least privilege, SonarQube, SAST, dependency scanning, "
    "image scanning, SSH keys, certificates, TLS. "
    
    "Databases: PostgreSQL, replication, backup, restore, Patroni, Redis, MongoDB, "
    "MariaDB, ClickHouse, migrations, Liquibase. "
    
    "Messaging: Kafka, topic, partition, consumer group, offset, ZooKeeper, RabbitMQ, "
    "queue, exchange. "
    
    "Storage: NFS, S3, Ceph, Harbor, Nexus, Artifactory, GitLab Registry. "
    
    "Web servers: Nginx, HAProxy, Apache, WebLogic, upstream, reverse proxy, rate limiting."
)
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
            language=None,
            task="transcribe",
            beam_size=3,
            best_of=3,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=WHISPER_INITIAL_PROMPT,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 450},
            no_speech_threshold=0.65,
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
        

@app.get("/status")
def status():
    return {
        "status": "working",
        "answer_model": MODEL,
        "vision_model": VISION_MODEL,
        "recovery_model": RECOVERY_MODEL,
        "mode": STEALTHWIRE_MODE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "ollama_url": OLLAMA_URL,
        "whisper_model": WHISPER_MODEL,
        "whisper_device": WHISPER_DEVICE,
        "whisper_compute_type": WHISPER_COMPUTE_TYPE,
    }


@app.get("/")
def root():
    return {
        "name": "Interview Assistant",
        "ui": "Run the desktop app: python -m app.desktop",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("STEALTHWIRE_HOST", "127.0.0.1"),
        port=int(os.getenv("STEALTHWIRE_PORT", "8000")),
    )
