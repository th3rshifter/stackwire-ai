import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import requests
from PySide6.QtCore import QObject, Signal, Slot


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


@dataclass(frozen=True)
class ModelRecommendation:
    name: str
    title: str
    size: str
    kind: str
    note: str
    min_ram_gb: int
    min_vram_gb: int = 0


MODELHUB_RECOMMENDED: tuple[ModelRecommendation, ...] = (
    ModelRecommendation("qwen3.6:latest", "Qwen 3.6", "latest", "balanced", "Current StackWire default; good general local model.", 16),
    ModelRecommendation("gemma3:4b", "Gemma 3 4B Vision", "4B", "vision", "Installed on this PC; use it for screenshot and image analysis.", 8),
    ModelRecommendation("qwen2.5vl:7b", "Qwen 2.5 VL 7B", "7B", "vision", "Better screenshot/OCR model if your PC can run it.", 16, 8),
    ModelRecommendation("qwen2.5:7b", "Qwen 2.5 7B", "7B", "balanced", "Good Russian/English general answers on common PCs.", 12),
    ModelRecommendation("qwen2.5-coder:7b", "Qwen 2.5 Coder 7B", "7B", "code", "Best starter choice for code and technical tasks.", 12),
    ModelRecommendation("llama3.2:3b", "Llama 3.2 3B", "3B", "fast", "Very fast fallback for weak CPU/RAM.", 8),
    ModelRecommendation("mistral:7b", "Mistral 7B", "7B", "balanced", "Good concise technical answers.", 12),
    ModelRecommendation("deepseek-r1:7b", "DeepSeek R1 7B", "7B", "reasoning", "Reasoning model; slower, useful for deeper analysis.", 16),
    ModelRecommendation("qwen2.5:14b", "Qwen 2.5 14B", "14B", "quality", "Better quality if you have enough RAM/VRAM.", 32, 12),
)


def normalized_model_name(model: str) -> str:
    value = str(model).strip().lower()
    if not value:
        return ""
    if ":" not in value:
        return f"{value}:latest"
    return value


def model_name_matches(left: str, right: str) -> bool:
    return normalized_model_name(left) == normalized_model_name(right)


VISION_MODEL_NAMES = frozenset(
    normalized_model_name(recommendation.name)
    for recommendation in MODELHUB_RECOMMENDED
    if recommendation.kind == "vision"
)


def is_vision_model(model: str) -> bool:
    return normalized_model_name(model) in VISION_MODEL_NAMES


def short_error(exc: BaseException, limit: int = 180) -> str:
    text = str(exc).strip().replace("\n", " ")
    return text[:limit] + ("..." if len(text) > limit else "")


def dedupe_models(models: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for model in models:
        value = str(model).strip()
        key = normalized_model_name(value)
        if not value or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def llm_provider() -> str:
    provider = os.getenv("STACKWIRE_LLM_PROVIDER", os.getenv("STACKWIRE_PROVIDER", "ollama")).strip().lower()
    if provider in {"openai", "openai-compatible", "openai_compatible", "compatible"}:
        return "openai_compatible"
    return "ollama"


def ollama_base_url(endpoint: str | None = None) -> str:
    value = (endpoint or os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)).strip().rstrip("/")
    for suffix in ("/api/chat", "/api/generate", "/api/tags", "/api/pull", "/api/version"):
        if value.endswith(suffix):
            return value[: -len(suffix)].rstrip("/")
    return value


def ollama_url(path: str, endpoint: str | None = None) -> str:
    return f"{ollama_base_url(endpoint)}{path}"


def ollama_chat_url(endpoint: str | None = None) -> str:
    return ollama_url("/api/chat", endpoint)


def ollama_tags_url(endpoint: str | None = None) -> str:
    return ollama_url("/api/tags", endpoint)


def openai_base_url(endpoint: str | None = None) -> str:
    value = (endpoint or os.getenv("STACKWIRE_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")).strip().rstrip("/")
    for suffix in ("/chat/completions", "/models"):
        if value.endswith(suffix):
            value = value[: -len(suffix)].rstrip("/")
    if not value.endswith("/v1"):
        value = f"{value}/v1"
    return value


def openai_chat_url(endpoint: str | None = None) -> str:
    return f"{openai_base_url(endpoint)}/chat/completions"


def openai_models_url(endpoint: str | None = None) -> str:
    return f"{openai_base_url(endpoint)}/models"


def openai_headers(api_key: str | None = None) -> dict[str, str]:
    key = (api_key if api_key is not None else os.getenv("STACKWIRE_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))).strip()
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def installed_ollama_models(endpoint: str | None = None) -> list[str]:
    session = requests.Session()
    session.trust_env = False
    try:
        # 4s: /api/tags can lag while Ollama is busy generating; 1.5s used to
        # time out and leave the Settings model dropdowns empty.
        response = session.get(ollama_tags_url(endpoint), timeout=4.0)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []
    models = payload.get("models") if isinstance(payload, dict) else []
    if not isinstance(models, list):
        return []
    names = [str(item.get("name", "")).strip() for item in models if isinstance(item, dict)]
    return dedupe_models(names)


def installed_openai_models(endpoint: str | None = None, api_key: str | None = None) -> list[str]:
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(openai_models_url(endpoint), headers=openai_headers(api_key), timeout=3)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []
    items = payload.get("data") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return []
    names = [str(item.get("id", "")).strip() for item in items if isinstance(item, dict)]
    return dedupe_models(names)


def system_ram_gb() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        return max(0, round(psutil.virtual_memory().total / (1024**3)))
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return max(0, round(status.ullTotalPhys / (1024**3)))
        except Exception:
            return 0
    return 0


def nvidia_vram_gb() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return 0
    values: list[int] = []
    for line in result.stdout.splitlines():
        try:
            values.append(int(line.strip()) // 1024)
        except ValueError:
            continue
    return max(values, default=0)


def hardware_summary() -> tuple[str, int, int]:
    ram_gb = system_ram_gb()
    vram_gb = nvidia_vram_gb()
    cpu = os.cpu_count() or 0
    parts = [f"CPU threads: {cpu or 'unknown'}"]
    parts.append(f"RAM: {ram_gb} GB" if ram_gb else "RAM: unknown")
    parts.append(f"NVIDIA VRAM: {vram_gb} GB" if vram_gb else "NVIDIA GPU: not detected")
    return " - ".join(parts), ram_gb, vram_gb


def hardware_recommendation_note(model: ModelRecommendation, ram_gb: int, vram_gb: int) -> str:
    if ram_gb and ram_gb < model.min_ram_gb and vram_gb < model.min_vram_gb:
        return "too heavy"
    if model.min_vram_gb and vram_gb >= model.min_vram_gb:
        return "best on your GPU"
    if vram_gb >= 8 and model.min_ram_gb <= 16:
        return "recommended on GPU"
    if ram_gb and ram_gb >= model.min_ram_gb:
        return "recommended"
    if not ram_gb and model.min_ram_gb <= 12:
        return "safe starter"
    return "optional"


class ModelHubRefreshWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, *, provider: str, ollama_endpoint: str, openai_endpoint: str, api_key: str) -> None:
        super().__init__()
        self.provider = provider
        self.ollama_endpoint = ollama_endpoint
        self.openai_endpoint = openai_endpoint
        self.api_key = api_key

    @Slot()
    def run(self) -> None:
        session = requests.Session()
        session.trust_env = False
        try:
            if self.provider == "openai_compatible":
                response = session.get(openai_models_url(self.openai_endpoint), headers=openai_headers(self.api_key), timeout=5)
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data") if isinstance(payload, dict) else []
                installed = [
                    str(item.get("id", "")).strip()
                    for item in data
                    if isinstance(item, dict) and str(item.get("id", "")).strip()
                ]
                self.finished.emit(
                    {
                        "provider": self.provider,
                        "online": True,
                        "version": "OpenAI-compatible",
                        "installed": dedupe_models(installed),
                        "message": "OpenAI-compatible endpoint is reachable.",
                    }
                )
                return

            version = "unknown"
            try:
                version_response = session.get(ollama_url("/api/version", self.ollama_endpoint), timeout=2)
                if version_response.ok:
                    version_payload = version_response.json()
                    version = str(version_payload.get("version", "unknown")) if isinstance(version_payload, dict) else "unknown"
            except Exception:
                version = "unknown"

            response = session.get(ollama_tags_url(self.ollama_endpoint), timeout=5)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("models") if isinstance(payload, dict) else []
            installed = [
                str(item.get("name", "")).strip()
                for item in models
                if isinstance(item, dict) and str(item.get("name", "")).strip()
            ]
            self.finished.emit(
                {
                    "provider": self.provider,
                    "online": True,
                    "version": version,
                    "installed": dedupe_models(installed),
                    "message": f"Ollama is reachable at {ollama_base_url(self.ollama_endpoint)}.",
                }
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(short_error(exc))


class ModelPullWorker(QObject):
    progress = Signal(str, str, int, int, str)
    finished = Signal(str)
    failed = Signal(str, str)

    def __init__(self, model: str, endpoint: str) -> None:
        super().__init__()
        self.model = model
        self.endpoint = endpoint

    @Slot()
    def run(self) -> None:
        session = requests.Session()
        session.trust_env = False
        last_completed = 0
        last_clock = time.perf_counter()
        try:
            with session.post(
                ollama_url("/api/pull", self.endpoint),
                json={"name": self.model, "stream": True},
                timeout=3600,
                stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError:
                        continue
                    status = str(data.get("status", "downloading")).strip() or "downloading"
                    total = int(data.get("total") or 0)
                    completed = int(data.get("completed") or 0)
                    now = time.perf_counter()
                    speed = ""
                    if completed and now > last_clock:
                        delta_bytes = max(0, completed - last_completed)
                        delta_time = max(0.001, now - last_clock)
                        mbps = delta_bytes / delta_time / (1024 * 1024)
                        speed = f"{mbps:.1f} MB/s"
                        last_completed = completed
                        last_clock = now
                    self.progress.emit(self.model, status, completed, total, speed)
            self.finished.emit(self.model)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self.model, short_error(exc))


class ModelTestWorker(QObject):
    finished = Signal(str, float, str)
    failed = Signal(str, str)

    def __init__(self, *, provider: str, model: str, ollama_endpoint: str, openai_endpoint: str, api_key: str) -> None:
        super().__init__()
        self.provider = provider
        self.model = model
        self.ollama_endpoint = ollama_endpoint
        self.openai_endpoint = openai_endpoint
        self.api_key = api_key

    @Slot()
    def run(self) -> None:
        session = requests.Session()
        session.trust_env = False
        started = time.perf_counter()
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Reply with exactly: OK"}]
        try:
            if self.provider == "openai_compatible":
                response = session.post(
                    openai_chat_url(self.openai_endpoint),
                    headers=openai_headers(self.api_key),
                    json={"model": self.model, "messages": messages, "stream": False, "max_tokens": 8, "temperature": 0},
                    timeout=60,
                )
                response.raise_for_status()
                payload = response.json()
                choices = payload.get("choices") if isinstance(payload, dict) else []
                message = choices[0].get("message", {}) if choices and isinstance(choices[0], dict) else {}
                text = str(message.get("content", "")).strip()
            else:
                response = session.post(
                    ollama_chat_url(self.ollama_endpoint),
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "think": False,
                        "options": {"num_predict": 8, "temperature": 0},
                    },
                    timeout=120,
                )
                response.raise_for_status()
                payload = response.json()
                text = str((payload.get("message") or {}).get("content", "")).strip()
            self.finished.emit(self.model, time.perf_counter() - started, text)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self.model, short_error(exc))
