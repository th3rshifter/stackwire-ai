from __future__ import annotations

import base64
import json
import logging
import os
import random
import re
from typing import Any, cast

import requests
from PySide6.QtCore import QBuffer, QIODevice, QObject, QThread, Signal, Slot
from requests import RequestException

from app.llm import AskResult, ExpandResult, OllamaClient
from app.question_recovery import RecoveryResult


LOGGER = logging.getLogger(__name__)
STACKWIRE_API_URL = ""
STACKWIRE_API_CONNECT_TIMEOUT = 5.0
STACKWIRE_API_TIMEOUT = 300.0
_auth_headers_func = lambda: {}
_remote_request_error_func = lambda prefix, api_url, exc: f"{prefix}: {exc}"


def configure_chat_workers(
    *,
    api_url: str,
    api_connect_timeout: float,
    api_timeout: float,
    auth_headers,
    remote_request_error,
) -> None:  # noqa: ANN001
    global STACKWIRE_API_URL, STACKWIRE_API_CONNECT_TIMEOUT, STACKWIRE_API_TIMEOUT
    global _auth_headers_func, _remote_request_error_func
    STACKWIRE_API_URL = api_url
    STACKWIRE_API_CONNECT_TIMEOUT = api_connect_timeout
    STACKWIRE_API_TIMEOUT = api_timeout
    _auth_headers_func = auth_headers
    _remote_request_error_func = remote_request_error


def _auth_headers() -> dict[str, str]:
    return _auth_headers_func()


def _remote_request_error(prefix: str, api_url: str, exc: RequestException) -> str:
    return _remote_request_error_func(prefix, api_url, exc)


def _parse_recovery_result(data: dict[str, Any], _as_float, _as_int) -> RecoveryResult:
    recovery_payload = data.get("recovery") or {}
    technical_entities = recovery_payload.get("technical_entities") or []
    ambiguities = recovery_payload.get("ambiguities") or []
    candidate_questions = recovery_payload.get("candidate_questions") or []
    candidate_details = recovery_payload.get("candidate_details") or []
    return RecoveryResult(
        confidence=_as_float(recovery_payload.get("confidence"), 0.0),
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


def _build_ask_result(data: dict[str, Any], raw_text: str, recovery: RecoveryResult, _as_float, _as_int) -> AskResult:
    return AskResult(
        raw_text=str(data.get("raw_text", raw_text)),
        recovery=recovery,
        answer=str(data.get("answer", "")),
        answered=bool(data.get("answered", False)),
        recovery_latency=_as_float(data.get("recovery_latency"), 0.0),
        answer_latency=_as_float(data.get("answer_latency"), 0.0),
        total_latency=_as_float(data.get("total_latency"), 0.0),
        question_id=_as_int(data.get("question_id")),
        answer_id=_as_int(data.get("answer_id")),
        plan_domain=str(data.get("plan_domain") or "") or None,
        plan_intent=str(data.get("plan_intent") or "") or None,
        answer_model=str(data.get("answer_model", "") or ""),
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
        recovery = _parse_recovery_result(data, self._as_float, self._as_int)
        return _build_ask_result(data, self.raw_text, recovery, self._as_float, self._as_int)

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
    delta = Signal(int, str)
    thinking = Signal(int, str)  # DeepThink: model reasoning chunks
    finished = Signal(int, object)
    failed = Signal(int, str)
    done = Signal()

    def __init__(
        self,
        raw_text: str,
        context: list[str],
        *,
        trusted_text: bool = False,
        storage_session_id: int | None = None,
        stream_generation: int = 0,
        creative: bool = False,
    ) -> None:
        super().__init__()
        self.raw_text = raw_text
        self.context = context
        self.trusted_text = trusted_text
        self.storage_session_id = storage_session_id
        self.stream_generation = stream_generation
        self.creative = creative
        self.api_url = STACKWIRE_API_URL
        self.client = None if self.api_url else OllamaClient(storage_session_id=storage_session_id)
        self.session = requests.Session()
        self.session.trust_env = False

    def _emit_delta(self, chunk: str) -> None:
        self.delta.emit(self.stream_generation, chunk)

    @Slot()
    def run(self) -> None:
        try:
            if self.api_url:
                self._ask_remote_stream()
                return
            if self.client is None:
                raise RuntimeError("Local Ollama client is not initialized")
            result = self.client.ask_stream(
                self.raw_text,
                self.context,
                trusted_text=self.trusted_text,
                on_recovery=self.recovered.emit,
                on_delta=self._emit_delta,
                on_thinking=lambda c: self.thinking.emit(self.stream_generation, c),
                should_stop=lambda: QThread.currentThread().isInterruptionRequested(),
                creative=self.creative,
            )
            self.finished.emit(self.stream_generation, result)
        except RequestException as exc:
            self.failed.emit(self.stream_generation, _remote_request_error("Processing request failed", self.api_url, exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self.stream_generation, str(exc))
        finally:
            self.done.emit()

    def _ask_remote_stream(self) -> None:
        """Read SSE from the server's /ask/stream endpoint, emitting deltas live."""
        import json as _json

        payload = _json.dumps(
            {"text": self.raw_text, "context": self.context, "trusted_text": self.trusted_text},
            ensure_ascii=False,
        )
        try:
            with self.session.post(
                f"{self.api_url}/ask/stream",
                data=payload.encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8", **_auth_headers()},
                timeout=(STACKWIRE_API_CONNECT_TIMEOUT, STACKWIRE_API_TIMEOUT),
                stream=True,
            ) as response:
                self._raise_for_status(response, "/ask/stream")
                result_data: dict[str, Any] = {}
                for line in response.iter_lines(decode_unicode=True):
                    if QThread.currentThread().isInterruptionRequested():
                        break
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        msg = _json.loads(line[6:])
                    except _json.JSONDecodeError:
                        continue
                    msg_type = msg.get("type", "")
                    if msg_type == "delta":
                        self._emit_delta(str(msg.get("content", "")))
                    elif msg_type == "thinking":
                        self.thinking.emit(self.stream_generation, str(msg.get("content", "")))
                    elif msg_type == "recovery":
                        self.recovered.emit(str(msg.get("content", "")))
                    elif msg_type == "done":
                        result_data = msg
                if result_data:
                    recovery = _parse_recovery_result(result_data, self._as_float, self._as_int)
                    result = _build_ask_result(result_data, self.raw_text, recovery, self._as_float, self._as_int)
                    self.finished.emit(self.stream_generation, result)
                else:
                    self.failed.emit(self.stream_generation, "Server returned no response")
        except RequestException as exc:
            self.failed.emit(self.stream_generation, _remote_request_error("Processing request failed", self.api_url, exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self.stream_generation, str(exc))
        finally:
            self.done.emit()


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
            answer_model=str(data.get("answer_model", "") or ""),
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
    delta = Signal(int, str)  # stream_generation, chunk — drives the same render pipeline as text
    finished = Signal(int, str)
    failed = Signal(int, str)
    done = Signal()

    def __init__(self, image_b64: str, prompt: str, *, image_generation: int = 0, stream_generation: int = 0, creative: bool = False) -> None:
        super().__init__()
        self.image_b64 = image_b64
        self.prompt = prompt
        self.image_generation = image_generation
        self.stream_generation = stream_generation
        self.creative = creative
        self.api_url = STACKWIRE_API_URL
        self.client = None if self.api_url else OllamaClient()
        self.session = requests.Session()
        self.session.trust_env = False

    def _emit_delta(self, chunk: str) -> None:
        self.delta.emit(self.stream_generation, chunk)

    @Slot()
    def run(self) -> None:
        try:
            if self.api_url:
                # Remote API has no streaming endpoint: emit the whole answer as one delta.
                answer = self._analyze_remote()
                if answer:
                    self._emit_delta(answer)
                self.finished.emit(self.image_generation, answer)
                return
            if self.client is None:
                raise RuntimeError("Local Ollama client is not initialized")
            answer = self.client.analyze_image_stream(self.image_b64, self.prompt, self._emit_delta, creative=self.creative)
            self.finished.emit(self.image_generation, answer)
        except RequestException as exc:
            self.failed.emit(self.image_generation, _remote_request_error("Image analysis request failed", self.api_url, exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self.image_generation, str(exc))
        finally:
            self.done.emit()

    def _analyze_remote(self) -> str:
        _img = self.image_b64[0] if isinstance(self.image_b64, (list, tuple)) and self.image_b64 else self.image_b64
        payload: dict[str, str] = {"image_b64": _img, "prompt": self.prompt}
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


class ImageGenWorker(QObject):
    """Background worker that generates an image from a text prompt.

    Backend selection (auto):
    - **Pollinations.ai** (DEFAULT, free, no API key) — used when nothing else is configured.
    - OpenAI DALL-E — used when STACKWIRE_OPENAI_API_KEY / OPENAI_API_KEY is set.
    - Local AUTOMATIC1111 / ComfyUI or any OpenAI-compatible endpoint — set STACKWIRE_IMAGE_GEN_URL.
    Force a backend with STACKWIRE_IMAGE_GEN_PROVIDER = pollinations | openai.
    """

    finished = Signal(int, str, str)  # generation, image_b64, prompt
    failed = Signal(int, str)
    done = Signal()

    def __init__(self, prompt: str, *, generation: int = 0) -> None:
        super().__init__()
        self.prompt = prompt
        self.generation = generation
        self.session = requests.Session()
        self.session.trust_env = False

    @Slot()
    def run(self) -> None:
        try:
            b64 = self._generate()
            self.finished.emit(self.generation, b64, self.prompt)
        except RequestException as exc:
            self.failed.emit(self.generation, _remote_request_error("Image generation failed", "image service", exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self.generation, str(exc))
        finally:
            self.done.emit()

    def _generate(self) -> str:
        provider = os.getenv("STACKWIRE_IMAGE_GEN_PROVIDER", "").strip().lower()
        custom_url = os.getenv("STACKWIRE_IMAGE_GEN_URL", "").strip()
        api_key = (os.getenv("STACKWIRE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")).strip()

        # Explicit choice / configuration wins; otherwise default to the free Pollinations service.
        if provider == "pollinations":
            return self._generate_pollinations()
        if custom_url:
            return self._generate_openai_compatible(custom_url, api_key)
        if provider == "openai" or api_key:
            return self._generate_openai_compatible("https://api.openai.com/v1/images/generations", api_key)
        return self._generate_pollinations()

    def _image_size(self) -> tuple[int, int]:
        size = os.getenv("STACKWIRE_IMAGE_GEN_SIZE", "1024x1024").strip() or "1024x1024"
        try:
            w_str, h_str = size.lower().split("x", 1)
            return max(64, int(w_str)), max(64, int(h_str))
        except Exception:
            return 1024, 1024

    def _generate_pollinations(self) -> str:
        """Free, key-less text-to-image via pollinations.ai. Returns raw image bytes as base64."""
        import urllib.parse

        width, height = self._image_size()
        model = os.getenv("STACKWIRE_POLLINATIONS_MODEL", "flux").strip() or "flux"
        encoded = urllib.parse.quote(self.prompt[:1800], safe="")
        url = f"https://image.pollinations.ai/prompt/{encoded}"
        params: dict[str, Any] = {
            "width": width,
            "height": height,
            "seed": random.randint(1, 1_000_000),
            "model": model,
            "nologo": "true",
            "referrer": "stackwire",
        }
        # Optional free token (https://enter.pollinations.ai) gives reliable, unthrottled
        # access without a paid key. Anonymous works too, just rate-limited to 1 at a time.
        headers: dict[str, str] = {}
        token = os.getenv("POLLINATIONS_TOKEN", "").strip()
        if token:
            params["token"] = token
            headers["Authorization"] = f"Bearer {token}"

        response = self.session.get(url, params=params, headers=headers, timeout=(10, 180))
        if response.status_code in (402, 429):
            raise RuntimeError(
                "Бесплатный сервис Pollinations сейчас перегружен (лимит запросов с этого IP).\n"
                "Подождите ~минуту и попробуйте снова, либо получите бесплатный токен на\n"
                "https://enter.pollinations.ai и задайте переменную окружения POLLINATIONS_TOKEN."
            )
        response.raise_for_status()
        content = response.content
        if not content or len(content) < 256:
            raise RuntimeError("Pollinations вернул пустое изображение — попробуйте ещё раз или измените промпт.")
        # Pollinations returns JPEG, but the chat embeds images as data:image/png. Normalize
        # to PNG so the inline thumbnail renders correctly. QImage is safe off the GUI thread.
        try:
            from PySide6.QtGui import QImage
            img = QImage.fromData(content)
            if not img.isNull():
                buf = QBuffer()
                buf.open(QIODevice.OpenModeFlag.WriteOnly)
                img.save(buf, "PNG")  # string format works off-thread; b"PNG" is broken in this PySide6
                return base64.b64encode(bytes(buf.data().data())).decode("ascii")
        except Exception:
            LOGGER.exception("pollinations PNG normalization failed; using raw bytes")
        return base64.b64encode(content).decode("ascii")

    def _generate_openai_compatible(self, url: str, api_key: str) -> str:
        if not api_key and "openai.com" in url:
            raise RuntimeError(
                "OpenAI генерация требует ключ. Задайте OPENAI_API_KEY,\n"
                "или оставьте всё пустым — тогда используется бесплатный Pollinations."
            )
        model = os.getenv("STACKWIRE_IMAGE_GEN_MODEL", "dall-e-3").strip()
        width, height = self._image_size()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": self.prompt,
            "n": 1,
            "size": f"{width}x{height}",
            "response_format": "b64_json",
        }
        response = self.session.post(url, json=cast(Any, payload), headers=headers, timeout=(10, 120))
        response.raise_for_status()
        data = response.json()

        # OpenAI / compatible format: {"data": [{"b64_json": "..."}]}
        items = data.get("data", [])
        if items and "b64_json" in items[0]:
            return str(items[0]["b64_json"])
        # AUTOMATIC1111 format: {"images": ["base64..."]}
        images = data.get("images", [])
        if images:
            return str(images[0])
        raise RuntimeError(
            f"Неизвестный формат ответа от сервиса генерации изображений: {list(data.keys())}"
        )


class SuggestionsWorker(QObject):
    """Generates 3 follow-up question suggestions for the last answer."""

    finished = Signal(list)   # list[str] — up to 3 suggestion strings
    done = Signal()

    def __init__(self, question: str, answer: str) -> None:
        super().__init__()
        self.question = question
        self.answer = answer

    @Slot()
    def run(self) -> None:
        try:
            import json as _json
            from app.llm import OllamaClient
            client = OllamaClient()
            prompt = (
                f"Question: {self.question[:280]}\n\n"
                f"Answer: {self.answer[:500]}\n\n"
                "Generate exactly 3 concise follow-up questions (max 9 words each) the user might want to ask next.\n"
                "Output ONLY a JSON array of 3 strings, nothing else.\n"
                'Example: ["What are the main risks?", "How to implement this?", "Any alternatives?"]'
            )
            raw = client._chat(
                [{"role": "user", "content": prompt}],
                {"num_predict": 140, "temperature": 0.5},
                timeout=25,
            )
            m = re.search(r"\[.*?\]", raw, re.DOTALL)
            if m:
                items = _json.loads(m.group(0))
                if isinstance(items, list) and len(items) >= 2:
                    self.finished.emit([str(s).strip() for s in items[:3]])
                    return
        except Exception:
            pass
        self.done.emit()


class AgentWorker(QObject):
    """Agent mode LLM step: ask the model what to do next given the agent transcript."""

    result = Signal(str)
    delta = Signal(str)
    failed = Signal(str)
    done = Signal()

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        super().__init__()
        self.messages = messages
        self.client = OllamaClient()

    @Slot()
    def run(self) -> None:
        try:
            text = self.client._chat_stream(
                self.messages,
                {"num_ctx": 8192, "num_predict": 1200, "temperature": 0.15, "top_p": 0.85},
                self.delta.emit,
                timeout=300,
                should_stop=lambda: QThread.currentThread().isInterruptionRequested(),
            )
            self.result.emit(text or "")
        except RequestException as exc:
            self.failed.emit(_remote_request_error_func("Agent request failed", STACKWIRE_API_URL, exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
        finally:
            self.done.emit()


class CommandWorker(QObject):
    """Runs an APPROVED shell command off the GUI thread and returns its output."""

    result = Signal(str)
    done = Signal()

    def __init__(self, command: str) -> None:
        super().__init__()
        self.command = command

    @Slot()
    def run(self) -> None:
        try:
            from app.agent import run_command

            out = run_command(self.command)
        except Exception as exc:  # noqa: BLE001
            out = f"[failed: {exc}]"
        self.result.emit(out)
        self.done.emit()
