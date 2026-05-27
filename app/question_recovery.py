import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from app.tech_terms import normalize_spoken_technical_terms

try:
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover - startup scripts install rapidfuzz
    fuzz = None  # type: ignore[assignment]
    process = None  # type: ignore[assignment]


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip()
STACKWIRE_MODE = os.getenv("STACKWIRE_MODE", "fast").strip().lower()
IS_FAST_MODE = STACKWIRE_MODE != "accurate"
RECOVERY_LOCAL_FAST_PATH = os.getenv("RECOVERY_LOCAL_FAST_PATH", "1" if IS_FAST_MODE else "0").strip().lower() not in {"0", "false", "no", "off"}
DEFAULT_MODEL = (
    os.getenv("FAST_RECOVERY_MODEL", os.getenv("RECOVERY_MODEL", os.getenv("OLLAMA_RECOVERY_MODEL", "llama3.2:latest")))
    if IS_FAST_MODE
    else os.getenv("RECOVERY_MODEL", os.getenv("OLLAMA_RECOVERY_MODEL", "llama3.2:latest"))
)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75" if IS_FAST_MODE else "0.80"))
MAX_CANDIDATES = 2 if IS_FAST_MODE else 3
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "3072" if IS_FAST_MODE else "4096"))
DEFAULT_RECOVERY_NUM_PREDICT = int(os.getenv("OLLAMA_RECOVERY_NUM_PREDICT", "160" if IS_FAST_MODE else "320"))
RECOVERY_CONTEXT_LINES = int(os.getenv("RECOVERY_CONTEXT_LINES", "6"))
_RICH_CONSOLE: Any | None = None

QUESTION_MARKERS = (
    "что",
    "чем",
    "как",
    "когда",
    "зачем",
    "почему",
    "где",
    "какой",
    "какая",
    "какие",
    "отлич",
    "сравни",
    "объясни",
    "расскажи",
    "покажи",
    "пример",
    "when",
    "what",
    "why",
    "how",
)

TECH_POSITION_MARKERS = (
    "что такое",
    "чем отличается",
    "как работает",
    "когда использовать",
    "покажи пример",
    "пример",
    "команда",
    "файл",
    "директория",
    "порт",
    "протокол",
    "сервис",
    "утилита",
    "модуль",
    "плагин",
    "метрика",
    "лог",
    "алерт",
    "config",
    "command",
    "file",
    "directory",
    "port",
    "protocol",
    "service",
    "module",
    "plugin",
    "metric",
    "log",
    "alert",
)

TRANSLITERATED_TECH_TERMS: frozenset[str] = frozenset({
    "ансибл", "плейбук", "хандлер", "хендлер", "таска", "темплейт", "инвентори",
    "кубернетес", "кубер", "кубектл", "хелм", "опеншифт", "деплоймент",
    "неймспейс", "нэймспейс", "под", "поды", "подов", "нода", "ноды", "ингресс",
    "сервисмеш", "истио",
    "докер", "докерфайл", "компоуз",
    "терраформ", "стейт", "провайдер",
    "дженкинс", "пайплайн", "гитхаб", "битбакет", "сонаркьюб",
    "прометеус", "прометей", "графана", "алертменеджер", "алертменеджере",
    "кибана", "логсташ", "флюентд", "флюентбит", "флуентд", "флуентбит",
    "опентелеметри", "джаегер", "джейгер", "темпо", "датадог",
    "постгрес", "постгресql", "оракл", "монго", "монгодб", "редис",
    "кликхаус", "ликвибейс", "патрони", "мариадб",
    "кафка", "зукипер", "зукепер", "рэббитмq", "рэббит",
    "нжинкс", "энджинкс", "хапрокси", "веблоджик",
    "цефс", "сеф", "ваулт", "волт", "харбор", "нексус", "артифактори",
    "системд", "сислог", "крон", "кронтаб",
    "тлс", "ссл", "мтлс", "ссш", "нфс", "айпи",
})

SUSPICIOUS_PHONETIC_PREFIXES: tuple[str, ...] = (
    "кубер",
    "кубе",
    "куб",
    "докер",
    "промет",
    "граф",
    "терра",
    "ансиб",
    "джин",
    "джен",
    "дженк",
    "кинс",
    "кино",
    "декор",
    "декларат",
    "постгр",
    "патрон",
    "эндж",
    "нгин",
    "кафк",
    "ваул",
    "волт",
)

RUSSIAN_TECH_WORDS: frozenset[str] = frozenset({
    "репликация",
    "балансировщик",
    "контейнер",
    "контейнеры",
    "сертификат",
    "сертификаты",
    "секрет",
    "секреты",
    "протокол",
    "протоколы",
    "метрика",
    "метрики",
    "алерт",
    "алерты",
    "лог",
    "логи",
    "очередь",
    "очереди",
    "индекс",
    "индексы",
    "кластер",
    "нода",
    "ноды",
    "под",
    "поды",
})

PATH_TRANSLITERATIONS: tuple[tuple[str, str], ...] = (
    (r"\bдев\s+и\s+прок\b", "/dev и /proc"),
    (r"\bдев\b", "/dev"),
    (r"\bпрок\b", "/proc"),
    (r"\bетс\b", "/etc"),
    (r"\bвар\s+лог\b", "/var/log"),
)

FUZZY_TERM_CATALOG: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Kubernetes", "Kubernetes", ("kubernetes", "кубернетес", "кубернетис", "кубернетик", "губернии тёс", "губерни тес")),
    ("kubectl", "Kubernetes", ("kubectl", "кубектл", "куб ctl", "кубеконтрол", "куб контрол")),
    ("kubeadm", "Kubernetes", ("kubeadm", "куб адм", "кубадм", "кубеадм")),
    ("Deployment", "Kubernetes", ("deployment", "деплоймент", "дипло и мент", "дипло и менты", "диплоймент")),
    ("Pod", "Kubernetes", ("pod", "pods", "под", "поды", "пад", "пады", "падаешь")),
    ("readinessProbe", "Kubernetes probes", ("readinessprobe", "readiness probe", "рединес", "рединесс", "готовность проба")),
    ("livenessProbe", "Kubernetes probes", ("livenessprobe", "liveness probe", "лайвнес", "лайвнесс", "живость проба")),
    ("startupProbe", "Kubernetes probes", ("startupprobe", "startup probe", "стартап проба", "старт ап проба")),
    ("Ingress", "Kubernetes", ("ingress", "ингресс", "один грея с", "ин грея с", "ингрея с")),
    ("Docker", "Containers", ("docker", "докер", "докерн", "докер контейнер")),
    ("Dockerfile", "Containers", ("dockerfile", "докерфайл", "докер файл")),
    ("docker-compose.yml", "Containers", ("docker compose", "docker-compose", "докер компоуз", "докеркомпоуз", "компоуз")),
    ("Prometheus", "Observability", ("prometheus", "прометей", "прометеус", "прометейс", "прометея")),
    ("Grafana", "Observability", ("grafana", "графана", "графаны", "грофана")),
    ("Alertmanager", "Observability", ("alertmanager", "алертменеджер", "алерт менеджер")),
    ("Terraform", "IaC", ("terraform", "терраформ", "тероформ", "тераформ")),
    ("Terraform state", "IaC", ("terraform state", "терраформ стейт", "tf state")),
    ("Ansible", "Configuration management", ("ansible", "ансибл", "энсибл")),
    ("playbook", "Configuration management", ("playbook", "playbooks", "плейбук", "плейбуки", "букет", "коль буки")),
    ("role", "Configuration management", ("role", "roles", "роль", "роли", "рауль", "рауля", "роули")),
    ("collection", "Configuration management", ("collection", "collections", "коллекция", "коллекции")),
    ("Jenkins Pipeline", "CI/CD", ("jenkins pipeline", "дженкинс пайплайн", "джин киноха", "дженкинс", "кинс")),
    ("declarative pipeline", "CI/CD", ("declarative pipeline", "декларативный пайплайн", "декоративный подход", "декларативный подход")),
    ("GitLab CI", "CI/CD", ("gitlab ci", "гитлаб ci", "гитлаб си ай")),
    ("runner", "CI/CD", ("runner", "раннер", "ранер")),
    ("pipeline", "CI/CD", ("pipeline", "пайплайн", "пайплан")),
    ("TCP", "Networking", ("tcp", "тсп", "тиси пи", "тс пи")),
    ("UDP", "Networking", ("udp", "юдипи", "юдп", "уди пи")),
    ("DNS", "Networking", ("dns", "днс", "диэнэс", "дэнээс")),
    ("TLS", "Networking security", ("tls", "тлс", "тиэлэс")),
    ("mTLS", "Networking security", ("mtls", "m tls", "мтлс", "эм тлс")),
    ("NFS", "Storage", ("nfs", "нфс", "эн эф эс")),
    ("S3", "Storage", ("s3", "с3", "эс три")),
    ("Vault", "Security", ("vault", "ваулт", "волт")),
    ("PostgreSQL", "Databases", ("postgresql", "postgres", "постгрес", "постгресql")),
    ("Patroni", "Databases", ("patroni", "патрони")),
    ("Redis", "Databases", ("redis", "редис")),
    ("Kafka", "Messaging", ("kafka", "кафка")),
    ("RabbitMQ", "Messaging", ("rabbitmq", "rabbit", "рэббит", "рэббитмq")),
    ("Nginx", "Web/proxy", ("nginx", "nginx", "нжинкс", "энджинкс")),
    ("systemd", "Linux", ("systemd", "системд")),
    ("journalctl", "Linux", ("journalctl", "джорнал си ти эл", "журнал ctl")),
    ("systemctl", "Linux", ("systemctl", "систем си ти эл")),
    ("D state", "Linux process state", ("d state", "д стейт", "ди стейт", "uninterruptible sleep")),
    ("df", "Linux filesystem", ("df", "ди эф")),
    ("du", "Linux filesystem", ("du", "ди ю", "дю")),
    ("OOMKilled", "Kubernetes troubleshooting", ("oomkilled", "oom killed", "ум килд", "оом килд")),
    ("CrashLoopBackOff", "Kubernetes troubleshooting", ("crashloopbackoff", "crash loop", "крашлуп", "краш луп")),
    ("load average", "Linux troubleshooting", ("load average", "лоад average", "лоад эвередж")),
)

FUZZY_MATCH_THRESHOLD = int(os.getenv("RECOVERY_FUZZY_MATCH_THRESHOLD", "86"))


def _configure_logging() -> logging.Logger:
    global _RICH_CONSOLE
    log_dir = Path(os.getenv("STACKWIRE_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "stackwire.log"

    root = logging.getLogger()
    root.setLevel(os.getenv("STACKWIRE_LOG_LEVEL", "INFO"))

    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(
        log_file,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    root.addHandler(file_handler)

    if os.getenv("STACKWIRE_RICH_LOGS", "1").strip() == "1":
        try:
            from rich.console import Console
            from rich.logging import RichHandler

            _RICH_CONSOLE = Console()
            rich_handler = RichHandler(
                console=_RICH_CONSOLE,
                rich_tracebacks=True,
                markup=False,
                show_path=False,
            )
            rich_handler.setFormatter(logging.Formatter("%(message)s"))
            root.addHandler(rich_handler)
        except ImportError:
            _RICH_CONSOLE = None

    return logging.getLogger(__name__)


LOGGER = _configure_logging()


@dataclass(frozen=True)
class RecoveryResult:
    confidence: float
    recovered_question: str
    detected_topic: str
    reason: str
    technical_entities: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    needs_manual_fix: bool = False
    candidate_questions: list[str] = field(default_factory=list)
    candidate_quality: str = "unclear"
    candidate_details: list[dict[str, Any]] = field(default_factory=list)


def normalize_lightweight(text: str) -> str:
    normalized = text.strip()
    for pattern, replacement in PATH_TRANSLITERATIONS:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    normalized = normalize_spoken_technical_terms(normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([?!,.:;])", r"\1", normalized)
    return normalized.strip()


class QuestionRecovery:
    def __init__(
        self,
        *,
        model: str | None = None,
        ollama_url: str | None = None,
        session: requests.Session | None = None,
        llm_complete: Callable[[str], str] | None = None,
        use_llm: bool = True,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.ollama_url = ollama_url or OLLAMA_URL
        self.session = session or requests.Session()
        self.session.trust_env = False
        self.llm_complete = llm_complete
        self.use_llm = use_llm

    def recover(self, raw_text: str, context: list[str]) -> RecoveryResult:
        raw_text = raw_text.strip()
        normalized = normalize_lightweight(raw_text)
        normalized_context = [normalize_lightweight(line) for line in context[-RECOVERY_CONTEXT_LINES:] if line.strip()]

        LOGGER.info("raw transcript=%r", raw_text)
        LOGGER.info("normalized transcript=%r", normalized)

        if not normalized:
            return RecoveryResult(0.0, "", "NEED_CLARIFICATION", "empty input", needs_manual_fix=True)

        if (
            self._is_clean_supported_question(normalized)
            and not self._has_distortion_signal(raw_text)
            and self.llm_complete is None
        ):
            recovered = self._ensure_question_punctuation(normalized)
            entities = self._extract_generic_entities(recovered)
            result = RecoveryResult(
                confidence=0.90,
                recovered_question=recovered,
                detected_topic="Technical",
                reason="local clean supported question",
                technical_entities=entities,
                ambiguities=[],
                needs_manual_fix=False,
                candidate_questions=[recovered],
                candidate_quality="raw_copy",
                candidate_details=[
                    {
                        "question": recovered,
                        "confidence": 0.90,
                        "candidate_quality": "raw_copy",
                    }
                ],
            )
            self._log_result(result)
            return result

        local = self._recover_locally(normalized)
        if self.use_llm and RECOVERY_LOCAL_FAST_PATH and self.llm_complete is None:
            validated_local = self._validate(local, raw_text, normalized, normalized_context)
            if (
                not validated_local.needs_manual_fix
                and validated_local.confidence >= CONFIDENCE_THRESHOLD
            ):
                LOGGER.info("using local recovery fast path")
                self._log_result(validated_local)
                return validated_local

        if self.use_llm:
            try:
                llm_result = self._recover_with_llm(raw_text, normalized, normalized_context)
                validated = self._validate(llm_result, raw_text, normalized, normalized_context)
                self._log_result(validated)
                return validated
            except Exception as exc:
                LOGGER.warning("question recovery LLM failed, using local fallback: %s", exc)

        validated = self._validate(local, raw_text, normalized, normalized_context)
        self._log_result(validated)
        return validated

    def _recover_locally(self, normalized: str) -> RecoveryResult:
        entities = self._extract_generic_entities(normalized)
        confidence = 0.55 if self._looks_question_like(normalized) and entities else 0.25
        token_count = len(re.findall(r"[A-Za-zА-Яа-яЁё0-9/+#.-]+", normalized))

        if self._looks_question_like(normalized) and entities and re.search(r"[A-Za-z/]", normalized):
            confidence = 0.84 if token_count <= 16 else 0.65

        if self._looks_question_like(normalized) and any(entity.startswith("/") for entity in entities):
            confidence = 0.82

        return RecoveryResult(
            confidence=confidence,
            recovered_question=self._ensure_question_punctuation(normalized),
            detected_topic="Technical" if entities else "NEED_CLARIFICATION",
            reason="local generic fallback",
            technical_entities=entities,
            needs_manual_fix=confidence < CONFIDENCE_THRESHOLD,
            candidate_questions=[self._ensure_question_punctuation(normalized)] if normalized else [],
            candidate_quality="unclear",
        )

    def _recover_with_llm(self, raw_text: str, normalized: str, context: list[str]) -> RecoveryResult:
        prompt = self._build_prompt(raw_text, normalized, context)
        content = self._complete(prompt)
        payload = self._parse_json(content)
        candidate_details = self._parse_candidate_details(payload)

        candidates = [
            detail["question"]
            for detail in candidate_details
            if str(detail.get("question", "")).strip()
        ]
        if not candidates:
            candidates = self._as_string_list(payload.get("candidate_questions") or payload.get("candidates"))

        recovered_question = str(payload.get("recovered_question", "")).strip()
        if recovered_question and recovered_question not in candidates:
            candidates.insert(0, recovered_question)

        candidate_quality = str(payload.get("candidate_quality", "")).strip() or (
            str(candidate_details[0].get("candidate_quality", "")).strip() if candidate_details else "unclear"
        )

        return RecoveryResult(
            confidence=self._clamp_float(payload.get("confidence", 0.0)),
            recovered_question=recovered_question,
            detected_topic=str(payload.get("detected_topic", "NEED_CLARIFICATION")).strip() or "NEED_CLARIFICATION",
            reason=str(payload.get("reason", "")).strip(),
            technical_entities=self._as_string_list(payload.get("technical_entities")),
            ambiguities=self._as_string_list(payload.get("ambiguities")),
            needs_manual_fix=bool(payload.get("needs_manual_fix", False)),
            candidate_questions=candidates[:MAX_CANDIDATES],
            candidate_quality=candidate_quality if candidate_quality else "unclear",
            candidate_details=candidate_details[:MAX_CANDIDATES],
        )

    def _complete(self, prompt: str) -> str:
        if self.llm_complete is not None:
            return self.llm_complete(prompt)

        payload_dict = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "format": "json",
                "options": {
                    "num_ctx": DEFAULT_NUM_CTX,
                    "num_predict": DEFAULT_RECOVERY_NUM_PREDICT,
                    "temperature": float(os.getenv("OLLAMA_RECOVERY_TEMPERATURE", "0.05")),
                    "top_p": 0.7,
                    "top_k": 20,
                    "repeat_penalty": 1.08,
                },
        }
        if OLLAMA_KEEP_ALIVE:
            payload_dict["keep_alive"] = OLLAMA_KEEP_ALIVE
        payload = json.dumps(payload_dict, ensure_ascii=False).encode("utf-8")

        response = self.session.post(
            self.ollama_url,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        return str(message.get("content", "")).strip()

    def _build_prompt(self, raw_text: str, normalized: str, context: list[str]) -> str:
        context_text = "\n".join(context[-RECOVERY_CONTEXT_LINES:]) if context else "(empty)"
        language_hints = self._language_hints(raw_text, normalized, context)
        generic_entities = self._extract_generic_entities(" ".join([*context[-RECOVERY_CONTEXT_LINES:], normalized]))
        generic_entities_text = ", ".join(generic_entities) if generic_entities else "(none)"
        fuzzy_hints = self._fuzzy_entity_hints(" ".join([*context[-RECOVERY_CONTEXT_LINES:], normalized]))
        fuzzy_hints_text = ", ".join(fuzzy_hints) if fuzzy_hints else "(none)"
        mode_hint = (
            "FAST mode: be concise. Return one repaired transcript, no alternatives."
            if IS_FAST_MODE
            else "ACCURATE mode: return one repaired transcript and note real ambiguities."
        )

        return f"""
You repair a noisy speech-to-text transcript from Russian technical content.
The content can contain mixed Russian and English DevOps/SRE/IT terminology.
{mode_hint}

Raw transcript:
{raw_text}

Light generic normalization:
{normalized}

Last context lines:
{context_text}

Detected language hints:
{language_hints}

Generic entity hints from regex patterns:
{generic_entities_text}

Fuzzy technical entity hints from rapidfuzz:
{fuzzy_hints_text}

Return strict JSON only:
{{
  "confidence": 0.0,
  "recovered_question": "...",
  "detected_topic": "...",
  "technical_entities": ["..."],
  "ambiguities": ["..."],
  "needs_manual_fix": false,
  "candidate_quality": "stt_repair",
  "candidate_questions": ["..."],
  "candidate_details": [
    {{"question": "...", "confidence": 0.0, "candidate_quality": "stt_repair"}}
  ],
  "reason": "short explanation"
}}

Transcript repair rules:
- Your task is not to guess a new question. Repair the STT transcript.
- Preserve all audible words, order, repeated fragments and the original question structure.
- You may only replace technical terms, acronyms, product names, commands, file paths and API object names when the replacement is phonetically close to raw transcript or explicitly supported by context.
- Do not add technologies, tools or concepts that are not present by sound in raw transcript or explicitly present in context.
- Do not rewrite fragments into a cleaner semantic question. Do not merge several fragments into one new question.
- Do not add clarifying words such as "роль", "используется", "в Linux", "в Kubernetes", "CI/CD", "networking" unless they are audible in raw transcript or present in context.
- If raw transcript is already readable, keep it and only normalize punctuation/capitalization.
- If confidence is not high, return the cleaned raw transcript as recovered_question, set needs_manual_fix=true, and put only that same text in candidate_questions.
- candidate_questions must contain at most one item: the same repaired transcript as recovered_question.
- candidate_quality must be one of: stt_repair, raw_copy, unclear.
- Preserve likely technical entities in English spelling when phonetically supported: product names, CLI tools, commands, file paths, acronyms, protocols, config keys, flags, service names and technology names.
- Fuzzy hints are only spelling hints. Ignore any fuzzy hint that would add a term not supported by raw transcript or context.
- Score confidence by literal transcript support, not by how plausible the final question sounds.
- Never answer the question. Only repair the transcript and score it.
""".strip()

    def _validate(
        self,
        result: RecoveryResult,
        raw_text: str,
        normalized: str,
        context: list[str],
    ) -> RecoveryResult:
        recovered = self._ensure_question_punctuation(normalize_lightweight(result.recovered_question.strip() or normalized))
        confidence = self._clamp_float(result.confidence)
        detected_topic = result.detected_topic.strip() or "Technical"
        reason = result.reason.strip() or "validated"

        entity_text = " ".join([raw_text, normalized, recovered, *context])
        generic_entities = self._extract_generic_entities(entity_text)
        fuzzy_entities = [
            match["term"]
            for match in self._fuzzy_term_matches(entity_text)
            if self._catalog_term_is_supported(str(match["term"]), entity_text)
        ]
        technical_entities = self._dedupe([*result.technical_entities, *generic_entities, *fuzzy_entities])
        ambiguities = self._dedupe(result.ambiguities)

        candidate_details = self._normalize_candidate_details(result, raw_text, normalized)
        candidates = self._dedupe(
            [
                self._ensure_question_punctuation(normalize_lightweight(candidate))
                for candidate in result.candidate_questions
                if candidate.strip()
            ]
        )
        detail_questions = [detail["question"] for detail in candidate_details]
        candidates = self._dedupe([*detail_questions, *candidates])

        if recovered and recovered not in candidates:
            candidates.insert(0, recovered)

        support_problem = self._transcript_support_problem(raw_text, normalized, recovered, context)
        if support_problem:
            recovered = self._format_recovered_question(normalized)
            candidates = [recovered]
            confidence = min(confidence, 0.55)
        else:
            recovered = self._format_recovered_question(recovered)
            supported_candidates = [
                self._format_recovered_question(candidate)
                for candidate in candidates
                if not self._transcript_support_problem(raw_text, normalized, candidate, context)
            ]
            candidates = self._dedupe([recovered, *supported_candidates])[:1]

        candidate_quality = self._candidate_quality(recovered, raw_text, normalized)

        question_like = self._looks_question_like(recovered)
        enough_signal = bool(
            technical_entities
            or self._has_technical_position(normalized)
            or self._has_technical_position(recovered)
        )
        contradiction_risk = self._contradiction_risk(raw_text, recovered, technical_entities)
        needs_manual_fix = result.needs_manual_fix

        if support_problem:
            needs_manual_fix = True
            reason = f"{reason}; unsupported transcript repair: {support_problem}"

        if self._is_clean_supported_question(recovered) and not needs_manual_fix and len(ambiguities) < 2:
            confidence = max(confidence, 0.86)
            needs_manual_fix = False
            reason = "clean supported question"

        if self._has_unresolved_cyrillic_tech_token(recovered):
            confidence = min(confidence, 0.64)
            needs_manual_fix = True
            reason = f"{reason}; unresolved Cyrillic technical term"

        if not question_like:
            confidence = min(confidence, 0.45)
            needs_manual_fix = True
            reason = f"{reason}; missing question structure"

        if not enough_signal:
            confidence = min(confidence, 0.55)
            needs_manual_fix = True
            reason = f"{reason}; insufficient technical signal"

        if self._is_very_noisy(raw_text) and confidence < 0.9:
            confidence = min(confidence, 0.55)
            needs_manual_fix = True
            reason = f"{reason}; raw transcript is very noisy"

        if contradiction_risk:
            confidence = min(confidence, 0.65)
            needs_manual_fix = True
            reason = f"{reason}; recovered question has weak support in raw transcript"

        if len(ambiguities) >= 2 and confidence < 0.9:
            confidence = min(confidence, 0.79)
            needs_manual_fix = True
            reason = f"{reason}; multiple plausible interpretations"

        if candidate_quality == "raw_copy" and self._has_distortion_signal(raw_text):
            confidence = min(confidence, 0.45)
            needs_manual_fix = True
            reason = f"{reason}; candidate is raw transcript copy of distorted input"

        if confidence < CONFIDENCE_THRESHOLD:
            needs_manual_fix = True

        if needs_manual_fix and detected_topic == "":
            detected_topic = "NEED_CLARIFICATION"

        return RecoveryResult(
            confidence=confidence,
            recovered_question=recovered,
            detected_topic=detected_topic,
            reason=reason,
            technical_entities=technical_entities,
            ambiguities=ambiguities,
            needs_manual_fix=needs_manual_fix,
            candidate_questions=candidates,
            candidate_quality=candidate_quality,
            candidate_details=[
                {
                    "question": candidates[0],
                    "confidence": confidence,
                    "candidate_quality": candidate_quality,
                }
            ] if candidates else [],
        )

    def _parse_json(self, content: str) -> dict[str, Any]:
        cleaned = content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            cleaned = self._repair_truncated_json(cleaned)
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
                if not match:
                    raise
                data = json.loads(match.group(0))

        if not isinstance(data, dict):
            msg = f"Recovery JSON must be an object, got {type(data).__name__}"
            raise ValueError(msg)

        return data

    def _repair_truncated_json(self, text: str) -> str:
        if text.count('"') % 2 != 0:
            text += '"'
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")
        text += "]" * open_brackets
        text += "}" * open_braces
        return text

    def _parse_candidate_details(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        raw_details = payload.get("candidate_details")
        if not isinstance(raw_details, list):
            return []

        details: list[dict[str, Any]] = []
        for item in raw_details:
            if not isinstance(item, dict):
                continue

            question = str(item.get("question", "")).strip()
            if not question:
                continue

            details.append(
                {
                    "question": self._ensure_question_punctuation(normalize_lightweight(question)),
                    "confidence": self._clamp_float(item.get("confidence", 0.0)),
                    "candidate_quality": str(item.get("candidate_quality", "unclear")).strip() or "unclear",
                }
            )

        return details

    def _normalize_candidate_details(
        self,
        result: RecoveryResult,
        raw_text: str,
        normalized: str,
    ) -> list[dict[str, Any]]:
        details = list(result.candidate_details)
        known_questions = {str(detail.get("question", "")).strip().lower() for detail in details}

        for question in result.candidate_questions:
            normalized_question = self._ensure_question_punctuation(normalize_lightweight(question))
            if normalized_question.lower() not in known_questions:
                details.append(
                    {
                        "question": normalized_question,
                        "confidence": result.confidence,
                        "candidate_quality": self._candidate_quality(normalized_question, raw_text, normalized),
                    }
                )
                known_questions.add(normalized_question.lower())

        normalized_details: list[dict[str, Any]] = []
        for detail in details:
            question = self._ensure_question_punctuation(normalize_lightweight(str(detail.get("question", "")).strip()))
            if not question:
                continue

            quality = str(detail.get("candidate_quality", "")).strip() or self._candidate_quality(
                question, raw_text, normalized
            )
            if self._is_raw_copy_candidate(question, raw_text, normalized):
                quality = "raw_copy"

            normalized_details.append(
                {
                    "question": question,
                    "confidence": self._clamp_float(detail.get("confidence", result.confidence)),
                    "candidate_quality": quality,
                }
            )

        normalized_details.sort(
            key=lambda detail: (
                0 if detail["candidate_quality"] == "stt_repair" else 1,
                -float(detail["confidence"]),
            )
        )
        return normalized_details

    def _sort_candidates(self, candidates: list[str], raw_text: str, normalized: str) -> list[str]:
        return sorted(
            candidates,
            key=lambda candidate: (
                1 if self._is_raw_copy_candidate(candidate, raw_text, normalized) else 0,
                self._question_shape_penalty(candidate, normalized or raw_text),
                -self._candidate_support_score(candidate, raw_text, normalized),
                -self._semantic_lift_score(candidate, raw_text),
            ),
        )

    def _candidate_support_score(self, candidate: str, raw_text: str, normalized: str) -> int:
        candidate_lower = candidate.lower()
        score = 0
        for index, match in enumerate(self._fuzzy_term_matches(f"{raw_text} {normalized}")[:8]):
            term = str(match["term"]).lower()
            if term and term in candidate_lower:
                score += int(float(match["score"])) + max(0, 8 - index)
        return score

    def _transcript_support_problem(
        self,
        raw_text: str,
        normalized: str,
        candidate: str,
        context: list[str],
    ) -> str:
        support_text = " ".join([raw_text, normalized, *context])
        unsupported_terms = self._unsupported_catalog_terms(candidate, support_text)
        if unsupported_terms:
            return "added terms " + ", ".join(unsupported_terms[:4])

        support_tokens = self._content_tokens(normalize_lightweight(support_text))
        candidate_tokens = self._content_tokens(candidate)
        allowed_repair_tokens: set[str] = set()
        for term in self._catalog_terms_in_text(candidate):
            if self._catalog_term_is_supported(term, support_text):
                allowed_repair_tokens.update(self._content_tokens(term))

        unsupported_tokens = sorted(candidate_tokens - support_tokens - allowed_repair_tokens)
        if unsupported_tokens:
            return "added words " + ", ".join(unsupported_tokens[:6])
        return ""

    def _unsupported_catalog_terms(self, candidate: str, support_text: str) -> list[str]:
        return [
            term
            for term in self._catalog_terms_in_text(candidate)
            if not self._catalog_term_is_supported(term, support_text)
        ]

    def _catalog_terms_in_text(self, text: str) -> list[str]:
        lowered = normalize_lightweight(text).lower()
        terms: list[str] = []
        for canonical, _topic, _aliases in FUZZY_TERM_CATALOG:
            if self._contains_catalog_phrase(lowered, canonical.lower()):
                terms.append(canonical)
        return self._dedupe(terms)

    def _catalog_term_is_supported(self, canonical: str, support_text: str) -> bool:
        lowered = normalize_lightweight(support_text).lower()
        for catalog_term, _topic, aliases in FUZZY_TERM_CATALOG:
            if catalog_term != canonical:
                continue
            if self._contains_catalog_phrase(lowered, catalog_term.lower()):
                return True
            return any(self._contains_catalog_phrase(lowered, alias.lower()) for alias in aliases)
        return False

    def _contains_catalog_phrase(self, lowered_text: str, phrase: str) -> bool:
        phrase = phrase.strip().lower()
        if not phrase:
            return False
        if re.fullmatch(r"[\w.+#/-]+", phrase):
            return bool(re.search(rf"(?<![\w.+#/-]){re.escape(phrase)}(?![\w.+#/-])", lowered_text))
        return phrase in lowered_text

    def _question_shape_penalty(self, candidate: str, raw_text: str) -> int:
        raw_lower = raw_text.lower()
        candidate_lower = candidate.lower()
        if (
            ("что такое" in raw_lower or "what is" in raw_lower)
            and ("чем" in candidate_lower or "отлич" in candidate_lower or "different" in candidate_lower)
        ):
            return 1
        shape_groups = (
            ("что такое", "объясни", "расскажи"),
            ("чем", "отлич"),
            ("когда", "использ"),
            ("как", "работает"),
            ("почему", "диагност"),
        )
        for group in shape_groups:
            if any(marker in raw_lower for marker in group):
                return 0 if any(marker in candidate_lower for marker in group) else 1
        return 0

    def _candidate_quality(self, candidate: str, raw_text: str, normalized: str) -> str:
        if self._is_raw_copy_candidate(candidate, raw_text, normalized):
            return "raw_copy"
        if self._semantic_lift_score(candidate, raw_text) > 0:
            return "stt_repair"
        return "unclear"

    def _is_raw_copy_candidate(self, candidate: str, raw_text: str, normalized: str) -> bool:
        if self._semantic_lift_score(candidate, raw_text) > 0:
            return False

        candidate_tokens = self._content_tokens(candidate)
        raw_tokens = self._content_tokens(normalized or raw_text)

        if not candidate_tokens or not raw_tokens:
            return False

        overlap = len(candidate_tokens & raw_tokens) / max(len(candidate_tokens | raw_tokens), 1)
        direct_copy = (
            self._strip_punctuation(candidate).lower()
            == self._strip_punctuation(normalized or raw_text).lower()
        )

        return (direct_copy or overlap >= 0.78) and self._has_distortion_signal(raw_text)

    def _semantic_lift_score(self, candidate: str, raw_text: str) -> int:
        raw_entities = set(self._extract_generic_entities(raw_text))
        candidate_entities = set(self._extract_generic_entities(candidate))
        return len({entity.lower() for entity in candidate_entities - raw_entities})

    def _has_distortion_signal(self, raw_text: str) -> bool:
        lowered = raw_text.lower()
        tokens = re.findall(r"[а-яё]+", lowered)

        if set(tokens) & TRANSLITERATED_TECH_TERMS:
            return True

        if self._has_technical_position(lowered) and any(len(token) >= 4 for token in tokens):
            latin_chars = len(re.findall(r"[A-Za-z]", raw_text))
            if latin_chars == 0:
                return True

        cyrillic_chars = len(re.findall(r"[а-яё]", lowered))
        latin_chars = len(re.findall(r"[A-Za-z]", raw_text))

        if (
            cyrillic_chars > 10
            and latin_chars == 0
            and self._looks_question_like(lowered)
            and self._has_technical_position(lowered)
        ):
            return True

        return False

    def _is_clean_supported_question(self, text: str) -> bool:
        normalized = normalize_lightweight(text)

        if not self._looks_question_like(normalized):
            return False

        if self._is_very_noisy(normalized):
            return False

        entities = self._extract_generic_entities(normalized)
        has_latin_or_path = bool(re.search(r"[A-Za-z]", normalized)) or "/" in normalized

        if self._has_unresolved_cyrillic_tech_token(normalized):
            return False

        return bool(has_latin_or_path)

    def _semantic_fallback_candidates(self, raw_text: str, normalized: str) -> list[str]:
        text = f"{raw_text} {normalized}".lower()
        candidates: list[str] = []

        if self._has_linux_d_state_signal(text):
            candidates.extend(
                [
                    "Что такое D state процесса в Linux?",
                    "Почему процесс в Linux может зависнуть в D state и как это диагностировать?",
                ]
            )

        if self._has_jenkins_declarative_signal(text):
            candidates.extend(
                [
                    "Что такое декларативный подход в Jenkins Pipeline?",
                    "Чем declarative pipeline отличается от scripted pipeline в Jenkins?",
                ]
            )

        if re.search(r"\bкуб[а-яёa-z]*\b", text):
            if re.search(r"адм|инициал|установ|bootstrap|init", text):
                candidates.append("Что такое kubeadm и когда его используют?")
            if re.search(r"команд|cli|утилит|apply|get|describe|exec", text):
                candidates.append("Что такое kubectl и для чего он нужен?")
            if re.search(r"редин|ready|лайв|liveness|стартап|startup|проб|probe", text):
                candidates.append("Чем отличаются readinessProbe, livenessProbe и startupProbe в Kubernetes?")
            candidates.extend(
                [
                    "Что такое Kubernetes?",
                    "Что такое kubectl и чем он отличается от Kubernetes?",
                ]
            )

        candidates.extend(self._fuzzy_fallback_candidates(text))

        return self._dedupe(candidates)[:MAX_CANDIDATES]

    def _has_linux_d_state_signal(self, text: str) -> bool:
        lowered = text.lower()
        has_linux_process_context = bool(
            re.search(r"\blinux\b|\bлинукс\b|\bпроцесс[а-яё]*\b|\bps\b|\btop\b|\bkernel\b|\bядр[оа]\b", lowered)
        )
        has_d_state = bool(
            re.search(r"\bd\s*state\b|\bд\s*стейт\b|\bди\s*стейт\b|\buninterruptible\s+sleep\b", lowered)
        )
        return has_linux_process_context and has_d_state

    def _fuzzy_fallback_candidates(self, text: str) -> list[str]:
        matches = self._fuzzy_term_matches(text)
        if not matches:
            return []

        terms = {str(match["term"]) for match in matches}
        candidates: list[str] = []
        lowered = text.lower()
        is_compare = bool(re.search(r"\bчем\b|\bотлича|vs|versus|сравни|разниц", lowered))
        is_troubleshooting = bool(
            re.search(r"почему|не работает|ошибк|timeout|failed|crash|latency|diagnos|диагност|дебаж", lowered)
        )
        is_example = bool(re.search(r"пример|покажи|как выглядит|config|yaml|manifest|dockerfile", lowered))

        if {"Prometheus", "Grafana"} <= terms:
            candidates.append("Чем Prometheus отличается от Grafana?")
        if {"TCP", "UDP"} <= terms:
            candidates.append("Чем TCP отличается от UDP?")
        if {"TLS", "mTLS"} <= terms:
            candidates.append("Чем TLS отличается от mTLS?")
        if {"NFS", "S3"} <= terms:
            candidates.append("Чем NFS отличается от S3?")
        if {"PostgreSQL", "Patroni"} <= terms:
            candidates.append("Чем PostgreSQL replication отличается от Patroni?")
        if {"readinessProbe", "livenessProbe"} & terms and (
            {"readinessProbe", "livenessProbe", "startupProbe"} & terms
        ):
            candidates.append("Чем отличаются readinessProbe, livenessProbe и startupProbe в Kubernetes?")
        if {"playbook", "role", "collection"} & terms and (
            "Ansible" in terms or len({"playbook", "role", "collection"} & terms) >= 2
        ):
            candidates.append("В Ansible чем отличаются playbook, role и collection, когда что использовать?")
        if {"Jenkins Pipeline", "declarative pipeline"} <= terms:
            candidates.append("Что такое декларативный подход в Jenkins Pipeline?")
        if "D state" in terms:
            candidates.append("Что такое D state процесса в Linux?")

        if is_troubleshooting:
            for match in matches[:2]:
                term = str(match["term"])
                if term in {"df", "du"} and {"df", "du"} <= terms:
                    candidates.append("Почему df и du показывают разное место на диске?")
                    continue
                if term in {"D state", "OOMKilled", "CrashLoopBackOff", "DNS", "load average"}:
                    candidates.append(f"Как диагностировать {term}?")

        if is_example:
            for match in matches[:2]:
                term = str(match["term"])
                if term in {"Dockerfile", "docker-compose.yml", "Terraform", "playbook", "Helm"}:
                    candidates.append(f"Покажи пример {term}.")

        if not candidates and matches:
            top_terms = [str(match["term"]) for match in matches[:2]]
            if is_compare and len(top_terms) >= 2:
                candidates.append(f"Чем {top_terms[0]} отличается от {top_terms[1]}?")
            else:
                candidates.append(f"Что такое {top_terms[0]}?")

        return self._dedupe(candidates)

    def _fuzzy_entity_hints(self, text: str) -> list[str]:
        return [
            f"{match['term']}:{float(match['score']) / 100:.2f}"
            for match in self._fuzzy_term_matches(text)[:8]
        ]

    def _fuzzy_term_matches(self, text: str) -> list[dict[str, Any]]:
        if fuzz is None or process is None:
            return []

        lowered = normalize_lightweight(text).lower()
        if not lowered:
            return []

        tokens = re.findall(r"[a-zа-яё0-9.+#/-]+", lowered)
        windows = self._dedupe(
            [
                *tokens,
                *(" ".join(tokens[index : index + 2]) for index in range(max(0, len(tokens) - 1))),
                *(" ".join(tokens[index : index + 3]) for index in range(max(0, len(tokens) - 2))),
            ]
        )
        if not windows:
            return []

        matches_by_term: dict[str, dict[str, Any]] = {}
        for canonical, topic, aliases in FUZZY_TERM_CATALOG:
            best_score = 0.0
            best_alias = ""
            best_window = ""
            for alias in aliases:
                alias_lower = alias.lower()
                if len(alias_lower) <= 3 and not re.search(rf"(?<!\w){re.escape(alias_lower)}(?!\w)", lowered):
                    continue

                window_match = process.extractOne(alias_lower, windows, scorer=fuzz.WRatio)
                window_score = float(window_match[1]) if window_match else 0.0
                partial_score = float(fuzz.partial_ratio(alias_lower, lowered)) if len(alias_lower) >= 5 else 0.0
                score = max(window_score, partial_score)
                if score > best_score:
                    best_score = score
                    best_alias = alias
                    best_window = str(window_match[0]) if window_match else ""

            threshold = FUZZY_MATCH_THRESHOLD
            if canonical in {"D state", "df", "du", "TCP", "UDP", "DNS", "TLS", "mTLS", "S3", "NFS"}:
                threshold = max(threshold, 90)
            if best_score >= threshold:
                current = matches_by_term.get(canonical)
                if current is None or best_score > float(current["score"]):
                    matches_by_term[canonical] = {
                        "term": canonical,
                        "topic": topic,
                        "score": best_score,
                        "alias": best_alias,
                        "source": best_window,
                    }

        return sorted(matches_by_term.values(), key=lambda match: float(match["score"]), reverse=True)

    def _has_jenkins_declarative_signal(self, text: str) -> bool:
        lowered = text.lower()
        declarative_signal = re.search(
            r"\bдекор[а-яё]*\b|\bдекларат[а-яё]*\b|\bdeclarative\b",
            lowered,
        )
        jenkins_signal = re.search(
            r"\bдж[еи]н[а-яё]*\b|\bдженк[а-яё]*\b|\bкинс[а-яё]*\b|\bкинох[а-яё]*\b|\bjenkins\b",
            lowered,
        )
        return bool(declarative_signal and jenkins_signal)

    def _strong_semantic_fallback_confidence(
        self,
        raw_text: str,
        normalized: str,
        recovered: str,
    ) -> float:
        text = f"{raw_text} {normalized}".lower()
        recovered_lower = recovered.lower()
        if (
            self._has_jenkins_declarative_signal(text)
            and "jenkins" in recovered_lower
            and ("декларатив" in recovered_lower or "declarative" in recovered_lower)
        ):
            return 0.86
        if (
            self._has_linux_d_state_signal(text)
            and "linux" in recovered_lower
            and ("d state" in recovered_lower or "uninterruptible" in recovered_lower)
        ):
            return 0.86
        fuzzy_matches = self._fuzzy_term_matches(text)
        if fuzzy_matches:
            recovered_terms = {
                str(match["term"]).lower()
                for match in fuzzy_matches
                if str(match["term"]).lower() in recovered_lower
            }
            if recovered_terms and max(float(match["score"]) for match in fuzzy_matches) >= 92:
                if len(recovered_terms) >= 2 or self._looks_question_like(recovered):
                    return 0.84
        return 0.0

    def _has_unresolved_cyrillic_tech_token(self, text: str) -> bool:
        lowered = text.lower()
        if re.search(r"[A-Za-z/]", text):
            return False

        marker_match = re.search(
            r"(что такое|чем отличается|как работает|когда использовать|покажи пример)\s+(.{1,80})",
            lowered,
        )
        if not marker_match:
            return False

        tail_tokens = re.findall(r"[а-яё]+", marker_match.group(2))[:5]
        for token in tail_tokens:
            if token in TRANSLITERATED_TECH_TERMS or token in RUSSIAN_TECH_WORDS:
                continue
            if any(token.startswith(prefix) for prefix in SUSPICIOUS_PHONETIC_PREFIXES):
                return True
        return False

    def _strip_punctuation(self, text: str) -> str:
        return re.sub(r"[^\w/+.#-]+", " ", text).strip()

    def _looks_question_like(self, text: str) -> bool:
        lowered = text.lower()
        return "?" in lowered or any(marker in lowered for marker in QUESTION_MARKERS)

    def _has_technical_position(self, text: str) -> bool:
        lowered = text.lower()
        if any(marker in lowered for marker in TECH_POSITION_MARKERS):
            return True

        tokens = set(re.findall(r"[а-яё]+", lowered))
        return bool(tokens & TRANSLITERATED_TECH_TERMS)

    def _extract_generic_entities(self, text: str) -> list[str]:
        entities: list[str] = []

        patterns = (
            r"(?<!\w)/(?:[A-Za-z0-9_.-]+/?)+",
            r"\b[A-ZА-ЯЁ]{2,}(?:/[A-ZА-ЯЁ]{2,})?\b",
            r"(?<!\w)--?[A-Za-z][A-Za-z0-9_-]*\b",
            r"\b[A-Za-z][A-Za-z0-9_.:+#/-]*\b",
            r"\b[A-Za-z]+[А-Яа-яЁё]+[A-Za-zА-Яа-яЁё0-9_.:+#/-]*\b",
            r"\b[А-Яа-яЁё]+[A-Za-z]+[A-Za-zА-Яа-яЁё0-9_.:+#/-]*\b",
        )

        for pattern in patterns:
            entities.extend(re.findall(pattern, text))

        lowered = text.lower()
        for marker in TECH_POSITION_MARKERS:
            for match in re.finditer(re.escape(marker), lowered):
                tail = text[match.end(): match.end() + 80]
                tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9_./:+#-]+", tail)
                entities.extend(token.strip(".,?!:;") for token in tokens[:4])

        stopwords = {
            "что",
            "такое",
            "чем",
            "как",
            "когда",
            "использовать",
            "работает",
            "отличается",
            "и",
            "или",
            "в",
            "на",
            "от",
            "the",
            "and",
            "or",
            "what",
            "how",
            "when",
        }

        return self._dedupe(
            entity
            for entity in entities
            if entity and entity.lower() not in stopwords and len(entity) > 1
        )

    def _language_hints(self, raw_text: str, normalized: str, context: list[str]) -> str:
        text = " ".join([raw_text, normalized, *context])
        cyrillic = len(re.findall(r"[А-Яа-яЁё]", text))
        latin = len(re.findall(r"[A-Za-z]", text))
        paths = re.findall(r"(?<!\w)/(?:[A-Za-z0-9_.-]+/?)+", text)
        acronyms = re.findall(r"\b[A-ZА-ЯЁ]{2,}\b", text)
        mixed = bool(cyrillic and latin)

        return (
            f"mixed_russian_english={mixed}; cyrillic_chars={cyrillic}; latin_chars={latin}; "
            f"paths={paths[:6]}; acronyms={acronyms[:8]}"
        )

    def _contradiction_risk(self, raw_text: str, recovered: str, entities: list[str]) -> bool:
        raw_tokens = self._content_tokens(normalize_lightweight(raw_text))
        recovered_tokens = self._content_tokens(recovered)

        if not raw_tokens or not recovered_tokens:
            return True

        if self._looks_question_like(raw_text) and self._looks_question_like(recovered) and entities:
            return False

        shared = raw_tokens & recovered_tokens
        shared_entities = [
            entity for entity in entities if entity.lower() in normalize_lightweight(raw_text).lower()
        ]
        overlap = len(shared) / max(len(recovered_tokens), 1)

        return overlap < 0.12 and not shared_entities

    def _content_tokens(self, text: str) -> set[str]:
        stopwords = {
            "что",
            "такое",
            "чем",
            "как",
            "когда",
            "использовать",
            "работает",
            "отличается",
            "между",
            "разница",
            "и",
            "или",
            "в",
            "на",
            "от",
            "для",
            "the",
            "and",
            "or",
        }
        tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9/+#.-]+", text.lower())
        return {token for token in tokens if len(token) > 2 and token not in stopwords}

    def _ensure_question_punctuation(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text
        if text[-1] not in ".?!":
            return f"{text}?"
        return text

    def _format_recovered_question(self, text: str) -> str:
        text = self._ensure_question_punctuation(text)
        match = re.fullmatch(
            r"чем отличается ([A-Za-z0-9_.:+#/-]+) от ([A-Za-z0-9_.:+#/-]+)\?",
            text.strip(),
            flags=re.IGNORECASE,
        )
        if match:
            return f"Чем {match.group(1)} отличается от {match.group(2)}?"
        return text[:1].upper() + text[1:] if text else text

    def _is_very_noisy(self, raw_text: str) -> bool:
        tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9/+#.-]+", raw_text)

        if len(tokens) < 2:
            return True

        filler = {
            "ну",
            "это",
            "там",
            "как",
            "бы",
            "вообще",
            "вот",
            "типа",
            "ээ",
            "мм",
            "а",
        }

        filler_count = sum(1 for token in tokens if token.lower() in filler)
        question_like = self._looks_question_like(raw_text)
        entities = self._extract_generic_entities(normalize_lightweight(raw_text))

        return filler_count / max(len(tokens), 1) > 0.55 and not (question_like and entities)

    def _as_string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def _dedupe(self, values: Any) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            item = str(value).strip()
            key = item.lower()
            if item and key not in seen:
                seen.add(key)
                result.append(item)

        return result

    def _clamp_float(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, number))

    def _log_result(self, result: RecoveryResult) -> None:
        LOGGER.info(
            "recovered_question=%r confidence=%.2f quality=%s topic=%s entities=%s ambiguities=%s manual=%s reason=%s",
            result.recovered_question,
            result.confidence,
            result.candidate_quality,
            result.detected_topic,
            result.technical_entities,
            result.ambiguities,
            result.needs_manual_fix,
            result.reason,
        )
        if _RICH_CONSOLE is not None and os.getenv("STACKWIRE_RICH_TABLES", "1").strip() == "1":
            try:
                from rich.table import Table

                table = Table(title="Question recovery", show_header=True, header_style="bold cyan")
                table.add_column("field", style="dim", no_wrap=True)
                table.add_column("value")
                table.add_row("question", result.recovered_question or "-")
                table.add_row("confidence", f"{result.confidence:.2f}")
                table.add_row("topic", result.detected_topic or "-")
                table.add_row("quality", result.candidate_quality or "-")
                table.add_row("manual_fix", str(result.needs_manual_fix))
                table.add_row("entities", ", ".join(result.technical_entities[:8]) or "-")
                _RICH_CONSOLE.print(table)
            except Exception:
                LOGGER.debug("rich recovery table render failed", exc_info=True)
