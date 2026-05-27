import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import requests

from app.answer_planner import AnswerPlan, build_answer_plan, normalize_question
from app.answer_validator import ValidationResult, validate_answer
from app.question_recovery import CONFIDENCE_THRESHOLD, QuestionRecovery, RecoveryResult
from app.rag import format_knowledge_chunks, retrieve_knowledge
from app.storage import create_session, log_answer, log_question, search_good_answers


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
MODEL = os.getenv("ANSWER_MODEL", os.getenv("OLLAMA_ANSWER_MODEL", os.getenv("OLLAMA_MODEL", "qwen3.6:latest")))
ANSWER_MODEL = MODEL
VISION_MODEL = os.getenv("VISION_MODEL", os.getenv("OLLAMA_VISION_MODEL", "gemma4:latest"))
ANSWER_MODE = os.getenv("ANSWER_MODE", os.getenv("STACKWIRE_ANSWER_MODE", "normal")).strip().lower()
if ANSWER_MODE not in {"normal", "deep"}:
    ANSWER_MODE = "normal"
ANSWER_PROMPT_PROFILE = os.getenv("ANSWER_PROMPT_PROFILE", "compact").strip().lower()
DEFAULT_NUM_CTX = max(int(os.getenv("OLLAMA_NUM_CTX", "4096")), 4096)
DEFAULT_ANSWER_NUM_PREDICT = max(
    int(os.getenv("OLLAMA_ANSWER_NUM_PREDICT", "1200" if ANSWER_MODE == "deep" else "760")),
    1100 if ANSWER_MODE == "deep" else 760,
)
ARTIFACT_ANSWER_NUM_PREDICT = max(
    int(os.getenv("OLLAMA_ARTIFACT_NUM_PREDICT", os.getenv("OLLAMA_CODE_NUM_PREDICT", "1200"))),
    1200,
)
EXPAND_ANSWER_NUM_PREDICT = max(int(os.getenv("OLLAMA_EXPAND_NUM_PREDICT", "1100")), 900)
DEFAULT_VISION_NUM_PREDICT = int(os.getenv("OLLAMA_VISION_NUM_PREDICT", "700"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip()

LOGGER = logging.getLogger(__name__)
NEED_MANUAL_FIX_MESSAGE = "Вопрос распознан ненадежно. Поправь текст вручную."

ANSWER_SYSTEM_PROMPT = """
You are a senior DevOps/SRE specialist.
Answer in Russian, concise and production-oriented.
Keep canonical English names for tools, protocols, API objects, commands, config keys and metrics.
Do not invent mechanisms. Do not use prior question context unless the current question explicitly asks for it.
Assume the question may come from noisy speech recognition: ignore filler words and answer only the supported technical core.
Style: direct, practical response. Start with a clear first sentence, then use compact bullets. Avoid long textbook introductions.
Use Russian section labels. Use fenced Markdown with a language tag for any code, config, command or query.
Do not write the language name as a separate line before code.
""".strip()

VISION_SYSTEM_PROMPT = """
Ты анализируешь выделенную область экрана для DevOps/SRE.

Ответь на русском, коротко и практично:
- что это за объект, экран, ошибка, код, конфиг или интерфейс;
- какие ключевые детали видны;
- если это ошибка/лог/конфиг, что проверить дальше;
- если текст плохо читается, явно скажи что уверенность низкая.

Не выдумывай невидимые строки и не делай длинную лекцию.
""".strip()


@dataclass(frozen=True)
class AskResult:
    raw_text: str
    recovery: RecoveryResult
    answer: str
    answered: bool
    recovery_latency: float
    answer_latency: float
    total_latency: float
    question_id: int | None = None
    answer_id: int | None = None
    plan_domain: str | None = None
    plan_intent: str | None = None


@dataclass(frozen=True)
class ExpandResult:
    question: str
    previous_answer: str
    answer: str
    mode: str
    latency: float
    question_id: int | None = None
    answer_id: int | None = None
    plan_domain: str | None = None
    plan_intent: str | None = None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _append_query_history(
    *,
    raw_text: str,
    recovered_question: str,
    answer: str,
    answered: bool,
    recovery_latency: float,
    answer_latency: float,
    total_latency: float,
) -> None:
    if os.getenv("STACKWIRE_QUERY_LOG", "1").strip().lower() in {"0", "false", "no", "off"}:
        return
    try:
        log_path = Path(os.getenv("STACKWIRE_QUERY_LOG_PATH", "logs/stackwire_queries.md"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        date = time.strftime("%Y-%m-%d")
        clock = time.strftime("%H:%M:%S")
        existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        chunks: list[str] = []
        if f"## {date}" not in existing:
            if existing and not existing.endswith("\n"):
                chunks.append("\n")
            chunks.append(f"\n## {date}\n")
        chunks.append(
            f"\n### {clock}\n"
            f"- answered: {answered}\n"
            f"- recovery_latency_ms: {recovery_latency * 1000:.0f}\n"
            f"- answer_latency_ms: {answer_latency * 1000:.0f}\n"
            f"- total_latency_ms: {total_latency * 1000:.0f}\n\n"
            f"Raw question:\n{raw_text.strip() or '-'}\n\n"
            f"Recovered question:\n{recovered_question.strip() or '-'}\n\n"
            f"Answer:\n{answer.strip() or '-'}\n"
        )
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(chunks))
    except Exception:
        LOGGER.debug("query history write failed", exc_info=True)


def _strip_model_noise(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^\s*#+\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", cleaned)
    return cleaned.strip()


def _repair_markdown_tail(text: str) -> str:
    cleaned = text.strip()
    if cleaned.count("```") % 2 != 0:
        cleaned += "\n```"
    if cleaned.count("**") % 2 != 0:
        cleaned += "**"
    return cleaned


def _repair_unfinished_sentence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned or cleaned[-1] in ".!?`":
        return cleaned
    if cleaned.count("```") % 2 != 0:
        return cleaned
    last_sentence_end = max(cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?"))
    if last_sentence_end > max(80, int(len(cleaned) * 0.55)):
        return cleaned[: last_sentence_end + 1].strip()
    return cleaned


def _default_code_language(question: str, plan: AnswerPlan) -> str:
    q = question.casefold()
    if plan.domain == "kubernetes" or any(term in q for term in ("yaml", "yml", "manifest", "values", "playbook", "pipeline")):
        return "yaml"
    if "dockerfile" in q:
        return "dockerfile"
    if plan.domain == "iac" and any(term in q for term in ("terraform", ".tf", "provider", "resource")):
        return "hcl"
    if plan.domain == "web_proxy":
        return "nginx"
    if plan.domain == "observability" and "promql" in q:
        return "promql"
    return "bash"


def _repair_bare_code(answer: str, question: str, plan: AnswerPlan) -> str:
    if "```" in answer:
        return answer

    code_markers = (
        "apiVersion:",
        "kind:",
        "FROM ",
        "RUN ",
        "COPY ",
        "CMD ",
        "ENTRYPOINT ",
        "server {",
        "upstream ",
        "provider ",
        "resource ",
        "stages:",
        "jobs:",
        "groups by",
        "rate(",
        "sum(",
        "kubectl ",
        "awk ",
        "sort ",
        "uniq ",
        "head ",
        "wc ",
        "df ",
        "du ",
        "ps ",
        "journalctl ",
        "systemctl ",
    )
    lines = answer.splitlines()
    code_start = next((idx for idx, line in enumerate(lines) if line.strip().startswith(code_markers)), -1)
    if code_start < 0:
        return answer

    code_end = len(lines)
    for idx, line in enumerate(lines[code_start + 1 :], start=code_start + 1):
        if line.strip().endswith(":") and not line.startswith((" ", "\t")):
            code_end = idx
            break

    prefix = "\n".join(lines[:code_start]).strip()
    code = "\n".join(lines[code_start:code_end]).strip()
    suffix = "\n".join(lines[code_end:]).strip()
    fenced = f"```{_default_code_language(question, plan)}\n{code}\n```"
    return "\n\n".join(part for part in (prefix, fenced, suffix) if part)


def _repair_answer(answer: str, question: str, plan: AnswerPlan) -> str:
    cleaned = _strip_model_noise(answer)
    cleaned = _repair_bare_code(cleaned, question, plan)
    cleaned = _repair_markdown_tail(cleaned)
    cleaned = _repair_unfinished_sentence(cleaned)
    return cleaned.strip()


def _question_allows_history(question: str) -> bool:
    lowered = question.casefold()
    return any(marker in lowered for marker in ("предыдущ", "выше", "с учетом", "с учётом", "в этом контексте", "как раньше", "previous", "context"))


def _history_section(question: str, context: list[str] | None) -> str:
    if not context or not _question_allows_history(question):
        return ""
    lines = [line.strip() for line in context[-6:] if line.strip()]
    if not lines:
        return ""
    return "\n\nAllowed previous context:\n" + "\n".join(f"- {line}" for line in lines)


def _format_tuple(values: tuple[str, ...]) -> str:
    return "\n".join(f"- {value}" for value in values) if values else "- none"


EXPAND_MODE_RULES: dict[str, tuple[str, ...]] = {
    "details": (
        "Пиши на русском.",
        "Не пересказывай previous_answer. Считай, что он уже прочитан.",
        "Добавь именно новые детали: больше механики, связей, production-нюансов и типичных ошибок.",
        "Если повторяешь термин из прошлого ответа, сразу добавляй новую информацию о нём.",
        "Не добавляй код, если исходный вопрос не про CLI/config/YAML/manifest/code.",
    ),
    "components": (
        "Пиши на русском.",
        "Не повторяй previous_answer, а разложи систему по компонентам глубже.",
        "Если применимо, раздели control plane и data plane.",
        "Покажи data path, control flow или lifecycle.",
        "Укажи кто хранит desired state/config, кто принимает решения, кто реально выполняет workload или пропускает traffic.",
    ),
    "example": (
        "Пиши на русском.",
        "Добавь один минимальный code/config/command/YAML пример.",
        "Пример обязательно должен быть fenced Markdown block с language tag.",
        "Внутри кода/конфига добавь 3-6 коротких комментариев, если синтаксис это позволяет.",
        "После кода добавь раздел 'Практические замечания' с 2-4 пунктами.",
    ),
    "compare": (
        "Пиши на русском.",
        "Сравни только с ближайшими релевантными аналогами.",
        "Явно покажи главное отличие.",
        "Объясни когда что использовать.",
        "Не уходи в unrelated domain и не добавляй YAML.",
    ),
    "troubleshoot": (
        "Пиши на русском.",
        "Это troubleshooting mode, не definition mode.",
        "Структура обязательна: 'Причины:', 'Проверки:', 'Fix:'.",
        "В 'Проверки' дай реальные команды, логи или метрики, если контекст Kubernetes, Linux, CI/CD или Network.",
        "Команды держи в fenced Markdown block с language tag.",
        "Не повторяй previous_answer, кроме короткой привязки к симптому.",
    ),
}


def _plan_contract(plan: AnswerPlan) -> str:
    return f"""
AnswerPlan:
- domain: {plan.domain}
- intent: {plan.intent}
- artifact_required: {plan.artifact_required}
- code_allowed: {plan.code_allowed}
- depth: {plan.depth}
- answer_shape: {plan.answer_shape}

Required concepts:
{_format_tuple(plan.required_concepts)}

Forbidden concepts:
{_format_tuple(plan.forbidden_concepts)}

Dangerous confusions:
{_format_tuple(plan.dangerous_confusions)}

Component model:
{plan.component_model or "none"}
""".strip()


def _format_good_answer_examples(question: str, plan: AnswerPlan, limit: int = 3) -> str:
    try:
        examples = search_good_answers(question, domain=plan.domain, limit=limit)
    except Exception:
        LOGGER.debug("good answer search failed", exc_info=True)
        return ""
    if not examples:
        return ""

    blocks: list[str] = []
    for index, example in enumerate(examples, start=1):
        answer = example.answer.strip()
        if len(answer) > 900:
            answer = answer[:900].rstrip() + "..."
        blocks.append(
            f"Example {index} (style/reference only, do not copy verbatim):\n"
            f"Question: {example.question.strip()}\n"
            f"Answer:\n{answer}"
        )
    return "\n\n".join(blocks)


def _format_rag_context(question: str, plan: AnswerPlan) -> str:
    try:
        chunks = retrieve_knowledge(question, plan, limit=3)
    except Exception:
        LOGGER.debug("rag retrieval failed", exc_info=True)
        return ""
    return format_knowledge_chunks(chunks, max_chars=3200)


def _question_requests_artifact(question: str) -> bool:
    lowered = question.casefold()
    return any(
        marker in lowered
        for marker in (
            "cli",
            "command",
            "kubectl",
            "yaml",
            "yml",
            "manifest",
            "config",
            "конфиг",
            "команд",
            "код",
            "пример",
            "dockerfile",
            "jenkinsfile",
            ".tf",
        )
    )


def _build_prompt(question: str, plan: AnswerPlan, context: list[str] | None = None) -> str:
    artifact_rule = (
        "If artifact_required=true, start with the fenced artifact/code/config block and then add 'Практические замечания'."
        if plan.artifact_required
        else "Do not add code blocks unless code_allowed=true and a command/query example is essential."
    )
    command_rule = (
        "For command_explain, include one practical fenced bash example and explain the pipeline/flags briefly."
        if plan.intent == "command_explain"
        else ""
    )
    compare_rule = (
        "For compare answers, never use section labels 'X:' or 'Y:'. Use the real names of compared technologies or concepts as headings."
        if plan.intent == "compare"
        else ""
    )
    rag_context = _format_rag_context(question, plan)
    rag_section = (
        "\n\nRelevant local markdown knowledge (guidance, not absolute truth; do not dump verbatim):\n"
        f"{rag_context}"
        if rag_context
        else ""
    )
    good_examples = _format_good_answer_examples(question, plan)
    good_examples_section = (
        "\n\nSaved good answers (style/reference examples only; do not copy verbatim):\n"
        f"{good_examples}"
        if good_examples
        else ""
    )
    return f"""
Question:
{question}
{_history_section(question, context)}

{_plan_contract(plan)}
{rag_section}
{good_examples_section}

Contract:
- Follow answer_shape.
- Include required_concepts naturally.
- Avoid forbidden_concepts and dangerous_confusions.
- Explain platform objects as platform objects, not generic concepts.
- Keep the answer concise, complete and logically finished.
- {artifact_rule}
- {command_rule}
- {compare_rule}
- Any code/config/query/command must be fenced Markdown with a language tag.
""".strip()


def _build_expand_prompt(question: str, previous_answer: str, mode: str, plan: AnswerPlan) -> str:
    mode = mode if mode in EXPAND_MODE_RULES else "details"
    rules = "\n".join(f"- {rule}" for rule in EXPAND_MODE_RULES[mode])
    rag_context = _format_rag_context(question, plan)
    rag_section = (
        "\n\nRelevant local markdown knowledge (guidance, not absolute truth; do not dump verbatim):\n"
        f"{rag_context}"
        if rag_context
        else ""
    )
    code_guard = (
        "Code/config is allowed only because mode=example or the original question clearly asked for an artifact."
        if mode == "example" or plan.artifact_required or _question_requests_artifact(question)
        else "Do not include code/config/YAML/commands in this expansion."
    )
    return f"""
Исходный вопрос:
{question}

Предыдущий ответ уже был показан пользователю. Не переписывай его:
{previous_answer}

{_plan_contract(plan)}
{rag_section}

Режим расширения: {mode}
AnswerPlan нужен только чтобы сохранить domain/intent и dangerous confusions. Для формы ответа следуй режиму расширения, а не answer_shape из AnswerPlan.
Правила режима:
{rules}
- Оставайся в том же domain и не меняй смысл исходного вопроса.
- Не запускай question recovery и не угадывай новый вопрос.
- Пиши как отдельное расширение, а не как замену предыдущего ответа.
- Не начинай заново с базового определения, если оно уже есть в previous_answer.
- Запрещено выдавать почти тот же текст другими словами.
- Сначала мысленно вычти из ответа всё, что уже было сказано, и добавь только недостающий слой.
- {code_guard}
- Any code/config/query/command must be fenced Markdown with a language tag.
""".strip()


def _build_retry_prompt(question: str, plan: AnswerPlan, previous_answer: str, validation: ValidationResult) -> str:
    return f"""
Previous answer violated contract:
{_format_tuple(validation.violations)}

Question:
{question}

{_plan_contract(plan)}

Previous answer:
{previous_answer}

Rewrite answer from scratch.
Stay in domain={plan.domain}, intent={plan.intent}.
Include required_concepts.
Avoid forbidden_concepts.
Follow answer_shape.
Do not mention validation.
""".strip()


def _build_expand_retry_prompt(
    question: str,
    plan: AnswerPlan,
    previous_answer: str,
    mode: str,
    failed_answer: str,
    validation: ValidationResult,
) -> str:
    return f"""
Expansion answer violated contract:
{_format_tuple(validation.violations)}

Исходный вопрос:
{question}

Предыдущий ответ, который нельзя повторять:
{previous_answer}

Неудачное расширение:
{failed_answer}

{_plan_contract(plan)}

Перепиши расширение с нуля.
Режим расширения: {mode}
Форма ответа должна соответствовать режиму расширения, а не answer_shape из AnswerPlan.
Оставайся в domain={plan.domain}, intent={plan.intent}.
Не повторяй previous_answer и не упоминай validation.
""".strip()


def _token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-zА-Яа-я0-9_./+-]{3,}", text.casefold())
        if token
        not in {
            "это",
            "как",
            "для",
            "что",
            "или",
            "and",
            "the",
            "with",
            "при",
            "если",
            "когда",
        }
    }


def _looks_too_similar(previous_answer: str, answer: str) -> bool:
    previous_tokens = _token_set(previous_answer)
    answer_tokens = _token_set(answer)
    if len(previous_tokens) < 12 or len(answer_tokens) < 12:
        return False
    overlap = len(previous_tokens & answer_tokens) / max(1, min(len(previous_tokens), len(answer_tokens)))
    return overlap >= 0.72


def _validate_expand_mode(
    answer: str,
    mode: str,
    validation: ValidationResult,
    *,
    previous_answer: str,
    plan: AnswerPlan,
) -> ValidationResult:
    violations = list(validation.violations)
    if _looks_too_similar(previous_answer, answer):
        violations.append("expand_repeats_previous_answer")
    if mode == "example" and "```" not in answer:
        violations.append("expand_example_missing_fenced_block")
    if mode == "compare" and re.search(r"(?m)^\s*(apiVersion|kind):", answer):
        violations.append("expand_compare_added_yaml")
    if mode == "compare" and not any(marker in answer.casefold() for marker in ("главное отличие", "когда использовать", "когда что", "vs", "аналог")):
        violations.append("expand_compare_missing_decision_points")
    if mode == "components" and not any(marker in answer.casefold() for marker in ("control plane", "data plane", "компонент", "data path", "control flow", "desired state")):
        violations.append("expand_components_missing_component_model")
    if mode == "troubleshoot":
        lowered = answer.casefold()
        if not all(marker in lowered for marker in ("причин", "провер", "fix")):
            violations.append("expand_troubleshoot_missing_causes_checks_fix")
        if plan.domain in {"kubernetes", "linux_fs", "linux_process", "linux_network", "ci_cd", "web_proxy", "observability"}:
            has_check_artifact = bool(
                re.search(
                    r"\b(kubectl|journalctl|systemctl|curl|dig|ss|tcpdump|df|du|lsof|grep|docker|gitlab|promql|rate\(|sum\(|logs?)\b",
                    lowered,
                )
            )
            if not has_check_artifact:
                violations.append("expand_troubleshoot_missing_real_checks")
    unique = tuple(dict.fromkeys(violations))
    return ValidationResult(ok=not unique, violations=unique)


class OllamaClient:
    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        question_recovery: QuestionRecovery | None = None,
        answer_generator: Callable[[str], str] | None = None,
        storage_session_id: int | None = None,
    ) -> None:
        self.session = session or requests.Session()
        self.question_recovery = question_recovery or QuestionRecovery(session=self.session)
        self.answer_generator = answer_generator
        try:
            self.storage_session_id = storage_session_id if storage_session_id is not None else create_session("StackWire")
        except Exception:
            LOGGER.debug("storage session creation failed", exc_info=True)
            self.storage_session_id = None

    def _chat(self, messages: list[dict[str, Any]], options: dict[str, Any], *, timeout: int = 300, model: str = ANSWER_MODEL) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": options,
        }
        if OLLAMA_KEEP_ALIVE:
            payload["keep_alive"] = OLLAMA_KEEP_ALIVE
        response = self.session.post(OLLAMA_URL, json=cast(Any, payload), timeout=timeout)
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        return str(message.get("content", "")).strip()

    def _answer_options(self, plan: AnswerPlan) -> dict[str, Any]:
        num_predict = ARTIFACT_ANSWER_NUM_PREDICT if plan.artifact_required else DEFAULT_ANSWER_NUM_PREDICT
        if plan.intent == "command_explain":
            num_predict = min(num_predict, 560)
        return {
            "num_ctx": _env_int("OLLAMA_ANSWER_NUM_CTX", DEFAULT_NUM_CTX),
            "num_predict": num_predict,
            "temperature": float(os.getenv("OLLAMA_ANSWER_TEMPERATURE", "0.04")),
            "top_p": 0.72,
            "repeat_penalty": 1.08,
            "top_k": 20,
        }

    def _expand_options(self, plan: AnswerPlan, mode: str) -> dict[str, Any]:
        options = self._answer_options(plan)
        mode_floor = {
            "details": 1200,
            "components": 1100,
            "example": 1300,
            "compare": 1000,
            "troubleshoot": 1300,
        }.get(mode, EXPAND_ANSWER_NUM_PREDICT)
        options["num_predict"] = max(int(options["num_predict"]), EXPAND_ANSWER_NUM_PREDICT, mode_floor)
        return options

    def _generate_answer(self, question: str, plan: AnswerPlan, prompt: str, *, options: dict[str, Any] | None = None) -> str:
        answer = self._chat(
            [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options or self._answer_options(plan),
            timeout=300,
        )
        return _repair_answer(answer, question, plan)

    def answer_question(self, recovered_question: str, context: list[str] | None = None) -> str:
        question = normalize_question(recovered_question)
        if not question:
            return "Вопрос нужно уточнить."

        if self.answer_generator is not None:
            return _repair_markdown_tail(_strip_model_noise(self.answer_generator(question)))

        plan = build_answer_plan(question)
        answer = self._generate_answer(question, plan, _build_prompt(question, plan, context))
        validation = validate_answer(question, answer, plan)

        if not validation.ok:
            LOGGER.info("answer validation retry violations=%s", validation.violations)
            retry_prompt = _build_retry_prompt(question, plan, answer, validation)
            answer = self._generate_answer(question, plan, retry_prompt)
            retry_validation = validate_answer(question, answer, plan)
            if not retry_validation.ok:
                LOGGER.warning("answer still violates contract violations=%s question=%r", retry_validation.violations, question)

        if not answer:
            return "Вопрос нужно уточнить."
        return answer

    def expand(self, question: str, previous_answer: str, mode: str) -> ExpandResult:
        started = time.perf_counter()
        normalized_question = normalize_question(question)
        normalized_mode = mode if mode in EXPAND_MODE_RULES else "details"
        plan = build_answer_plan(normalized_question)
        prompt = _build_expand_prompt(normalized_question, previous_answer, normalized_mode, plan)
        answer = self._generate_answer(normalized_question, plan, prompt, options=self._expand_options(plan, normalized_mode))
        validation = _validate_expand_mode(
            answer,
            normalized_mode,
            validate_answer(normalized_question, answer, plan),
            previous_answer=previous_answer,
            plan=plan,
        )

        if not validation.ok:
            LOGGER.info("expand validation retry mode=%s violations=%s", normalized_mode, validation.violations)
            retry_prompt = _build_expand_retry_prompt(
                normalized_question,
                plan,
                previous_answer,
                normalized_mode,
                answer,
                validation,
            )
            answer = self._generate_answer(normalized_question, plan, retry_prompt, options=self._expand_options(plan, normalized_mode))
            validation = _validate_expand_mode(
                answer,
                normalized_mode,
                validate_answer(normalized_question, answer, plan),
                previous_answer=previous_answer,
                plan=plan,
            )
            if not validation.ok:
                LOGGER.warning(
                    "expand still violates contract mode=%s violations=%s question=%r",
                    normalized_mode,
                    validation.violations,
                    normalized_question,
                )

        question_id: int | None = None
        answer_id: int | None = None
        latency = time.perf_counter() - started
        try:
            question_id = log_question(
                session_id=self.storage_session_id,
                raw_text=normalized_question,
                recovered_question=normalized_question,
                trusted_text=True,
                source="expand",
                recovery_confidence=1.0,
                detected_topic=plan.domain,
            )
            answer_id = log_answer(
                question_id=question_id,
                answer=answer,
                answer_type="expand",
                expand_mode=normalized_mode,
                model=MODEL,
                answer_mode=ANSWER_MODE,
                latency_ms=latency * 1000,
                validator_ok=validation.ok,
                validator_violations=validation.violations,
                plan_domain=plan.domain,
                plan_intent=plan.intent,
                artifact_required=plan.artifact_required,
            )
        except Exception:
            LOGGER.debug("expand storage logging failed", exc_info=True)

        LOGGER.info(
            "expand mode=%s latency_ms=%.0f question_id=%s answer_id=%s",
            normalized_mode,
            latency * 1000,
            question_id,
            answer_id,
        )
        return ExpandResult(
            question=normalized_question,
            previous_answer=previous_answer,
            answer=answer,
            mode=normalized_mode,
            latency=latency,
            question_id=question_id,
            answer_id=answer_id,
            plan_domain=plan.domain,
            plan_intent=plan.intent,
        )

    def expand_answer(self, question: str, previous_answer: str, mode: str) -> str:
        return self.expand(question, previous_answer, mode).answer

    def _log_main_answer(
        self,
        *,
        raw_text: str,
        recovered_question: str,
        trusted_text: bool,
        recovery_confidence: float | None,
        detected_topic: str | None,
        answer: str,
        answer_latency: float,
    ) -> tuple[int | None, int | None, AnswerPlan | None]:
        try:
            plan = build_answer_plan(recovered_question)
            validation = validate_answer(recovered_question, answer, plan)
            question_id = log_question(
                session_id=self.storage_session_id,
                raw_text=raw_text,
                recovered_question=recovered_question,
                trusted_text=trusted_text,
                source="manual" if trusted_text else "stt",
                recovery_confidence=recovery_confidence,
                detected_topic=detected_topic or plan.domain,
            )
            answer_id = log_answer(
                question_id=question_id,
                answer=answer,
                answer_type="main",
                model=MODEL,
                answer_mode=ANSWER_MODE,
                latency_ms=answer_latency * 1000,
                validator_ok=validation.ok,
                validator_violations=validation.violations,
                plan_domain=plan.domain,
                plan_intent=plan.intent,
                artifact_required=plan.artifact_required,
            )
            return question_id, answer_id, plan
        except Exception:
            LOGGER.debug("main answer storage logging failed", exc_info=True)
            return None, None, None

    def ask(self, raw_text: str, context: list[str] | None = None, *, trusted_text: bool = False) -> AskResult:
        context = context or []
        raw_text = raw_text.strip()
        pipeline_started = time.perf_counter()

        if trusted_text:
            recovery = RecoveryResult(
                confidence=1.0,
                recovered_question=raw_text,
                detected_topic="Manual input",
                reason="trusted manual input",
                technical_entities=[],
                ambiguities=[],
                needs_manual_fix=False,
                candidate_questions=[raw_text],
                candidate_quality="manual",
            )
            started = time.perf_counter()
            answer = self.answer_question(raw_text, None)
            answer_latency = time.perf_counter() - started
            total_latency = time.perf_counter() - pipeline_started
            LOGGER.info("trusted_text answer_latency_ms=%.0f total_latency_ms=%.0f", answer_latency * 1000, total_latency * 1000)
            _append_query_history(
                raw_text=raw_text,
                recovered_question=raw_text,
                answer=answer,
                answered=True,
                recovery_latency=0.0,
                answer_latency=answer_latency,
                total_latency=total_latency,
            )
            question_id, answer_id, plan = self._log_main_answer(
                raw_text=raw_text,
                recovered_question=raw_text,
                trusted_text=True,
                recovery_confidence=1.0,
                detected_topic="Manual input",
                answer=answer,
                answer_latency=answer_latency,
            )
            return AskResult(
                raw_text=raw_text,
                recovery=recovery,
                answer=answer,
                answered=True,
                recovery_latency=0.0,
                answer_latency=answer_latency,
                total_latency=total_latency,
                question_id=question_id,
                answer_id=answer_id,
                plan_domain=plan.domain if plan else None,
                plan_intent=plan.intent if plan else None,
            )

        recovery_started = time.perf_counter()
        recovery = self.question_recovery.recover(raw_text, context)
        recovery_latency = time.perf_counter() - recovery_started

        if (
            recovery.needs_manual_fix
            or recovery.confidence < CONFIDENCE_THRESHOLD
            or recovery.detected_topic == "NEED_CLARIFICATION"
        ):
            LOGGER.info("skip answer generation confidence=%.2f recovered_question=%r", recovery.confidence, recovery.recovered_question)
            return AskResult(
                raw_text=raw_text,
                recovery=recovery,
                answer=NEED_MANUAL_FIX_MESSAGE,
                answered=False,
                recovery_latency=recovery_latency,
                answer_latency=0.0,
                total_latency=time.perf_counter() - pipeline_started,
            )

        started = time.perf_counter()
        answer = self.answer_question(recovery.recovered_question, context)
        answer_latency = time.perf_counter() - started
        total_latency = time.perf_counter() - pipeline_started
        LOGGER.info(
            "recovery_latency_ms=%.0f answer_latency_ms=%.0f total_latency_ms=%.0f",
            recovery_latency * 1000,
            answer_latency * 1000,
            total_latency * 1000,
        )
        _append_query_history(
            raw_text=raw_text,
            recovered_question=recovery.recovered_question,
            answer=answer,
            answered=True,
            recovery_latency=recovery_latency,
            answer_latency=answer_latency,
            total_latency=total_latency,
        )
        question_id, answer_id, plan = self._log_main_answer(
            raw_text=raw_text,
            recovered_question=recovery.recovered_question,
            trusted_text=False,
            recovery_confidence=recovery.confidence,
            detected_topic=recovery.detected_topic,
            answer=answer,
            answer_latency=answer_latency,
        )
        return AskResult(
            raw_text=raw_text,
            recovery=recovery,
            answer=answer,
            answered=True,
            recovery_latency=recovery_latency,
            answer_latency=answer_latency,
            total_latency=total_latency,
            question_id=question_id,
            answer_id=answer_id,
            plan_domain=plan.domain if plan else None,
            plan_intent=plan.intent if plan else None,
        )

    def analyze_image(self, image_b64: str, prompt: str | None = None) -> str:
        image_b64 = image_b64.strip()
        if image_b64.startswith("data:image"):
            image_b64 = image_b64.split(",", 1)[-1]
        if not image_b64:
            return "Не удалось получить изображение."

        user_prompt = (prompt or "").strip() or (
            "Определи, что находится на выделенной области экрана. "
            "Если это DevOps/SRE материал, объясни смысл и что важно сказать."
        )
        answer = self._chat(
            [
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt, "images": [image_b64]},
            ],
            {
                "num_ctx": _env_int("OLLAMA_VISION_NUM_CTX", DEFAULT_NUM_CTX),
                "num_predict": DEFAULT_VISION_NUM_PREDICT,
                "temperature": 0.05,
                "top_p": 0.75,
            },
            timeout=300,
            model=VISION_MODEL,
        )
        return _strip_model_noise(answer) or "Не удалось распознать содержимое области."
