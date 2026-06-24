import json
import logging
import os
import random
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
from app.transcript_repair import condense_spoken_question
from app.web_search import format_results_for_prompt, format_results_markdown, search_duckduckgo


DEFAULT_STACKWIRE_MODEL = "qwen3.6:latest"
DEFAULT_STACKWIRE_VISION_MODEL = "gemma3:4b"


def current_answer_model() -> str:
    return os.getenv("ANSWER_MODEL", os.getenv("OLLAMA_ANSWER_MODEL", os.getenv("OLLAMA_MODEL", DEFAULT_STACKWIRE_MODEL))).strip() or DEFAULT_STACKWIRE_MODEL


def current_vision_model() -> str:
    return os.getenv("VISION_MODEL", os.getenv("OLLAMA_VISION_MODEL", DEFAULT_STACKWIRE_VISION_MODEL)).strip() or DEFAULT_STACKWIRE_VISION_MODEL


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
MODEL = current_answer_model()
ANSWER_MODEL = MODEL
VISION_MODEL = current_vision_model()


def current_llm_provider() -> str:
    provider = os.getenv("STACKWIRE_LLM_PROVIDER", os.getenv("STACKWIRE_PROVIDER", "ollama")).strip().lower()
    if provider in {"openai", "openai-compatible", "openai_compatible", "compatible"}:
        return "openai_compatible"
    return "ollama"


def current_vision_provider() -> str:
    """Provider for screenshot/vision requests. Defaults to local Ollama (where the
    gemma vision model lives) even when text answers use a remote API without vision
    (e.g. DeepSeek). Override with STACKWIRE_VISION_PROVIDER."""
    raw = os.getenv("STACKWIRE_VISION_PROVIDER", "").strip().lower()
    if raw in {"openai", "openai-compatible", "openai_compatible", "compatible"}:
        return "openai_compatible"
    return "ollama"


def current_ollama_chat_url() -> str:
    return os.getenv("OLLAMA_URL", OLLAMA_URL).strip() or OLLAMA_URL


def current_openai_chat_url() -> str:
    base = os.getenv("STACKWIRE_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/chat/completions"


def _openai_headers() -> dict[str, str]:
    key = os.getenv("STACKWIRE_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _openai_payload_options(options: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if "num_predict" in options:
        payload["max_tokens"] = int(options["num_predict"])
    if "temperature" in options:
        payload["temperature"] = float(options["temperature"])
    if "top_p" in options:
        payload["top_p"] = float(options["top_p"])
    return payload


def _openai_message_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") if isinstance(data, dict) else []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip()


ANSWER_MODE = os.getenv("ANSWER_MODE", os.getenv("STACKWIRE_ANSWER_MODE", "normal")).strip().lower()
if ANSWER_MODE not in {"normal", "deep"}:
    ANSWER_MODE = "normal"
ANSWER_PROMPT_PROFILE = os.getenv("ANSWER_PROMPT_PROFILE", "balanced").strip().lower()
# 8192 by default: the system prompt (~1.5k tokens) + RAG (up to 3.2k chars) +
# chat history + a full ~950-token answer must all fit in one window. At 4096 a
# populated RAG/history made the total overflow, so Ollama either cut answers off
# mid-sentence or silently truncated the OLDEST tokens — the system prompt — losing
# the instructions. All bundled models (qwen3.6, qwen2.5*, gemma3, llama3.2) handle
# 8k easily; weak hardware can still set OLLAMA_NUM_CTX=4096.
DEFAULT_NUM_CTX = max(int(os.getenv("OLLAMA_NUM_CTX", "8192")), 4096)
DEFAULT_ANSWER_NUM_PREDICT = max(
    int(os.getenv("OLLAMA_ANSWER_NUM_PREDICT", "1100" if ANSWER_MODE == "deep" else "950")),
    1100 if ANSWER_MODE == "deep" else 900,
)
ARTIFACT_ANSWER_NUM_PREDICT = max(
    int(os.getenv("OLLAMA_ARTIFACT_NUM_PREDICT", os.getenv("OLLAMA_CODE_NUM_PREDICT", "1200"))),
    1200,
)
EXPAND_ANSWER_NUM_PREDICT = max(int(os.getenv("OLLAMA_EXPAND_NUM_PREDICT", "1100")), 900)
DEFAULT_VISION_NUM_PREDICT = int(os.getenv("OLLAMA_VISION_NUM_PREDICT", "1200"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m").strip()

LOGGER = logging.getLogger(__name__)
NEED_MANUAL_FIX_MESSAGE = "Вопрос распознан ненадежно. Поправь текст вручную."


def _recovery_is_unusable(recovery: "RecoveryResult", raw_text: str) -> bool:
    """True only when there is genuinely nothing to answer (noise/empty).

    Low recovery confidence is NOT a reason to refuse: in a live conversation the
    user cannot 'fix the text manually'. The answer model is instructed to
    reconstruct the most likely intended question from fragments, so we answer
    best-effort instead of bouncing the question back."""
    best = (recovery.recovered_question or raw_text or "").strip()
    if not best:
        return True
    # Fewer than 2 meaningful tokens → noise (e.g. "ну", "ага").
    return len(re.findall(r"[\w-]{2,}", best)) < 2


ANSWER_SYSTEM_PROMPT = """
You are StackWire, a sharp and genuinely helpful assistant. Use your own judgment to understand what each question is really about and answer it well — everyday life, science, history, cooking, health, money, software, or anything else.

Be practical and concrete. Infer what the person most likely wants and answer THAT fully — don't ask for clarification on short or vague requests; add a brief note on a real alternative only when it genuinely matters.

Shape each answer to the question:
- Lead with the direct answer, or the usable artifact itself (code, recipe, steps, calculation), then a tight breakdown of the key parts. No filler introductions.
- Match the depth to the request — a small question gets a short answer; a hard or technical one gets real depth and the non-obvious specifics.
- Answer the whole question: if it has several parts, conditions or comparisons, cover all of them.
- Use natural section headings only where they truly help; never template-style labels.

Formatting:
- Put any code, config, command or query in fenced Markdown with a language tag.
- If the user asks for a table, give a Markdown table. If they ask for a diagram/scheme, give a clear ASCII diagram in a fenced block (or a ```mermaid / ```dot block); otherwise don't add diagrams.

Live speech: questions may arrive as noisy speech fragments — reconstruct the intended question and answer it directly, like a knowledgeable person speaking out loud (the user may read it aloud, so make the first 1-2 sentences self-sufficient). Don't lecture. If you already covered something, add the new angle instead of repeating.
""".strip()

VISION_SYSTEM_PROMPT = """
You are a sharp, detail-oriented visual assistant. Analyze the attached image thoroughly and answer the user's actual question. Answer in Russian.

## Principle of maximum usefulness
If the user gave no specific question (or just "what is this?"), do NOT reply with one vague sentence. Infer what a person sharing THIS kind of screenshot most likely wants, and give a rich, well-structured answer.

Adapt to what the image actually is:
- **Code / terminal / config / logs**: transcribe the key lines exactly (in a code block), say what the code/command does, point out errors, warnings, bugs or smells, and give the concrete fix or next step.
- **Error message / stack trace / crash**: state the exact error, the most likely cause, and how to fix it step by step.
- **UI / website / app / dashboard**: describe the layout and the meaningful elements (buttons, fields, menus, data), what state it is in, and what the user can do here.
- **Diagram / architecture / chart / graph**: explain what it represents, the components and how they connect, and read off concrete numbers/labels/axes.
- **Document / table / text**: extract and summarize the actual content; reproduce important text and numbers faithfully.
- **Photo / general image**: describe the scene, the notable objects, text on signs/labels, and anything contextually important.

## Structure
Use short headings or bullet points so the answer is easy to scan. A good default shape:
1. Что на экране — concise overview of what the image shows.
2. Детали / ключевые элементы — the important specifics (transcribe text/code/numbers verbatim where it matters).
3. Что это значит / что делать — interpretation, the answer to the implied question, or the fix/next step.

## Accuracy rules
- Transcribe visible text, code, numbers and labels exactly as shown — do not paraphrase identifiers.
- Describe only what is actually visible. Never invent prices, names, dates, hidden text, or off-screen content.
- If something is cut off, blurry, or ambiguous, say so plainly and explain what extra context would help.
- Do not pad with generic filler or mention categories that are irrelevant to this image.
""".strip()

DEFAULT_VISION_USER_PROMPT = (
    "Подробно разбери, что изображено на скриншоте. Дай структурированный ответ: "
    "что на экране, ключевые детали (точно процитируй важный текст, код, числа), "
    "и что это значит или что с этим делать."
)


def _answer_language_directive() -> str:
    """Force the answer language to the View → Language setting. Appended after the
    base prompt so it overrides the default 'Answer in Russian'. Read live from env."""
    lang = os.getenv("STACKWIRE_ANSWER_LANGUAGE", "ru").strip().lower()
    if lang == "en":
        return "\n\nLANGUAGE OVERRIDE (highest priority): Reply ONLY in English, regardless of the question's language."
    return "\n\nLANGUAGE (highest priority): Reply in Russian."


def _answer_system_prompt() -> str:
    # The base prompt is language-neutral; one clean line up front sets the language.
    lang = os.getenv("STACKWIRE_ANSWER_LANGUAGE", "ru").strip().lower()
    language = "Reply ONLY in English." if lang == "en" else "Reply in Russian."
    # Optional user "custom instructions" (style/tone/role) — appended so they steer the
    # response without overriding accuracy/safety. Empty by default.
    custom = os.getenv("STACKWIRE_CUSTOM_INSTRUCTIONS", "").strip()
    custom_block = (
        "\n\nUser's custom instructions (follow them as long as they don't conflict with "
        f"accuracy or safety):\n{custom}"
        if custom
        else ""
    )
    return f"{language}\n\n{ANSWER_SYSTEM_PROMPT}{custom_block}"


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
    answer_model: str = ""


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
    answer_model: str = ""


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


def _question_allows_history(question: str) -> bool:  # noqa: ARG001
    # Always allow recent chat history — the LLM needs it for follow-up questions.
    return True


def _history_section(question: str, context: list[str] | None) -> str:
    if not context or not _question_allows_history(question):
        return ""
    lines = [line.strip() for line in context[-14:] if line.strip()]
    if not lines:
        return ""
    return (
        "\n\nRecent conversation history (use it to understand follow-up questions AND to avoid repeating yourself):\n"
        + "\n".join(f"  {line}" for line in lines)
        + "\n\nIMPORTANT about history: if a similar question was already answered above, do NOT repeat the same material. "
        "Briefly acknowledge what was already covered in one short sentence, then add ONLY the new angle, the missing details, "
        "or the next logical step. The user is often in a live conversation — repeated identical answers are useless to them."
    )


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
        "Не уходи в нерелевантную тему.",
    ),
    "troubleshoot": (
        "Пиши на русском.",
        "Это troubleshooting mode, не definition mode.",
        "Структура обязательна: 'Причины:', 'Проверки:', 'Fix:'.",
        "В 'Проверки' дай реальные команды, проверки, логи, метрики или шаги диагностики, уместные для темы.",
        "Команды держи в fenced Markdown block с language tag.",
        "Не повторяй previous_answer, кроме короткой привязки к симптому.",
    ),
}


EXPAND_OUTPUT_SHAPES: dict[str, tuple[str, ...]] = {
    "details": (
        "Новый слой:",
        "Связи и механика:",
        "Production-нюансы:",
        "Типичные ошибки:",
    ),
    "components": (
        "Компоненты:",
        "Control plane / data plane:",
        "Data path / control flow:",
        "Кто за что отвечает:",
    ),
    "example": (
        "Минимальный пример:",
        "Практические замечания:",
    ),
    "compare": (
        "Главное отличие:",
        "Ближайшие аналоги:",
        "Когда что использовать:",
        "Нюанс:",
    ),
    "troubleshoot": (
        "Причины:",
        "Проверки:",
        "Fix:",
    ),
}


EXPAND_MODE_INTENT: dict[str, str] = {
    "details": "definition",
    "components": "architecture",
    "example": "example",
    "compare": "compare",
    "troubleshoot": "troubleshoot",
}


def _expand_output_shape(mode: str) -> str:
    return "\n".join(f"- {heading}" for heading in EXPAND_OUTPUT_SHAPES.get(mode, EXPAND_OUTPUT_SHAPES["details"]))


def _expand_validation_plan(plan: AnswerPlan, mode: str) -> AnswerPlan:
    mode = mode if mode in EXPAND_MODE_RULES else "details"
    intent = EXPAND_MODE_INTENT[mode]
    artifact_required = mode == "example"
    code_allowed = mode in {"example", "troubleshoot"} or plan.code_allowed
    return AnswerPlan(
        domain=plan.domain,
        intent=intent,
        artifact_required=artifact_required,
        code_allowed=code_allowed,
        answer_shape=" -> ".join(EXPAND_OUTPUT_SHAPES[mode]),
        required_concepts=plan.required_concepts,
        forbidden_concepts=plan.forbidden_concepts,
        dangerous_confusions=plan.dangerous_confusions,
        component_model=plan.component_model,
        depth="deep" if mode in {"details", "components", "troubleshoot"} else plan.depth,
    )


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
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    # Semantic recall first (saved good answers + remembered Q/A), then lexical.
    try:
        from app import vectorstore

        for hit in vectorstore.search_memory(question, limit=limit):
            q = hit.question.strip() or hit.title.strip()
            a = hit.answer.strip() or hit.text.strip()
            if q and a and q.casefold() not in seen:
                seen.add(q.casefold())
                pairs.append((q, a))
    except Exception:
        LOGGER.debug("vector good-answer search failed", exc_info=True)

    if len(pairs) < limit:
        try:
            for example in search_good_answers(question, domain=plan.domain, limit=limit):
                q = example.question.strip()
                if q and q.casefold() not in seen:
                    seen.add(q.casefold())
                    pairs.append((q, example.answer.strip()))
        except Exception:
            LOGGER.debug("good answer search failed", exc_info=True)

    if not pairs:
        return ""

    blocks: list[str] = []
    for index, (q, answer) in enumerate(pairs[:limit], start=1):
        if len(answer) > 900:
            answer = answer[:900].rstrip() + "..."
        blocks.append(
            f"Example {index} (style/reference only, do not copy verbatim):\n"
            f"Question: {q}\n"
            f"Answer:\n{answer}"
        )
    return "\n\n".join(blocks)


def _remember_answer(question: str, answer: str, plan: AnswerPlan, *, valid: bool) -> None:
    """Persist a good answer into the local vector store so the app 'remembers' it."""
    if not valid:
        return
    if os.getenv("STACKWIRE_REMEMBER_ANSWERS", "1").strip().lower() in {"0", "false", "no", "off"}:
        return
    try:
        from app import vectorstore

        vectorstore.remember(question, answer, domain=plan.domain, intent=plan.intent)
    except Exception:
        LOGGER.debug("vector remember skipped", exc_info=True)


def _format_rag_context(question: str, plan: AnswerPlan) -> str:
    """Hybrid retrieval: fuse semantic (vector) + lexical results with reciprocal-rank
    fusion, so we get semantic recall AND exact-term precision (commands, flags, paths).
    Degrades gracefully to whichever source is available."""
    semantic: list = []
    try:
        from app import vectorstore

        semantic = vectorstore.search_knowledge(question, limit=5)
    except Exception:
        LOGGER.debug("vector rag retrieval failed", exc_info=True)
    lexical: list = []
    try:
        lexical = retrieve_knowledge(question, plan, limit=5)
    except Exception:
        LOGGER.debug("lexical rag retrieval failed", exc_info=True)
    fused = _fuse_rag(semantic, lexical, limit=3)
    if not fused:
        return ""
    parts: list[str] = []
    used = 0
    for title, source, text in fused:
        header = f"[{source} :: {title}]" if title else f"[{source}]"
        block = header + chr(10) + text
        if used + len(block) > 3200:
            break
        parts.append(block)
        used += len(block)
    return (chr(10) + chr(10)).join(parts)


def _fuse_rag(semantic, lexical, *, k: int = 60, limit: int = 3):
    """Reciprocal-rank fusion of vector hits (VectorHit) and lexical chunks
    (KnowledgeChunk) into a deduped, re-ranked list of (title, source, text) tuples."""
    scores: dict[str, float] = {}
    meta: dict[str, tuple[str, str, str]] = {}

    def _add(items, getter):
        for rank, item in enumerate(items):
            title, source, text = getter(item)
            text = (text or "").strip()
            if not text:
                continue
            key = text[:120].casefold().strip()
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            meta.setdefault(key, (title or "", source or "", text))

    _add(semantic, lambda h: (h.title, h.source, h.text))
    _add(lexical, lambda c: (c.heading, c.source_file, c.text))
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [meta[key] for key, _ in ranked]


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
    # General question: don't wrap it in a rigid technical "contract" (domain / required
    # concepts / component model) — that scaffolding made every answer feel narrow and
    # un-ChatGPT-like. The system prompt already says how to answer well, so just hand the
    # model the question and the conversation history. Technical domains keep the contract.
    if plan.domain == "generic_software" and not plan.required_concepts:
        return f"{question}{_history_section(question, context)}".strip()

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
    example_breakdown_rule = (
        "For 'example' intent: after the artifact, add a breakdown section (table or bullet list) explaining each key part. "
        "Then add a 'Частые вариации' section listing 2-3 other common scenarios. "
        "End with a single clarifying question only if the topic has genuinely important branches."
        if plan.intent == "example"
        else ""
    )
    definition_breakdown_rule = (
        "For 'definition' intent: after the explanation, add a 'Ключевые компоненты' section as a short table or bullet list. "
        "Include a minimal practical example to ground the concept."
        if plan.intent == "definition"
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
- answer_shape is a CHECKLIST of what a complete answer usually covers — NOT a template. Adapt the structure freely to this specific question. NEVER copy shape labels (like "Что это (1 предложение)") as literal headings. Write natural headings or none at all.
- If the question is fragmentary or conversational (e.g. captured from speech), answer it the way a strong senior specialist would answer in a live conversation: directly, starting from the most likely intended meaning. Do not deliver a textbook lecture.
- Answer the full Question field, not only the first detected technical term.
- Include required_concepts naturally.
- Avoid forbidden_concepts and dangerous_confusions.
- Explain platform objects as platform objects, not generic concepts.
- Keep the answer focused, complete and logically finished; do not compress away important steps.
- {artifact_rule}
- {command_rule}
- {compare_rule}
- {example_breakdown_rule}
- {definition_breakdown_rule}
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
Обязательная форма ответа:
{_expand_output_shape(mode)}

Начни сразу с первой секции из обязательной формы. Не добавляй общий вступительный абзац перед ней.
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
Обязательная форма ответа:
{_expand_output_shape(mode)}

Начни сразу с первой секции из обязательной формы.
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
    return overlap >= 0.58


def _has_required_expand_sections(answer: str, mode: str) -> bool:
    lowered = answer.casefold()
    headings = EXPAND_OUTPUT_SHAPES.get(mode, ())
    return all(heading.casefold().rstrip(":") in lowered for heading in headings)


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
    if not _has_required_expand_sections(answer, mode):
        violations.append(f"expand_{mode}_missing_required_sections")
    if mode == "example" and "```" not in answer:
        violations.append("expand_example_missing_fenced_block")
    if mode != "example" and "```" in answer and not (mode == "troubleshoot" and plan.domain in {"kubernetes", "linux_fs", "linux_process", "linux_network", "ci_cd", "web_proxy", "observability"}):
        violations.append(f"expand_{mode}_unexpected_code_block")
    if mode == "compare" and ("```" in answer or re.search(r"(?m)^\s*(apiVersion|kind):", answer)):
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


WEB_SEARCH_UNCERTAINTY_MARKERS: tuple[str, ...] = (
    "не знаю",
    "не уверен",
    "недостаточно информаци",
    "не могу ответить",
    "нет информации",
    "не располага",
    "не нашёл",
    "не нашел",
    "затрудняюсь ответить",
    "неизвестно",
    "не известно",
    "i don't know",
    "cannot answer",
    "no information",
    "not sure",
)


def _web_search_enabled() -> bool:
    return os.getenv("STACKWIRE_WEB_SEARCH", "1").strip().lower() not in {"0", "false", "no", "off"}


def _deepthink_enabled() -> bool:
    """DeepThink: ask the model to reason first and surface that reasoning. Read live so
    the rail toggle applies immediately. Only reasoning models (e.g. qwen3) actually
    produce a separate thinking stream; others just answer normally."""
    return os.getenv("STACKWIRE_DEEPTHINK", "0").strip().lower() in {"1", "true", "yes", "on"}


def _answer_is_uncertain(answer: str) -> bool:
    text = answer.strip().casefold()
    if not text:
        return True
    head = text[:240]
    return any(marker in head for marker in WEB_SEARCH_UNCERTAINTY_MARKERS)


def _build_web_prompt(question: str, web_context: str) -> str:
    return f"""
Вопрос:
{question}

Свежие результаты веб-поиска DuckDuckGo. Используй их как источник фактов и не выдумывай:
{web_context}

Ответь на русском, опираясь на эти результаты. Если они не отвечают на вопрос — честно скажи об этом.
Не вставляй сырые URL в текст: список источников будет добавлен отдельно.
""".strip()


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

    def _chat(self, messages: list[dict[str, Any]], options: dict[str, Any], *, timeout: int = 300, model: str | None = None, provider: str | None = None) -> str:
        if (provider or current_llm_provider()) == "openai_compatible":
            payload: dict[str, Any] = {
                "model": model or current_answer_model(),
                "messages": messages,
                "stream": False,
                **_openai_payload_options(options),
            }
            response = self.session.post(current_openai_chat_url(), json=cast(Any, payload), headers=_openai_headers(), timeout=timeout)
            response.raise_for_status()
            return _openai_message_content(response.json())

        payload: dict[str, Any] = {
            "model": model or current_answer_model(),
            "messages": messages,
            "stream": False,
            "think": _deepthink_enabled(),
            "options": options,
        }
        if OLLAMA_KEEP_ALIVE:
            payload["keep_alive"] = OLLAMA_KEEP_ALIVE
        response = self.session.post(current_ollama_chat_url(), json=cast(Any, payload), timeout=timeout)
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        return str(message.get("content") or "").strip()

    def _chat_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        on_delta: Callable[[str], None] | None,
        *,
        timeout: int = 300,
        model: str | None = None,
        on_thinking: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        provider: str | None = None,
    ) -> str:
        if (provider or current_llm_provider()) == "openai_compatible":
            payload: dict[str, Any] = {
                "model": model or current_answer_model(),
                "messages": messages,
                "stream": True,
                **_openai_payload_options(options),
            }
            parts: list[str] = []
            with self.session.post(current_openai_chat_url(), json=cast(Any, payload), headers=_openai_headers(), timeout=timeout, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if should_stop is not None and should_stop():
                        break
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    try:
                        data = json.loads(line)
                    except ValueError:
                        continue
                    choices = data.get("choices") if isinstance(data, dict) else []
                    if not choices or not isinstance(choices[0], dict):
                        continue
                    delta = choices[0].get("delta") or {}
                    reasoning = str(delta.get("reasoning_content") or delta.get("reasoning") or "")
                    if reasoning and on_thinking is not None:
                        on_thinking(reasoning)
                    chunk = str(delta.get("content") or "")
                    if chunk:
                        parts.append(chunk)
                        if on_delta is not None:
                            on_delta(chunk)
            return "".join(parts).strip()

        payload: dict[str, Any] = {
            "model": model or current_answer_model(),
            "messages": messages,
            "stream": True,
            "think": _deepthink_enabled(),
            "options": options,
        }
        if OLLAMA_KEEP_ALIVE:
            payload["keep_alive"] = OLLAMA_KEEP_ALIVE
        parts: list[str] = []
        with self.session.post(current_ollama_chat_url(), json=cast(Any, payload), timeout=timeout, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if should_stop is not None and should_stop():
                    break
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                message = data.get("message") or {}
                thinking = str(message.get("thinking") or "")
                if thinking and on_thinking is not None:
                    on_thinking(thinking)
                chunk = str(message.get("content") or "")
                if chunk:
                    parts.append(chunk)
                    if on_delta is not None:
                        on_delta(chunk)
                if data.get("done"):
                    break
        return "".join(parts).strip()

    def _answer_options(self, plan: AnswerPlan) -> dict[str, Any]:
        num_predict = ARTIFACT_ANSWER_NUM_PREDICT if plan.artifact_required else DEFAULT_ANSWER_NUM_PREDICT
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
                {"role": "system", "content": _answer_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            options or self._answer_options(plan),
            timeout=300,
        )
        return _repair_answer(answer, question, plan)

    def _generate_answer_stream(
        self,
        question: str,
        plan: AnswerPlan,
        prompt: str,
        on_delta: Callable[[str], None] | None,
        *,
        options: dict[str, Any] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> str:
        answer = self._chat_stream(
            [
                {"role": "system", "content": _answer_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            options or self._answer_options(plan),
            on_delta,
            timeout=300,
            on_thinking=on_thinking,
            should_stop=should_stop,
        )
        return _repair_answer(answer, question, plan)

    def _web_fallback(self, question: str, plan: AnswerPlan, prior_answer: str, on_delta: Callable[[str], None] | None) -> str:
        try:
            results = search_duckduckgo(question)
        except Exception:
            LOGGER.debug("web search failed", exc_info=True)
            results = []
        if not results:
            note = "\n\nВеб-поиск не дал результатов."
            if on_delta is not None:
                on_delta(note)
            return prior_answer + note

        if on_delta is not None:
            on_delta("\n\nДополняю из веба:\n\n")
        grounded = self._chat_stream(
            [
                {"role": "system", "content": _answer_system_prompt()},
                {"role": "user", "content": _build_web_prompt(question, format_results_for_prompt(results))},
            ],
            self._answer_options(plan),
            on_delta,
        )
        grounded = _repair_answer(grounded, question, plan)
        sources = format_results_markdown(results)
        if on_delta is not None and sources:
            on_delta(f"\n\n{sources}")
        if not grounded:
            return f"{prior_answer}\n\n{sources}".strip()
        return f"{grounded}\n\n{sources}".strip()

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
        validation_plan = _expand_validation_plan(plan, normalized_mode)
        prompt = _build_expand_prompt(normalized_question, previous_answer, normalized_mode, plan)
        answer = self._generate_answer(normalized_question, plan, prompt, options=self._expand_options(plan, normalized_mode))
        validation = _validate_expand_mode(
            answer,
            normalized_mode,
            validate_answer(normalized_question, answer, validation_plan),
            previous_answer=previous_answer,
            plan=validation_plan,
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
                validate_answer(normalized_question, answer, validation_plan),
                previous_answer=previous_answer,
                plan=validation_plan,
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
                model=current_answer_model(),
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
            answer_model=current_answer_model(),
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
                model=current_answer_model(),
                answer_mode=ANSWER_MODE,
                latency_ms=answer_latency * 1000,
                validator_ok=validation.ok,
                validator_violations=validation.violations,
                plan_domain=plan.domain,
                plan_intent=plan.intent,
                artifact_required=plan.artifact_required,
            )
            _remember_answer(recovered_question, answer, plan, valid=validation.ok)
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
                answer_model=current_answer_model(),
            )

        recovery_started = time.perf_counter()
        recovery_input = condense_spoken_question(raw_text)
        if recovery_input and recovery_input != raw_text:
            LOGGER.info("condensed transcript for recovery raw_len=%s condensed_len=%s", len(raw_text), len(recovery_input))
        recovery = self.question_recovery.recover(recovery_input or raw_text, context)
        recovery_latency = time.perf_counter() - recovery_started

        if _recovery_is_unusable(recovery, raw_text):
            LOGGER.info("skip answer generation confidence=%.2f recovered_question=%r", recovery.confidence, recovery.recovered_question)
            return AskResult(
                raw_text=raw_text,
                recovery=recovery,
                answer=NEED_MANUAL_FIX_MESSAGE,
                answered=False,
                recovery_latency=recovery_latency,
                answer_latency=0.0,
                total_latency=time.perf_counter() - pipeline_started,
                answer_model=current_answer_model(),
            )
        if recovery.confidence < CONFIDENCE_THRESHOLD:
            LOGGER.info(
                "low recovery confidence=%.2f — answering best-effort with %r",
                recovery.confidence,
                recovery.recovered_question or raw_text,
            )

        started = time.perf_counter()
        answer = self.answer_question(recovery.recovered_question or raw_text, context)
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
            answer_model=current_answer_model(),
        )

    def ask_stream(
        self,
        raw_text: str,
        context: list[str] | None = None,
        *,
        trusted_text: bool = False,
        on_recovery: Callable[[str], None] | None = None,
        on_delta: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        creative: bool = False,
    ) -> AskResult:
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
            recovery_latency = 0.0
            # Trusted (typed) input still needs the chat history — dropping it here
            # was why follow-up questions "forgot" previous messages.
            answer_context: list[str] | None = context
            question = raw_text
        else:
            recovery_started = time.perf_counter()
            recovery_input = condense_spoken_question(raw_text)
            recovery = self.question_recovery.recover(recovery_input or raw_text, context)
            recovery_latency = time.perf_counter() - recovery_started
            if _recovery_is_unusable(recovery, raw_text):
                return AskResult(
                    raw_text=raw_text,
                    recovery=recovery,
                    answer=NEED_MANUAL_FIX_MESSAGE,
                    answered=False,
                    recovery_latency=recovery_latency,
                    answer_latency=0.0,
                    total_latency=time.perf_counter() - pipeline_started,
                    answer_model=current_answer_model(),
                )
            if recovery.confidence < CONFIDENCE_THRESHOLD:
                LOGGER.info(
                    "low recovery confidence=%.2f — streaming best-effort answer for %r",
                    recovery.confidence,
                    recovery.recovered_question or raw_text,
                )
            answer_context = context
            question = recovery.recovered_question or raw_text

        normalized = normalize_question(question)
        if not normalized:
            return AskResult(
                raw_text=raw_text,
                recovery=recovery,
                answer="Вопрос нужно уточнить.",
                answered=False,
                recovery_latency=recovery_latency,
                answer_latency=0.0,
                total_latency=time.perf_counter() - pipeline_started,
                answer_model=current_answer_model(),
            )

        if on_recovery is not None:
            on_recovery(normalized)

        started = time.perf_counter()
        plan = build_answer_plan(normalized)
        prompt = _build_prompt(normalized, plan, answer_context)
        # Regeneration: bump temperature + random seed so each click yields a fresh variant
        # rather than the near-deterministic default (temperature ~0.04).
        regen_options: dict[str, Any] | None = None
        if creative:
            regen_options = self._answer_options(plan)
            regen_options["temperature"] = float(os.getenv("OLLAMA_REGEN_TEMPERATURE", "0.7"))
            regen_options["top_p"] = 0.95
            regen_options["seed"] = random.randint(1, 2_000_000_000)
        answer = self._generate_answer_stream(normalized, plan, prompt, on_delta, options=regen_options, on_thinking=on_thinking, should_stop=should_stop) or "Вопрос нужно уточнить."
        if _web_search_enabled() and _answer_is_uncertain(answer) and not (should_stop is not None and should_stop()):
            answer = self._web_fallback(normalized, plan, answer, on_delta)
        answer_latency = time.perf_counter() - started
        total_latency = time.perf_counter() - pipeline_started

        _append_query_history(
            raw_text=raw_text,
            recovered_question=normalized,
            answer=answer,
            answered=True,
            recovery_latency=recovery_latency,
            answer_latency=answer_latency,
            total_latency=total_latency,
        )
        question_id, answer_id, logged_plan = self._log_main_answer(
            raw_text=raw_text,
            recovered_question=normalized,
            trusted_text=trusted_text,
            recovery_confidence=recovery.confidence,
            detected_topic=recovery.detected_topic,
            answer=answer,
            answer_latency=answer_latency,
        )
        effective_plan = logged_plan or plan
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
            plan_domain=effective_plan.domain,
            plan_intent=effective_plan.intent,
            answer_model=current_answer_model(),
        )

    def _vision_request(self, image_b64, prompt: str | None, *, creative: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:  # noqa: ANN001
        """Build the (messages, options) for a vision call. image_b64 may be a single
        base64 string or a list of them (several attached images). None if none usable."""
        raw_images = image_b64 if isinstance(image_b64, (list, tuple)) else [image_b64]
        images: list[str] = []
        for img in raw_images:
            img = (img or "").strip()
            if img.startswith("data:image"):
                img = img.split(",", 1)[-1]
            if img:
                images.append(img)
        if not images:
            return None
        user_prompt = (prompt or "").strip() or DEFAULT_VISION_USER_PROMPT
        messages = [
            {"role": "system", "content": VISION_SYSTEM_PROMPT + _answer_language_directive()},
            {"role": "user", "content": user_prompt, "images": images},
        ]
        options = {
            "num_ctx": _env_int("OLLAMA_VISION_NUM_CTX", DEFAULT_NUM_CTX),
            "num_predict": DEFAULT_VISION_NUM_PREDICT,
            "temperature": 0.05,
            "top_p": 0.75,
        }
        if creative:
            # Regenerate: bump temperature + random seed so each click yields a fresh
            # variant instead of repeating the near-deterministic default answer.
            options["temperature"] = float(os.getenv("OLLAMA_REGEN_TEMPERATURE", "0.7"))
            options["top_p"] = 0.95
            options["seed"] = random.randint(1, 2_000_000_000)
        return messages, options

    def analyze_image(self, image_b64: str, prompt: str | None = None) -> str:
        request = self._vision_request(image_b64, prompt)
        if request is None:
            return "Could not read the image."
        messages, options = request
        answer = self._chat(messages, options, timeout=300, model=current_vision_model(), provider=current_vision_provider())
        return _strip_model_noise(answer) or "Не удалось распознать содержимое области."

    def analyze_image_stream(
        self,
        image_b64: str,
        prompt: str | None = None,
        on_delta: Callable[[str], None] | None = None,
        *,
        timeout: int = 300,
        creative: bool = False,
    ) -> str:
        """Stream a vision answer token-by-token (Ollama). Returns the cleaned full text."""
        request = self._vision_request(image_b64, prompt, creative=creative)
        if request is None:
            return "Could not read the image."
        messages, options = request
        answer = self._chat_stream(messages, options, on_delta, timeout=timeout, model=current_vision_model(), provider=current_vision_provider())
        return _strip_model_noise(answer) or "Не удалось распознать содержимое области."
