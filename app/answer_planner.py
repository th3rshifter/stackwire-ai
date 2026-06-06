import re
from dataclasses import dataclass

from app.knowledge_base import DOMAIN_PROFILES, INFRASTRUCTURE_DOMAINS
from app.tech_terms import normalize_spoken_technical_terms


@dataclass(frozen=True)
class AnswerPlan:
    domain: str
    intent: str
    artifact_required: bool
    code_allowed: bool
    answer_shape: str
    required_concepts: tuple[str, ...]
    forbidden_concepts: tuple[str, ...]
    dangerous_confusions: tuple[str, ...]
    component_model: str
    depth: str


ANSWER_SHAPES: dict[str, str] = {
    "definition": "Коротко -> Как работает -> Компоненты -> Практика -> Нюанс",
    "compare": "Коротко -> Названные варианты -> Главное отличие -> Когда что использовать -> Нюанс",
    "troubleshoot": "Что означает симптом -> Частые причины -> Как проверить -> Fix",
    "configure": "Что настраиваем -> Компоненты -> Шаги -> Проверка -> Подводные камни",
    "architecture": "Контекст -> Компоненты -> Control plane/Data plane -> Компромиссы -> Failure modes",
    "example": "Артефакт -> Практические замечания",
    "analogy": "Коротко -> Аналогия -> Где аналогия ломается -> Практический смысл",
    "command_explain": "Что делает команда -> Пример -> Ключевые флаги/поля -> Нюанс",
}

EXPLICIT_ARTIFACT_RE = re.compile(
    r"\b(example|show|write|yaml|yml|manifest|config|dockerfile|playbook|pipeline|values|template|command)\b|"
    r"\.tf\b|"
    r"\b(пример|покажи|напиши|написать|код|конфиг|конфигурац|манифест|ямл|плейбук|пайплайн|команд[ауые]?|values|template)\b",
    re.IGNORECASE,
)

TROUBLESHOOT_RE = re.compile(
    r"\b(почему|как\s+дебажить|как\s+диагностировать|не\s+работает|timeout|failed|error|backoff|crashloopbackoff|imagepullbackoff|imagepolicybackoff|errimagepull)\b",
    re.IGNORECASE,
)

CONFIGURE_RE = re.compile(r"\b(как\s+настроить|настроить|configure|setup|install|deploy|сконфигур)\b", re.IGNORECASE)
COMPARE_RE = re.compile(r"\b(чем|различие|разница|отлича|сравни|vs|versus|между)\b", re.IGNORECASE)
ARCHITECTURE_RE = re.compile(r"\b(архитектур|system\s*design|проектировал|внедрял|строил)\b", re.IGNORECASE)
ANALOGY_RE = re.compile(r"\b(аналогия|простыми\s+словами|как\s+представить)\b", re.IGNORECASE)
DEFINITION_RE = re.compile(r"\b(что\s+такое|что\s+за|объясни|как\s+работает)\b", re.IGNORECASE)

COMMAND_DOMAIN_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("linux_fs", ("df", "du", "lsof", "find", "awk", "sort", "uniq", "head", "tail", "wc", "/var/log")),
    ("linux_process", ("ps", "top", "htop", "kill", "strace", "journalctl", "systemctl")),
    ("linux_network", ("netstat", "ss", "tcpdump", "ip route", "ip addr", "dig", "curl", "nslookup")),
)

COMMAND_RE = re.compile(
    r"(^|\s)(wc|netstat|ss|lsof|df|du|ps|journalctl|systemctl|awk|sort|uniq|head|tail|grep|cat|tcpdump|dig|curl)(\s|$|-)",
    re.IGNORECASE,
)


def normalize_question(question: str) -> str:
    normalized = normalize_spoken_technical_terms(question.strip())
    replacements = (
        (r"\bservish\s+mesh\b", "service mesh"),
        (r"\bservis\s+mesh\b", "service mesh"),
        (r"\bservise\s+mesh\b", "service mesh"),
        (r"\bсервис\s*меш\b", "service mesh"),
        (r"\bстейтфул+\s*сет\b", "StatefulSet"),
        (r"\bстейтфул+\s*set\b", "StatefulSet"),
        (r"\bдеплоймент[аеоы]?\b", "Deployment"),
        (r"\bингресс\b", "Ingress"),
        (r"\bгейтвей\b", "Gateway"),
        (r"\bпромете(?:й|ус|йс|я)\b", "Prometheus"),
        (r"\bграфан[ауы]?\b", "Grafana"),
    )
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", normalized).strip()


def build_answer_plan(question: str) -> AnswerPlan:
    normalized = normalize_question(question)
    lowered = normalized.casefold()
    domain = _detect_domain(lowered)
    intent = _detect_intent(lowered)
    artifact_required = bool(EXPLICIT_ARTIFACT_RE.search(lowered))
    code_allowed = artifact_required or intent == "command_explain"

    profile = DOMAIN_PROFILES[domain]
    required = list(profile.required_concepts)
    forbidden = list(profile.forbidden_concepts)
    dangerous = list(profile.dangerous_confusions)

    if domain in INFRASTRUCTURE_DOMAINS and profile.component_model:
        required.append("components")

    if domain == "kubernetes" and {"deployment", "statefulset"} <= set(re.findall(r"deployment|statefulset", lowered)):
        intent = "compare"
        required.extend(
            (
                "Deployment as Kubernetes workload controller",
                "StatefulSet stable identity",
                "ordinal Pod names",
                "Headless Service",
                "stable PVC",
            )
        )

    if domain == "kubernetes" and "ingress" in lowered:
        required.extend(("Ingress API object", "Ingress Controller", "Service backend", "host/path routing", "TLS"))

    if domain == "kubernetes" and ("gateway" in lowered or "httproute" in lowered):
        required.extend(("Gateway API or Istio gateway context", "Gateway", "HTTPRoute", "listener", "backendRef"))

    if domain == "kubernetes" and "crashloopbackoff" in lowered:
        required.extend(("kubectl describe pod", "kubectl logs --previous", "events", "exit code/config/secrets/probes"))

    if "imagepolicybackoff" in lowered:
        domain = "kubernetes"
        profile = DOMAIN_PROFILES[domain]
        intent = "troubleshoot"
        required.extend(("ImagePolicyBackOff ambiguity", "ImagePullBackOff", "ErrImagePull", "admission/image policy only if explicit"))
        dangerous.extend(profile.dangerous_confusions)

    if domain == "observability" and ("promql" in lowered or "label" in lowered or "лейбл" in lowered):
        required.extend(("metric name", "label selector", 'metric{label="value"} syntax'))
    if "burn rate" in lowered or "burnrate" in lowered:
        required.extend(("SLO", "error budget", "burn rate ratio"))

    if _is_request_diff_question(lowered):
        required.extend(
            (
                "compare the same request on both servers",
                "HTTP method/path/query/body/headers",
                "status code/response body/timing",
                "access logs or correlation id",
                "upstream/app config differences",
            )
        )
        code_allowed = True

    if intent == "command_explain":
        code_allowed = True
        if COMMAND_RE.search(lowered) or any(marker in lowered for marker in ("как вывести", "как посчитать", "топ ip", "/var/log")):
            required.extend(("command example", "fenced bash block", "pipeline explanation"))

    depth = "deep" if intent in {"architecture", "troubleshoot"} else "normal"
    if intent == "command_explain":
        depth = "normal"

    return AnswerPlan(
        domain=domain,
        intent=intent,
        artifact_required=artifact_required,
        code_allowed=code_allowed,
        answer_shape=ANSWER_SHAPES[intent],
        required_concepts=tuple(dict.fromkeys(required)),
        forbidden_concepts=tuple(dict.fromkeys(forbidden)),
        dangerous_confusions=tuple(dict.fromkeys(dangerous)),
        component_model=profile.component_model,
        depth=depth,
    )


def _detect_domain(lowered_question: str) -> str:
    if _is_request_diff_question(lowered_question):
        return "web_proxy"

    if {"deployment", "statefulset"} <= set(re.findall(r"deployment|statefulset", lowered_question)):
        return "kubernetes"

    if any(term in lowered_question for term in ("promql", "prometheus", "grafana", "burn rate", "slo", "sli", "metric", "label", "лейбл")):
        return "observability"

    if any(term in lowered_question for term in ("service mesh", "istio", "linkerd", "consul connect", "virtualservice", "destinationrule")):
        return "service_mesh"

    if any(term in lowered_question for term in ("ingress", "httproute", "gateway api")):
        return "kubernetes"

    for command_domain, commands in COMMAND_DOMAIN_HINTS:
        if any(_contains_trigger(lowered_question, command) for command in commands):
            return command_domain

    scores: dict[str, int] = {}
    for domain, profile in DOMAIN_PROFILES.items():
        score = 0
        matched = 0
        for trigger in profile.triggers:
            if _contains_trigger(lowered_question, trigger):
                matched += 1
                score += 3 if " " in trigger or len(trigger) >= 8 else 1
        if matched:
            scores[domain] = score + min(matched, 3)

    if not scores:
        return "generic_software"
    best_domain, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score <= 1 and best_domain not in {"linux_fs", "linux_process", "linux_network"}:
        return "generic_software"
    return best_domain


def _detect_intent(lowered_question: str) -> str:
    if _is_command_question(lowered_question):
        return "command_explain"
    if _is_request_diff_question(lowered_question):
        return "troubleshoot"
    if TROUBLESHOOT_RE.search(lowered_question):
        return "troubleshoot"
    if ARCHITECTURE_RE.search(lowered_question):
        return "architecture"
    if COMPARE_RE.search(lowered_question):
        return "compare"
    if CONFIGURE_RE.search(lowered_question):
        return "configure"
    if EXPLICIT_ARTIFACT_RE.search(lowered_question):
        return "example"
    if ANALOGY_RE.search(lowered_question):
        return "analogy"
    if DEFINITION_RE.search(lowered_question):
        return "definition"
    return "definition"


def _is_command_question(lowered_question: str) -> bool:
    if COMMAND_RE.search(lowered_question):
        return True
    return any(marker in lowered_question for marker in ("как вывести", "как посчитать", "топ ip", "top ip")) and any(
        file_marker in lowered_question for file_marker in ("/var/log", ".log", "лог")
    )


def _is_request_diff_question(lowered_question: str) -> bool:
    has_request = any(term in lowered_question for term in ("запрос", "request", "http", "api", "curl"))
    has_server_context = any(term in lowered_question for term in ("сервер", "server", "host", "хост", "endpoint", "эндпоинт"))
    has_diff_or_check = any(
        term in lowered_question
        for term in (
            "разница",
            "различие",
            "отлич",
            "сравни",
            "сравнить",
            "определить",
            "проверить",
            "найти",
            "different",
            "compare",
            "diff",
        )
    )
    return has_request and has_server_context and has_diff_or_check


def _contains_trigger(lowered_question: str, trigger: str) -> bool:
    trigger = trigger.casefold()
    if not trigger:
        return False
    if re.search(r"[^\w\s/.-]", trigger):
        return trigger in lowered_question
    if " " in trigger or "/" in trigger or "." in trigger or "-" in trigger:
        return trigger in lowered_question
    return bool(re.search(rf"(?<![\w-]){re.escape(trigger)}(?![\w-])", lowered_question))
