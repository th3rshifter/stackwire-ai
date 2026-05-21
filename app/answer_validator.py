import re
from dataclasses import dataclass

from app.answer_planner import AnswerPlan, normalize_question
from app.knowledge_base import INFRASTRUCTURE_DOMAINS


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    violations: tuple[str, ...]


FENCED_BLOCK_RE = re.compile(r"```[A-Za-z0-9_-]+\n.*?\n```", re.DOTALL)
ANY_FENCE_RE = re.compile(r"```")

CODE_LINE_RE = re.compile(
    r"(?m)^\s*(apiVersion:|kind:|FROM\s+|RUN\s+|COPY\s+|CMD\s+|ENTRYPOINT\s+|"
    r"server\s*\{|location\s+|upstream\s+|provider\s+|resource\s+|stages:|jobs:|"
    r"awk\s+|sort\s+|uniq\s+|head\s+|kubectl\s+|curl\s+|df\s+|du\s+|wc\s+|"
    r"ps\s+|journalctl\s+|systemctl\s+|groups? by\s*\(|rate\s*\(|sum\s*\()",
    re.IGNORECASE,
)


def validate_answer(question: str, answer: str, plan: AnswerPlan) -> ValidationResult:
    normalized_question = normalize_question(question)
    q = normalized_question.casefold()
    a = answer.casefold()
    violations: list[str] = []

    for forbidden in plan.forbidden_concepts:
        if forbidden and forbidden.casefold() in a:
            violations.append(f"forbidden_concept:{forbidden}")

    if plan.artifact_required and not FENCED_BLOCK_RE.search(answer):
        violations.append("artifact_without_fenced_markdown")

    if CODE_LINE_RE.search(answer) and not FENCED_BLOCK_RE.search(answer):
        violations.append("code_or_config_not_fenced")

    if answer.strip() and answer.strip()[-1] not in ".!?`":
        violations.append("unfinished_answer")

    if plan.domain == "kubernetes":
        _validate_kubernetes(q, a, violations)

    if plan.domain == "observability":
        _validate_observability(q, a, violations)

    if plan.intent == "troubleshoot":
        _validate_troubleshoot(a, violations)

    _validate_context_leak(q, a, plan, violations)

    if plan.intent in {"definition", "compare"} and plan.domain in INFRASTRUCTURE_DOMAINS:
        min_words = 50 if plan.intent == "compare" else 55
        if len(re.findall(r"\w+", answer)) < min_words:
            violations.append("too_generic")

    if "```" in answer and not _all_fences_are_language_tagged(answer):
        violations.append("fenced_block_without_language")

    return ValidationResult(ok=not violations, violations=tuple(dict.fromkeys(violations)))


def _validate_kubernetes(q: str, a: str, violations: list[str]) -> None:
    if "statefulset" in q:
        stateful_markers = (
            "stable identity",
            "стабильн",
            "ordinal",
            "порядков",
            "headless service",
            "pvc",
            "persistentvolumeclaim",
            "ordered rollout",
            "упорядоч",
        )
        if sum(1 for marker in stateful_markers if marker in a) < 2:
            violations.append("statefulset_missing_stable_identity_model")

    if "deployment" in q:
        deployment_markers = (
            "kubernetes",
            "workload controller",
            "контроллер",
            "replicaset",
            "pod",
            "desired state",
        )
        generic_bad = ("generic deployment", "процесс деплоя", "процесс развертывания")
        if not any(marker in a for marker in deployment_markers) or any(marker in a for marker in generic_bad):
            violations.append("deployment_not_kubernetes_workload_controller")

    if "ingress" in q:
        ingress_markers = ("ingress", "object", "объект", "controller", "контроллер", "service", "host", "path", "tls")
        if sum(1 for marker in ingress_markers if marker in a) < 4:
            violations.append("ingress_missing_object_controller_service_model")
        if any(marker in a for marker in ("statefulset", "deployment")) and not any(marker in q for marker in ("statefulset", "deployment")):
            violations.append("context_leak")

    if "gateway" in q or "httproute" in q:
        gateway_markers = ("gateway api", "httproute", "ingress", "controller", "route")
        if not any(marker in a for marker in gateway_markers):
            violations.append("gateway_context_missing")

    if "imagepolicybackoff" in q:
        ambiguity_markers = ("suspicious", "подозр", "ambiguous", "не стандарт", "imagepullbackoff", "errimagepull", "admission", "image policy")
        invented_markers = ("standard kubernetes status", "стандартный статус", "imagepolicy controller", "контроллер imagepolicybackoff")
        if not any(marker in a for marker in ambiguity_markers):
            violations.append("imagepolicybackoff_ambiguity_missing")
        if any(marker in a for marker in invented_markers):
            violations.append("imagepolicybackoff_invented_mechanism")


def _validate_observability(q: str, a: str, violations: list[str]) -> None:
    if "promql" in q or "label" in q or "лейбл" in q:
        has_metric = "metric" in a or "метрик" in a
        has_label = "label" in a or "лейбл" in a
        has_selector = bool(re.search(r"\w+\s*\{[^}]+=", a))
        if not (has_metric and has_label):
            violations.append("promql_metric_label_not_distinguished")
        if ("по лейблу" in q or "label" in q or "лейбл" in q) and not has_selector:
            violations.append("promql_label_selector_missing")

    if "burn rate" in q or "burnrate" in q:
        if not any(marker in a for marker in ("slo", "error budget", "бюджет ошибок", "не классический burn rate", "не классический")):
            violations.append("burn_rate_not_tied_to_slo_error_budget")


def _validate_troubleshoot(a: str, violations: list[str]) -> None:
    has_check = any(marker in a for marker in ("kubectl", "journalctl", "logs", "describe", "events", "провер", "команд", "curl", "dig", "ps ", "df ", "du "))
    has_fix = any(marker in a for marker in ("fix", "исправ", "почин", "перезапу", "rollback", "увелич", "замен", "обнов"))
    if not has_check or not has_fix:
        violations.append("troubleshoot_missing_checks_or_fix")
    if "crashloopbackoff" in a and "kubectl exec" in a and "logs --previous" not in a and "describe pod" not in a:
        violations.append("crashloopbackoff_bad_first_check")


def _validate_context_leak(q: str, a: str, plan: AnswerPlan, violations: list[str]) -> None:
    if "ingress" in q and "deployment" not in q and "statefulset" not in q:
        if "deployment" in a or "statefulset" in a:
            violations.append("context_leak")

    if plan.domain == "kubernetes" and not any(term in q for term in ("inode", "df", "du", "lsof", "filesystem", "диск")):
        if any(term in a for term in ("inode", "lsof", "file descriptor", "дескриптор")):
            violations.append("context_leak")

    if plan.domain != "kubernetes" and "kubernetes" not in q:
        leaked = ("statefulset", "imagepullbackoff", "crashloopbackoff")
        if any(term in a for term in leaked):
            violations.append("context_leak")


def _all_fences_are_language_tagged(answer: str) -> bool:
    fence_count = 0
    for match in re.finditer(r"```([^\n`]*)", answer):
        fence_count += 1
        if fence_count % 2 == 1 and not match.group(1).strip():
            return False
    return True
