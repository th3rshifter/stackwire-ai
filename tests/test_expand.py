from app.llm import OllamaClient


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, dict[str, str]]:
        return {"message": {"content": self.content}}


class _FakeSession:
    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.payloads: list[dict] = []

    def post(self, _url: str, *, json: dict, timeout: int) -> _FakeResponse:
        self.payloads.append(json)
        index = min(len(self.payloads) - 1, len(self.answers) - 1)
        return _FakeResponse(self.answers[index])


def test_expand_example_retries_until_fenced_block(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STACKWIRE_DB_PATH", str(tmp_path / "stackwire.db"))
    session = _FakeSession(
        [
            "Пример: kubectl get pods",
            """
```bash
# namespace with the workload
kubectl -n default get pods
# inspect one pod
kubectl -n default describe pod app-0
# previous container logs
kubectl -n default logs app-0 --previous
```

Практические замечания:
- Сначала смотри events.
- Для рестартов важны previous logs.
""",
        ]
    )
    client = OllamaClient(session=session)

    result = client.expand("как проверить Pod в Kubernetes", "Pod запускает workload.", "example")

    assert result.mode == "example"
    assert "```bash" in result.answer
    assert len(session.payloads) == 2
    retry_prompt = session.payloads[1]["messages"][1]["content"]
    assert "expand_example_missing_fenced_block" in retry_prompt


def test_expand_details_retries_when_answer_repeats_previous(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STACKWIRE_DB_PATH", str(tmp_path / "stackwire.db"))
    previous = (
        "Ingress - это Kubernetes API object для HTTP routing. "
        "Ingress Controller читает объект и настраивает proxy. "
        "Service является backend для Pod."
    )
    better = (
        "Новый слой: в production важно разделять ownership Ingress object и controller. "
        "Команда приложения обычно меняет host/path/TLS rules, а platform-команда держит controller, "
        "ingressClass, default backend, лимиты body size, timeout и reload behaviour."
    )
    session = _FakeSession([previous, better])
    client = OllamaClient(session=session)

    result = client.expand("что такое Ingress", previous, "details")

    assert result.answer == better
    assert len(session.payloads) == 2
    assert "expand_repeats_previous_answer" in session.payloads[1]["messages"][1]["content"]


def test_expand_troubleshoot_requires_causes_checks_fix(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STACKWIRE_DB_PATH", str(tmp_path / "stackwire.db"))
    bad = "Ingress - это объект для HTTP routing."
    good = """
Причины:
- Не тот ingressClass.
- Backend Service не имеет endpoints.

Проверки:
```bash
kubectl describe ingress app
kubectl get endpoints app
kubectl logs -n ingress-nginx deploy/ingress-nginx-controller
```

Fix:
Исправить ingressClass, Service selector или TLS secret.
"""
    session = _FakeSession([bad, good])
    client = OllamaClient(session=session)

    result = client.expand("почему Ingress не работает в Kubernetes", "Ingress routes HTTP traffic.", "troubleshoot")

    assert "Причины" in result.answer
    assert "```bash" in result.answer
    assert len(session.payloads) == 2
    assert "expand_troubleshoot_missing_causes_checks_fix" in session.payloads[1]["messages"][1]["content"]


def test_expand_compare_requires_compare_sections(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STACKWIRE_DB_PATH", str(tmp_path / "stackwire.db"))
    bad = "Ingress и Gateway API используются для HTTP routing."
    good = """
Главное отличие:
Ingress задает простой host/path routing, а Gateway API разделяет инфраструктурный Gateway и прикладные Routes.

Ближайшие аналоги:
- Ingress Controller.
- Gateway API с HTTPRoute.

Когда что использовать:
- Ingress проще для базового HTTP routing.
- Gateway API лучше, когда нужны роли, shared Gateway и более строгая модель маршрутов.

Нюанс:
Оба варианта все равно требуют controller, который реально программирует proxy.
"""
    session = _FakeSession([bad, good])
    client = OllamaClient(session=session)

    result = client.expand("сравни Ingress и Gateway API", "Ingress routes HTTP traffic.", "compare")

    assert "Главное отличие" in result.answer
    assert len(session.payloads) == 2
    assert "expand_compare_missing_required_sections" in session.payloads[1]["messages"][1]["content"]
