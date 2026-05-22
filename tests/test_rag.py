from app.answer_planner import build_answer_plan
from app.rag import format_knowledge_chunks, load_chunks, retrieve_knowledge


def test_rag_retrieves_relevant_markdown_chunks(tmp_path, monkeypatch) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "kubernetes.md").write_text(
        "# Kubernetes\n\n"
        "## Ingress\n"
        "Ingress is an API object. Ingress Controller programs Nginx or Envoy and routes traffic to Service backends.\n\n"
        "## StatefulSet\n"
        "StatefulSet keeps stable identity and PVCs.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STACKWIRE_KNOWLEDGE_DIR", str(knowledge_dir))

    chunks = load_chunks()
    plan = build_answer_plan("что такое Ingress в Kubernetes")
    matches = retrieve_knowledge("что такое Ingress Controller Service", plan, limit=2)
    formatted = format_knowledge_chunks(matches)

    assert len(chunks) == 2
    assert matches
    assert matches[0].source_file == "kubernetes.md"
    assert "Ingress Controller" in formatted


def test_rag_empty_directory_is_safe(tmp_path, monkeypatch) -> None:
    knowledge_dir = tmp_path / "empty"
    knowledge_dir.mkdir()
    monkeypatch.setenv("STACKWIRE_KNOWLEDGE_DIR", str(knowledge_dir))

    plan = build_answer_plan("что такое TCP")

    assert retrieve_knowledge("что такое TCP", plan) == []
