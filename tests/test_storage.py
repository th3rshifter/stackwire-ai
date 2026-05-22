from app import storage


def test_storage_logs_feedback_good_answer_and_exports_session(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STACKWIRE_DB_PATH", str(tmp_path / "stackwire.db"))

    storage.init_db()
    session_id = storage.create_session("test session")
    question_id = storage.log_question(
        session_id=session_id,
        raw_text="что такое Ingress",
        recovered_question="что такое Ingress",
        trusted_text=True,
        source="manual",
        recovery_confidence=1.0,
        detected_topic="kubernetes",
    )
    answer_id = storage.log_answer(
        question_id=question_id,
        answer="Ingress is a Kubernetes API object plus controller.",
        answer_type="main",
        model="test-model",
        answer_mode="normal",
        latency_ms=12.0,
        validator_ok=True,
        validator_violations=[],
        plan_domain="kubernetes",
        plan_intent="definition",
        artifact_required=False,
    )

    feedback_id = storage.log_feedback(answer_id, "good")
    good_id = storage.save_good_answer(
        question="что такое Ingress",
        answer="Ingress is a Kubernetes API object plus controller.",
        domain="kubernetes",
        intent="definition",
        tags=["k8s"],
    )
    matches = storage.search_good_answers("Ingress controller Service", domain="kubernetes", limit=3)
    recent = storage.get_recent_questions(limit=5)
    exported = storage.export_session_markdown(session_id)

    assert feedback_id > 0
    assert good_id > 0
    assert matches and matches[0].id == good_id
    assert recent and recent[0]["id"] == question_id
    assert "test session" in exported
    assert "Ingress is a Kubernetes API object" in exported
