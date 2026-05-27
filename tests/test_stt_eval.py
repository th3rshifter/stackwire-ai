from app.stt_eval import normalize_for_score, term_accuracy, word_error_rate


def test_normalize_for_score_repairs_spoken_terms() -> None:
    assert normalize_for_score("чем тиси пи отличается от юдипи?") == [
        "чем",
        "tcp",
        "отличается",
        "от",
        "udp",
    ]


def test_word_error_rate_is_zero_for_equal_normalized_text() -> None:
    assert word_error_rate("Kubernetes Deployment", "kubernetes deployment") == 0.0


def test_term_accuracy_reports_missing_terms() -> None:
    accuracy, missing = term_accuracy(
        ("Kubernetes", "Deployment", "Ingress"),
        "Расскажи про Kubernetes и Deployment",
    )

    assert accuracy == 2 / 3
    assert missing == ("Ingress",)
