from dataclasses import replace

from app.config import STTSettings, update_stt_language_lock, whisper_language, whisper_model_attempts
from app.stt_eval import EvalResult, normalize_for_score, term_accuracy, word_error_rate, write_training_report


def _settings(language_mode: str = "auto") -> STTSettings:
    return STTSettings(
        backend="whisper",
        allow_vosk_fallback=False,
        allow_cpu_whisper_fallback=True,
        mic_signal_threshold=0.003,
        loopback_signal_threshold=0.00025,
        probe_loopback_devices=False,
        live_max_words=900,
        context_lines=60,
        model="large-v3-turbo",
        device="cpu",
        compute_type="int8",
        chunk_seconds=3.5,
        chunk_overlap_seconds=1.0,
        sample_rate=16000,
        language_mode=language_mode,
        beam_size=5,
        best_of=5,
        vad_filter=True,
        retry_without_vad=True,
        vad_threshold=0.20,
        vad_min_speech_ms=100,
        vad_min_silence_ms=650,
        vad_speech_pad_ms=450,
        no_speech_threshold=0.75,
        log_prob_threshold=-2.0,
        compression_ratio_threshold=3.0,
        repetition_penalty=1.08,
        no_repeat_ngram_size=3,
        hallucination_silence_threshold=1.0,
        hotwords=None,
    )


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


def test_auto_language_mode_locks_only_after_good_text() -> None:
    settings = _settings("auto")

    assert whisper_language(settings) is None
    assert update_stt_language_lock(settings, None, "en", "", bad_text=False) is None
    assert update_stt_language_lock(settings, None, "en", "How do you scale ECS?", bad_text=False) == "en"
    assert whisper_language(settings, locked_language="en") == "en"


def test_fixed_language_mode_ignores_detected_language() -> None:
    settings = _settings("ru")

    assert whisper_language(settings) == "ru"
    assert update_stt_language_lock(settings, None, "en", "hello", bad_text=False) is None


def test_auto_device_prefers_cuda_then_cpu_fallback() -> None:
    settings = replace(_settings("auto"), device="auto", compute_type="auto")

    attempts = whisper_model_attempts(settings)

    assert attempts[0][0] == "cuda"
    assert attempts[-1] == ("cpu", "int8")


def test_write_training_report_highlights_weak_cases(tmp_path) -> None:
    report = tmp_path / "training.md"
    results = [
        EvalResult(
            case_id="weak",
            expected="question",
            transcript="bad transcript",
            repaired="bad transcript",
            wer=0.5,
            term_accuracy=0.5,
            missing_terms=("Kubernetes",),
            latency_ms=100.0,
            recovered_question="bad transcript?",
            recovery_confidence=0.4,
            recovery_needs_manual_fix=True,
        )
    ]

    write_training_report(
        results,
        report,
        min_average_term_accuracy=0.9,
        min_case_term_accuracy=0.75,
    )

    text = report.read_text(encoding="utf-8")
    assert "weak cases: 1" in text
    assert "Kubernetes: 1" in text
