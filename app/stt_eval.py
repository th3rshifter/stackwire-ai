import argparse
import json
import subprocess
import tempfile
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_stt_settings, is_cuda_whisper_error, load_local_env, whisper_language, whisper_model_attempts, whisper_vad_parameters
from app.tech_terms import WHISPER_TECHNICAL_PROMPT
from app.transcript_repair import clean_stt_output, condense_spoken_question

load_local_env()
STT_SETTINGS = get_stt_settings()

DEFAULT_OUTPUT_DIR = Path("data/stt_eval")
DEFAULT_REPORT = Path("logs/stt_eval_report.md")
DEFAULT_TRAINING_REPORT = Path("logs/stt_training_report.md")
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    text: str
    expected_terms: tuple[str, ...]
    language: str = "ru"


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    expected: str
    transcript: str
    repaired: str
    wer: float
    term_accuracy: float
    missing_terms: tuple[str, ...]
    latency_ms: float
    recovery_input: str = ""
    recovered_question: str = ""
    recovery_confidence: float = 0.0
    recovery_needs_manual_fix: bool = False


DEFAULT_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        "kubernetes_deployment_ingress",
        "Расскажи, что такое кубернетес, деплоймент, под и ингресс, и как они связаны в продакшене.",
        ("Kubernetes", "Deployment", "Pod", "Ingress"),
        "ru",
    ),
    EvalCase(
        "network_tcp_udp_tls",
        "Чем TCP отличается от UDP, и где в этой схеме используются TLS и mTLS?",
        ("TCP", "UDP", "TLS", "mTLS"),
        "ru",
    ),
    EvalCase(
        "linux_systemd_journalctl",
        "Как через систем си ти эл и журнал си ти эл понять, почему сервис в Linux не стартует?",
        ("systemctl", "journalctl", "Linux"),
        "ru",
    ),
    EvalCase(
        "ci_gitlab_jenkins",
        "Сравни GitLab CI и Jenkins Pipeline, когда что лучше использовать.",
        ("GitLab CI", "Jenkins Pipeline"),
        "ru",
    ),
    EvalCase(
        "observability_prometheus_grafana",
        "Как Прометей, Графана и алерт менеджер работают вместе для мониторинга и алертов?",
        ("Prometheus", "Grafana", "Alertmanager"),
        "ru",
    ),
    EvalCase(
        "broken_spoken_kubernetes",
        "Что такое губернии тёс, дипло и менты, поды и один грея с?",
        ("Kubernetes", "Deployment", "Pod", "Ingress"),
        "ru",
    ),
    EvalCase(
        "english_aws_autoscaling",
        "How do you implement auto scaling in AWS with ECS, CloudWatch alarms and target tracking?",
        ("AWS", "ECS", "CloudWatch", "auto scaling"),
        "en",
    ),
    EvalCase(
        "english_kubernetes_crashloop",
        "How do you troubleshoot a Kubernetes pod in CrashLoopBackOff using kubectl logs and describe?",
        ("Kubernetes", "Pod", "CrashLoopBackOff", "kubectl"),
        "en",
    ),
    EvalCase(
        "docker_compose_dockerfile",
        "Объясни, чем докерфайл отличается от докер компоуз, где контейнер, image, volume и network.",
        ("Dockerfile", "docker-compose", "container", "image", "volume", "network"),
        "ru",
    ),
    EvalCase(
        "terraform_state_plan_apply",
        "Что такое терраформ стейт, провайдер, ресурс, модуль, план и эплай?",
        ("Terraform", "state", "provider", "resource", "module", "plan", "apply"),
        "ru",
    ),
    EvalCase(
        "ansible_playbook_inventory",
        "Как ансибл использует инвентори, плейбук, role, task, handler и template?",
        ("Ansible", "inventory", "playbook", "role", "task", "handler", "template"),
        "ru",
    ),
    EvalCase(
        "gitlab_runner_artifact_cache",
        "Объясни гитлаб си ай, раннер, стейджи, джобы, артефакт, кэш, variables и environment.",
        ("GitLab CI", "runner", "stages", "jobs", "artifact", "cache", "variables", "environment"),
        "ru",
    ),
    EvalCase(
        "jenkins_declarative_pipeline",
        "Чем дженкинс declarative pipeline отличается от scripted pipeline, где агент, стейджи и steps?",
        ("Jenkins", "declarative pipeline", "scripted pipeline", "agent", "stages", "steps"),
        "ru",
    ),
    EvalCase(
        "prometheus_promql_slo",
        "Как прометей, пром кью эл, лейблы, рейт, алерт менеджер, SLI и SLO связаны с алертами?",
        ("Prometheus", "PromQL", "labels", "rate", "Alertmanager", "SLI", "SLO"),
        "ru",
    ),
    EvalCase(
        "networking_dns_tls_mtls",
        "Как DNS, TCP, UDP, HTTP, HTTPS, TLS, mTLS, NAT и load balancer связаны в сети?",
        ("DNS", "TCP", "UDP", "HTTP", "HTTPS", "TLS", "mTLS", "NAT", "load balancer"),
        "ru",
    ),
    EvalCase(
        "security_vault_rbac_sast",
        "Explain Vault, secrets, RBAC, least privilege, SAST, image scanning and TLS certificates.",
        ("Vault", "secrets", "RBAC", "least privilege", "SAST", "image scanning", "TLS", "certificates"),
        "en",
    ),
    EvalCase(
        "linux_network_cli",
        "Как через кёрл, диг, ss, tcpdump, journalctl и systemctl диагностировать Linux network issue?",
        ("curl", "dig", "ss", "tcpdump", "journalctl", "systemctl", "Linux", "network"),
        "ru",
    ),
    EvalCase(
        "english_sre_observability",
        "Explain SLI, SLO, error budget, Prometheus, Grafana and Alertmanager in an SRE interview.",
        ("SLI", "SLO", "error budget", "Prometheus", "Grafana", "Alertmanager", "SRE"),
        "en",
    ),
    EvalCase(
        "english_ci_cd_pipeline",
        "Compare GitLab CI, GitHub Actions and Jenkins Pipeline for CI CD automation.",
        ("GitLab CI", "GitHub Actions", "Jenkins Pipeline", "CI", "CD"),
        "en",
    ),
)


def normalize_for_score(text: str) -> list[str]:
    cleaned = clean_stt_output(text).lower()
    normalized = "".join(char if char.isalnum() or char in {"/", ".", "-"} else " " for char in cleaned)
    return [token for token in normalized.split() if token]


def word_error_rate(expected: str, actual: str) -> float:
    expected_tokens = normalize_for_score(expected)
    actual_tokens = normalize_for_score(actual)
    if not expected_tokens:
        return 0.0 if not actual_tokens else 1.0
    return _edit_distance(expected_tokens, actual_tokens) / len(expected_tokens)


def term_accuracy(expected_terms: tuple[str, ...], transcript: str) -> tuple[float, tuple[str, ...]]:
    if not expected_terms:
        return 1.0, ()
    normalized = " ".join(normalize_for_score(transcript))
    missing = tuple(term for term in expected_terms if " ".join(normalize_for_score(term)) not in normalized)
    return (len(expected_terms) - len(missing)) / len(expected_terms), missing


def write_fixtures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for case in DEFAULT_CASES:
            handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    print(f"wrote {manifest}")


def synthesize_windows_sapi(output_dir: Path, voice: str = "", english_voice: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_fixtures(output_dir)
    voice = voice or _select_sapi_voice("ru")
    english_voice = english_voice or _select_sapi_voice("en")
    print(f"using ru voice: {voice or 'default'}")
    print(f"using en voice: {english_voice or voice or 'default'}")
    for case in DEFAULT_CASES:
        wav_path = output_dir / f"{case.case_id}.wav"
        case_voice = english_voice if case.language == "en" and english_voice else voice
        _synthesize_one_sapi(case.text, wav_path, case_voice)
        print(f"wrote {wav_path}")


def load_cases(manifest: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            cases.append(
                EvalCase(
                    case_id=str(payload["case_id"]),
                    text=str(payload["text"]),
                    expected_terms=tuple(str(term) for term in payload.get("expected_terms", ())),
                    language=str(payload.get("language") or "ru"),
                )
            )
    return cases


def run_eval(
    audio_dir: Path,
    manifest: Path,
    report_path: Path,
    *,
    use_manifest_language: bool = True,
    with_recovery: bool = False,
) -> list[EvalResult]:
    cases = load_cases(manifest)
    model = _load_whisper_model()
    recovery = None
    if with_recovery:
        from app.question_recovery import QuestionRecovery

        recovery = QuestionRecovery(use_llm=False)
    results: list[EvalResult] = []
    for case in cases:
        wav_path = audio_dir / f"{case.case_id}.wav"
        if not wav_path.exists():
            print(f"skip {case.case_id}: missing {wav_path}")
            continue
        audio = load_wav_mono_float32(wav_path)
        started = time.perf_counter()
        transcript = transcribe_audio(model, audio, language=case.language if use_manifest_language else None)
        latency_ms = (time.perf_counter() - started) * 1000
        repaired = clean_stt_output(transcript)
        wer = word_error_rate(case.text, repaired)
        accuracy, missing = term_accuracy(case.expected_terms, repaired)
        recovered_question = ""
        recovery_confidence = 0.0
        recovery_needs_manual_fix = False
        recovery_input = ""
        if recovery is not None:
            recovery_input = condense_spoken_question(repaired) or repaired
            recovery_result = recovery.recover(recovery_input, [])
            recovered_question = recovery_result.recovered_question
            recovery_confidence = recovery_result.confidence
            recovery_needs_manual_fix = recovery_result.needs_manual_fix
        result = EvalResult(
            case_id=case.case_id,
            expected=case.text,
            transcript=transcript,
            repaired=repaired,
            wer=wer,
            term_accuracy=accuracy,
            missing_terms=missing,
            latency_ms=latency_ms,
            recovery_input=recovery_input,
            recovered_question=recovered_question,
            recovery_confidence=recovery_confidence,
            recovery_needs_manual_fix=recovery_needs_manual_fix,
        )
        results.append(result)
        recovery_text = ""
        if recovery is not None:
            recovery_status = "manual_fix" if recovery_needs_manual_fix else "ok"
            recovery_text = f" recovery={recovery_status}/{recovery_confidence:.2f}"
        print(f"{case.case_id}: terms={accuracy:.0%} wer={wer:.2f} latency={latency_ms:.0f}ms{recovery_text}")
    write_report(results, report_path)
    return results


def run_full_eval(
    output_dir: Path,
    report_path: Path,
    *,
    voice: str = "",
    english_voice: str = "",
    use_manifest_language: bool = True,
    min_average_term_accuracy: float = 0.90,
    min_case_term_accuracy: float = 0.75,
) -> bool:
    synthesize_windows_sapi(output_dir, voice=voice, english_voice=english_voice)
    results = run_eval(output_dir, output_dir / "manifest.jsonl", report_path, use_manifest_language=use_manifest_language)
    if not results:
        print("FAIL: no STT eval results")
        return False
    average_terms = sum(result.term_accuracy for result in results) / len(results)
    worst = min(results, key=lambda result: result.term_accuracy)
    ok = average_terms >= min_average_term_accuracy and worst.term_accuracy >= min_case_term_accuracy
    status = "PASS" if ok else "FAIL"
    print(
        f"{status}: average_terms={average_terms:.1%} worst_case={worst.case_id} "
        f"worst_terms={worst.term_accuracy:.1%}"
    )
    return ok


def run_training_eval(
    output_dir: Path,
    report_path: Path,
    training_report_path: Path,
    *,
    voice: str = "",
    english_voice: str = "",
    use_manifest_language: bool = True,
    min_average_term_accuracy: float = 0.90,
    min_case_term_accuracy: float = 0.75,
) -> bool:
    synthesize_windows_sapi(output_dir, voice=voice, english_voice=english_voice)
    results = run_eval(
        output_dir,
        output_dir / "manifest.jsonl",
        report_path,
        use_manifest_language=use_manifest_language,
        with_recovery=True,
    )
    write_training_report(
        results,
        training_report_path,
        min_average_term_accuracy=min_average_term_accuracy,
        min_case_term_accuracy=min_case_term_accuracy,
    )
    if not results:
        print("FAIL: no STT training results")
        return False
    average_terms = sum(result.term_accuracy for result in results) / len(results)
    worst = min(results, key=lambda result: result.term_accuracy)
    recovery_failures = sum(1 for result in results if result.recovery_needs_manual_fix)
    ok = (
        average_terms >= min_average_term_accuracy
        and worst.term_accuracy >= min_case_term_accuracy
        and recovery_failures == 0
    )
    status = "PASS" if ok else "FAIL"
    print(
        f"{status}: average_terms={average_terms:.1%} worst_case={worst.case_id} "
        f"worst_terms={worst.term_accuracy:.1%} recovery_failures={recovery_failures}"
    )
    return ok


def transcribe_audio(model: Any, audio: np.ndarray, language: str | None = None) -> str:
    segments, _info = model.transcribe(
        audio,
        language=whisper_language(STT_SETTINGS, requested_language=language),
        task="transcribe",
        beam_size=STT_SETTINGS.beam_size,
        best_of=STT_SETTINGS.best_of,
        condition_on_previous_text=False,
        initial_prompt=WHISPER_TECHNICAL_PROMPT,
        vad_filter=STT_SETTINGS.vad_filter,
        vad_parameters=whisper_vad_parameters(STT_SETTINGS) if STT_SETTINGS.vad_filter else None,
        no_speech_threshold=STT_SETTINGS.no_speech_threshold,
        log_prob_threshold=STT_SETTINGS.log_prob_threshold,
        compression_ratio_threshold=STT_SETTINGS.compression_ratio_threshold,
        repetition_penalty=STT_SETTINGS.repetition_penalty,
        no_repeat_ngram_size=STT_SETTINGS.no_repeat_ngram_size,
        hotwords=STT_SETTINGS.hotwords,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def load_wav_mono_float32(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    if sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return _resample(audio.astype(np.float32), sample_rate, SAMPLE_RATE)


def write_report(results: list[EvalResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    average_wer = sum(result.wer for result in results) / len(results) if results else 0.0
    average_terms = sum(result.term_accuracy for result in results) / len(results) if results else 0.0
    has_recovery = any(result.recovered_question for result in results)
    recovery_failures = sum(1 for result in results if result.recovery_needs_manual_fix) if has_recovery else 0
    lines = [
        "# STT Eval Report",
        "",
        f"- cases: {len(results)}",
        f"- average term accuracy: {average_terms:.1%}",
        f"- average WER-like score: {average_wer:.2f}",
    ]
    if has_recovery:
        lines.append(f"- local recovery manual-fix cases: {recovery_failures}")
    lines.append("")
    for result in results:
        lines.extend(
            [
                f"## {result.case_id}",
                "",
                f"- term accuracy: {result.term_accuracy:.1%}",
                f"- WER-like score: {result.wer:.2f}",
                f"- latency: {result.latency_ms:.0f} ms",
                f"- missing terms: {', '.join(result.missing_terms) or '-'}",
                "",
                f"Expected: {result.expected}",
                "",
                f"Transcript: {result.transcript}",
                "",
                f"Repaired: {result.repaired}",
                "",
            ]
        )
        if result.recovered_question:
            lines.extend(
                [
                    f"Recovery input: {result.recovery_input or '-'}",
                    "",
                    f"Recovered question: {result.recovered_question}",
                    "",
                    f"Recovery confidence: {result.recovery_confidence:.2f}",
                    "",
                    f"Recovery needs manual fix: {result.recovery_needs_manual_fix}",
                    "",
                ]
            )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report_path}")


def write_training_report(
    results: list[EvalResult],
    report_path: Path,
    *,
    min_average_term_accuracy: float,
    min_case_term_accuracy: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    average_terms = sum(result.term_accuracy for result in results) / len(results) if results else 0.0
    weak = [
        result
        for result in results
        if result.term_accuracy < min_case_term_accuracy or result.recovery_needs_manual_fix
    ]
    missing_counts: dict[str, int] = {}
    for result in results:
        for term in result.missing_terms:
            missing_counts[term] = missing_counts.get(term, 0) + 1

    lines = [
        "# STT Training Report",
        "",
        "This is a local regression loop, not model fine-tuning.",
        "",
        f"- cases: {len(results)}",
        f"- required average term accuracy: {min_average_term_accuracy:.1%}",
        f"- actual average term accuracy: {average_terms:.1%}",
        f"- required per-case term accuracy: {min_case_term_accuracy:.1%}",
        f"- weak cases: {len(weak)}",
        "",
        "## Missing Terms",
        "",
    ]
    if missing_counts:
        for term, count in sorted(missing_counts.items(), key=lambda item: (-item[1], item[0].lower())):
            lines.append(f"- {term}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Weak Cases", ""])
    if not weak:
        lines.append("- none")
    for result in weak:
        lines.extend(
            [
                f"### {result.case_id}",
                "",
                f"- term accuracy: {result.term_accuracy:.1%}",
                f"- missing terms: {', '.join(result.missing_terms) or '-'}",
                f"- recovery confidence: {result.recovery_confidence:.2f}",
                f"- recovery needs manual fix: {result.recovery_needs_manual_fix}",
                "",
                f"Transcript: {result.transcript}",
                "",
                f"Repaired: {result.repaired}",
                "",
                f"Recovery input: {result.recovery_input or '-'}",
                "",
                f"Recovered question: {result.recovered_question or '-'}",
                "",
                "Suggested action: add or adjust a phonetic normalization in app/tech_terms.py only if the term is audible in Transcript/Repaired.",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report_path}")


def list_sapi_voices() -> list[dict[str, str]]:
    script = """
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.GetInstalledVoices() | ForEach-Object {
  $v = $_.VoiceInfo
  [PSCustomObject]@{ Name = $v.Name; Culture = $v.Culture.Name; Gender = $v.Gender.ToString() }
} | ConvertTo-Json -Compress
$synth.Dispose()
""".strip()
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8-sig",
        )
    except Exception:
        return []
    output = completed.stdout.strip()
    if not output:
        return []
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []
    voices: list[dict[str, str]] = []
    for item in payload:
        if isinstance(item, dict):
            voices.append({key: str(item.get(key, "")) for key in ("Name", "Culture", "Gender")})
    return voices


def _select_sapi_voice(language_prefix: str) -> str:
    for voice in list_sapi_voices():
        if voice.get("Culture", "").lower().startswith(language_prefix.lower()):
            return voice.get("Name", "")
    return ""


def _load_whisper_model() -> Any:
    from faster_whisper import WhisperModel

    last_exc: Exception | None = None
    attempts = whisper_model_attempts(STT_SETTINGS)
    for attempt_index, (device, compute_type) in enumerate(attempts):
        try:
            model = WhisperModel(STT_SETTINGS.model, device=device, compute_type=compute_type)
            _warmup_whisper_model(model, device)
            return model
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            has_retry = attempt_index + 1 < len(attempts)
            if has_retry and device == "cuda" and is_cuda_whisper_error(exc):
                print(f"CUDA Whisper unavailable, falling back to CPU/int8: {exc}")
                continue
            raise
    raise RuntimeError(f"Whisper model failed to load: {last_exc}")


def _warmup_whisper_model(model: Any, device: str) -> None:
    if device != "cuda":
        return
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    segments, _info = model.transcribe(
        audio,
        language="en",
        task="transcribe",
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
        vad_filter=False,
        no_speech_threshold=0.95,
    )
    list(segments)


def _synthesize_one_sapi(text: str, wav_path: Path, voice: str = "") -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        text_path = Path(temp_dir) / "text.txt"
        script_path = Path(temp_dir) / "synth.ps1"
        text_path.write_text(text, encoding="utf-8-sig")
        safe_voice = voice.replace("'", "''")
        voice_line = f"$synth.SelectVoice('{safe_voice}')" if voice else ""
        script_path.write_text(
            "\n".join(
                [
                    "Add-Type -AssemblyName System.Speech",
                    "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer",
                    "$text = Get-Content -Raw -Encoding UTF8 $args[0]",
                    voice_line.rstrip(),
                    "$synth.SetOutputToWaveFile($args[1])",
                    "$synth.Speak($text)",
                    "$synth.Dispose()",
                ]
            ),
            encoding="utf-8-sig",
        )
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
                str(text_path),
                str(wav_path),
            ],
            check=True,
        )


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    target_len = max(1, int(round(audio.size * target_rate / source_rate)))
    source_positions = np.arange(audio.size, dtype=np.float32)
    target_positions = np.linspace(0, audio.size - 1, target_len, dtype=np.float32)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="StackWire STT evaluation harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_parser = subparsers.add_parser("write-fixtures")
    write_parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)

    subparsers.add_parser("list-voices")

    synth_parser = subparsers.add_parser("synth")
    synth_parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    synth_parser.add_argument("--voice", default="")
    synth_parser.add_argument("--english-voice", default="")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--audio-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "manifest.jsonl")
    run_parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    run_parser.add_argument("--auto-language", action="store_true")

    full_parser = subparsers.add_parser("full")
    full_parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    full_parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    full_parser.add_argument("--voice", default="")
    full_parser.add_argument("--english-voice", default="")
    full_parser.add_argument("--auto-language", action="store_true")
    full_parser.add_argument("--min-average-term-accuracy", type=float, default=0.90)
    full_parser.add_argument("--min-case-term-accuracy", type=float, default=0.75)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    train_parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    train_parser.add_argument("--training-report", type=Path, default=DEFAULT_TRAINING_REPORT)
    train_parser.add_argument("--voice", default="")
    train_parser.add_argument("--english-voice", default="")
    train_parser.add_argument("--auto-language", action="store_true")
    train_parser.add_argument("--min-average-term-accuracy", type=float, default=0.90)
    train_parser.add_argument("--min-case-term-accuracy", type=float, default=0.75)

    args = parser.parse_args()
    if args.command == "write-fixtures":
        write_fixtures(args.out)
    elif args.command == "list-voices":
        for voice in list_sapi_voices():
            print(f"{voice.get('Name')} | {voice.get('Culture')} | {voice.get('Gender')}")
    elif args.command == "synth":
        synthesize_windows_sapi(args.out, args.voice, args.english_voice)
    elif args.command == "run":
        run_eval(args.audio_dir, args.manifest, args.report, use_manifest_language=not args.auto_language)
    elif args.command == "full":
        ok = run_full_eval(
            args.out,
            args.report,
            voice=args.voice,
            english_voice=args.english_voice,
            use_manifest_language=not args.auto_language,
            min_average_term_accuracy=args.min_average_term_accuracy,
            min_case_term_accuracy=args.min_case_term_accuracy,
        )
        return 0 if ok else 1
    elif args.command == "train":
        ok = run_training_eval(
            args.out,
            args.report,
            args.training_report,
            voice=args.voice,
            english_voice=args.english_voice,
            use_manifest_language=not args.auto_language,
            min_average_term_accuracy=args.min_average_term_accuracy,
            min_case_term_accuracy=args.min_case_term_accuracy,
        )
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
