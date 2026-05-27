import argparse
import json
import os
import subprocess
import tempfile
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.config import load_local_env
from app.tech_terms import WHISPER_TECHNICAL_PROMPT
from app.transcript_repair import clean_stt_output

load_local_env()

DEFAULT_OUTPUT_DIR = Path("data/stt_eval")
DEFAULT_REPORT = Path("logs/stt_eval_report.md")
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    text: str
    expected_terms: tuple[str, ...]


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


DEFAULT_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        "kubernetes_deployment_ingress",
        "Расскажи, что такое Kubernetes, Deployment, Pod и Ingress, и как они связаны в продакшене.",
        ("Kubernetes", "Deployment", "Pod", "Ingress"),
    ),
    EvalCase(
        "network_tcp_udp_tls",
        "Чем TCP отличается от UDP, и где в этой схеме используются TLS и mTLS?",
        ("TCP", "UDP", "TLS", "mTLS"),
    ),
    EvalCase(
        "linux_systemd_journalctl",
        "Как через systemctl и journalctl понять, почему сервис в Linux не стартует?",
        ("systemctl", "journalctl", "Linux"),
    ),
    EvalCase(
        "ci_gitlab_jenkins",
        "Сравни GitLab CI и Jenkins Pipeline, когда что лучше использовать.",
        ("GitLab CI", "Jenkins Pipeline"),
    ),
    EvalCase(
        "observability_prometheus_grafana",
        "Как Prometheus, Grafana и Alertmanager работают вместе для мониторинга и алертов?",
        ("Prometheus", "Grafana", "Alertmanager"),
    ),
    EvalCase(
        "broken_spoken_kubernetes",
        "Что такое губернии тёс, дипло и менты, поды и один грея с?",
        ("Kubernetes", "Deployment", "Pod", "Ingress"),
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


def synthesize_windows_sapi(output_dir: Path, voice: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_fixtures(output_dir)
    for case in DEFAULT_CASES:
        wav_path = output_dir / f"{case.case_id}.wav"
        _synthesize_one_sapi(case.text, wav_path, voice)
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
                )
            )
    return cases


def run_eval(audio_dir: Path, manifest: Path, report_path: Path) -> list[EvalResult]:
    cases = load_cases(manifest)
    model = _load_whisper_model()
    results: list[EvalResult] = []
    for case in cases:
        wav_path = audio_dir / f"{case.case_id}.wav"
        if not wav_path.exists():
            print(f"skip {case.case_id}: missing {wav_path}")
            continue
        audio = load_wav_mono_float32(wav_path)
        started = time.perf_counter()
        transcript = transcribe_audio(model, audio)
        latency_ms = (time.perf_counter() - started) * 1000
        repaired = clean_stt_output(transcript)
        wer = word_error_rate(case.text, repaired)
        accuracy, missing = term_accuracy(case.expected_terms, repaired)
        result = EvalResult(
            case_id=case.case_id,
            expected=case.text,
            transcript=transcript,
            repaired=repaired,
            wer=wer,
            term_accuracy=accuracy,
            missing_terms=missing,
            latency_ms=latency_ms,
        )
        results.append(result)
        print(f"{case.case_id}: terms={accuracy:.0%} wer={wer:.2f} latency={latency_ms:.0f}ms")
    write_report(results, report_path)
    return results


def transcribe_audio(model: Any, audio: np.ndarray) -> str:
    segments, _info = model.transcribe(
        audio,
        language=os.getenv("WHISPER_LANGUAGE", "ru").strip() or "ru",
        task="transcribe",
        beam_size=int(os.getenv("WHISPER_BEAM_SIZE", "5")),
        best_of=int(os.getenv("WHISPER_BEST_OF", "5")),
        condition_on_previous_text=False,
        initial_prompt=WHISPER_TECHNICAL_PROMPT,
        vad_filter=os.getenv("WHISPER_VAD_FILTER", "1").strip().lower() not in {"0", "false", "no", "off"},
        vad_parameters={
            "threshold": float(os.getenv("WHISPER_VAD_THRESHOLD", "0.20")),
            "min_speech_duration_ms": int(os.getenv("WHISPER_VAD_MIN_SPEECH_MS", "100")),
            "min_silence_duration_ms": int(os.getenv("WHISPER_VAD_MIN_SILENCE_MS", "650")),
            "speech_pad_ms": int(os.getenv("WHISPER_VAD_SPEECH_PAD_MS", "450")),
        },
        no_speech_threshold=float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.75")),
        log_prob_threshold=float(os.getenv("WHISPER_LOG_PROB_THRESHOLD", "-2.0")),
        compression_ratio_threshold=float(os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "3.0")),
        repetition_penalty=float(os.getenv("WHISPER_REPETITION_PENALTY", "1.08")),
        no_repeat_ngram_size=int(os.getenv("WHISPER_NO_REPEAT_NGRAM_SIZE", "3")),
        hotwords=os.getenv("WHISPER_HOTWORDS", "").strip() or None,
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
    lines = [
        "# STT Eval Report",
        "",
        f"- cases: {len(results)}",
        f"- average term accuracy: {average_terms:.1%}",
        f"- average WER-like score: {average_wer:.2f}",
        "",
    ]
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
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report_path}")


def _load_whisper_model() -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(
        os.getenv("WHISPER_MODEL", "large-v3-turbo"),
        device=os.getenv("WHISPER_DEVICE", "cpu"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
    )


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

    synth_parser = subparsers.add_parser("synth")
    synth_parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    synth_parser.add_argument("--voice", default="")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--audio-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "manifest.jsonl")
    run_parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)

    args = parser.parse_args()
    if args.command == "write-fixtures":
        write_fixtures(args.out)
    elif args.command == "synth":
        synthesize_windows_sapi(args.out, args.voice)
    elif args.command == "run":
        run_eval(args.audio_dir, args.manifest, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
