import re

from app.tech_terms import normalize_spoken_technical_terms


QUESTION_OR_TECH_MARKERS: tuple[str, ...] = (
    "что",
    "чем",
    "как",
    "почему",
    "зачем",
    "когда",
    "где",
    "kubernetes",
    "kubectl",
    "deployment",
    "pod",
    "ingress",
    "docker",
    "linux",
    "tcp",
    "udp",
    "http",
    "https",
    "tls",
    "dns",
    "web server",
    "nginx",
    "haproxy",
    "apache",
    "load balancer",
    "reverse proxy",
    "proxy",
    "gitlab",
    "jenkins",
    "terraform",
    "ansible",
    "prometheus",
    "grafana",
)

COMMON_WHISPER_HALLUCINATION_PATTERNS: tuple[str, ...] = (
    "субтитры сделал",
    "субтитры создавал",
    "субтитры добавил",
    "продолжение следует",
    "спасибо за просмотр",
    "thanks for watching",
    "thank you for watching",
    "amara.org",
    "подписывайтесь",
)

LEADING_NOISE_PATTERNS: tuple[str, ...] = (
    r"\bу\s+меня\s+вопрос(?:\s+такой)?\b",
    r"\bвопрос\s+такой\b",
    r"\bсобственно\b",
    r"\bслушай\b",
    r"\bсмотри\b",
    r"\bможешь\s+(?:мне\s+)?(?:рассказать|объяснить|показать)\b",
    r"\bрасскажи\s+пожалуйста\b",
    r"\bобъясни\s+пожалуйста\b",
)

FILLER_WORDS: frozenset[str] = frozenset(
    {
        "ну",
        "вот",
        "типа",
        "короче",
        "значит",
        "окей",
        "ладно",
        "вообще",
        "какбы",
        "как-бы",
        "э",
        "ээ",
        "эээ",
        "м",
        "мм",
        "ммм",
        "а",
    }
)

FILLER_PHRASES: tuple[str, ...] = (
    "в общем",
    "на самом деле",
    "как бы",
    "то есть",
    "так сказать",
)

TRAILING_NOISE_PATTERNS: tuple[str, ...] = (
    r"\bда\b",
    r"\bнет\b",
    r"\bзнаешь\b",
    r"\bпонимаешь\b",
    r"\bвот так\b",
)


def repair_live_transcript(text: str) -> str:
    repaired = normalize_spoken_technical_terms(text.strip())
    if not repaired:
        return ""

    repaired = _remove_noise_phrases(repaired)
    repaired = _remove_filler_words(repaired)
    repaired = collapse_repeated_phrases(repaired)
    repaired = _fix_spacing(repaired)
    return repaired.strip(" ,.;")


def clean_stt_output(text: str) -> str:
    repaired = repair_live_transcript(text)
    repaired = _drop_consecutive_duplicate_words(repaired)
    repaired = collapse_repeated_phrases(repaired)
    return _fix_spacing(repaired).strip(" ,.;")


def is_probable_stt_hallucination(
    text: str,
    *,
    avg_logprob: float | None = None,
    no_speech_prob: float | None = None,
    compression_ratio: float | None = None,
    rms: float | None = None,
) -> bool:
    cleaned = _fix_spacing(text).strip(" ,.;").lower()
    if not cleaned:
        return False

    if any(pattern in cleaned for pattern in COMMON_WHISPER_HALLUCINATION_PATTERNS):
        return True

    words = [_word_key(word) for word in cleaned.split()]
    words = [word for word in words if word]
    if not words:
        return True

    has_question_or_tech = "?" in cleaned or any(marker in cleaned for marker in QUESTION_OR_TECH_MARKERS)
    unique_ratio = len(set(words)) / len(words)
    duplicate_ratio = _consecutive_duplicate_ratio(words)

    if len(words) >= 4 and duplicate_ratio >= 0.35:
        return True
    if len(words) >= 6 and unique_ratio <= 0.32 and not has_question_or_tech:
        return True
    if _max_repeated_ngram_count(words) >= 3:
        return True

    low_signal = rms is not None and rms < 0.0015
    high_no_speech = no_speech_prob is not None and no_speech_prob >= 0.72

    if no_speech_prob is not None and no_speech_prob >= 0.88 and not has_question_or_tech:
        return True
    if avg_logprob is not None and avg_logprob <= -1.25 and not has_question_or_tech and (high_no_speech or low_signal):
        return True
    if compression_ratio is not None and compression_ratio >= 2.6 and not has_question_or_tech:
        return True
    if rms is not None and rms < 0.0007 and len(words) <= 5 and not has_question_or_tech:
        return True

    repeated_drupal = sum(1 for word in words if word == "drupal")
    if repeated_drupal >= 2 and len(set(words)) <= 3:
        return True

    return False


def collapse_repeated_phrases(text: str, max_ngram: int = 8) -> str:
    words = text.split()
    if len(words) < 2:
        return text.strip()

    changed = True
    while changed:
        changed = False
        max_size = min(max_ngram, len(words) // 2)
        for size in range(max_size, 0, -1):
            index = 0
            while index + (size * 2) <= len(words):
                left = [_word_key(word) for word in words[index : index + size]]
                right = [_word_key(word) for word in words[index + size : index + (size * 2)]]
                if left == right:
                    del words[index + size : index + (size * 2)]
                    changed = True
                    continue
                index += 1
            if changed:
                break

    return " ".join(words).strip()


def _drop_consecutive_duplicate_words(text: str) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    kept: list[str] = []
    for word in words:
        if kept and _word_key(kept[-1]) == _word_key(word):
            continue
        kept.append(word)
    return " ".join(kept)


def _remove_noise_phrases(text: str) -> str:
    cleaned = text
    for phrase in FILLER_PHRASES:
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b[,\s]*", " ", cleaned, flags=re.IGNORECASE)
    for pattern in LEADING_NOISE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    for pattern in TRAILING_NOISE_PATTERNS:
        cleaned = re.sub(rf"(?:,?\s*{pattern}\??)+$", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _remove_filler_words(text: str) -> str:
    tokens = text.split()
    kept: list[str] = []
    for token in tokens:
        key = _word_key(token)
        if key in FILLER_WORDS:
            continue
        kept.append(token)
    return " ".join(kept)


def _word_key(word: str) -> str:
    return re.sub(r"^[^\w/.-]+|[^\w/.-]+$", "", word.lower(), flags=re.UNICODE)


def _consecutive_duplicate_ratio(words: list[str]) -> float:
    if len(words) < 2:
        return 0.0
    duplicates = sum(1 for index in range(1, len(words)) if words[index] == words[index - 1])
    return duplicates / max(len(words) - 1, 1)


def _max_repeated_ngram_count(words: list[str], max_ngram: int = 5) -> int:
    best = 1
    for size in range(1, min(max_ngram, len(words)) + 1):
        counts: dict[tuple[str, ...], int] = {}
        for index in range(0, len(words) - size + 1):
            ngram = tuple(words[index : index + size])
            counts[ngram] = counts.get(ngram, 0) + 1
        if counts:
            best = max(best, max(counts.values()))
    return best


def _fix_spacing(text: str) -> str:
    text = re.sub(r"\s+([?!,.:;])", r"\1", text)
    text = re.sub(r"([?!]){2,}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()
