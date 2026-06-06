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


def condense_spoken_question(text: str, *, max_words: int = 120) -> str:
    cleaned = clean_stt_output(text)
    if not cleaned:
        return ""

    chunks = _split_spoken_chunks(cleaned)
    if not chunks:
        return _limit_words(cleaned, max_words)

    scored = [(index, _spoken_chunk_score(chunk), chunk) for index, chunk in enumerate(chunks)]
    question_chunks = [(index, score, chunk) for index, score, chunk in scored if _looks_like_question_chunk(chunk)]
    if question_chunks:
        anchor_index, _score, anchor = question_chunks[-1]
        selected = [anchor]
        if anchor_index > 0:
            previous = chunks[anchor_index - 1]
            previous_has_terms = _latin_token_count(previous) > 0
            anchor_has_terms = _latin_token_count(anchor) > 0
            if _spoken_chunk_score(previous) >= 1.2 and (
                len(previous.split()) <= 12
                or (previous_has_terms and not anchor_has_terms and len(previous.split()) <= 35)
            ):
                selected.insert(0, previous)
        condensed = " ".join(selected)
    else:
        scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
        condensed = scored[0][2] if scored else cleaned

    condensed = _trim_to_question_window(condensed)
    condensed = collapse_repeated_phrases(condensed)
    condensed = _limit_words(_fix_spacing(condensed).strip(" ,.;"), max_words)
    return _capitalize_cyrillic_start(condensed)


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


def _split_spoken_chunks(text: str) -> list[str]:
    normalized = re.sub(r"[\r\n]+", ". ", text)
    marker_pattern = (
        r"\b(?:что|чем|как|почему|зачем|когда|где|какой|какая|какие|"
        r"объясни|расскажи|сравни|покажи|what|how|why|when|where|compare|explain|troubleshoot)\b"
    )
    normalized = re.sub(
        rf"(.{{25,}}?)\s+({marker_pattern})",
        lambda match: f"{match.group(1).strip()}. {match.group(2)}",
        normalized,
        flags=re.IGNORECASE,
    )
    chunks = [chunk.strip(" ,.;") for chunk in re.split(r"[.?!]+", normalized) if chunk.strip(" ,.;")]
    return chunks


def _spoken_chunk_score(chunk: str) -> float:
    lowered = chunk.casefold()
    score = 0.0
    if _looks_like_question_chunk(chunk):
        score += 3.0
    score += min(4.0, sum(1 for marker in QUESTION_OR_TECH_MARKERS if marker in lowered) * 0.8)
    score += min(3.0, len(re.findall(r"\b[A-Za-z][A-Za-z0-9./+-]{1,}\b", chunk)) * 0.35)
    if any(pattern in lowered for pattern in COMMON_WHISPER_HALLUCINATION_PATTERNS):
        score -= 4.0
    if len(chunk.split()) <= 3:
        score -= 1.0
    return score


def _looks_like_question_chunk(chunk: str) -> bool:
    lowered = chunk.casefold()
    return bool(
        re.search(
            r"\b(что|чем|как|почему|зачем|когда|где|какой|какая|какие|"
            r"объясни|расскажи|сравни|покажи|what|how|why|when|where|compare|explain|troubleshoot)\b",
            lowered,
            flags=re.IGNORECASE,
        )
    )


def _trim_to_question_window(text: str) -> str:
    matches = list(
        re.finditer(
            r"\b(что|чем|как|почему|зачем|когда|где|какой|какая|какие|"
            r"объясни|расскажи|сравни|покажи|what|how|why|when|where|compare|explain|troubleshoot)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    if not matches:
        return text
    last = matches[-1]
    prefix = text[: last.start()].strip(" ,.;")
    suffix = text[last.start() :].strip(" ,.;")
    if prefix and _latin_token_count(prefix) > 0 and _question_depends_on_previous_subject(suffix):
        return text
    if prefix and _latin_token_count(prefix) > 0 and _latin_token_count(suffix) == 0 and len(prefix.split()) <= 35:
        return text
    if prefix and _spoken_chunk_score(prefix) >= 1.2 and len(prefix.split()) <= 8:
        return text
    return text[last.start() :].strip(" ,.;")


def _question_depends_on_previous_subject(text: str) -> bool:
    return bool(
        re.match(
            r"^(?:как|чем|почему|зачем|когда|где|how|why|when|where)\s+"
            r"(?:он|она|оно|они|это|эти|их|его|ее|её|it|they|these|those|that)\b",
            text.strip(),
            flags=re.IGNORECASE,
        )
    )


def _latin_token_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z][A-Za-z0-9./+-]*\b", text))


def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if max_words <= 0 or len(words) <= max_words:
        return text.strip()
    return " ".join(words[-max_words:]).strip(" ,.;")


def _capitalize_cyrillic_start(text: str) -> str:
    if text and re.match(r"[а-яё]", text[0]):
        return text[0].upper() + text[1:]
    return text


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
