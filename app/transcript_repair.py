import re

from app.tech_terms import normalize_spoken_technical_terms


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


def _fix_spacing(text: str) -> str:
    text = re.sub(r"\s+([?!,.:;])", r"\1", text)
    text = re.sub(r"([?!]){2,}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()
