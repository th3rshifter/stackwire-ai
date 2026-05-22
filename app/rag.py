import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.answer_planner import AnswerPlan, build_answer_plan
from app.config import ROOT_DIR


LOGGER = logging.getLogger(__name__)
DEFAULT_KNOWLEDGE_DIR = ROOT_DIR / "docs" / "knowledge"
MAX_CHUNK_CHARS = int(os.getenv("STACKWIRE_RAG_CHUNK_CHARS", "1100"))


@dataclass(frozen=True)
class KnowledgeChunk:
    source_file: str
    heading: str
    text: str
    score: float


def knowledge_dir() -> Path:
    configured = os.getenv("STACKWIRE_KNOWLEDGE_DIR", "").strip()
    return Path(configured) if configured else DEFAULT_KNOWLEDGE_DIR


def load_chunks(directory: Path | None = None) -> list[KnowledgeChunk]:
    base = directory or knowledge_dir()
    if not base.exists():
        return []

    chunks: list[KnowledgeChunk] = []
    for path in sorted(base.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        chunks.extend(_chunk_markdown(path, text))
    return chunks


def retrieve_knowledge(question: str, plan: AnswerPlan, limit: int = 3) -> list[KnowledgeChunk]:
    chunks = load_chunks()
    if not chunks:
        LOGGER.info("rag retrieve skipped: no knowledge chunks")
        return []

    query_tokens = _tokens(question)
    if not query_tokens:
        return []

    domain_hints = _domain_hints(plan.domain)
    scored: list[KnowledgeChunk] = []
    for chunk in chunks:
        source_lower = chunk.source_file.casefold()
        heading_lower = chunk.heading.casefold()
        chunk_tokens = _tokens(f"{chunk.heading} {chunk.text}")
        source_tokens = _tokens(f"{chunk.source_file} {chunk.heading}")
        overlap = len(query_tokens & chunk_tokens)
        source_overlap = len(query_tokens & source_tokens)
        if overlap == 0 and source_overlap == 0:
            continue
        score = overlap / max(1, len(query_tokens))
        if any(hint in source_lower for hint in domain_hints):
            score += 0.25
        if any(hint in heading_lower for hint in domain_hints):
            score += 0.12
        if score <= 0:
            continue
        scored.append(
            KnowledgeChunk(
                source_file=chunk.source_file,
                heading=chunk.heading,
                text=chunk.text,
                score=score,
            )
        )

    scored.sort(key=lambda item: item.score, reverse=True)
    result = _limit_total_chars(scored[: max(limit * 2, limit)], limit=limit)
    LOGGER.info("rag retrieve question=%r domain=%s chunks=%s", question[:80], plan.domain, len(result))
    return result


def format_knowledge_chunks(chunks: Iterable[KnowledgeChunk], max_chars: int = 3200) -> str:
    parts: list[str] = []
    used = 0
    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue
        block = f"[{chunk.source_file} :: {chunk.heading}]\n{text}"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining < 240:
                break
            block = block[:remaining].rstrip()
        parts.append(block)
        used += len(block)
        if used >= max_chars:
            break
    return "\n\n".join(parts)


def _chunk_markdown(path: Path, text: str) -> list[KnowledgeChunk]:
    sections: list[tuple[str, str]] = []
    current_heading = path.stem
    current_lines: list[str] = []

    for line in text.splitlines():
        heading_match = re.match(r"^(#{1,4})\s+(.+?)\s*$", line)
        if heading_match:
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = heading_match.group(2).strip()
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    chunks: list[KnowledgeChunk] = []
    for heading, section_text in sections:
        for part in _split_text(section_text):
            if part.strip():
                chunks.append(
                    KnowledgeChunk(
                        source_file=path.name,
                        heading=heading,
                        text=part.strip(),
                        score=0.0,
                    )
                )
    return chunks


def _split_text(text: str) -> list[str]:
    cleaned = text.strip()
    if len(cleaned) <= MAX_CHUNK_CHARS:
        return [cleaned]

    paragraphs = re.split(r"\n\s*\n", cleaned)
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) + 2 > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = paragraph
        elif current:
            current += "\n\n" + paragraph
        else:
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def _limit_total_chars(chunks: list[KnowledgeChunk], *, limit: int) -> list[KnowledgeChunk]:
    result: list[KnowledgeChunk] = []
    used = 0
    max_total = int(os.getenv("STACKWIRE_RAG_MAX_CHARS", "3200"))
    for chunk in chunks:
        if len(result) >= limit:
            break
        if used + len(chunk.text) > max_total and result:
            break
        result.append(chunk)
        used += len(chunk.text)
    return result


def _domain_hints(domain: str) -> tuple[str, ...]:
    mapping = {
        "kubernetes": ("kubernetes", "k8s"),
        "docker": ("docker",),
        "git": ("git",),
        "linux_fs": ("linux",),
        "linux_process": ("linux",),
        "linux_network": ("linux", "networking"),
        "service_mesh": ("kubernetes", "networking", "security"),
        "ci_cd": ("gitlab-ci", "jenkins", "ci"),
        "iac": ("terraform", "ansible"),
        "observability": ("prometheus", "observability"),
        "web_proxy": ("networking",),
        "security": ("security",),
    }
    return mapping.get(domain, (domain.replace("_", "-"),))


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-zА-Яа-я0-9_./+-]{2,}", text.casefold())
        if token not in {"the", "and", "for", "что", "как", "это", "или", "при", "для", "with"}
    }


def _index_command() -> int:
    chunks = load_chunks()
    sources = sorted({chunk.source_file for chunk in chunks})
    print(f"knowledge_dir={knowledge_dir()}")
    print(f"files={len(sources)}")
    print(f"chunks={len(chunks)}")
    for source in sources:
        count = sum(1 for chunk in chunks if chunk.source_file == source)
        print(f"- {source}: {count}")
    return 0


def _search_command(query: str) -> int:
    plan = build_answer_plan(query)
    chunks = retrieve_knowledge(query, plan, limit=5)
    print(f"domain={plan.domain} intent={plan.intent}")
    for chunk in chunks:
        print(f"\n## {chunk.source_file} :: {chunk.heading} ({chunk.score:.2f})")
        print(chunk.text)
    return 0


def validate_knowledge(directory: Path | None = None) -> list[str]:
    base = directory or knowledge_dir()
    issues: list[str] = []
    if not base.exists():
        return issues

    markdown_files = sorted(base.glob("*.md"))
    chunks = load_chunks(base)
    chunks_by_file: dict[str, list[KnowledgeChunk]] = {}
    for chunk in chunks:
        chunks_by_file.setdefault(chunk.source_file, []).append(chunk)

    for path in markdown_files:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            issues.append(f"{path.name}: empty file")
            continue
        file_chunks = chunks_by_file.get(path.name, [])
        if not file_chunks:
            issues.append(f"{path.name}: no chunks")
            continue
        headings = [chunk.heading for chunk in file_chunks[:3]]
        query = " ".join([path.stem.replace("-", " "), *headings])
        plan = build_answer_plan(query)
        matches = retrieve_knowledge(query, plan, limit=3)
        if not any(match.source_file == path.name for match in matches):
            issues.append(f"{path.name}: not retrievable by topic query")
    return issues


def _test_command() -> int:
    issues = validate_knowledge()
    if not issues:
        print("knowledge self-test ok")
        return 0
    print("knowledge self-test failed", file=sys.stderr)
    for issue in issues:
        print(f"- {issue}", file=sys.stderr)
    return 1


def _main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "index":
        return _index_command()
    if len(argv) >= 3 and argv[1] == "search":
        return _search_command(" ".join(argv[2:]))
    if len(argv) >= 2 and argv[1] == "test":
        return _test_command()
    print('Usage: python -m app.rag index | python -m app.rag search "ingress gateway" | python -m app.rag test', file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
