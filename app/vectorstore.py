"""Unified local vector store (one RAG over knowledge + saved answers + memory).

Everything is stored in a single embedded Qdrant collection created automatically
on each machine under ``data/qdrant`` — no server, no manual setup. Embeddings are
produced by fastembed (a small multilingual ONNX model downloaded once, then fully
offline on CPU).

The store holds three kinds of points in one collection:
- ``knowledge``   — chunks of the bundled ``docs/knowledge/*.md`` cheatsheets
- ``good_answer`` — curated good answers migrated from the SQLite store
- ``memory``      — every answered question/answer pair, so the app "remembers"

Everything degrades gracefully: if qdrant-client / fastembed are not installed, or
the store cannot be opened, all functions become no-ops / empty and the caller
falls back to the previous lexical search. That keeps the app and tests working
without the heavy optional dependencies.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.config import ROOT_DIR

LOGGER = logging.getLogger(__name__)

QDRANT_PATH = Path(os.getenv("STACKWIRE_QDRANT_PATH", "").strip() or (ROOT_DIR / "data" / "qdrant"))
COLLECTION = os.getenv("STACKWIRE_QDRANT_COLLECTION", "stackwire_rag").strip() or "stackwire_rag"
# Multilingual (RU + EN) small model — ~120MB, downloaded once by fastembed.
EMBED_MODEL = os.getenv("STACKWIRE_EMBED_MODEL", "").strip() or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_NAMESPACE = uuid.UUID("a3f1c0de-0000-4000-8000-0000573ac4e1")

_LOCK = threading.Lock()
_client = None  # qdrant client (lazy)
_client_init_done = False
_available: bool | None = None
_indexed = False


@dataclass(frozen=True)
class VectorHit:
    kind: str
    text: str
    title: str
    source: str
    score: float
    question: str = ""
    answer: str = ""
    domain: str | None = None


def is_available() -> bool:
    """True when qdrant-client + fastembed import and the store can be opened."""
    global _available
    if _available is not None:
        return _available
    if os.getenv("STACKWIRE_DISABLE_VECTOR", "").strip().lower() in {"1", "true", "yes", "on"}:
        _available = False
        return False
    # Stay out of the way during tests (isolation + speed), unless explicitly forced.
    if "PYTEST_CURRENT_TEST" in os.environ and os.getenv("STACKWIRE_FORCE_VECTOR", "").strip().lower() not in {"1", "true", "yes", "on"}:
        _available = False
        return False
    try:
        import qdrant_client  # noqa: F401
        from fastembed import TextEmbedding  # noqa: F401
    except Exception:
        LOGGER.info("vector store disabled: qdrant-client/fastembed not installed")
        _available = False
        return False
    _available = _get_client() is not None
    return _available


def _get_client():
    global _client, _client_init_done
    if _client_init_done:
        return _client
    with _LOCK:
        if _client_init_done:
            return _client
        _client_init_done = True
        try:
            from qdrant_client import QdrantClient

            QDRANT_PATH.mkdir(parents=True, exist_ok=True)
            # Stable cache dir so the embedding model is downloaded only once
            # (not into a Temp folder that the OS may clear).
            cache_dir = Path(os.getenv("STACKWIRE_EMBED_CACHE", "").strip() or (ROOT_DIR / "data" / "embed_cache"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            client = QdrantClient(path=str(QDRANT_PATH))
            client.set_model(EMBED_MODEL, cache_dir=str(cache_dir))
            _client = client
            LOGGER.info("vector store ready path=%s model=%s", QDRANT_PATH, EMBED_MODEL)
        except Exception:
            LOGGER.warning("vector store open failed; falling back to lexical search", exc_info=True)
            _client = None
    return _client


def _point_id(key: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, key))


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def _knowledge_fingerprint() -> str:
    """Hash of all knowledge files so we only re-embed when they change."""
    from app.rag import knowledge_dir

    base = knowledge_dir()
    if not base.exists():
        return "none"
    hasher = hashlib.sha1()
    for path in sorted(base.glob("*.md")):
        try:
            stat = path.stat()
            hasher.update(path.name.encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("ascii"))
            hasher.update(str(stat.st_size).encode("ascii"))
        except OSError:
            continue
    return hasher.hexdigest()


def _fingerprint_file() -> Path:
    return QDRANT_PATH / ".knowledge_fingerprint"


def ensure_indexed(force: bool = False) -> None:
    """Index knowledge + migrate good answers once per process (and per content change)."""
    global _indexed
    if _indexed and not force:
        return
    if not is_available():
        _indexed = True
        return
    client = _get_client()
    if client is None:
        _indexed = True
        return

    with _LOCK:
        if _indexed and not force:
            return
        try:
            fingerprint = _knowledge_fingerprint()
            marker = _fingerprint_file()
            previous = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
            if force or previous != fingerprint:
                count_k = _index_knowledge(client)
                count_g = _migrate_good_answers(client)
                marker.write_text(fingerprint, encoding="utf-8")
                LOGGER.info("vector store indexed knowledge=%s good_answers=%s", count_k, count_g)
        except Exception:
            LOGGER.warning("vector index failed", exc_info=True)
        finally:
            _indexed = True


def _index_knowledge(client) -> int:
    from app.rag import load_chunks

    chunks = load_chunks()
    if not chunks:
        return 0
    documents: list[str] = []
    metadata: list[dict] = []
    ids: list[str] = []
    for index, chunk in enumerate(chunks):
        documents.append(f"{chunk.heading}\n{chunk.text}")
        metadata.append(
            {
                "kind": "knowledge",
                "text": chunk.text,
                "title": chunk.heading,
                "source": chunk.source_file,
            }
        )
        ids.append(_point_id(f"knowledge:{chunk.source_file}:{chunk.heading}:{index}"))
    client.add(collection_name=COLLECTION, documents=documents, metadata=metadata, ids=ids)
    return len(documents)


def _migrate_good_answers(client) -> int:
    try:
        from app.storage import all_good_answers

        answers = all_good_answers()
    except Exception:
        LOGGER.debug("good answer migration skipped", exc_info=True)
        return 0
    if not answers:
        return 0
    documents: list[str] = []
    metadata: list[dict] = []
    ids: list[str] = []
    for row in answers:
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue
        documents.append(question)
        metadata.append(
            {
                "kind": "good_answer",
                "text": answer,
                "title": question,
                "source": "good_answers",
                "question": question,
                "answer": answer,
                "domain": row.get("domain"),
            }
        )
        ids.append(_point_id(f"good_answer:{row.get('id', question)}"))
    if not documents:
        return 0
    client.add(collection_name=COLLECTION, documents=documents, metadata=metadata, ids=ids)
    return len(documents)


# --------------------------------------------------------------------------- #
# Writing (remember everything)
# --------------------------------------------------------------------------- #
def remember(question: str, answer: str, *, domain: str | None = None, intent: str | None = None) -> None:
    """Store an answered Q/A pair so future similar questions can reuse it."""
    question = (question or "").strip()
    answer = (answer or "").strip()
    if not question or not answer or len(answer) < 40:
        return
    if not is_available():
        return
    client = _get_client()
    if client is None:
        return
    try:
        ensure_indexed()
        key = hashlib.sha1(f"{question}\n{answer}".encode("utf-8")).hexdigest()
        client.add(
            collection_name=COLLECTION,
            documents=[question],
            metadata=[
                {
                    "kind": "memory",
                    "text": answer,
                    "title": question,
                    "source": "memory",
                    "question": question,
                    "answer": answer,
                    "domain": domain,
                    "intent": intent,
                }
            ],
            ids=[_point_id(f"memory:{key}")],
        )
    except Exception:
        LOGGER.debug("vector remember failed", exc_info=True)


def remember_good_answer(answer_id: int, question: str, answer: str, *, domain: str | None = None) -> None:
    question = (question or "").strip()
    answer = (answer or "").strip()
    if not question or not answer:
        return
    if not is_available():
        return
    client = _get_client()
    if client is None:
        return
    try:
        ensure_indexed()
        client.add(
            collection_name=COLLECTION,
            documents=[question],
            metadata=[
                {
                    "kind": "good_answer",
                    "text": answer,
                    "title": question,
                    "source": "good_answers",
                    "question": question,
                    "answer": answer,
                    "domain": domain,
                }
            ],
            ids=[_point_id(f"good_answer:{answer_id}")],
        )
    except Exception:
        LOGGER.debug("vector remember_good_answer failed", exc_info=True)


def _chunk_text(text: str, *, size: int = 800, max_chunks: int = 80) -> list[str]:
    paragraphs = [p.strip() for p in text.split(chr(10) + chr(10)) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > size:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), size):
                chunks.append(para[i : i + size])
        elif len(current) + len(para) + 2 <= size:
            current = (current + chr(10) + chr(10) + para) if current else para
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks[:max_chunks]


def remember_document(name: str, text: str) -> None:
    """Index an attached document's text (kind='document') so the model can retrieve
    from it across the conversation. Persisted in the local store, like a project file."""
    name = (name or "").strip()
    text = (text or "").strip()
    if not name or len(text) < 40 or not is_available():
        return
    client = _get_client()
    if client is None:
        return
    try:
        ensure_indexed()
        chunks = _chunk_text(text)
        documents: list[str] = []
        metadata: list[dict] = []
        ids: list[str] = []
        for index, chunk in enumerate(chunks):
            documents.append(chunk)
            metadata.append({"kind": "document", "text": chunk, "title": name, "source": f"attachment:{name}"})
            digest = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:12]
            ids.append(_point_id(f"document:{name}:{index}:{digest}"))
        if documents:
            client.add(collection_name=COLLECTION, documents=documents, metadata=metadata, ids=ids)
            LOGGER.info("vector store indexed document=%r chunks=%s", name, len(documents))
    except Exception:
        LOGGER.debug("vector remember_document failed", exc_info=True)


# --------------------------------------------------------------------------- #
# Searching
# --------------------------------------------------------------------------- #
def _search(query: str, *, kinds: tuple[str, ...], limit: int, score_threshold: float) -> list[VectorHit]:
    query = (query or "").strip()
    if not query or not is_available():
        return []
    client = _get_client()
    if client is None:
        return []
    try:
        ensure_indexed()
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        query_filter = Filter(must=[FieldCondition(key="kind", match=MatchAny(any=list(kinds)))]) if kinds else None
        responses = client.query(
            collection_name=COLLECTION,
            query_text=query,
            limit=max(1, limit),
            query_filter=query_filter,
        )
    except Exception:
        LOGGER.debug("vector search failed", exc_info=True)
        return []

    hits: list[VectorHit] = []
    for item in responses:
        score = float(getattr(item, "score", 0.0) or 0.0)
        if score < score_threshold:
            continue
        meta = getattr(item, "metadata", None) or {}
        hits.append(
            VectorHit(
                kind=str(meta.get("kind", "")),
                text=str(meta.get("text", "")),
                title=str(meta.get("title", "")),
                source=str(meta.get("source", "")),
                score=score,
                question=str(meta.get("question", "")),
                answer=str(meta.get("answer", "")),
                domain=meta.get("domain"),
            )
        )
    return hits


def search_knowledge(query: str, *, limit: int = 3, score_threshold: float = 0.2) -> list[VectorHit]:
    return _search(query, kinds=("knowledge", "document"), limit=limit, score_threshold=score_threshold)


def search_memory(query: str, *, limit: int = 3, score_threshold: float = 0.45) -> list[VectorHit]:
    return _search(query, kinds=("good_answer", "memory"), limit=limit, score_threshold=score_threshold)


def close() -> None:
    """Close the embedded Qdrant client cleanly (avoids a noisy __del__ at exit)."""
    global _client, _client_init_done
    client = _client
    _client = None
    _client_init_done = False
    if client is not None:
        try:
            client.close()
        except Exception:
            LOGGER.debug("vector store close failed", exc_info=True)


def stats() -> dict[str, int | str | bool]:
    info: dict[str, int | str | bool] = {"available": is_available(), "model": EMBED_MODEL, "path": str(QDRANT_PATH)}
    if not is_available():
        return info
    client = _get_client()
    if client is None:
        return info
    try:
        ensure_indexed()
        info["points"] = int(client.count(collection_name=COLLECTION).count)
    except Exception:
        LOGGER.debug("vector stats failed", exc_info=True)
    return info
