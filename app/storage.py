import json
import logging
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from app.config import ROOT_DIR


LOGGER = logging.getLogger(__name__)
_LOCK = Lock()


@dataclass(frozen=True)
class GoodAnswer:
    id: int
    question: str
    answer: str
    domain: str | None
    intent: str | None
    tags: list[str]
    rating: int
    score: float = 0.0


def _db_path() -> Path:
    configured = os.getenv("STACKWIRE_DB_PATH", "").strip()
    return Path(configured) if configured else ROOT_DIR / "data" / "stackwire.db"


def _now() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _LOCK:
        with _connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    title TEXT
                );

                CREATE TABLE IF NOT EXISTS questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    created_at TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    recovered_question TEXT NOT NULL,
                    trusted_text INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL,
                    recovery_confidence REAL,
                    detected_topic TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS answers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    answer_type TEXT NOT NULL,
                    expand_mode TEXT,
                    model TEXT,
                    answer_mode TEXT,
                    latency_ms REAL,
                    validator_ok INTEGER,
                    validator_violations TEXT,
                    plan_domain TEXT,
                    plan_intent TEXT,
                    artifact_required INTEGER,
                    FOREIGN KEY(question_id) REFERENCES questions(id)
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    answer_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    label TEXT NOT NULL,
                    note TEXT,
                    FOREIGN KEY(answer_id) REFERENCES answers(id)
                );

                CREATE TABLE IF NOT EXISTS good_answers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    domain TEXT,
                    intent TEXT,
                    tags TEXT,
                    created_at TEXT NOT NULL,
                    rating INTEGER NOT NULL DEFAULT 5
                );
                """
            )
    LOGGER.info("storage db ready path=%s", _db_path())


def create_session(title: str | None = None) -> int:
    init_db()
    with _LOCK:
        with _connect() as db:
            cursor = db.execute(
                "INSERT INTO sessions(started_at, title) VALUES (?, ?)",
                (_now(), title),
            )
            session_id = int(cursor.lastrowid)
    LOGGER.info("storage session created id=%s", session_id)
    return session_id


def log_question(
    *,
    raw_text: str,
    recovered_question: str,
    trusted_text: bool = False,
    source: str = "manual",
    session_id: int | None = None,
    recovery_confidence: float | None = None,
    detected_topic: str | None = None,
) -> int:
    init_db()
    with _LOCK:
        with _connect() as db:
            cursor = db.execute(
                """
                INSERT INTO questions(
                    session_id, created_at, raw_text, recovered_question,
                    trusted_text, source, recovery_confidence, detected_topic
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    _now(),
                    raw_text,
                    recovered_question,
                    1 if trusted_text else 0,
                    source,
                    recovery_confidence,
                    detected_topic,
                ),
            )
            question_id = int(cursor.lastrowid)
    LOGGER.info("storage question logged id=%s source=%s topic=%s", question_id, source, detected_topic)
    return question_id


def log_answer(
    *,
    question_id: int,
    answer: str,
    answer_type: str = "main",
    expand_mode: str | None = None,
    model: str | None = None,
    answer_mode: str | None = None,
    latency_ms: float | None = None,
    validator_ok: bool | None = None,
    validator_violations: list[str] | tuple[str, ...] | None = None,
    plan_domain: str | None = None,
    plan_intent: str | None = None,
    artifact_required: bool | None = None,
) -> int:
    init_db()
    violations_json = json.dumps(list(validator_violations or ()), ensure_ascii=False)
    with _LOCK:
        with _connect() as db:
            cursor = db.execute(
                """
                INSERT INTO answers(
                    question_id, created_at, answer, answer_type, expand_mode,
                    model, answer_mode, latency_ms, validator_ok,
                    validator_violations, plan_domain, plan_intent, artifact_required
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question_id,
                    _now(),
                    answer,
                    answer_type,
                    expand_mode,
                    model,
                    answer_mode,
                    latency_ms,
                    None if validator_ok is None else (1 if validator_ok else 0),
                    violations_json,
                    plan_domain,
                    plan_intent,
                    None if artifact_required is None else (1 if artifact_required else 0),
                ),
            )
            answer_id = int(cursor.lastrowid)
    LOGGER.info(
        "storage answer logged id=%s question_id=%s type=%s expand_mode=%s validator_ok=%s",
        answer_id,
        question_id,
        answer_type,
        expand_mode,
        validator_ok,
    )
    return answer_id


def log_feedback(answer_id: int, label: str, note: str | None = None) -> int:
    init_db()
    with _LOCK:
        with _connect() as db:
            cursor = db.execute(
                "INSERT INTO feedback(answer_id, created_at, label, note) VALUES (?, ?, ?, ?)",
                (answer_id, _now(), label, note),
            )
            feedback_id = int(cursor.lastrowid)
    LOGGER.info("storage feedback logged id=%s answer_id=%s label=%s", feedback_id, answer_id, label)
    return feedback_id


def save_good_answer(
    *,
    question: str,
    answer: str,
    domain: str | None = None,
    intent: str | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
    rating: int = 5,
) -> int:
    init_db()
    tags_json = json.dumps(list(tags or ()), ensure_ascii=False)
    with _LOCK:
        with _connect() as db:
            cursor = db.execute(
                """
                INSERT INTO good_answers(question, answer, domain, intent, tags, created_at, rating)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (question, answer, domain, intent, tags_json, _now(), int(rating)),
            )
            good_answer_id = int(cursor.lastrowid)
    LOGGER.info("storage good answer saved id=%s domain=%s intent=%s", good_answer_id, domain, intent)
    return good_answer_id


def search_good_answers(query: str, domain: str | None = None, limit: int = 3) -> list[GoodAnswer]:
    init_db()
    query_tokens = _tokens(query)
    if not query_tokens:
        return []

    with _connect() as db:
        if domain:
            rows = db.execute(
                """
                SELECT id, question, answer, domain, intent, tags, rating
                FROM good_answers
                WHERE domain = ? OR domain IS NULL
                ORDER BY rating DESC, id DESC
                LIMIT 80
                """,
                (domain,),
            ).fetchall()
        else:
            rows = db.execute(
                """
                SELECT id, question, answer, domain, intent, tags, rating
                FROM good_answers
                ORDER BY rating DESC, id DESC
                LIMIT 80
                """
            ).fetchall()

    scored: list[GoodAnswer] = []
    for row in rows:
        question = str(row["question"])
        answer = str(row["answer"])
        haystack = f"{question} {answer}"
        score = _lexical_score(query_tokens, haystack)
        if domain and row["domain"] == domain:
            score += 0.15
        if score <= 0:
            continue
        scored.append(
            GoodAnswer(
                id=int(row["id"]),
                question=question,
                answer=answer,
                domain=row["domain"],
                intent=row["intent"],
                tags=_loads_tags(row["tags"]),
                rating=int(row["rating"] or 5),
                score=score,
            )
        )

    scored.sort(key=lambda item: (item.score, item.rating, item.id), reverse=True)
    result = scored[: max(0, limit)]
    LOGGER.info("storage good answer search query=%r domain=%s matches=%s", query[:80], domain, len(result))
    return result


def all_good_answers(limit: int = 5000) -> list[dict[str, Any]]:
    """Return every saved good answer (used to migrate into the vector store)."""
    init_db()
    with _connect() as db:
        rows = db.execute(
            """
            SELECT id, question, answer, domain, intent, tags, rating
            FROM good_answers
            ORDER BY id ASC
            LIMIT ?
            """,
            (max(0, limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def get_recent_questions(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    with _connect() as db:
        rows = db.execute(
            """
            SELECT id, session_id, created_at, raw_text, recovered_question, trusted_text,
                   source, recovery_confidence, detected_topic
            FROM questions
            ORDER BY id DESC
            LIMIT ?
            """,
            (max(0, limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def export_session_markdown(session_id: int) -> str:
    init_db()
    with _connect() as db:
        session = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        questions = db.execute(
            """
            SELECT * FROM questions
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
        answers_by_question: dict[int, list[sqlite3.Row]] = {}
        for answer in db.execute(
            """
            SELECT a.* FROM answers a
            JOIN questions q ON q.id = a.question_id
            WHERE q.session_id = ?
            ORDER BY a.id ASC
            """,
            (session_id,),
        ).fetchall():
            answers_by_question.setdefault(int(answer["question_id"]), []).append(answer)

    if session is None:
        raise ValueError(f"Session {session_id} not found")

    title = session["title"] or f"StackWire session {session_id}"
    lines = [
        f"# {title}",
        "",
        f"- session_id: {session_id}",
        f"- started_at: {session['started_at']}",
        f"- ended_at: {session['ended_at'] or '-'}",
        "",
    ]
    for index, question in enumerate(questions, start=1):
        lines.extend(
            [
                f"## Question {index}",
                "",
                f"- source: {question['source']}",
                f"- detected_topic: {question['detected_topic'] or '-'}",
                "",
                "Raw:",
                "",
                question["raw_text"] or "-",
                "",
                "Recovered:",
                "",
                question["recovered_question"] or "-",
                "",
            ]
        )
        for answer in answers_by_question.get(int(question["id"]), []):
            mode = f" ({answer['expand_mode']})" if answer["expand_mode"] else ""
            lines.extend(
                [
                    f"### Answer: {answer['answer_type']}{mode}",
                    "",
                    answer["answer"] or "-",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _loads_tags(raw: Any) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-zА-Яа-я0-9_./+-]{2,}", text.casefold())
        if token not in {"the", "and", "for", "что", "как", "это", "или", "при", "для"}
    }


def _lexical_score(query_tokens: set[str], text: str) -> float:
    text_tokens = _tokens(text)
    if not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    score = overlap / max(1, len(query_tokens))
    if score == 0:
        try:
            from rapidfuzz import fuzz

            score = fuzz.partial_ratio(" ".join(sorted(query_tokens)), text.casefold()) / 100.0
            if score < 0.55:
                return 0.0
            return score * 0.45
        except Exception:
            return 0.0
    return score


def _main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "export-session":
        if len(argv) < 3:
            print("Usage: python -m app.storage export-session <session_id>", file=sys.stderr)
            return 2
        try:
            print(export_session_markdown(int(argv[2])), end="")
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if len(argv) >= 2 and argv[1] == "init":
        init_db()
        print(_db_path())
        return 0
    print("Usage: python -m app.storage export-session <session_id>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
