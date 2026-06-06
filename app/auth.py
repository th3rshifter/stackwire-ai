"""Server-side user accounts and session tokens (stdlib only).

Accounts live in the same SQLite database as the rest of the app. Passwords are
stored as salted PBKDF2-HMAC-SHA256 hashes; login returns an opaque bearer token
whose SHA-256 is stored server-side. The desktop caches that token locally so the
user authenticates once.

This module is intentionally dependency-free so the FastAPI backend (app/main.py)
can enforce auth and the desktop can talk to it over HTTP.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock

from app.config import ROOT_DIR

LOGGER = logging.getLogger(__name__)
_LOCK = Lock()

_PBKDF2_ITERATIONS = 200_000
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.@+-]{3,64}$")
MIN_PASSWORD_LEN = 6


class AuthError(Exception):
    """Raised for invalid credentials, duplicate users, or bad input."""


@dataclass(frozen=True)
class AuthUser:
    id: int
    username: str


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


def init_auth_db() -> None:
    with _LOCK:
        with _connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auth_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );
                """
            )


def _hash_password(password: str, *, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = stored.split("$", 3)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    except ValueError:
        return False
    return secrets.compare_digest(candidate, expected)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _validate_username(username: str) -> str:
    username = (username or "").strip()
    if not _USERNAME_RE.match(username):
        raise AuthError("Имя пользователя: 3–64 символа, латиница/цифры/._@+-")
    return username


def _issue_token(db: sqlite3.Connection, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    db.execute(
        "INSERT INTO auth_tokens(token_hash, user_id, created_at, last_seen) VALUES (?, ?, ?, ?)",
        (_hash_token(token), user_id, _now(), _now()),
    )
    return token


def register(username: str, password: str) -> str:
    """Create a new account and return a fresh session token."""
    init_auth_db()
    username = _validate_username(username)
    if len(password or "") < MIN_PASSWORD_LEN:
        raise AuthError(f"Пароль не короче {MIN_PASSWORD_LEN} символов")
    with _LOCK:
        with _connect() as db:
            existing = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if existing:
                raise AuthError("Пользователь уже существует")
            cursor = db.execute(
                "INSERT INTO users(username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, _hash_password(password), _now()),
            )
            user_id = int(cursor.lastrowid)
            token = _issue_token(db, user_id)
    LOGGER.info("auth register username=%s user_id=%s", username, user_id)
    return token


def login(username: str, password: str) -> str:
    """Validate credentials and return a fresh session token."""
    init_auth_db()
    username = (username or "").strip()
    with _LOCK:
        with _connect() as db:
            row = db.execute(
                "SELECT id, password_hash FROM users WHERE username = ?", (username,)
            ).fetchone()
            if row is None or not _verify_password(password, str(row["password_hash"])):
                raise AuthError("Неверное имя пользователя или пароль")
            token = _issue_token(db, int(row["id"]))
    LOGGER.info("auth login username=%s", username)
    return token


def verify_token(token: str) -> AuthUser | None:
    """Return the user for a valid token, or None."""
    token = (token or "").strip()
    if not token:
        return None
    init_auth_db()
    with _LOCK:
        with _connect() as db:
            row = db.execute(
                """
                SELECT u.id AS id, u.username AS username
                FROM auth_tokens t
                JOIN users u ON u.id = t.user_id
                WHERE t.token_hash = ?
                """,
                (_hash_token(token),),
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE auth_tokens SET last_seen = ? WHERE token_hash = ?",
                (_now(), _hash_token(token)),
            )
    return AuthUser(id=int(row["id"]), username=str(row["username"]))


def logout(token: str) -> None:
    token = (token or "").strip()
    if not token:
        return
    init_auth_db()
    with _LOCK:
        with _connect() as db:
            db.execute("DELETE FROM auth_tokens WHERE token_hash = ?", (_hash_token(token),))


def user_count() -> int:
    init_auth_db()
    with _connect() as db:
        row = db.execute("SELECT COUNT(*) AS n FROM users").fetchone()
    return int(row["n"]) if row else 0
