"""Desktop-side authentication: talk to the auth server and cache the token locally.

The user logs in / registers once; the bearer token is cached in
``data/credentials.json`` so subsequent launches sign in automatically. All chat
actions are gated on having a valid token (verified against the server, with the
cached token trusted if the server is unreachable).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import requests

from app.config import ROOT_DIR

LOGGER = logging.getLogger(__name__)

CREDENTIALS_FILE = Path(os.getenv("STACKWIRE_CREDENTIALS_PATH", "").strip() or (ROOT_DIR / "data" / "credentials.json"))
_TIMEOUT = float(os.getenv("STACKWIRE_AUTH_TIMEOUT", "10"))


def auth_base_url() -> str:
    """Auth server base URL: explicit override, else the API URL, else local default."""
    explicit = os.getenv("STACKWIRE_AUTH_URL", "").strip().rstrip("/")
    if explicit:
        return explicit
    api = os.getenv("STACKWIRE_API_URL", "").strip().rstrip("/")
    if api:
        return api
    return "http://127.0.0.1:8000"


@dataclass
class Credentials:
    username: str
    token: str


class AuthClientError(Exception):
    """User-facing auth failure with a clean message."""


def load_credentials() -> Credentials | None:
    try:
        if not CREDENTIALS_FILE.exists():
            return None
        data = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
        username = str(data.get("username", "")).strip()
        token = str(data.get("token", "")).strip()
        if username and token:
            return Credentials(username=username, token=token)
    except Exception:
        LOGGER.debug("credentials load failed", exc_info=True)
    return None


def save_credentials(credentials: Credentials) -> None:
    try:
        CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        CREDENTIALS_FILE.write_text(
            json.dumps({"username": credentials.username, "token": credentials.token}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        LOGGER.warning("credentials save failed", exc_info=True)


def clear_credentials() -> None:
    try:
        if CREDENTIALS_FILE.exists():
            CREDENTIALS_FILE.unlink()
    except Exception:
        LOGGER.debug("credentials clear failed", exc_info=True)


def _post(path: str, payload: dict) -> dict:
    url = f"{auth_base_url()}{path}"
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post(url, json=payload, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        raise AuthClientError(
            f"Сервер авторизации недоступен ({auth_base_url()}). Запустите сервер или укажите адрес в настройках."
        ) from exc
    if response.status_code in (400, 401):
        detail = _detail(response)
        raise AuthClientError(detail or "Неверные данные для входа")
    if response.status_code >= 400:
        raise AuthClientError(_detail(response) or f"Ошибка сервера ({response.status_code})")
    try:
        return response.json()
    except ValueError as exc:
        raise AuthClientError("Некорректный ответ сервера авторизации") from exc


def _detail(response: requests.Response) -> str:
    try:
        data = response.json()
        detail = data.get("detail")
        if isinstance(detail, str):
            return detail
    except Exception:
        pass
    return ""


def register(username: str, password: str) -> Credentials:
    data = _post("/auth/register", {"username": username.strip(), "password": password})
    credentials = Credentials(username=str(data.get("username", username)).strip(), token=str(data.get("token", "")))
    save_credentials(credentials)
    return credentials


def login(username: str, password: str) -> Credentials:
    data = _post("/auth/login", {"username": username.strip(), "password": password})
    credentials = Credentials(username=str(data.get("username", username)).strip(), token=str(data.get("token", "")))
    save_credentials(credentials)
    return credentials


def verify(token: str) -> bool:
    """Verify a token against the server. On network failure, trust the cached token."""
    url = f"{auth_base_url()}/auth/me"
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=_TIMEOUT)
    except requests.RequestException:
        LOGGER.info("auth verify: server unreachable, trusting cached token")
        return True
    return response.status_code == 200


def logout(token: str) -> None:
    url = f"{auth_base_url()}/auth/logout"
    session = requests.Session()
    session.trust_env = False
    try:
        session.post(url, headers={"Authorization": f"Bearer {token}"}, timeout=_TIMEOUT)
    except requests.RequestException:
        LOGGER.debug("auth logout request failed", exc_info=True)
    clear_credentials()
