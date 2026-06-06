import importlib

import pytest


@pytest.fixture()
def auth_module(tmp_path, monkeypatch):
    monkeypatch.setenv("STACKWIRE_DB_PATH", str(tmp_path / "auth_test.db"))
    import app.auth as auth

    importlib.reload(auth)
    auth.init_auth_db()
    return auth


def test_register_then_verify(auth_module):
    token = auth_module.register("alice", "secret123")
    assert token
    user = auth_module.verify_token(token)
    assert user is not None
    assert user.username == "alice"


def test_login_roundtrip(auth_module):
    auth_module.register("bob", "hunter2x")
    token = auth_module.login("bob", "hunter2x")
    assert auth_module.verify_token(token) is not None


def test_wrong_password_rejected(auth_module):
    auth_module.register("carol", "rightpass")
    with pytest.raises(auth_module.AuthError):
        auth_module.login("carol", "wrongpass")


def test_duplicate_user_rejected(auth_module):
    auth_module.register("dave", "passpass")
    with pytest.raises(auth_module.AuthError):
        auth_module.register("dave", "passpass2")


def test_short_password_rejected(auth_module):
    with pytest.raises(auth_module.AuthError):
        auth_module.register("erin", "123")


def test_invalid_username_rejected(auth_module):
    with pytest.raises(auth_module.AuthError):
        auth_module.register("ab", "longenough")  # too short
    with pytest.raises(auth_module.AuthError):
        auth_module.register("has space", "longenough")


def test_bad_token_returns_none(auth_module):
    assert auth_module.verify_token("not-a-real-token") is None
    assert auth_module.verify_token("") is None


def test_logout_invalidates_token(auth_module):
    token = auth_module.register("frank", "passpass")
    assert auth_module.verify_token(token) is not None
    auth_module.logout(token)
    assert auth_module.verify_token(token) is None


def test_password_hash_not_plaintext(auth_module):
    token = auth_module.register("grace", "topsecretpw")
    # The stored hash must not contain the raw password.
    import sqlite3

    conn = sqlite3.connect(auth_module._db_path())
    row = conn.execute("SELECT password_hash FROM users WHERE username = 'grace'").fetchone()
    conn.close()
    assert row is not None
    assert "topsecretpw" not in row[0]
    assert row[0].startswith("pbkdf2_sha256$")
    assert token  # sanity
