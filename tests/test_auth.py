from datetime import datetime, timedelta, timezone

import pytest

from conftest import PASSWORD, csrf, execute, login, logout, query, register


# ---------- registration ----------

def test_register_page_loads(client):
    r = client.get("/register")
    assert r.status_code == 200
    assert b"Create an account" in r.data


def test_register_creates_user_and_logs_in(client, app):
    r = register(client)
    assert r.status_code == 302
    assert r.headers["Location"].endswith("/dashboard")

    rows = query(app, "SELECT * FROM users")
    assert len(rows) == 1
    assert rows[0]["name"] == "Ananya Rao"
    # password is stored hashed, never as plain text
    assert rows[0]["password_hash"] != PASSWORD
    assert rows[0]["password_hash"].startswith(("scrypt:", "pbkdf2:"))

    assert client.get("/dashboard").status_code == 200


def test_register_lowercases_email(client, app):
    register(client, email="  Ananya@Example.COM ")
    assert query(app, "SELECT email FROM users")[0]["email"] == "ananya@example.com"


def test_register_rejects_duplicate_email_case_insensitive(client, user, app):
    r = register(client, email="ANANYA@example.com")
    assert r.status_code == 400
    assert b"already exists" in r.data
    assert len(query(app, "SELECT id FROM users")) == 1


@pytest.mark.parametrize("field,value,message", [
    ("name", "", b"Name is required"),
    ("name", "x" * 61, b"longer than 60"),
    ("email", "", b"Email is required"),
    ("email", "not-an-email", b"valid email"),
    ("email", "a@b", b"valid email"),
    ("password", "short1", b"at least 8 characters"),
    ("password", "allletters", b"one letter and one number"),
    ("password", "12345678", b"one letter and one number"),
    ("password", "a1" * 65, b"longer than 128"),
])
def test_register_validation(client, app, field, value, message):
    data = {"name": "Ananya", "email": "ananya@example.com", "password": PASSWORD}
    data[field] = value
    r = register(client, **data)
    assert r.status_code == 400
    assert message in r.data
    assert query(app, "SELECT id FROM users") == []


def test_register_password_mismatch(client):
    r = register(client, confirm="Different2024")
    assert r.status_code == 400
    assert b"match" in r.data


def test_register_keeps_entered_values_on_error(client):
    r = register(client, name="Kabir", email="kabir@example.com", password="weak")
    assert b'value="Kabir"' in r.data
    assert b'value="kabir@example.com"' in r.data
    assert b"weak" not in r.data  # password is never echoed back


# ---------- login / logout ----------

def test_login_success(client, user, app):
    r = login(client)
    assert r.status_code == 302
    assert r.headers["Location"].endswith("/dashboard")
    assert query(app, "SELECT last_login_at FROM users")[0][0] is not None


def test_login_wrong_password(client, user):
    r = login(client, password="WrongPass99")
    assert r.status_code == 401
    assert b"Incorrect email or password" in r.data
    assert client.get("/dashboard").status_code == 302


def test_login_unknown_email_gives_same_message(client, user):
    r = login(client, email="nobody@example.com")
    assert r.status_code == 401
    assert b"Incorrect email or password" in r.data


def test_login_is_case_insensitive_on_email(client, user):
    assert login(client, email="ANANYA@EXAMPLE.com").status_code == 302


def test_account_locks_after_five_failures(client, user, app):
    for _ in range(4):
        assert b"Incorrect" in login(client, password="nope1234").data
    r = login(client, password="nope1234")
    assert b"locked for 15 minutes" in r.data

    # even the right password is refused while locked
    r = login(client)
    assert r.status_code == 401
    assert b"locked" in r.data
    assert client.get("/dashboard").status_code == 302


def test_lock_expires(client, user, app):
    for _ in range(5):
        login(client, password="nope1234")
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
    execute(app, "UPDATE users SET locked_until = ?", (past,))
    assert login(client).status_code == 302


def test_successful_login_resets_failure_count(client, user, app):
    for _ in range(3):
        login(client, password="nope1234")
    assert query(app, "SELECT failed_logins FROM users")[0][0] == 3
    login(client)
    assert query(app, "SELECT failed_logins FROM users")[0][0] == 0


def test_logout(auth_client):
    r = logout(auth_client)
    assert r.status_code == 302
    assert auth_client.get("/dashboard").status_code == 302


def test_logout_requires_post(auth_client):
    assert auth_client.get("/logout").status_code == 405
    assert auth_client.get("/dashboard").status_code == 200


def test_remember_me_sets_persistent_cookie(client, user):
    r = login(client, remember="on")
    cookie = r.headers["Set-Cookie"]
    assert "Expires=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=Lax" in cookie


def test_without_remember_cookie_is_session_only(client, user):
    r = login(client)
    assert "Expires=" not in r.headers["Set-Cookie"]


def test_login_redirects_to_next(client, user):
    r = client.get("/saved")
    assert "/login?next=/saved" in r.headers["Location"]
    r = login(client, next="/saved")
    assert r.headers["Location"].endswith("/saved")


@pytest.mark.parametrize("bad", ["https://evil.example", "//evil.example", "/\\evil.example", "javascript:alert(1)"])
def test_login_ignores_external_next(client, user, bad):
    r = login(client, next=bad)
    assert r.status_code == 302
    assert r.headers["Location"].endswith("/dashboard")


def test_logged_in_user_skips_login_and_register(auth_client):
    assert auth_client.get("/login").headers["Location"].endswith("/dashboard")
    assert auth_client.get("/register").headers["Location"].endswith("/dashboard")


def test_root_redirects_by_auth_state(client, user):
    assert client.get("/").headers["Location"].endswith("/login")
    login(client)
    assert client.get("/").headers["Location"].endswith("/dashboard")
