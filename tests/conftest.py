import pytest

from app import create_app
from app.db import get_db, init_db

PASSWORD = "Sunrise2024"


@pytest.fixture
def app(tmp_path):
    app = create_app({
        "TESTING": True,
        "SECRET_KEY": "test-secret",
        "PASSWORD_HASH_METHOD": "pbkdf2:sha256:1000",
        "DATABASE": str(tmp_path / "test.sqlite3"),
    })
    with app.app_context():
        init_db()
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def csrf(client):
    """Return the CSRF token for this client's session, creating one if needed."""
    client.get("/login")  # make sure a session cookie exists
    with client.session_transaction() as s:
        if "_csrf" not in s:
            s["_csrf"] = "fixed-test-token"
        return s["_csrf"]


def register(client, name="Ananya Rao", email="ananya@example.com",
             password=PASSWORD, confirm=None):
    return client.post("/register", data={
        "csrf_token": csrf(client),
        "name": name,
        "email": email,
        "password": password,
        "confirm": password if confirm is None else confirm,
    })


def login(client, email="ananya@example.com", password=PASSWORD, **extra):
    data = {"csrf_token": csrf(client), "email": email, "password": password}
    data.update(extra)
    query = f"?next={extra.pop('next')}" if "next" in extra else ""
    return client.post("/login" + query, data=data)


def logout(client):
    return client.post("/logout", data={"csrf_token": csrf(client)})


@pytest.fixture
def user(client):
    """A registered user, logged out again so each test starts clean."""
    register(client)
    logout(client)
    return {"email": "ananya@example.com", "password": PASSWORD}


@pytest.fixture
def auth_client(client, user):
    login(client)
    return client


def query(app, sql, params=()):
    with app.app_context():
        return get_db().execute(sql, params).fetchall()


def execute(app, sql, params=()):
    with app.app_context():
        db = get_db()
        db.execute(sql, params)
        db.commit()
