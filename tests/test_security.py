import pytest

from conftest import PASSWORD, csrf, execute, login, query, register


PROTECTED = ["/dashboard", "/vehicles", "/vehicles/1", "/saved", "/trips", "/account"]
API = ["/api/me", "/api/vehicles", "/api/vehicles/1", "/api/stats", "/api/saved",
       "/api/estimate?vehicle_id=1&distance_km=10"]


@pytest.mark.parametrize("path", PROTECTED)
def test_pages_need_login(client, path):
    r = client.get(path)
    assert r.status_code == 302
    assert "/login" in r.headers["Location"]


@pytest.mark.parametrize("path", API)
def test_api_needs_login(client, path):
    r = client.get(path)
    assert r.status_code == 401
    assert r.get_json() == {"error": "authentication required"}


@pytest.mark.parametrize("path", ["/vehicles/1/save", "/trips", "/trips/1/delete", "/account"])
def test_protected_posts_need_login(client, path):
    r = client.post(path, data={"csrf_token": csrf(client)})
    assert r.status_code == 302
    assert "/login" in r.headers["Location"]


def test_post_without_csrf_token_rejected(client):
    r = client.post("/register", data={"name": "X", "email": "x@example.com",
                                       "password": PASSWORD, "confirm": PASSWORD})
    assert r.status_code == 400
    assert b"form expired" in r.data


def test_post_with_wrong_csrf_token_rejected(client, user):
    csrf(client)
    r = client.post("/login", data={"csrf_token": "forged", "email": user["email"],
                                    "password": user["password"]})
    assert r.status_code == 400


def test_logout_without_csrf_is_rejected(auth_client):
    assert auth_client.post("/logout").status_code == 400
    assert auth_client.get("/dashboard").status_code == 200


def test_login_starts_a_fresh_session(client, user):
    csrf(client)
    with client.session_transaction() as s:
        s["planted"] = "attacker value"
        old_token = s["_csrf"]
    login(client)
    with client.session_transaction() as s:
        assert "planted" not in s
        assert s["_csrf"] != old_token


def test_password_change_signs_out_other_sessions(app, user):
    laptop, phone = app.test_client(), app.test_client()
    login(laptop)
    login(phone)
    assert phone.get("/dashboard").status_code == 200

    laptop.post("/account", data={"csrf_token": csrf(laptop), "action": "password",
                                  "current": PASSWORD, "new": "NewPass2025", "confirm": "NewPass2025"})
    assert laptop.get("/dashboard").status_code == 200   # the device that changed it stays in
    assert phone.get("/dashboard").status_code == 302    # everyone else is out


def test_sign_out_everywhere(app, user):
    a, b = app.test_client(), app.test_client()
    login(a)
    login(b)
    a.post("/account", data={"csrf_token": csrf(a), "action": "signout_all"})
    assert a.get("/dashboard").status_code == 302
    assert b.get("/dashboard").status_code == 302


def test_tampered_session_cookie_is_ignored(client, user):
    login(client)
    client.set_cookie("session", "eyJ1aWQiOjF9.forged.signature")
    assert client.get("/dashboard").status_code == 302


def test_deleted_user_session_is_dropped(app, auth_client):
    execute(app, "DELETE FROM users")
    assert auth_client.get("/dashboard").status_code == 302


def test_security_headers(client):
    r = client.get("/login")
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]


def test_authenticated_pages_not_cached(auth_client):
    assert auth_client.get("/dashboard").headers["Cache-Control"] == "no-store"


def test_names_are_html_escaped(client):
    register(client, name="<script>alert(1)</script>")
    r = client.get("/dashboard")
    assert b"<script>alert(1)" not in r.data
    assert b"&lt;script&gt;" in r.data


def test_sql_injection_in_login_fails(client, user):
    r = login(client, email="' OR '1'='1", password="' OR '1'='1")
    assert r.status_code == 401


def test_sql_injection_in_search_is_treated_as_text(auth_client):
    r = auth_client.get("/vehicles?q=' OR 1=1 --")
    assert r.status_code == 200
    assert b"0 matches" in r.data


def test_sql_injection_in_sort_falls_back_to_default(auth_client):
    r = auth_client.get("/api/vehicles?sort=km_per_kwh;DROP TABLE users")
    assert r.status_code == 200
    assert r.get_json()["total"] == 1197


def test_cannot_delete_someone_elses_trip(app, client):
    register(client, name="Owner", email="owner@example.com")
    client.post("/trips", data={"csrf_token": csrf(client), "vehicle_id": 1,
                                "distance_km": 50, "price_per_kwh": 8})
    trip_id = query(app, "SELECT id FROM trips")[0][0]

    other = app.test_client()
    register(other, name="Intruder", email="intruder@example.com")
    r = other.post(f"/trips/{trip_id}/delete", data={"csrf_token": csrf(other)})
    assert r.status_code == 404
    assert len(query(app, "SELECT id FROM trips")) == 1


def test_users_only_see_their_own_data(app, client):
    register(client, name="Owner", email="owner@example.com")
    client.post("/vehicles/5/save", data={"csrf_token": csrf(client)})
    client.post("/trips", data={"csrf_token": csrf(client), "vehicle_id": 5,
                                "distance_km": 42, "price_per_kwh": 8, "label": "Secret run"})

    other = app.test_client()
    register(other, name="Other", email="other@example.com")
    assert other.get("/api/saved").get_json()["results"] == []
    assert b"Secret run" not in other.get("/trips").data
