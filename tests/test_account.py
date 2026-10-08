import pytest

from conftest import PASSWORD, csrf, login, logout, query


def post_account(client, **data):
    data["csrf_token"] = csrf(client)
    return client.post("/account", data=data)


def test_account_page(auth_client):
    r = auth_client.get("/account")
    assert r.status_code == 200
    assert b"ananya@example.com" in r.data


def test_update_profile(auth_client, app):
    r = post_account(auth_client, action="profile", name="Ananya R.", price_per_kwh="9.5")
    assert r.status_code == 302
    row = query(app, "SELECT name, price_per_kwh FROM users")[0]
    assert row["name"] == "Ananya R."
    assert row["price_per_kwh"] == 9.5


@pytest.mark.parametrize("price", ["0", "-3", "101", "abc", "", "nan"])
def test_update_profile_rejects_bad_price(auth_client, app, price):
    r = post_account(auth_client, action="profile", name="Ananya", price_per_kwh=price)
    assert r.status_code == 400
    assert b"between 0 and 100" in r.data
    assert query(app, "SELECT price_per_kwh FROM users")[0][0] == 8.0


def test_update_profile_rejects_empty_name(auth_client):
    r = post_account(auth_client, action="profile", name="  ", price_per_kwh="8")
    assert r.status_code == 400


def test_change_password_flow(auth_client):
    r = post_account(auth_client, action="password", current=PASSWORD,
                     new="Monsoon2025", confirm="Monsoon2025")
    assert r.status_code == 302
    logout(auth_client)
    assert login(auth_client, password=PASSWORD).status_code == 401
    assert login(auth_client, password="Monsoon2025").status_code == 302


@pytest.mark.parametrize("current,new,confirm,message", [
    ("WrongOld11", "Monsoon2025", "Monsoon2025", b"Current password is wrong"),
    (PASSWORD, "short", "short", b"at least 8"),
    (PASSWORD, "Monsoon2025", "Monsoon2026", b"match"),
    (PASSWORD, PASSWORD, PASSWORD, b"must be different"),
])
def test_change_password_errors(auth_client, current, new, confirm, message):
    r = post_account(auth_client, action="password", current=current, new=new, confirm=confirm)
    assert r.status_code == 400
    assert message in r.data


def test_unknown_account_action(auth_client):
    assert post_account(auth_client, action="explode").status_code == 400


def test_delete_account_needs_password(auth_client, app):
    r = post_account(auth_client, action="delete", password="wrong")
    assert r.status_code == 400
    assert len(query(app, "SELECT id FROM users")) == 1


def test_delete_account_removes_everything(auth_client, app):
    auth_client.post("/vehicles/3/save", data={"csrf_token": csrf(auth_client)})
    auth_client.post("/trips", data={"csrf_token": csrf(auth_client), "vehicle_id": 3,
                                     "distance_km": 20, "price_per_kwh": 8})
    r = post_account(auth_client, action="delete", password=PASSWORD)
    assert r.status_code == 302
    assert query(app, "SELECT id FROM users") == []
    assert query(app, "SELECT * FROM favourites") == []
    assert query(app, "SELECT * FROM trips") == []
    assert auth_client.get("/dashboard").status_code == 302
    assert login(auth_client).status_code == 401
