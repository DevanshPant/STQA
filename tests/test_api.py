import pytest

from conftest import csrf, execute, query


def test_me(auth_client):
    data = auth_client.get("/api/me").get_json()
    assert data["email"] == "ananya@example.com"
    assert data["price_per_kwh"] == 8.0
    assert "password_hash" not in data


def test_vehicles_shape(auth_client):
    data = auth_client.get("/api/vehicles").get_json()
    assert data["total"] == 1197
    assert set(data["results"][0]) == {"id", "year", "make", "model", "class",
                                       "motor_kw", "recharge_hours", "km_per_kwh"}


@pytest.mark.parametrize("per_page,expected", [("5", 5), ("500", 100), ("0", 1)])
def test_per_page_is_bounded(auth_client, per_page, expected):
    data = auth_client.get(f"/api/vehicles?per_page={per_page}").get_json()
    assert len(data["results"]) == expected


@pytest.mark.parametrize("qs", ["year=twenty", "min_eff=-2", "page=x", "per_page=lots", "min_eff=nan"])
def test_bad_params_return_400(auth_client, qs):
    r = auth_client.get(f"/api/vehicles?{qs}")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_single_vehicle(auth_client):
    data = auth_client.get("/api/vehicles/1").get_json()
    assert data["make"] == "Mitsubishi" and data["model"] == "i-MiEV" and data["year"] == 2012


def test_single_vehicle_404(auth_client):
    r = auth_client.get("/api/vehicles/424242")
    assert r.status_code == 404
    assert r.get_json()["error"] == "vehicle not found"


def test_stats(auth_client, app):
    data = auth_client.get("/api/stats").get_json()
    avg = query(app, "SELECT AVG(km_per_kwh) FROM vehicles")[0][0]
    assert data["vehicles"] == 1197
    assert data["makes"] == 35
    assert data["years"] == [2012, 2026]
    assert data["average_km_per_kwh"] == round(avg, 3)
    assert sum(y["count"] for y in data["by_year"]) == 1197


def test_estimate_uses_user_price_by_default(auth_client, app):
    eff = query(app, "SELECT km_per_kwh FROM vehicles WHERE id = 3")[0][0]
    data = auth_client.get("/api/estimate?vehicle_id=3&distance_km=250").get_json()
    assert data["price_per_kwh"] == 8.0
    assert data["energy_kwh"] == round(250 / eff, 2)
    assert data["cost"] == round(250 / eff * 8, 2)


def test_estimate_respects_profile_price(auth_client, app):
    execute(app, "UPDATE users SET price_per_kwh = 12")
    data = auth_client.get("/api/estimate?vehicle_id=3&distance_km=100").get_json()
    assert data["price_per_kwh"] == 12


@pytest.mark.parametrize("qs,status", [
    ("distance_km=10", 400),
    ("vehicle_id=3", 400),
    ("vehicle_id=3&distance_km=-5", 400),
    ("vehicle_id=3&distance_km=10&price_per_kwh=500", 400),
    ("vehicle_id=99999&distance_km=10", 404),
])
def test_estimate_errors(auth_client, qs, status):
    assert auth_client.get(f"/api/estimate?{qs}").status_code == status


def test_saved_list(auth_client):
    auth_client.post("/vehicles/7/save", data={"csrf_token": csrf(auth_client)})
    ids = [v["id"] for v in auth_client.get("/api/saved").get_json()["results"]]
    assert ids == [7]


def test_api_is_read_only(auth_client):
    assert auth_client.post("/api/vehicles", data={"csrf_token": csrf(auth_client)}).status_code == 405
