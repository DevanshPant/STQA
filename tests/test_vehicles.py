import pytest

from conftest import csrf, query


def test_dataset_loaded(app):
    assert query(app, "SELECT COUNT(*) FROM vehicles")[0][0] == 1197


def test_make_names_are_trimmed(app):
    makes = [r[0] for r in query(app, "SELECT DISTINCT make FROM vehicles")]
    assert all(m == m.strip() for m in makes)
    assert len(makes) == 35


def test_dashboard(auth_client):
    r = auth_client.get("/dashboard")
    assert r.status_code == 200
    assert b"Hi Ananya." in r.data
    assert b"1,197" in r.data
    assert b"Most efficient, 2026" in r.data


def test_vehicle_list_default(auth_client):
    r = auth_client.get("/vehicles")
    assert r.status_code == 200
    assert b"1,197 matches" in r.data
    assert b"Page 1 of 48" in r.data


@pytest.mark.parametrize("params,expected_sql,args", [
    ("make=Tesla", "make = ?", ("Tesla",)),
    ("vclass=Two-seater", "vehicle_class = ?", ("Two-seater",)),
    ("year=2020", "model_year = ?", (2020,)),
    ("min_eff=6", "km_per_kwh >= ?", (6,)),
    ("make=Hyundai&year=2024", "make = ? AND model_year = ?", ("Hyundai", 2024)),
])
def test_filters_match_database(auth_client, app, params, expected_sql, args):
    expected = query(app, f"SELECT COUNT(*) FROM vehicles WHERE {expected_sql}", args)[0][0]
    data = auth_client.get(f"/api/vehicles?{params}&per_page=100").get_json()
    assert data["total"] == expected
    html = auth_client.get(f"/vehicles?{params}").data.decode()
    assert f"{expected:,} match" in html


def test_search_matches_make_or_model(auth_client):
    data = auth_client.get("/api/vehicles?q=leaf&per_page=100").get_json()
    assert data["total"] > 0
    assert all("leaf" in (v["make"] + v["model"]).lower() for v in data["results"])


@pytest.mark.parametrize("sort,key,reverse", [
    ("efficiency", "km_per_kwh", True),
    ("efficiency_asc", "km_per_kwh", False),
    ("newest", "year", True),
    ("oldest", "year", False),
    ("power", "motor_kw", True),
    ("charge", "recharge_hours", False),
])
def test_sorting(auth_client, sort, key, reverse):
    res = auth_client.get(f"/api/vehicles?sort={sort}&per_page=100").get_json()["results"]
    values = [v[key] for v in res]
    assert values == sorted(values, reverse=reverse)


def test_pagination(auth_client):
    p1 = auth_client.get("/api/vehicles?page=1").get_json()
    p2 = auth_client.get("/api/vehicles?page=2").get_json()
    assert len(p1["results"]) == 25
    assert {v["id"] for v in p1["results"]}.isdisjoint({v["id"] for v in p2["results"]})


def test_page_past_end_is_clamped(auth_client):
    data = auth_client.get("/api/vehicles?page=999").get_json()
    assert data["page"] == data["pages"] == 48
    assert len(data["results"]) == 1197 - 47 * 25


def test_bad_filter_values_redirect_with_message(auth_client):
    r = auth_client.get("/vehicles?year=abc")
    assert r.status_code == 302
    r = auth_client.get("/vehicles?min_eff=-1", follow_redirects=True)
    assert b"weren&#39;t valid numbers" in r.data


def test_no_results(auth_client):
    r = auth_client.get("/vehicles?q=zzzzzz")
    assert b"No vehicles match" in r.data


def test_vehicle_detail(auth_client):
    r = auth_client.get("/vehicles/2")
    assert r.status_code == 200
    assert b"Nissan LEAF" in r.data
    assert b"Against its class" in r.data


def test_vehicle_detail_cost_per_100km(auth_client, app):
    v = query(app, "SELECT km_per_kwh FROM vehicles WHERE id = 2")[0][0]
    cost = round(100 / v * 8.0, 2)
    assert f"₹{cost:,.2f}".encode() in auth_client.get("/vehicles/2").data


def test_vehicle_not_found(auth_client):
    r = auth_client.get("/vehicles/99999")
    assert r.status_code == 404
    assert b"doesn't exist" in r.data


def test_save_and_unsave(auth_client, app):
    token = csrf(auth_client)
    r = auth_client.post("/vehicles/10/save", data={"csrf_token": token}, follow_redirects=True)
    assert b"Saved to your list" in r.data
    assert len(query(app, "SELECT * FROM favourites")) == 1
    assert b"Saved" in auth_client.get("/saved").data

    r = auth_client.post("/vehicles/10/save", data={"csrf_token": token}, follow_redirects=True)
    assert b"Removed from your list" in r.data
    assert query(app, "SELECT * FROM favourites") == []


def test_save_returns_to_safe_next_only(auth_client):
    token = csrf(auth_client)
    r = auth_client.post("/vehicles/10/save", data={"csrf_token": token, "next": "/vehicles?page=3"})
    assert r.headers["Location"].endswith("/vehicles?page=3")
    r = auth_client.post("/vehicles/10/save", data={"csrf_token": token, "next": "https://evil.example"})
    assert r.headers["Location"].endswith("/vehicles/10")


def test_save_missing_vehicle(auth_client):
    assert auth_client.post("/vehicles/99999/save", data={"csrf_token": csrf(auth_client)}).status_code == 404


def test_saved_page_empty(auth_client):
    assert b"haven't saved anything" in auth_client.get("/saved").data


# ---------- trips ----------

def add_trip(client, **data):
    payload = {"csrf_token": csrf(client), "vehicle_id": 2, "distance_km": 150,
               "price_per_kwh": 8, "label": ""}
    payload.update(data)
    return client.post("/trips", data=payload)


def test_log_trip_calculates_cost(auth_client, app):
    eff = query(app, "SELECT km_per_kwh FROM vehicles WHERE id = 2")[0][0]
    r = add_trip(auth_client, distance_km=150, price_per_kwh=8)
    assert r.status_code == 302
    t = query(app, "SELECT * FROM trips")[0]
    assert t["energy_kwh"] == round(150 / eff, 2)
    assert t["cost"] == round(150 / eff * 8, 2)
    assert t["label"] == "150 km in the LEAF"


def test_trip_custom_label_and_totals(auth_client):
    add_trip(auth_client, label="Pune to Mumbai", distance_km=150)
    add_trip(auth_client, label="Office", distance_km=50)
    r = auth_client.get("/trips")
    assert b"Pune to Mumbai" in r.data and b"Office" in r.data
    assert b"200<small> km</small>" in r.data


@pytest.mark.parametrize("field,value,message", [
    ("vehicle_id", "", b"Pick a vehicle"),
    ("vehicle_id", "99999", b"Pick a vehicle"),
    ("vehicle_id", "abc", b"Pick a vehicle"),
    ("distance_km", "0", b"between 0 and 5000"),
    ("distance_km", "-10", b"between 0 and 5000"),
    ("distance_km", "6000", b"between 0 and 5000"),
    ("distance_km", "far", b"between 0 and 5000"),
    ("price_per_kwh", "0", b"between 0 and 100"),
    ("price_per_kwh", "inf", b"between 0 and 100"),
    ("label", "x" * 61, b"under 60"),
])
def test_trip_validation(auth_client, app, field, value, message):
    r = add_trip(auth_client, **{field: value})
    assert r.status_code == 400
    assert message in r.data
    assert query(app, "SELECT * FROM trips") == []


def test_delete_trip(auth_client, app):
    add_trip(auth_client)
    trip_id = query(app, "SELECT id FROM trips")[0][0]
    r = auth_client.post(f"/trips/{trip_id}/delete", data={"csrf_token": csrf(auth_client)})
    assert r.status_code == 302
    assert query(app, "SELECT * FROM trips") == []


def test_delete_missing_trip(auth_client):
    assert auth_client.post("/trips/12345/delete", data={"csrf_token": csrf(auth_client)}).status_code == 404


def test_trip_page_preselects_vehicle(auth_client):
    r = auth_client.get("/trips?vehicle=2")
    assert b'value="2" selected' in r.data
