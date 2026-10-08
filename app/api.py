from flask import Blueprint, current_app, g, jsonify, request

from . import queries
from .db import get_db
from .security import api_login_required

bp = Blueprint("api", __name__, url_prefix="/api")


def vehicle_json(row):
    return {
        "id": row["id"],
        "year": row["model_year"],
        "make": row["make"],
        "model": row["model"],
        "class": row["vehicle_class"],
        "motor_kw": row["motor_kw"],
        "recharge_hours": row["recharge_hours"],
        "km_per_kwh": row["km_per_kwh"],
    }


@bp.get("/me")
@api_login_required
def me():
    u = g.user
    return jsonify(id=u["id"], name=u["name"], email=u["email"],
                   price_per_kwh=u["price_per_kwh"], created_at=u["created_at"])


@bp.get("/vehicles")
@api_login_required
def vehicles():
    try:
        f = queries.parse_filters(request.args)
    except queries.FilterError as e:
        return jsonify(error=str(e)), 400
    try:
        per_page = min(100, max(1, int(request.args.get("per_page", current_app.config["PER_PAGE"]))))
    except ValueError:
        return jsonify(error="per_page must be a number"), 400
    rows, total, page, pages = queries.search_vehicles(f, per_page)
    return jsonify(total=total, page=page, pages=pages, per_page=per_page,
                   results=[vehicle_json(r) for r in rows])


@bp.get("/vehicles/<int:vehicle_id>")
@api_login_required
def vehicle(vehicle_id):
    v = queries.get_vehicle(vehicle_id)
    if v is None:
        return jsonify(error="vehicle not found"), 404
    return jsonify(vehicle_json(v))


@bp.get("/stats")
@api_login_required
def stats():
    data = queries.overview()
    t = data["totals"]
    return jsonify(
        vehicles=t["n"],
        makes=t["makes"],
        average_km_per_kwh=round(t["avg_eff"], 3),
        years=[t["first_year"], t["last_year"]],
        by_year=[{"year": r["model_year"], "count": r["n"], "avg_km_per_kwh": round(r["avg_eff"], 3)}
                 for r in data["by_year"]],
    )


@bp.get("/estimate")
@api_login_required
def estimate():
    try:
        vehicle_id = int(request.args["vehicle_id"])
        distance = float(request.args["distance_km"])
        price = float(request.args.get("price_per_kwh", g.user["price_per_kwh"]))
    except (KeyError, ValueError):
        return jsonify(error="vehicle_id and distance_km are required numbers"), 400
    if not (0 < distance <= 5000) or not (0 < price <= 100):
        return jsonify(error="distance_km must be 0-5000 and price_per_kwh 0-100"), 400
    v = queries.get_vehicle(vehicle_id)
    if v is None:
        return jsonify(error="vehicle not found"), 404
    energy, cost = queries.trip_estimate(v["km_per_kwh"], distance, price)
    return jsonify(vehicle=vehicle_json(v), distance_km=distance, price_per_kwh=price,
                   energy_kwh=energy, cost=cost)


@bp.get("/saved")
@api_login_required
def saved():
    rows = get_db().execute(
        "SELECT v.* FROM favourites f JOIN vehicles v ON v.id = f.vehicle_id WHERE f.user_id = ?",
        (g.user["id"],),
    ).fetchall()
    return jsonify(results=[vehicle_json(r) for r in rows])
