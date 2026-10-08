"""Read-side SQL for the vehicle data, shared by the HTML views and the API."""
import math

from .db import get_db

SORTS = {
    "efficiency": "km_per_kwh DESC, model_year DESC",
    "efficiency_asc": "km_per_kwh ASC, model_year DESC",
    "newest": "model_year DESC, make, model",
    "oldest": "model_year ASC, make, model",
    "power": "motor_kw DESC, km_per_kwh DESC",
    "charge": "recharge_hours ASC, km_per_kwh DESC",
    "name": "make, model, model_year DESC",
}


class FilterError(ValueError):
    pass


def parse_filters(args):
    """Turn query-string args into a clean filter dict. Bad numbers raise FilterError."""
    f = {
        "q": (args.get("q") or "").strip()[:80],
        "make": (args.get("make") or "").strip(),
        "vclass": (args.get("vclass") or "").strip(),
        "year": None,
        "min_eff": None,
        "sort": args.get("sort") if args.get("sort") in SORTS else "efficiency",
        "page": 1,
    }
    try:
        if args.get("year"):
            f["year"] = int(args["year"])
        if args.get("min_eff"):
            f["min_eff"] = float(args["min_eff"])
            if not math.isfinite(f["min_eff"]) or f["min_eff"] < 0:
                raise ValueError
        if args.get("page"):
            f["page"] = max(1, int(args["page"]))
    except (TypeError, ValueError):
        raise FilterError("year, min_eff and page must be valid positive numbers")
    return f


def search_vehicles(f, per_page=25):
    where, params = [], []
    if f["q"]:
        where.append("(make LIKE ? OR model LIKE ?)")
        params += [f"%{f['q']}%", f"%{f['q']}%"]
    if f["make"]:
        where.append("make = ?")
        params.append(f["make"])
    if f["vclass"]:
        where.append("vehicle_class = ?")
        params.append(f["vclass"])
    if f["year"] is not None:
        where.append("model_year = ?")
        params.append(f["year"])
    if f["min_eff"] is not None:
        where.append("km_per_kwh >= ?")
        params.append(f["min_eff"])

    clause = ("WHERE " + " AND ".join(where)) if where else ""
    db = get_db()
    total = db.execute(f"SELECT COUNT(*) FROM vehicles {clause}", params).fetchone()[0]
    pages = max(1, -(-total // per_page))
    page = min(f["page"], pages)
    rows = db.execute(
        f"SELECT * FROM vehicles {clause} ORDER BY {SORTS[f['sort']]} LIMIT ? OFFSET ?",
        params + [per_page, (page - 1) * per_page],
    ).fetchall()
    return rows, total, page, pages


def filter_options():
    db = get_db()
    return {
        "makes": [r[0] for r in db.execute("SELECT DISTINCT make FROM vehicles ORDER BY make COLLATE NOCASE")],
        "classes": [r[0] for r in db.execute("SELECT DISTINCT vehicle_class FROM vehicles ORDER BY vehicle_class")],
        "years": [r[0] for r in db.execute("SELECT DISTINCT model_year FROM vehicles ORDER BY model_year DESC")],
    }


def get_vehicle(vehicle_id):
    return get_db().execute("SELECT * FROM vehicles WHERE id = ?", (vehicle_id,)).fetchone()


def overview():
    db = get_db()
    totals = db.execute(
        "SELECT COUNT(*) AS n, COUNT(DISTINCT make) AS makes, AVG(km_per_kwh) AS avg_eff,"
        " MIN(model_year) AS first_year, MAX(model_year) AS last_year FROM vehicles"
    ).fetchone()
    by_year = db.execute(
        "SELECT model_year, COUNT(*) AS n, AVG(km_per_kwh) AS avg_eff"
        " FROM vehicles GROUP BY model_year ORDER BY model_year"
    ).fetchall()
    by_class = db.execute(
        "SELECT vehicle_class, COUNT(*) AS n, AVG(km_per_kwh) AS avg_eff"
        " FROM vehicles GROUP BY vehicle_class ORDER BY avg_eff DESC"
    ).fetchall()
    top_latest = db.execute(
        "SELECT * FROM vehicles WHERE model_year = ? ORDER BY km_per_kwh DESC LIMIT 5",
        (totals["last_year"],),
    ).fetchall()
    return {"totals": totals, "by_year": by_year, "by_class": by_class, "top_latest": top_latest}


def class_average(vehicle_class, year):
    return get_db().execute(
        "SELECT AVG(km_per_kwh) FROM vehicles WHERE vehicle_class = ? AND model_year = ?",
        (vehicle_class, year),
    ).fetchone()[0]


def trip_estimate(km_per_kwh, distance_km, price_per_kwh):
    energy = distance_km / km_per_kwh
    return round(energy, 2), round(energy * price_per_kwh, 2)
