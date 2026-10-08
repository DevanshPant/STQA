from flask import (Blueprint, abort, current_app, flash, g, redirect,
                   render_template, request, url_for)

from . import queries
from .db import get_db
from .security import login_required, safe_next

bp = Blueprint("views", __name__)


@bp.route("/")
def index():
    return redirect(url_for("views.dashboard" if g.user else "auth.login"))


@bp.route("/dashboard")
@login_required
def dashboard():
    data = queries.overview()
    db = get_db()
    saved = db.execute(
        "SELECT v.* FROM favourites f JOIN vehicles v ON v.id = f.vehicle_id"
        " WHERE f.user_id = ? ORDER BY f.created_at DESC LIMIT 4",
        (g.user["id"],),
    ).fetchall()
    recent_trips = db.execute(
        "SELECT t.*, v.make, v.model FROM trips t JOIN vehicles v ON v.id = t.vehicle_id"
        " WHERE t.user_id = ? ORDER BY t.id DESC LIMIT 4",
        (g.user["id"],),
    ).fetchall()
    peak = max((r["avg_eff"] for r in data["by_year"]), default=1)
    return render_template("dashboard.html", **data, saved=saved,
                           recent_trips=recent_trips, peak=peak)


@bp.route("/vehicles")
@login_required
def vehicles():
    try:
        f = queries.parse_filters(request.args)
    except queries.FilterError:
        flash("Some filters weren't valid numbers, so they were cleared.", "warn")
        return redirect(url_for("views.vehicles"))
    rows, total, page, pages = queries.search_vehicles(f, current_app.config["PER_PAGE"])
    saved_ids = {r[0] for r in get_db().execute(
        "SELECT vehicle_id FROM favourites WHERE user_id = ?", (g.user["id"],))}
    args = {k: v for k, v in request.args.items() if k != "page" and v}
    return render_template("vehicles.html", rows=rows, total=total, page=page, pages=pages,
                           f=f, opts=queries.filter_options(), saved_ids=saved_ids, args=args)


@bp.route("/vehicles/<int:vehicle_id>")
@login_required
def vehicle(vehicle_id):
    v = queries.get_vehicle(vehicle_id)
    if v is None:
        abort(404)
    avg = queries.class_average(v["vehicle_class"], v["model_year"])
    is_saved = get_db().execute(
        "SELECT 1 FROM favourites WHERE user_id = ? AND vehicle_id = ?",
        (g.user["id"], vehicle_id),
    ).fetchone() is not None
    per100 = queries.trip_estimate(v["km_per_kwh"], 100, g.user["price_per_kwh"])
    others = get_db().execute(
        "SELECT id, model_year, km_per_kwh FROM vehicles"
        " WHERE make = ? AND model = ? AND id != ? ORDER BY model_year DESC LIMIT 8",
        (v["make"], v["model"], vehicle_id),
    ).fetchall()
    best = get_db().execute("SELECT MAX(km_per_kwh) FROM vehicles").fetchone()[0]
    return render_template("vehicle.html", v=v, avg=avg, best=best, is_saved=is_saved,
                           per100=per100, others=others)


@bp.route("/vehicles/<int:vehicle_id>/save", methods=["POST"])
@login_required
def toggle_save(vehicle_id):
    if queries.get_vehicle(vehicle_id) is None:
        abort(404)
    db = get_db()
    removed = db.execute("DELETE FROM favourites WHERE user_id = ? AND vehicle_id = ?",
                         (g.user["id"], vehicle_id)).rowcount
    if not removed:
        db.execute("INSERT INTO favourites (user_id, vehicle_id) VALUES (?, ?)",
                   (g.user["id"], vehicle_id))
    db.commit()
    flash("Removed from your list." if removed else "Saved to your list.", "ok")
    return redirect(safe_next(request.form.get("next")) or url_for("views.vehicle", vehicle_id=vehicle_id))


@bp.route("/saved")
@login_required
def saved():
    rows = get_db().execute(
        "SELECT v.*, f.created_at AS saved_at FROM favourites f"
        " JOIN vehicles v ON v.id = f.vehicle_id WHERE f.user_id = ?"
        " ORDER BY v.km_per_kwh DESC",
        (g.user["id"],),
    ).fetchall()
    return render_template("saved.html", rows=rows)


def _parse_trip(form):
    errors = {}
    try:
        vehicle_id = int(form.get("vehicle_id", ""))
    except ValueError:
        vehicle_id = None
    v = queries.get_vehicle(vehicle_id) if vehicle_id else None
    if v is None:
        errors["vehicle_id"] = "Pick a vehicle."

    def number(name, lo, hi, msg):
        try:
            val = float(form.get(name, ""))
            if lo < val <= hi:
                return val
        except ValueError:
            pass
        errors[name] = msg

    distance = number("distance_km", 0, 5000, "Distance must be between 0 and 5000 km.")
    price = number("price_per_kwh", 0, 100, "Price must be between 0 and 100.")
    label = form.get("label", "").strip()
    if len(label) > 60:
        errors["label"] = "Keep the label under 60 characters."
    return v, distance, price, label, errors


@bp.route("/trips", methods=["GET", "POST"])
@login_required
def trips():
    db = get_db()
    errors = {}
    if request.method == "POST":
        v, distance, price, label, errors = _parse_trip(request.form)
        if not errors:
            energy, cost = queries.trip_estimate(v["km_per_kwh"], distance, price)
            db.execute(
                "INSERT INTO trips (user_id, vehicle_id, label, distance_km, price_per_kwh,"
                " energy_kwh, cost) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (g.user["id"], v["id"], label or f"{distance:g} km in the {v['model']}",
                 distance, price, energy, cost),
            )
            db.commit()
            flash(f"Trip logged: {energy} kWh, about ₹{cost:,.2f}.", "ok")
            return redirect(url_for("views.trips"))

    rows = db.execute(
        "SELECT t.*, v.make, v.model, v.model_year FROM trips t"
        " JOIN vehicles v ON v.id = t.vehicle_id WHERE t.user_id = ? ORDER BY t.id DESC",
        (g.user["id"],),
    ).fetchall()
    totals = db.execute(
        "SELECT COALESCE(SUM(distance_km), 0) AS km, COALESCE(SUM(energy_kwh), 0) AS kwh,"
        " COALESCE(SUM(cost), 0) AS cost FROM trips WHERE user_id = ?",
        (g.user["id"],),
    ).fetchone()
    picks = db.execute(
        "SELECT v.id, v.make, v.model, v.model_year FROM favourites f"
        " JOIN vehicles v ON v.id = f.vehicle_id WHERE f.user_id = ? ORDER BY v.make, v.model",
        (g.user["id"],),
    ).fetchall()
    preselect = request.form.get("vehicle_id") or request.args.get("vehicle", "")
    if preselect and preselect.isdigit() and not any(p["id"] == int(preselect) for p in picks):
        extra = queries.get_vehicle(int(preselect))
        if extra:
            picks = [extra] + list(picks)
    latest = db.execute(
        "SELECT id, make, model, model_year FROM vehicles"
        " WHERE model_year = (SELECT MAX(model_year) FROM vehicles) ORDER BY make, model"
    ).fetchall()
    return render_template("trips.html", rows=rows, totals=totals, picks=picks, latest=latest,
                           errors=errors, form=request.form, preselect=preselect), (400 if errors else 200)


@bp.route("/trips/<int:trip_id>/delete", methods=["POST"])
@login_required
def delete_trip(trip_id):
    db = get_db()
    # scoped to the owner, so someone else's trip id just 404s
    gone = db.execute("DELETE FROM trips WHERE id = ? AND user_id = ?",
                      (trip_id, g.user["id"])).rowcount
    db.commit()
    if not gone:
        abort(404)
    flash("Trip removed.", "ok")
    return redirect(url_for("views.trips"))
