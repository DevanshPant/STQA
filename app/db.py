import csv
import sqlite3
from pathlib import Path

import click
from flask import current_app, g
from flask.cli import with_appcontext


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def load_vehicles(db, csv_path):
    """Read the EV dataset into the vehicles table. Returns rows inserted."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for rec in csv.DictReader(fh):
            # a few makes in the source file have trailing spaces ("Tesla ")
            rows.append((
                int(rec["Model year"]),
                rec["Make"].strip(),
                rec["Model"].strip(),
                rec["Vehicle class"].strip(),
                int(float(rec["Motor (kW)"])),
                float(rec["Recharge time (h)"]),
                round(float(rec["Energy Efficiency (km/kWh)"]), 3),
            ))
    db.executemany(
        "INSERT INTO vehicles (model_year, make, model, vehicle_class, motor_kw,"
        " recharge_hours, km_per_kwh) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    db.commit()
    return len(rows)


def init_db():
    db = get_db()
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    db.executescript(schema)
    return load_vehicles(db, current_app.config["DATASET"])


def ensure_db():
    """Create the database on first run so `flask run` just works."""
    path = Path(current_app.config["DATABASE"])
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    init_db()


@click.command("init-db")
@with_appcontext
def init_db_command():
    """Wipe the database and reload the vehicle data."""
    n = init_db()
    click.echo(f"Database reset. Loaded {n} vehicles.")


def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
