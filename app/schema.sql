DROP TABLE IF EXISTS trips;
DROP TABLE IF EXISTS favourites;
DROP TABLE IF EXISTS vehicles;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    email           TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash   TEXT    NOT NULL,
    price_per_kwh   REAL    NOT NULL DEFAULT 8.0,
    failed_logins   INTEGER NOT NULL DEFAULT 0,
    locked_until    TEXT,
    session_version INTEGER NOT NULL DEFAULT 1,
    last_login_at   TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE vehicles (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    model_year     INTEGER NOT NULL,
    make           TEXT    NOT NULL,
    model          TEXT    NOT NULL,
    vehicle_class  TEXT    NOT NULL,
    motor_kw       INTEGER NOT NULL,
    recharge_hours REAL    NOT NULL,
    km_per_kwh     REAL    NOT NULL
);

CREATE INDEX idx_vehicles_make ON vehicles(make);
CREATE INDEX idx_vehicles_year ON vehicles(model_year);

CREATE TABLE favourites (
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    vehicle_id INTEGER NOT NULL REFERENCES vehicles(id) ON DELETE CASCADE,
    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, vehicle_id)
);

CREATE TABLE trips (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    vehicle_id    INTEGER NOT NULL REFERENCES vehicles(id) ON DELETE CASCADE,
    label         TEXT    NOT NULL,
    distance_km   REAL    NOT NULL,
    price_per_kwh REAL    NOT NULL,
    energy_kwh    REAL    NOT NULL,
    cost          REAL    NOT NULL,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);
