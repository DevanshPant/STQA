import os
import secrets
from datetime import timedelta
from pathlib import Path

from flask import Flask, render_template

from . import db
from .security import check_csrf, csrf_token, load_user, set_security_headers

BASE_DIR = Path(__file__).resolve().parent.parent


def _secret_key(instance_path):
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    # keep the key stable across restarts so people stay logged in
    key_file = Path(instance_path) / "secret_key"
    if key_file.exists():
        return key_file.read_text().strip()
    key = secrets.token_hex(32)
    key_file.write_text(key)
    return key


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    app.config.update(
        DATABASE=str(Path(app.instance_path) / "voltwise.sqlite3"),
        DATASET=str(BASE_DIR / "data" / "ev_efficiency.csv"),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE") == "1",
        PERMANENT_SESSION_LIFETIME=timedelta(days=14),
        MAX_LOGIN_ATTEMPTS=5,
        LOCKOUT_MINUTES=15,
        PER_PAGE=25,
        PASSWORD_HASH_METHOD="scrypt",
    )
    if test_config:
        app.config.update(test_config)
    if "SECRET_KEY" not in (test_config or {}):
        app.config["SECRET_KEY"] = _secret_key(app.instance_path)

    db.init_app(app)

    from . import api, auth, views
    app.register_blueprint(auth.bp)
    app.register_blueprint(views.bp)
    app.register_blueprint(api.bp)

    @app.before_request
    def _before():
        load_user()
        check_csrf()

    app.after_request(set_security_headers)
    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.template_filter("money")
    def money(value):
        return f"₹{value:,.2f}"

    for code in (400, 403, 404, 405, 500):
        app.register_error_handler(
            code,
            lambda e, code=code: (render_template("errors/error.html", error=e, code=code), code),
        )

    if not app.config.get("TESTING"):
        with app.app_context():
            db.ensure_db()

    return app
