import math
from datetime import datetime, timedelta, timezone

from flask import (Blueprint, current_app, flash, g, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from .db import get_db
from .security import (login_required, safe_next, start_session, validate_email,
                       validate_name, validate_password)

bp = Blueprint("auth", __name__)

# used when the email doesn't exist, so a miss takes as long as a wrong password
_DUMMY_HASH = generate_password_hash("not-a-real-password-1")

TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def _hash(password):
    # scrypt by default; the test config swaps in a cheaper method to keep the suite fast
    return generate_password_hash(password, method=current_app.config["PASSWORD_HASH_METHOD"])


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


@bp.route("/register", methods=["GET", "POST"])
def register():
    if g.user:
        return redirect(url_for("views.dashboard"))

    form = {"name": "", "email": ""}
    errors = {}
    if request.method == "POST":
        form["name"] = request.form.get("name", "").strip()
        form["email"] = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if err := validate_name(form["name"]):
            errors["name"] = err
        if err := validate_email(form["email"]):
            errors["email"] = err
        if err := validate_password(password):
            errors["password"] = err
        elif password != confirm:
            errors["confirm"] = "Passwords don't match."

        db = get_db()
        if "email" not in errors and db.execute(
            "SELECT 1 FROM users WHERE email = ?", (form["email"],)
        ).fetchone():
            errors["email"] = "An account with this email already exists."

        if not errors:
            cur = db.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (form["name"], form["email"], _hash(password)),
            )
            db.commit()
            user = db.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
            start_session(user)
            flash(f"Account created. Hi {user['name'].split()[0]}!", "ok")
            return redirect(url_for("views.dashboard"))

    status = 400 if errors else 200
    return render_template("auth/register.html", form=form, errors=errors), status


@bp.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("views.dashboard"))

    email = ""
    error = None
    next_url = safe_next(request.args.get("next") or request.form.get("next"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = request.form.get("remember") == "on"
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        now = _now()

        if user and user["locked_until"] and datetime.strptime(user["locked_until"], TS_FORMAT) > now:
            left = datetime.strptime(user["locked_until"], TS_FORMAT) - now
            mins = max(1, math.ceil(left.total_seconds() / 60))
            error = (f"Too many failed attempts. This account is locked for "
                     f"another {mins} minute{'s' if mins != 1 else ''}.")
        elif user is None:
            check_password_hash(_DUMMY_HASH, password)
            error = "Incorrect email or password."
        elif not check_password_hash(user["password_hash"], password):
            fails = user["failed_logins"] + 1
            if fails >= current_app.config["MAX_LOGIN_ATTEMPTS"]:
                until = now + timedelta(minutes=current_app.config["LOCKOUT_MINUTES"])
                db.execute("UPDATE users SET failed_logins = 0, locked_until = ? WHERE id = ?",
                           (until.strftime(TS_FORMAT), user["id"]))
                error = (f"Too many failed attempts. This account is locked for "
                         f"{current_app.config['LOCKOUT_MINUTES']} minutes.")
            else:
                db.execute("UPDATE users SET failed_logins = ? WHERE id = ?", (fails, user["id"]))
                error = "Incorrect email or password."
            db.commit()
        else:
            db.execute(
                "UPDATE users SET failed_logins = 0, locked_until = NULL, last_login_at = ? WHERE id = ?",
                (now.strftime(TS_FORMAT), user["id"]),
            )
            db.commit()
            start_session(user, remember=remember)
            return redirect(next_url or url_for("views.dashboard"))

    status = 401 if error else 200
    return render_template("auth/login.html", email=email, error=error, next=next_url), status


@bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You've been logged out.", "ok")
    return redirect(url_for("auth.login"))


@bp.route("/account", methods=["GET", "POST"])
@login_required
def account():
    db = get_db()
    errors = {}
    action = request.form.get("action")

    if request.method == "POST" and action == "profile":
        name = request.form.get("name", "").strip()
        price_raw = request.form.get("price_per_kwh", "").strip()
        if err := validate_name(name):
            errors["name"] = err
        try:
            price = float(price_raw)
            if not (0 < price <= 100):
                raise ValueError
        except ValueError:
            errors["price_per_kwh"] = "Enter a price between 0 and 100."
        if not errors:
            db.execute("UPDATE users SET name = ?, price_per_kwh = ? WHERE id = ?",
                       (name, round(price, 2), g.user["id"]))
            db.commit()
            flash("Profile saved.", "ok")
            return redirect(url_for("auth.account"))

    elif request.method == "POST" and action == "password":
        current = request.form.get("current", "")
        new = request.form.get("new", "")
        confirm = request.form.get("confirm", "")
        if not check_password_hash(g.user["password_hash"], current):
            errors["current"] = "Current password is wrong."
        elif err := validate_password(new):
            errors["new"] = err
        elif new != confirm:
            errors["confirm"] = "Passwords don't match."
        elif new == current:
            errors["new"] = "New password must be different from the old one."
        if not errors:
            db.execute(
                "UPDATE users SET password_hash = ?, session_version = session_version + 1 WHERE id = ?",
                (_hash(new), g.user["id"]),
            )
            db.commit()
            # other devices get signed out; keep this one going
            user = db.execute("SELECT * FROM users WHERE id = ?", (g.user["id"],)).fetchone()
            start_session(user, remember=session.permanent)
            flash("Password changed. Other devices have been signed out.", "ok")
            return redirect(url_for("auth.account"))

    elif request.method == "POST" and action == "signout_all":
        db.execute("UPDATE users SET session_version = session_version + 1 WHERE id = ?",
                   (g.user["id"],))
        db.commit()
        session.clear()
        flash("Signed out on every device.", "ok")
        return redirect(url_for("auth.login"))

    elif request.method == "POST" and action == "delete":
        if not check_password_hash(g.user["password_hash"], request.form.get("password", "")):
            errors["delete"] = "Password is wrong, account not deleted."
        else:
            db.execute("DELETE FROM users WHERE id = ?", (g.user["id"],))
            db.commit()
            session.clear()
            flash("Your account and all its data have been deleted.", "ok")
            return redirect(url_for("auth.login"))

    elif request.method == "POST":
        errors["form"] = "Unknown action."

    stats = db.execute(
        "SELECT (SELECT COUNT(*) FROM favourites WHERE user_id = :u) AS saved,"
        "       (SELECT COUNT(*) FROM trips WHERE user_id = :u) AS trips",
        {"u": g.user["id"]},
    ).fetchone()
    status = 400 if errors else 200
    return render_template("account.html", errors=errors, stats=stats, form=request.form), status
