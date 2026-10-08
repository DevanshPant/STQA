import functools
import hmac
import re
import secrets
from urllib.parse import urlparse

from flask import abort, g, jsonify, redirect, request, session, url_for

from .db import get_db

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


# --- CSRF -----------------------------------------------------------------

def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_urlsafe(32)
    return session["_csrf"]


def check_csrf():
    if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
        return
    sent = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
    expected = session.get("_csrf", "")
    if not expected or not hmac.compare_digest(sent, expected):
        abort(400, description="Your form expired. Go back, refresh and try again.")


# --- current user ---------------------------------------------------------

def load_user():
    g.user = None
    uid = session.get("uid")
    if uid is None:
        return
    user = get_db().execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    # session_version gets bumped on password change / logout-everywhere,
    # which kills any cookie issued before that.
    if user is None or user["session_version"] != session.get("sv"):
        session.clear()
        return
    g.user = user


def start_session(user, remember=False):
    session.clear()  # drop anything set before login (fixation)
    session["uid"] = user["id"]
    session["sv"] = user["session_version"]
    session.permanent = bool(remember)
    csrf_token()


def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))
        return view(*args, **kwargs)
    return wrapped


def api_login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            return jsonify(error="authentication required"), 401
        return view(*args, **kwargs)
    return wrapped


def safe_next(target):
    """Only allow redirects back into this site."""
    if not target:
        return None
    parts = urlparse(target)
    if parts.scheme or parts.netloc or not target.startswith("/") or target.startswith("//"):
        return None
    if "\\" in target:
        return None
    return target


# --- validation -----------------------------------------------------------

def validate_email(email):
    if not email:
        return "Email is required."
    if len(email) > 254 or not EMAIL_RE.match(email):
        return "That doesn't look like a valid email address."
    return None


def validate_password(pw):
    if len(pw) < 8:
        return "Password must be at least 8 characters."
    if len(pw) > 128:
        return "Password can't be longer than 128 characters."
    if not re.search(r"[A-Za-z]", pw) or not re.search(r"\d", pw):
        return "Password needs at least one letter and one number."
    return None


def validate_name(name):
    if not name:
        return "Name is required."
    if len(name) > 60:
        return "Name can't be longer than 60 characters."
    return None


def set_security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    resp.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; style-src 'self' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; img-src 'self' data:; frame-ancestors 'none'",
    )
    if g.get("user") is not None:
        resp.headers["Cache-Control"] = "no-store"
    return resp
