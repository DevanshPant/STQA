"""WSGI entry point for production hosts (PythonAnywhere, gunicorn, etc.)."""
import os

# the live site is served over HTTPS, so only send the session cookie over HTTPS
os.environ.setdefault("COOKIE_SECURE", "1")

from app import create_app  # noqa: E402

application = app = create_app()
