"""Unit tests for the small pure helpers (no HTTP involved)."""
import pytest

from app.queries import FilterError, parse_filters, trip_estimate
from app.security import safe_next, validate_email, validate_name, validate_password


@pytest.mark.parametrize("pw,ok", [
    ("abcdefg1", True),          # exactly 8, boundary
    ("abcdef1", False),          # 7, just under
    ("A1" + "b" * 126, True),    # 128, upper boundary
    ("A1" + "b" * 127, False),   # 129
    ("abcdefgh", False),
    ("12345678", False),
    ("Pass word 9", True),
])
def test_validate_password(pw, ok):
    assert (validate_password(pw) is None) is ok


@pytest.mark.parametrize("email,ok", [
    ("a@b.co", True),
    ("first.last+tag@uni.ac.in", True),
    ("", False),
    ("plain", False),
    ("a@b", False),
    ("a b@c.com", False),
    ("a@@b.com", False),
    ("x" * 250 + "@a.io", False),
])
def test_validate_email(email, ok):
    assert (validate_email(email) is None) is ok


@pytest.mark.parametrize("name,ok", [("A", True), ("x" * 60, True), ("x" * 61, False), ("", False)])
def test_validate_name(name, ok):
    assert (validate_name(name) is None) is ok


@pytest.mark.parametrize("target,expected", [
    ("/saved", "/saved"),
    ("/vehicles?make=Kia", "/vehicles?make=Kia"),
    ("", None),
    (None, None),
    ("saved", None),
    ("//evil.com", None),
    ("https://evil.com/x", None),
    ("/\\evil.com", None),
])
def test_safe_next(target, expected):
    assert safe_next(target) == expected


def test_trip_estimate():
    assert trip_estimate(5.0, 100, 8) == (20.0, 160.0)
    assert trip_estimate(4.739, 37, 7.5) == (7.81, 58.56)


def test_parse_filters_defaults():
    f = parse_filters({})
    assert f == {"q": "", "make": "", "vclass": "", "year": None, "min_eff": None,
                 "sort": "efficiency", "page": 1}


def test_parse_filters_cleans_input():
    f = parse_filters({"q": "  ioniq  ", "year": "2021", "min_eff": "5.5", "page": "-4", "sort": "bogus"})
    assert f["q"] == "ioniq"
    assert f["year"] == 2021
    assert f["min_eff"] == 5.5
    assert f["page"] == 1
    assert f["sort"] == "efficiency"


def test_parse_filters_truncates_long_search():
    assert len(parse_filters({"q": "a" * 500})["q"]) == 80


@pytest.mark.parametrize("args", [{"year": "abc"}, {"min_eff": "-1"}, {"page": "1.5"}, {"min_eff": "inf"}])
def test_parse_filters_rejects_bad_numbers(args):
    with pytest.raises(FilterError):
        parse_filters(args)


def test_production_config_uses_scrypt(tmp_path):
    from app import create_app
    app = create_app({"SECRET_KEY": "k", "DATABASE": str(tmp_path / "prod.sqlite3")})
    assert app.config["PASSWORD_HASH_METHOD"] == "scrypt"
    assert app.config["SESSION_COOKIE_HTTPONLY"] is True


def test_first_run_creates_database(tmp_path):
    import sqlite3
    from app import create_app
    db_path = tmp_path / "fresh.sqlite3"
    create_app({"SECRET_KEY": "k", "DATABASE": str(db_path)})
    assert sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM vehicles").fetchone()[0] == 1197


def test_init_db_command(app):
    result = app.test_cli_runner().invoke(args=["init-db"])
    assert "Loaded 1197 vehicles" in result.output
