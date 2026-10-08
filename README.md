# Voltwise

A small Flask web app built on the *EV Energy Efficiency* dataset (1,197 electric
vehicles, model years 2012–2026). Made for the STQA mini project, so it comes with a
full automated test suite.

## What it does

- **Accounts**: register, log in, log out, "keep me logged in", change password,
  sign out on every device, delete account.
- **Overview**: dataset stats, average efficiency per model year, the most efficient
  cars of the latest year, and averages by vehicle class.
- **Vehicles**: search, filter (make / class / year / minimum km per kWh), sort, paginate.
- **Vehicle page**: specs, cost per 100 km at your electricity price, comparison with
  its class average.
- **Saved**: a personal shortlist.
- **Trip log**: pick a car, enter a distance and price, and it works out the kWh used and
  the cost, then keeps a running total.
- **JSON API** (needs login): `/api/me`, `/api/vehicles`, `/api/vehicles/<id>`,
  `/api/stats`, `/api/estimate`, `/api/saved`.

## Running it

```bash
pip install -r requirements.txt
python run.py
```

Then open http://localhost:5000. The SQLite database is created in `instance/` the
first time you run it, and the CSV in `data/` is loaded into it. To wipe it and start
over:

```bash
flask --app run init-db
```

## Deploying (PythonAnywhere)

1. In a PythonAnywhere **Bash console**:
   ```bash
   git clone https://github.com/DevanshPant/STQA.git
   pip install --user flask
   ```
2. **Web** tab → *Add a new web app* → *Manual configuration* → latest Python.
3. Set **Source code** and **Working directory** to `/home/<username>/STQA`.
4. Open the **WSGI configuration file** link, delete everything in it and put:
   ```python
   import sys
   sys.path.insert(0, "/home/<username>/STQA")
   from wsgi import application
   ```
5. Turn on **Force HTTPS**, then hit **Reload**.

To update the live site later: `cd ~/STQA && git pull`, then Reload.

## Running the tests

```bash
python -m pytest --cov=app --cov-report=term-missing
```

Latest run: **184 passed, 0 failed, 98% line coverage** (reports are in `reports/`:
`test-output.txt`, `junit.xml`, and an HTML coverage report in `reports/coverage/index.html`).

| File | Tests | What it covers |
|------|------:|----------------|
| `test_auth.py` | 33 | registration, input validation, login/logout, lockout, remember-me, redirect after login |
| `test_security.py` | 32 | access control, CSRF, session fixation, session invalidation, XSS, SQL injection, users' data kept separate |
| `test_vehicles.py` | 42 | dataset loading, filters, sorting, pagination, saving, trip calculations and validation |
| `test_api.py` | 22 | JSON responses, parameter validation, 401/404 handling |
| `test_account.py` | 17 | profile update, password change, account deletion (and cascade) |
| `test_units.py` | 38 | validators, redirect checks, filter parsing, cost formula, boundary values |

Testing techniques used: equivalence partitioning and boundary value analysis (password
length 7/8/128/129, price 0/100/101, distance 0/5000/6000), negative testing, security
testing, and state-based tests for the lockout.

## Security notes

- Passwords hashed with scrypt (Werkzeug), never stored or echoed in plain text.
- Signed, HttpOnly, SameSite=Lax session cookie. The session is cleared on login, which
  prevents session fixation.
- A per-user `session_version` is bumped on password change / "sign out everywhere",
  which invalidates old cookies.
- CSRF token on every POST form; logout is POST-only.
- 5 wrong passwords lock the account for 15 minutes. "Wrong email" and "wrong password"
  give the same message so the form can't be used to discover accounts.
- `?next=` redirects only allow paths on this site (no open redirects).
- All SQL is parameterised; sort keys come from a whitelist.
- CSP, X-Frame-Options, nosniff and no-store headers on every response.

## Layout

```
app/
  __init__.py   app factory, config, error pages
  auth.py       register / login / logout / account
  views.py      dashboard, vehicles, saved, trips
  api.py        JSON endpoints
  queries.py    shared SQL for vehicle data
  security.py   CSRF, login guard, validators, headers
  db.py         SQLite connection + CSV loader
  schema.sql
  templates/ static/
data/ev_efficiency.csv
tests/
```

## Demo account

Used for the manual browser check on localhost: `demo@voltwise.test` / `Blush2026demo`.
It only exists in your local `instance/voltwise.sqlite3`. Delete that file to reset it.
