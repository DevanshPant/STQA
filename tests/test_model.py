import json
import re
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent / "app" / "static" / "ml"


def test_model_page_needs_login(client):
    r = client.get("/model")
    assert r.status_code == 302
    assert "/login" in r.headers["Location"]


def test_model_page_shows_training_results(auth_client):
    metrics = json.loads((ML_DIR / "metrics.json").read_text(encoding="utf-8"))
    r = auth_client.get("/model")
    assert r.status_code == 200
    html = r.data.decode()
    assert "How the model was trained" in html
    assert metrics["best"] in html
    best = next(x for x in metrics["regression"] if x["name"] == metrics["best"])
    assert f"{best['test']['r2']:.3f}" in html
    assert "tensorflow" in html


def test_every_graph_on_the_page_exists(auth_client):
    html = auth_client.get("/model").data.decode()
    images = set(re.findall(r'src="/static/ml/([^"]+)"', html))
    assert len(images) == 8
    for name in images:
        assert (ML_DIR / name).is_file(), name
        assert auth_client.get(f"/static/ml/{name}").status_code == 200


def test_model_link_in_navbar(auth_client):
    assert b'href="/model"' in auth_client.get("/dashboard").data


def test_model_page_without_results(app, auth_client, tmp_path):
    app.config["ML_DIR"] = str(tmp_path)
    r = auth_client.get("/model")
    assert r.status_code == 200
    assert b"aren't here yet" in r.data
