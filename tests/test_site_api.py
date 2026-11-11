"""Tests for the hosted (Vercel) API, built into a temp folder exactly as it is deployed."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_site

    out = tmp_path_factory.mktemp("web")
    assert build_site.main(out=out, replays=False) == 0
    spec = importlib.util.spec_from_file_location("site_api_under_test", out / "api" / "index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return out, TestClient(mod.app)


def test_build_contains_pages_api_package_and_policies(site):
    out, _ = site
    for rel in ("index.html", "app.html", "docs.html", "404.html", "vercel.json", "requirements.txt",
                "api/index.py", "breakpoint/scenario.py", "policies/policy_v2field.npz", "reports/final.html",
                "assets/three.min.js", "assets/img/world-rubble.jpg"):
        assert (out / rel).is_file(), rel
    assert not (out / "breakpoint" / "server.py").exists()  # websocket server is local-only
    cfg = json.loads((out / "vercel.json").read_text())
    assert "api/index.py" in cfg["functions"]


def test_listing_endpoints(site):
    _, c = site
    names = [p["name"] for p in c.get("/api/policies").json()["policies"]]
    assert names[0] == "gated-v2field" and "cpg" in names
    assert {p["name"] for p in c.get("/api/profiles").json()["profiles"]} >= {"warehouse", "teleop"}
    assert any(a["key"] == "push_left" for a in c.get("/api/axes").json()["axes"])


def test_simulate_caps_duration_and_explains(site):
    _, c = site
    r = c.post("/api/simulate", json={"policy": "gated-v2field", "scenario": {
        "duration_s": 30, "pushes": [{"t": 1.0, "force": 40, "direction_deg": -90, "duration": 0.1}]}})
    assert r.status_code == 200
    d = r.json()
    assert d["scenario"]["duration_s"] == 10.0
    assert d["explanation"] and d["frames"] and d["terrain"]["rows"] > 0


def test_errors_are_actionable(site):
    _, c = site
    r = c.post("/api/simulate", json={"policy": "nope", "scenario": {}})
    assert r.status_code == 404 and "Choose one of" in r.json()["detail"]
    assert c.post("/api/deploy", json={"profile": "moon"}).status_code == 404
    assert c.post("/api/deploy", json={"profile": "home", "trials": 999}).status_code == 422
    assert c.post("/api/probe", json={"axis": "gravity"}).status_code == 404
    assert c.post("/api/compile_image", json={"image_b64": "!!"}).status_code == 400


def test_probe_matches_cli_settings(site):
    _, c = site
    r = c.post("/api/probe", json={"policy": "cpg", "axis": "payload_kg"}).json()
    assert r["breaking_point"] == pytest.approx(3.25) and r["unit"] == "kg"


def test_compile_and_deploy(site):
    _, c = site
    s = c.post("/api/compile", json={"text": "icy ramp, 10 degrees"}).json()["scenario"]
    assert s["terrain"] == "slope" and s["friction"] < 0.2
    d = c.post("/api/deploy", json={"policy": "cpg", "profile": "home", "trials": 2}).json()
    assert d["trials"] == 2 and 0 <= d["failure_rate_upper_95"] <= 1
