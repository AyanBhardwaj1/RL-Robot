import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from breakpoint.compiler import compile_image, compile_rules, compile_text
from breakpoint.env import OBS_DIM, QuadrupedEnv
from breakpoint.evaluate import failure_upper_bound, rollout
from breakpoint.hunt import decode, shrink
from breakpoint.nn import MLP
from breakpoint.policy import CPGPolicy, ResidualPolicy
from breakpoint.probe import apply_axis, probe_axis
from breakpoint.report import render_html, scorecard
from breakpoint.scenario import Scenario


# ---------------------------------------------------------------- schema
def test_from_dict_clamps_out_of_range_and_garbage():
    s = Scenario.from_dict({"friction": -5, "payload_kg": "lots", "terrain": "lava", "latency_steps": 2.6,
                            "pushes": [{"force": 1e9, "t": "x"}]})
    assert s.friction == 0.05
    assert s.payload_kg == 0.0
    assert s.terrain == "flat"
    assert s.latency_steps == 3.0
    assert s.pushes[0].force == 400.0 and s.pushes[0].t == 2.0


def test_from_dict_rejects_non_objects():
    with pytest.raises(ValueError):
        Scenario.from_dict(["not", "a", "dict"])
    with pytest.raises(ValueError):
        Scenario.from_dict({"pushes": "kick"})


def test_with_returns_new_validated_copy():
    a = Scenario()
    b = a.with_(friction=0.0)
    assert a.friction == 1.0 and b.friction == 0.05


def test_json_roundtrip():
    s = Scenario.from_dict({"terrain": "rough", "roughness": 0.03, "pushes": [{"t": 1, "force": 50}]})
    assert Scenario.from_json(s.to_json()) == s


def test_severity_zero_for_nominal_and_grows():
    assert Scenario().severity() == 0.0
    assert Scenario().with_(friction=0.2).severity() > Scenario().with_(friction=0.6).severity() > 0


# -------------------------------------------------------------- compiler
def test_rules_parse_compound_description():
    c = compile_rules("carrying a 4kg box across an icy lot, kicked from the right at 3s with 120 N")
    s = c.scenario
    assert s.payload_kg == 4.0 and s.friction < 0.2
    assert s.pushes[0].direction_deg == -90.0 and s.pushes[0].force == 120.0 and s.pushes[0].t == 3.0


def test_rules_parse_slope_motor_latency():
    s = compile_rules("downhill 8 degree ramp, worn motors at 55%, 80ms lag").scenario
    assert s.slope_deg == -8.0 and s.motor_strength == 0.55 and s.latency_steps == 4.0


def test_compile_text_falls_back_without_llm(monkeypatch):
    monkeypatch.delenv("BREAKPOINT_LLM_BASE_URL", raising=False)
    assert compile_text("stairs with 7 cm steps").source == "rules"


def test_compile_text_falls_back_when_llm_returns_garbage(monkeypatch):
    monkeypatch.setenv("BREAKPOINT_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("BREAKPOINT_LLM_API_KEY", "k")
    monkeypatch.setenv("BREAKPOINT_LLM_MODEL", "m")
    monkeypatch.setattr("breakpoint.compiler._chat", lambda msgs: "sorry, no json here")
    c = compile_text("icy floor")
    assert c.source == "rules" and c.scenario.friction < 0.2


def test_compile_text_uses_llm_and_clamps(monkeypatch):
    monkeypatch.setenv("BREAKPOINT_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("BREAKPOINT_LLM_API_KEY", "k")
    monkeypatch.setenv("BREAKPOINT_LLM_MODEL", "m")
    monkeypatch.setattr("breakpoint.compiler._chat", lambda msgs: json.dumps({"terrain": "rough", "friction": 99}))
    c = compile_text("anything")
    assert c.source == "llm" and c.scenario.terrain == "rough" and c.scenario.friction == 2.0


def test_llm_base_url_must_be_https(monkeypatch):
    monkeypatch.setenv("BREAKPOINT_LLM_BASE_URL", "http://evil.example/v1")
    monkeypatch.setenv("BREAKPOINT_LLM_API_KEY", "k")
    monkeypatch.setenv("BREAKPOINT_LLM_MODEL", "m")
    with pytest.raises(ValueError):
        compile_text("icy floor")


def _png(color, noise=0.0):
    from io import BytesIO

    from PIL import Image

    rng = np.random.default_rng(0)
    a = np.clip(np.ones((64, 64, 3)) * np.array(color) + rng.normal(0, noise, (64, 64, 3)), 0, 1)
    buf = BytesIO()
    Image.fromarray((a * 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_image_heuristic_detects_grass_and_ice():
    assert compile_image(_png([0.2, 0.6, 0.2], 0.05), use_llm=False).scenario.friction == 0.6
    assert compile_image(_png([0.8, 0.9, 1.0]), use_llm=False).scenario.friction == 0.15
    assert compile_image(_png([0.5, 0.5, 0.5], 0.3), use_llm=False).scenario.terrain == "rough"


# ------------------------------------------------------------ sim + eval
def test_env_obs_shape_and_cpg_walks_forward():
    out, frames = rollout(CPGPolicy(), Scenario(duration_s=4.0), seed=0, record=True)
    assert out.survived and out.distance > 1.0
    assert len(frames) == 200
    env = QuadrupedEnv(Scenario())
    assert env.reset().shape == (OBS_DIM,)


def test_huge_side_push_topples_robot():
    scn = Scenario(duration_s=4.0).with_(pushes=[{"t": 1.0, "force": 400, "direction_deg": 90, "duration": 0.4}])
    out, _ = rollout(CPGPolicy(), scn, seed=0)
    assert out.fell and not out.passed and out.margin() < 0


def test_failure_upper_bound_rule_of_three():
    assert abs(failure_upper_bound(100, 0) - 0.0295) < 1e-3
    assert failure_upper_bound(0) == 1.0
    assert 0.1 < failure_upper_bound(20, 2) < 0.35


# ------------------------------------------------------------------ nn
def test_mlp_backward_matches_finite_difference():
    rng = np.random.default_rng(0)
    net = MLP([5, 7, 3], rng)
    x = rng.standard_normal((4, 5))
    g_out = rng.standard_normal((4, 3))
    y, acts = net.forward(x)
    grads = net.backward(acts, g_out)
    eps = 1e-6
    w = net.W[0]
    w[1, 2] += eps
    up = float(np.sum(net(x) * g_out))
    w[1, 2] -= 2 * eps
    down = float(np.sum(net(x) * g_out))
    w[1, 2] += eps
    assert abs((up - down) / (2 * eps) - grads[0][1, 2]) < 1e-5


def test_policy_save_load_roundtrip(tmp_path):
    p = ResidualPolicy(seed=3, name="t")
    p.norm.update(np.random.default_rng(0).standard_normal((10, OBS_DIM)))
    p.save(tmp_path / "p.npz")
    q = ResidualPolicy.load(tmp_path / "p.npz")
    o = np.ones(OBS_DIM)
    assert np.allclose(p.act(o), q.act(o)) and q.name == "t"


# ------------------------------------------------------ probe + hunt
def test_probe_axis_finds_a_push_breaking_point():
    r = probe_axis((None, Scenario().to_dict(), "push_left", [0], 4))
    assert 0.0 < r["breaking_point"] < 40.0 and not r["fails_nominal"]


def test_apply_axis_builds_expected_world():
    assert apply_axis(Scenario(), "slope_down", 5).slope_deg == -5
    assert apply_axis(Scenario(), "push_rear", 10).pushes[0].impulse == pytest.approx(10)


def test_decode_maps_unit_cube_to_valid_scenarios():
    lo = decode(np.zeros(11))
    hi = decode(np.ones(11))
    assert lo.friction == 1.0 and hi.friction == pytest.approx(0.1)
    assert hi.pushes and hi.pushes[0].impulse == pytest.approx(30.0)


def test_shrink_returns_milder_failure():
    bad = Scenario(duration_s=4.0).with_(pushes=[{"t": 1.0, "force": 400, "direction_deg": 90, "duration": 0.1}])
    minimal, alpha = shrink(None, bad, seed=0, iters=4)
    assert alpha < 1.0
    assert minimal.pushes[0].force < 400
    assert not rollout(CPGPolicy(), minimal, seed=0)[0].passed


# ---------------------------------------------------------------- report
def test_scorecard_and_html():
    probe = [{"key": "push_left", "label": "Push from left", "unit": "N·s", "breaking_point": 5, "fraction": 0.125},
             {"key": "friction", "label": "Low friction", "unit": "μ", "breaking_point": 0.2, "fraction": 0.8}]
    nominal = [{"passed": True}] * 9 + [{"passed": False}]
    card = scorecard(probe, nominal, {})
    assert card["nominal_failures"] == 1 and 0 < card["score"] <= 100
    page = render_html("T <script>", {"a": card}, {"a": probe}, {}, None)
    assert "<script>" not in page and "T &lt;script&gt;" in page


# ---------------------------------------------------------------- server
def test_server_endpoints():
    from breakpoint.server import app

    c = TestClient(app)
    assert c.get("/").status_code == 200
    assert "cpg" in c.get("/api/policies").json()["policies"]
    r = c.post("/api/compile", json={"text": "icy floor"}).json()
    assert r["scenario"]["friction"] < 0.2
    sim = c.post("/api/simulate", json={"scenario": {"duration_s": 1.0}, "policy": "cpg"}).json()
    assert sim["outcome"]["survived"] and sim["frames"]
    assert c.post("/api/simulate", json={"scenario": {}, "policy": "../../etc/passwd"}).status_code == 400
    from fastapi import HTTPException

    from breakpoint.server import report

    for bad in ("..", "../etc", "a/b"):
        with pytest.raises(HTTPException):
            report(bad)
    assert c.post("/api/compile_image", json={"image_b64": "!!!"}).status_code == 400


def test_websocket_shove_game():
    from breakpoint.server import app

    c = TestClient(app)
    with c.websocket_connect("/ws/play") as ws:
        ws.send_text(json.dumps({"scenario": {}, "policy": "cpg"}))
        assert json.loads(ws.receive_text())["type"] == "init"
        ws.send_text(json.dumps({"type": "shove", "force": 400, "direction_deg": 90}))
        ws.send_text(json.dumps({"type": "shove", "force": 400, "direction_deg": 90}))
        msg = {}
        for _ in range(400):
            msg = json.loads(ws.receive_text())
            if msg["type"] == "over":
                break
        assert msg["type"] == "over" and msg["fell"] and msg["shoves"] >= 1


# ---------------------------------------------------------------- red team
PROBE_FIXTURE = [
    {"key": "push_left", "label": "Push from left", "unit": "N·s", "breaking_point": 5.0, "fraction": 0.125},
    {"key": "friction", "label": "Low friction", "unit": "μ", "breaking_point": 0.3, "fraction": 0.7},
    {"key": "payload_kg", "label": "Payload", "unit": "kg", "breaking_point": 3.0, "fraction": 0.4},
]


def test_interactions_scale_each_axis_by_its_own_breaking_point():
    from breakpoint.redteam import interactions

    cases = interactions(None, PROBE_FIXTURE, top_k=3, workers=1, seeds=[0])
    assert len(cases) == 3 * 2  # 3 pairs x 2 fractions
    half = next(c for c in cases if c["fraction"] == 0.5 and "payload_kg" in c["settings"] and "friction" in c["settings"])
    assert half["settings"]["payload_kg"] == pytest.approx(1.5)
    assert half["settings"]["friction"] == pytest.approx(0.65)
    assert all(0.0 <= c["pass_rate"] <= 1.0 for c in cases)


def test_boundary_map_grid_shape_and_benign_corner_passes():
    from breakpoint.redteam import boundary_map

    bm = boundary_map(None, "friction", "push_left", n=3, workers=1)
    assert len(bm["pass_rate"]) == 3 and len(bm["pass_rate"][0]) == 3
    assert bm["pass_rate"][0][0] == 1.0  # full friction, no push


def test_propose_offline_templates_are_valid_and_target_weak_spots(monkeypatch):
    from breakpoint.redteam import propose

    monkeypatch.delenv("BREAKPOINT_LLM_BASE_URL", raising=False)
    scns, src = propose(PROBE_FIXTURE, n=3)
    assert src == "templates" and len(scns) == 3
    assert any(s.friction < 1.0 and s.pushes for s in scns)


def test_propose_uses_llm_and_validates(monkeypatch):
    from breakpoint import redteam

    monkeypatch.setattr(redteam, "_llm_config", lambda: ("https://x", "k", "m"))
    monkeypatch.setattr(redteam, "_chat", lambda msgs: json.dumps({"scenarios": [{"name": "x", "friction": -3}, "junk"]}))
    scns, src = redteam.propose(PROBE_FIXTURE, n=3)
    assert src == "llm" and len(scns) == 1 and scns[0].friction == 0.05


def test_report_renders_compound_sections():
    from breakpoint.redteam import boundary_map

    bm = boundary_map(None, "friction", "push_left", n=2, workers=1)
    extras = {"interactions": [{"a": "A", "b": "B", "fraction": 0.5, "pass_rate": 0.0, "compound_failure": True}],
              "boundary": bm, "redteam": {"source": "templates", "attacks": [{"scenario": {"name": "<b>x</b>"}, "pass_rate": 0.5}]}}
    card = scorecard(PROBE_FIXTURE, [{"passed": True}], {})
    page = render_html("t", {"a": card}, {"a": PROBE_FIXTURE}, {}, None, extras=extras)
    assert "Compound stress" in page and "Failure boundary" in page and "&lt;b&gt;x" in page
