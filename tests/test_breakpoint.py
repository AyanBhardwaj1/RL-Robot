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


@pytest.mark.parametrize("url", ["http://evil.example/v1", "http://localhost.evil.com/v1", "ftp://x"])
def test_insecure_llm_url_disables_llm_and_falls_back(monkeypatch, url):
    monkeypatch.setenv("BREAKPOINT_LLM_BASE_URL", url)
    monkeypatch.setenv("BREAKPOINT_LLM_API_KEY", "k")
    monkeypatch.setenv("BREAKPOINT_LLM_MODEL", "m")
    called = []
    monkeypatch.setattr("breakpoint.compiler._chat", lambda msgs: called.append(1) or "{}")
    with pytest.warns(UserWarning):
        c = compile_text("icy floor")
    assert c.source == "rules" and not called


@pytest.mark.parametrize("reply_exc", [IndexError, TypeError])
def test_malformed_llm_reply_falls_back(monkeypatch, reply_exc):
    monkeypatch.setenv("BREAKPOINT_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("BREAKPOINT_LLM_API_KEY", "k")
    monkeypatch.setenv("BREAKPOINT_LLM_MODEL", "m")

    def boom(msgs):
        raise reply_exc("bad shape")

    monkeypatch.setattr("breakpoint.compiler._chat", boom)
    assert compile_text("icy floor").source == "rules"


def test_oversized_image_is_rejected_before_decoding():
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("1", (5000, 4000)).save(buf, format="PNG")
    with pytest.raises(ValueError):
        compile_image(buf.getvalue(), use_llm=False)


def test_stair_clearance_uses_upper_step():
    from breakpoint.world import ground_height

    scn = Scenario(terrain="stairs", step_height=0.1)
    # just past a riser at x = 0.6 + 0.35 the ground must read as the upper step
    assert ground_height(scn, 0.6 + 0.35 + 0.01, 0.0) >= 0.06 + 0.1 - 1e-6


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
    _, acts = net.forward(x)
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


def test_websocket_ignores_nan_shoves_and_rejects_foreign_origin():
    from starlette.websockets import WebSocketDisconnect

    from breakpoint.server import app

    c = TestClient(app)
    with c.websocket_connect("/ws/play") as ws:
        ws.send_text(json.dumps({"scenario": {}, "policy": "cpg"}))
        ws.receive_text()
        ws.send_text('{"type": "shove", "force": NaN, "direction_deg": 90}')
        ws.send_text("1")
        for _ in range(5):
            frame = json.loads(ws.receive_text())
        assert frame["type"] == "frame" and frame["shoves"] == 0 and frame["impulse"] == 0.0
    with pytest.raises(WebSocketDisconnect), c.websocket_connect("/ws/play", headers={"origin": "https://evil.example"}) as ws:
        ws.receive_text()


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


def test_fixed_gait_phase_overrides_seed_randomisation():
    scn = Scenario(duration_s=1.0, phase0_deg=90.0)
    a = QuadrupedEnv(scn, seed=0)
    a.reset(seed=1)
    b = QuadrupedEnv(scn, seed=0)
    b.reset(seed=2)
    assert a.phase == pytest.approx(np.radians(90.0)) == pytest.approx(b.phase)


def test_push_probe_sweeps_gait_phases():
    from breakpoint.probe import PUSH_PHASES, push_phase_at_impact

    assert len(PUSH_PHASES) == 4
    assert 0.0 <= push_phase_at_impact(0.0) < 360.0
    r = probe_axis((None, Scenario().to_dict(), "push_rear", [0], 2))
    assert r["rollouts"] % len(PUSH_PHASES) == 0


def test_atlas_grid_marks_found_cells():
    from breakpoint.report import _atlas_grid

    svg = _atlas_grid({"payload|1": {"scenario": {"name": "heavy <box>"}}})
    assert svg.count("var(--bad)") == 1 and "heavy &lt;box&gt;" in svg
    assert _atlas_grid({}) == ""


def test_deterministic_score_and_checkpoint_callback():
    from breakpoint.ppo import PPOConfig, ScenarioSampler, deterministic_score, train

    pol = ResidualPolicy(seed=0)
    s = deterministic_score(pol, seeds=(0,))
    assert 0.0 <= s <= 1.25
    seen = []
    train(pol, ScenarioSampler(seed=0), PPOConfig(total_steps=256, rollout_steps=128, minibatch=64, epochs=1),
          checkpoint=lambda p, row: seen.append(row["det_score"]), eval_every=1)
    assert len(seen) == 2 and all(0.0 <= v <= 1.25 for v in seen)


def test_profiles_sample_valid_worlds_matching_their_terrain_mix():
    from breakpoint.profiles import PROFILES

    rng = np.random.default_rng(0)
    for prof in PROFILES.values():
        worlds = [prof.sample(rng) for _ in range(40)]
        assert {w.terrain for w in worlds} <= set(prof.terrains)
        assert all(prof.friction[0] - 1e-9 <= w.friction <= prof.friction[1] + 1e-9 for w in worlds)


def test_deployment_check_reports_rate_and_bound():
    from breakpoint.profiles import deployment_check

    r = deployment_check(None, "home", trials=6, workers=1)
    assert r["trials"] == 6 and 0 <= r["failures"] <= 6
    assert r["failure_rate_upper_95"] >= r["failure_rate"]
    with pytest.raises(ValueError):
        deployment_check(None, "moon", trials=1)


def test_sampler_draws_from_deployment_profiles():
    from breakpoint.ppo import ScenarioSampler
    from breakpoint.profiles import PROFILES

    s = ScenarioSampler(seed=0, profiles=[PROFILES["construction"]], widen_frac=1.0)
    worlds = [s.sample(6.0) for _ in range(20)]
    assert all(w.name == "construction sample" for w in worlds)


def test_deploy_endpoint_validates_and_runs():
    from breakpoint.server import app

    c = TestClient(app)
    assert {p["name"] for p in c.get("/api/profiles").json()["profiles"]} >= {"warehouse", "home"}
    assert c.post("/api/deploy", json={"profile": "moon"}).status_code == 404
    assert c.post("/api/deploy", json={"profile": "home", "trials": 10_000}).status_code == 422
    r = c.post("/api/deploy", json={"profile": "home", "trials": 4, "policy": "cpg"}).json()
    assert r["trials"] == 4 and "failures_by_dominant_stressor" in r


def test_explain_cites_push_direction_and_factors():
    from breakpoint.explain import direction_name, explain

    assert direction_name(90) == "the left" and direction_name(0) == "the rear" and direction_name(180) == "the front"
    scn = Scenario(friction=0.2, payload_kg=3.0).with_(pushes=[{"t": 2.0, "force": 120, "direction_deg": -90, "duration": 0.1}])
    out, _ = rollout(CPGPolicy(), scn, seed=0)
    text = explain(scn, out.to_dict())
    if out.fell:
        assert "shove from the right" in text and "3.0 kg" in text and "μ=0.20" in text
    assert explain(Scenario(), {"passed": True, "distance": 3.0, "time_s": 6.0}).startswith("Walked 3.0 m")
    stalled = explain(Scenario(motor_strength=0.3), {"passed": False, "fell": False, "distance": 0.1, "time_s": 8})
    assert "stalled" in stalled and "too weak" in stalled


def test_gymnasium_adapter_api():
    from gymnasium.utils.env_checker import check_env

    from breakpoint.gym_env import BreakpointEnv

    env = BreakpointEnv({"duration_s": 1.0})
    check_env(env, skip_render_check=True)
    env.reset(seed=0)
    total_steps, done = 0, False
    while not done:
        _, _, term, trunc, info = env.step(env.action_space.sample())
        total_steps += 1
        done = term or trunc
    assert total_steps <= 50 and "x" in info


def test_bring_your_own_policy_spec():
    from breakpoint.policy import CallablePolicy, load_policy

    pol = load_policy("py:examples.my_controller:make_policy")
    assert isinstance(pol, CallablePolicy)
    assert pol.act(np.zeros(OBS_DIM)).shape == (12,)
    with pytest.raises(ValueError):
        load_policy("py:nocolon")
    out, _ = rollout(pol, Scenario(duration_s=2.0), seed=0)
    assert out.survived


def test_rules_understand_imperial_units():
    s = compile_rules("hauling 10 lbs down a stairway with 2 inch steps").scenario
    assert s.payload_kg == pytest.approx(4.54, abs=0.01)
    assert s.step_height == pytest.approx(0.0508, abs=1e-3)


# ------------------------------------------------------------------- cli
def test_cli_compile_run_and_regress(tmp_path, capsys, monkeypatch):
    from breakpoint.cli import main

    monkeypatch.delenv("BREAKPOINT_LLM_BASE_URL", raising=False)
    scn_path = tmp_path / "scn.json"
    assert main(["compile", "stairs with 5 cm steps", "--offline", "--out", str(scn_path)]) == 0
    assert json.loads(capsys.readouterr().out)["scenario"]["terrain"] == "stairs"
    short = Scenario.from_json(scn_path.read_text()).with_(duration_s=2.0)
    scn_path.write_text(short.to_json())
    assert main(["run", "--scenario", str(scn_path)]) in (0, 1)
    assert "passed" in capsys.readouterr().out

    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "case_00.json").write_text(json.dumps({"scenario": Scenario(duration_s=2.0).to_dict(), "seed": 0}))
    assert main(["regress", "--suite", str(suite)]) == 0
    assert "1/1 regression cases pass" in capsys.readouterr().out
    assert main(["regress", "--suite", str(tmp_path / "empty")]) == 2


def test_cli_deploy_gates_on_failure_budget(capsys):
    from breakpoint.cli import main

    rc = main(["deploy", "--profile", "home", "--trials", "4", "--workers", "1", "--max-fail", "1.0"])
    assert rc == 0 and "home:" in capsys.readouterr().out
    assert main(["deploy", "--profile", "home", "--trials", "4", "--workers", "1", "--max-fail", "0.0"]) == 1


def test_hunt_small_budget_builds_atlas_and_suite(tmp_path):
    from breakpoint.cli import save_suite
    from breakpoint.hunt import hunt

    res = hunt(None, generations=2, population=8, workers=1, seed=3)
    assert res.evaluations == 16 and len(res.history) == 2
    for cell in res.atlas.values():
        assert not cell["outcome"]["passed"] and "|" in cell["cell"]
    paths = save_suite(dict(list(res.atlas.items())[:1]), None, tmp_path / "suite", shrink_iters=2)
    assert len(paths) == min(1, len(res.atlas))
    for p in paths:
        case = json.loads(p.read_text())
        assert 0 < case["alpha"] <= 1.0 and "scenario" in case


def test_compile_image_vision_llm_and_caption_merge(monkeypatch):
    monkeypatch.setenv("BREAKPOINT_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("BREAKPOINT_LLM_API_KEY", "k")
    monkeypatch.setenv("BREAKPOINT_LLM_MODEL", "m")
    sent = {}

    def fake_chat(msgs):
        sent["content"] = msgs[-1]["content"]
        return json.dumps({"terrain": "rough", "roughness": 0.04, "friction": 0.5})

    monkeypatch.setattr("breakpoint.compiler._chat", fake_chat)
    c = compile_image(_png([0.5, 0.5, 0.5]), caption="gravel lot")
    assert c.source == "llm-vision" and c.scenario.roughness == 0.04
    assert sent["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    monkeypatch.delenv("BREAKPOINT_LLM_BASE_URL")
    merged = compile_image(_png([0.2, 0.6, 0.2], 0.05), caption="carrying 3kg, kicked from the left")
    assert merged.source == "image-heuristic"
    assert merged.scenario.friction == 0.6 and merged.scenario.payload_kg == 3.0 and merged.scenario.pushes
