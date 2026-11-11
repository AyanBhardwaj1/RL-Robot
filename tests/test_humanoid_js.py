"""The browser demo (site/assets/humanoid.js) must agree with breakpoint/humanoid.py."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from breakpoint.humanoid import Conditions, Humanoid, WheeledBase, breaking_point, recover, recover_wheeled

JS = Path(__file__).resolve().parent.parent / "site" / "assets" / "humanoid.js"
CASES = [
    ("biped", 0.0, 0.0, {}), ("biped", 40.0, 0.0, {}), ("biped", 60.0, -90.0, {}),
    ("biped", 70.0, 0.0, {"friction": 0.15}), ("biped", 50.0, -90.0, {"latency_s": 0.15}),
    ("biped", 30.0, 45.0, {"payload_kg": 20.0}), ("biped", 0.0, 0.0, {"floor_slope_deg": 4.0}),
    ("biped", 0.0, 0.0, {"floor_step_m": 0.25}),
    ("biped", 60.0, 180.0, {"latency_s": 0.025}), ("biped", 40.0, -135.0, {"latency_s": 0.1, "payload_kg": 10.0}),
    ("biped", 45.0, 90.0, {}), ("biped", 61.0, 103.0, {"friction": 0.5, "floor_slope_deg": -2.7, "payload_kg": 15.0}),
    ("biped", 60.0, 180.0, {"friction": 0.1}),
    ("biped", 64.1, 38.3, {"friction": 0.5, "payload_kg": 16.3, "floor_slope_deg": 2.5, "latency_s": 0.023, "actuator_strength": 0.83}), ("biped", 40.0, 180.0, {"friction": 0.35}), ("biped", 30.0, -90.0, {}), ("biped", 35.0, 180.0, {"actuator_strength": 0.6}),
    ("wheeled", 100.0, -90.0, {}), ("wheeled", 90.0, 0.0, {"latency_s": 0.025}), ("wheeled", 0.0, 0.0, {"floor_step_m": 0.10}), ("wheeled", 80.0, 135.0, {"friction": 0.3}),
]


def run_js(expr: str):
    node = shutil.which("node")
    if node is None:
        pytest.fail("node is required to check the browser port")
    script = f"const H = require({json.dumps(str(JS))}); console.log(JSON.stringify({expr}));"
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60, check=True)
    return json.loads(out.stdout)


def test_js_recovery_matches_python():
    calls = ",".join(f"H.recover({json.dumps(b)}, {j}, {d}, {json.dumps(c)}, false)" for b, j, d, c in CASES)
    js = run_js(f"[{calls}]")
    for (body, imp, deg, c), got in zip(CASES, js):
        py = (recover_wheeled(imp, deg, WheeledBase(), Conditions(**c)) if body == "wheeled"
              else recover(imp, deg, Humanoid(), Conditions(**c)))
        assert got["recovered"] == py.recovered, (body, imp, deg, c, got["reason"], py.reason)
        assert got["steps"] == py.steps
        assert got["reason"].split(":")[0] == py.reason.split(":")[0]
        assert got["time_s"] == pytest.approx(py.time_s, abs=0.003)


def test_js_breaking_points_match_python():
    js = run_js('[H.breakingPoint("biped", -90, {}, 200, 8), H.breakingPoint("wheeled", 180, {}, 200, 8),'
                ' H.breakingPoint("biped", 0, {"latency_s": 0.1}, 200, 8)]')
    py = [breaking_point(-90.0, Humanoid(), iters=8), breaking_point(180.0, WheeledBase(), iters=8),
          breaking_point(0.0, Humanoid(), Conditions(latency_s=0.1), iters=8)]
    for a, b in zip(js, py):
        assert a == pytest.approx(b, abs=0.05)


def test_js_capturability_map_matches_python():
    from breakpoint.humanoid import capturability_map

    js = run_js('H.capturabilityMap("biped", {"latency_s": 0.05}, 9, 1.6)')
    py = capturability_map(Humanoid(), Conditions(latency_s=0.05), n=9, v_max=1.6)
    assert js == py
    flat = [c for row in py for c in row]
    assert flat[len(flat) // 2] == 0  # standing still: ankles only
    assert -1 in flat and max(flat) >= 1  # some shoves need steps, the fastest fail


def test_js_design_lever_overrides_match_python():
    from dataclasses import replace

    js = run_js('[H.breakingPoint(Object.assign({}, H.BIPED, {kind: "biped", min_swing: H.BIPED.min_swing * 0.85}), -90, {}, 200, 10),'
                ' H.breakingPoint(Object.assign({}, H.BIPED, {kind: "biped", max_lateral_step: 0.5}), -90, {}, 200, 10)]')
    r = Humanoid()
    py = [breaking_point(-90.0, replace(r, min_swing=r.min_swing * 0.85), iters=10),
          breaking_point(-90.0, replace(r, max_lateral_step=0.5), iters=10)]
    assert js == pytest.approx(py, abs=0.05)
    assert py[0] > breaking_point(-90.0, r, iters=10)


def test_js_episode_filter_matches_python():
    from breakpoint.episodes import filter_episodes, synthetic_episodes

    eps = synthetic_episodes(60, seed=3)
    eps.append({"episode": "gap", "latency_ms": [20, None, 400, 20], "contacts": [{"t_s": 1, "impulse_ns": 40, "direction_deg": -90}]})
    ep_js = Path(__file__).resolve().parent.parent / "site" / "assets" / "episodes.js"
    node = shutil.which("node")
    if node is None:
        pytest.fail("node is required to check the browser port")
    script = (f"const E = require({json.dumps(str(ep_js))}); const eps = {json.dumps(eps)};"
              " console.log(JSON.stringify(E.filterEpisodes(eps).verdicts.map((v) => v.tag)));"
              " let err = ''; try { E.filterEpisodes([{contacts: [{impulse_ns: null}]}]); } catch (e) { err = e.message; }"
              " console.log(JSON.stringify(err));")
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60, check=True).stdout.splitlines()
    assert json.loads(out[0]) == [v["tag"] for v in filter_episodes(eps)["verdicts"]]
    assert "record 1" in json.loads(out[1]) and "impulse_ns" in json.loads(out[1])
