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
    ("wheeled", 100.0, -90.0, {}), ("wheeled", 0.0, 0.0, {"floor_step_m": 0.10}), ("wheeled", 80.0, 135.0, {"friction": 0.3}),
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


def test_js_breaking_points_match_python():
    js = run_js('[H.breakingPoint("biped", -90, {}, 200, 8), H.breakingPoint("wheeled", 180, {}, 200, 8),'
                ' H.breakingPoint("biped", 0, {"latency_s": 0.1}, 200, 8)]')
    py = [breaking_point(-90.0, Humanoid(), iters=8), breaking_point(180.0, WheeledBase(), iters=8),
          breaking_point(0.0, Humanoid(), Conditions(latency_s=0.1), iters=8)]
    for a, b in zip(js, py):
        assert a == pytest.approx(b, abs=0.05)
