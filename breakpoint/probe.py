"""Breaking-point probe: per-stressor robustness margins by bisection.

For each stress axis we search for the largest stress level the policy still passes
under every seed. Seeds randomise the gait phase at which disturbances land, which
research shows strongly affects push recovery. The result is a margin per axis, e.g.
"survives side pushes up to 14 N·s" or "walks on friction down to 0.22".
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from .evaluate import rollout
from .policy import load_policy
from .scenario import Push, Scenario

PUSH_DURATION = 0.1
PUSH_TIME = 2.0
# Pushes land at these gait phases (deg). Recovery depends strongly on stance/swing timing,
# so a push margin is only as good as its worst phase.
PUSH_PHASES = (0.0, 90.0, 180.0, 270.0)


@dataclass(frozen=True)
class Axis:
    key: str
    label: str
    unit: str
    lo: float  # benign end
    hi: float  # extreme end
    integer: bool = False


PUSH_DIRS = {"rear": 0.0, "left": 90.0, "front": 180.0, "right": -90.0,
             "rear-left": 45.0, "front-left": 135.0, "front-right": -135.0, "rear-right": -45.0}

AXES: tuple[Axis, ...] = (
    *(Axis(f"push_{n}", f"Push from {n}", "N·s", 0.0, 40.0) for n in PUSH_DIRS),
    Axis("friction", "Low friction", "μ", 1.0, 0.05),
    Axis("payload_kg", "Payload", "kg", 0.0, 8.0),
    Axis("motor_strength", "Motor weakness", "× torque", 1.0, 0.2),
    Axis("latency_steps", "Action latency", "×20 ms", 0.0, 6.0, integer=True),
    Axis("obs_noise", "Sensor noise", "σ", 0.0, 0.5),
    Axis("slope_up", "Uphill slope", "deg", 0.0, 20.0),
    Axis("slope_down", "Downhill slope", "deg", 0.0, 20.0),
    Axis("step_height", "Stairs", "m", 0.0, 0.10),
    Axis("roughness", "Rough ground", "m", 0.0, 0.08),
)
AXIS_BY_KEY = {a.key: a for a in AXES}


def apply_axis(base: Scenario, key: str, value: float) -> Scenario:
    if key.startswith("push_"):
        direction = PUSH_DIRS[key[5:]]
        force = value / PUSH_DURATION
        return base.with_(pushes=[asdict(Push(t=PUSH_TIME, force=force, direction_deg=direction,
                                               duration=PUSH_DURATION))])
    if key == "slope_up":
        return base.with_(terrain="slope", slope_deg=value)
    if key == "slope_down":
        return base.with_(terrain="slope", slope_deg=-value)
    if key == "step_height":
        return base.with_(terrain="stairs", step_height=max(value, 1e-4))
    if key == "roughness":
        return base.with_(terrain="rough", roughness=max(value, 1e-4))
    return base.with_(**{key: value})


def push_phase_at_impact(phase0_deg: float) -> float:
    from .env import GAIT_FREQ

    return (phase0_deg + 360.0 * GAIT_FREQ * PUSH_TIME) % 360.0


def _passes(spec: str | None, base: Scenario, key: str, value: float, seeds: list[int]) -> bool:
    policy = load_policy(spec)
    scn = apply_axis(base, key, value)
    if key.startswith("push_"):
        # sweep the gait phase at which the push lands (phase at impact = PUSH_PHASES)
        worlds = [scn.with_(phase0_deg=(ph - push_phase_at_impact(0.0)) % 360.0) for ph in PUSH_PHASES]
        return all(rollout(policy, w, seed=seeds[0])[0].passed for w in worlds)
    return all(rollout(policy, scn, seed=s)[0].passed for s in seeds)


def probe_axis(args: tuple[str | None, dict, str, list[int], int]) -> dict:
    spec, base_d, key, seeds, iters = args
    base = Scenario.from_dict(base_d)
    ax = AXIS_BY_KEY[key]
    if not _passes(spec, base, key, ax.lo, seeds):
        return {"key": key, "label": ax.label, "unit": ax.unit, "breaking_point": ax.lo,
                "fraction": 0.0, "fails_nominal": True, "rollouts": len(seeds)}
    per_check = len(PUSH_PHASES) if key.startswith("push_") else len(seeds)
    rollouts = per_check
    if ax.integer:
        good = ax.lo
        for v in np.arange(ax.lo + 1, ax.hi + 1):
            rollouts += per_check
            if not _passes(spec, base, key, float(v), seeds):
                break
            good = float(v)
    else:
        if _passes(spec, base, key, ax.hi, seeds):
            good = ax.hi
            rollouts += per_check
        else:
            rollouts += per_check
            good, bad = ax.lo, ax.hi
            for _ in range(iters):
                mid = 0.5 * (good + bad)
                rollouts += per_check
                if _passes(spec, base, key, mid, seeds):
                    good = mid
                else:
                    bad = mid
    frac = (good - ax.lo) / (ax.hi - ax.lo)
    return {"key": key, "label": ax.label, "unit": ax.unit, "breaking_point": round(float(good), 4),
            "fraction": round(float(frac), 4), "fails_nominal": False, "rollouts": rollouts}


def probe(policy_spec: str | None, base: Scenario | None = None, seeds: list[int] | None = None,
          iters: int = 6, workers: int = 3, keys: list[str] | None = None) -> list[dict]:
    base = (base or Scenario()).with_(duration_s=6.0)
    seeds = seeds or [0, 1]
    keys = keys or [a.key for a in AXES]
    jobs = [(policy_spec, base.to_dict(), k, seeds, iters) for k in keys]
    if workers <= 1:
        return [probe_axis(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(probe_axis, jobs))


def push_polar(results: list[dict]) -> dict[str, float]:
    """Direction -> max recoverable impulse (N·s), for the report's polar plot."""
    return {r["key"][5:]: r["breaking_point"] for r in results if r["key"].startswith("push_")}
