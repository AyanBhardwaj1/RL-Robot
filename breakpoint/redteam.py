"""Compound-stress testing: interactions, failure-boundary maps, and an LLM red team.

Single-axis margins miss the most common real-world failure: two stressors that are
each survivable alone but fatal together (a moderate shove on a slightly slick floor).

* ``interactions`` takes the weakest axes from a probe and tests every pair with each
  stressor set to a *fraction of its own breaking point*. A pair that fails at 60% +
  60% is a compound fragility the single-axis report would never show.
* ``boundary_map`` sweeps a 2-D grid of two axes and records pass rates, giving the
  failure frontier the report draws as a heatmap.
* ``propose`` asks an LLM red-teamer for realistic multi-stressor scenarios aimed at
  the weak spots; offline it falls back to deterministic templates. Every proposal is
  validated through ``Scenario.from_dict`` like any other untrusted input.
"""

from __future__ import annotations

import itertools
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .compiler import _chat, _llm_config, _parse_json
from .evaluate import rollout
from .policy import load_policy
from .probe import AXIS_BY_KEY, apply_axis
from .scenario import Scenario

INTERACTION_FRACTIONS = (0.5, 0.75)


def _compose(base: Scenario, settings: dict[str, float]) -> Scenario:
    scn = base
    pushes: list[dict] = []
    for key, value in settings.items():
        nxt = apply_axis(scn, key, value)
        if key.startswith("push_"):
            pushes.extend(p.__dict__ for p in nxt.pushes)
        scn = nxt
    return scn.with_(pushes=pushes) if pushes else scn


def _value_at(axis_key: str, breaking_point: float, frac: float) -> float:
    ax = AXIS_BY_KEY[axis_key]
    v = ax.lo + frac * (breaking_point - ax.lo)
    return float(round(v)) if ax.integer else float(v)


def _pass_job(args: tuple[str | None, dict, list[int]]) -> float:
    spec, scn_d, seeds = args
    pol = load_policy(spec)
    scn = Scenario.from_dict(scn_d)
    return float(np.mean([rollout(pol, scn, seed=s)[0].passed for s in seeds]))


def _run(jobs: list, workers: int) -> list[float]:
    if workers <= 1:
        return [_pass_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_pass_job, jobs))


def interactions(spec: str | None, probe_results: list[dict], top_k: int = 4, seeds: list[int] | None = None,
                 workers: int = 3) -> list[dict]:
    seeds = seeds or [0, 1]
    base = Scenario(duration_s=6.0)
    usable = [r for r in probe_results if not r.get("fails_nominal") and r["fraction"] > 0]
    # one push direction is enough to represent "push"; prefer the weakest
    pushes = sorted((r for r in usable if r["key"].startswith("push_")), key=lambda r: r["fraction"])[:1]
    others = sorted((r for r in usable if not r["key"].startswith("push_")), key=lambda r: r["fraction"])
    weakest = (pushes + others)[:top_k]
    cases, jobs = [], []
    for a, b in itertools.combinations(weakest, 2):
        for frac in INTERACTION_FRACTIONS:
            settings = {a["key"]: _value_at(a["key"], a["breaking_point"], frac),
                        b["key"]: _value_at(b["key"], b["breaking_point"], frac)}
            cases.append({"a": a["label"], "b": b["label"], "fraction": frac, "settings": settings})
            jobs.append((spec, _compose(base, settings).to_dict(), seeds))
    rates = _run(jobs, workers)
    for c, r in zip(cases, rates):
        c["pass_rate"] = r
        c["compound_failure"] = r < 1.0
    return cases


def boundary_map(spec: str | None, x_key: str, y_key: str, n: int = 6, seeds: list[int] | None = None,
                 workers: int = 3) -> dict:
    seeds = seeds or [0]
    ax, ay = AXIS_BY_KEY[x_key], AXIS_BY_KEY[y_key]
    xs = np.linspace(ax.lo, ax.hi, n)
    ys = np.linspace(ay.lo, ay.hi, n)
    base = Scenario(duration_s=6.0)
    jobs = [(spec, _compose(base, {x_key: float(x), y_key: float(y)}).to_dict(), seeds) for y in ys for x in xs]
    rates = _run(jobs, workers)
    grid = [rates[i * n:(i + 1) * n] for i in range(n)]
    return {"x": {"key": x_key, "label": ax.label, "unit": ax.unit, "values": [round(float(v), 4) for v in xs]},
            "y": {"key": y_key, "label": ay.label, "unit": ay.unit, "values": [round(float(v), 4) for v in ys]},
            "pass_rate": grid}


# --------------------------------------------------------------- red team
REDTEAM_PROMPT = (
    "You are a red-team test engineer for a 6 kg quadruped robot. Given its measured weak spots, "
    "propose {n} REALISTIC deployment situations (warehouse, sidewalk, farm, home...) that combine "
    "2-3 moderate stressors likely to make it fall. Each must be plausible, not extreme. Reply with "
    'ONLY {{"scenarios": [ ... ]}} where each item follows this JSON schema:\n{schema}\n'
    "Weak spots (stressor: breaking point):\n{weak}"
)


def _templates(weak: list[dict], n: int) -> list[dict]:
    """Offline fallback: realistic stories built around the measured weak spots."""
    keys = {r["key"] for r in weak}
    lib = [
        ({"friction", "push_left", "push_right"}, {"name": "wet warehouse aisle, forklift brushes past",
          "friction": 0.35, "pushes": [{"t": 2.5, "force": 70, "direction_deg": 90, "duration": 0.1}]}),
        ({"payload_kg", "slope_up"}, {"name": "delivery up a loading ramp", "terrain": "slope", "slope_deg": 8,
          "payload_kg": 3.0}),
        ({"latency_steps", "roughness"}, {"name": "gravel yard on congested wifi", "terrain": "rough",
          "roughness": 0.03, "latency_steps": 3}),
        ({"motor_strength", "step_height"}, {"name": "end of shift: low battery on porch steps", "terrain": "stairs",
          "step_height": 0.05, "motor_strength": 0.7}),
        ({"obs_noise", "push_rear", "push_front"}, {"name": "vibrating factory floor, bumped by a cart", "obs_noise": 0.12,
          "pushes": [{"t": 3.0, "force": 80, "direction_deg": 0, "duration": 0.1}]}),
        ({"friction", "slope_down"}, {"name": "icy driveway, downhill", "terrain": "slope", "slope_deg": -6,
          "friction": 0.25}),
        ({"payload_kg", "push_left", "push_right", "push_rear-left"}, {"name": "carrying groceries, child grabs it",
          "payload_kg": 2.5, "payload_offset_x": 0.08, "pushes": [{"t": 2.0, "force": 60, "direction_deg": -90, "duration": 0.15}]}),
    ]
    ranked = sorted(lib, key=lambda t: -len(t[0] & keys))
    return [dict(d) for _, d in ranked[:n]]


def propose(probe_results: list[dict], n: int = 5, use_llm: bool = True) -> tuple[list[Scenario], str]:
    weak = sorted(probe_results, key=lambda r: r["fraction"])[:6]
    if use_llm and _llm_config() is not None:
        try:
            from .scenario import json_schema

            text = "\n".join(f"- {r['label']}: {r['breaking_point']} {r['unit']}" for r in weak)
            reply = _chat([{"role": "user", "content": REDTEAM_PROMPT.format(
                n=n, schema=json.dumps(json_schema()), weak=text)}])
            items = _parse_json(reply).get("scenarios", [])
            scns = [Scenario.from_dict(d).with_(duration_s=6.0) for d in items[:n] if isinstance(d, dict)]
            if scns:
                return scns, "llm"
        except Exception:  # noqa: BLE001 - any LLM failure degrades to templates
            pass
    return [Scenario.from_dict({**d, "duration_s": 6.0}) for d in _templates(weak, n)], "templates"


def red_team(spec: str | None, probe_results: list[dict], n: int = 5, seeds: list[int] | None = None,
             workers: int = 3, use_llm: bool = True) -> dict:
    seeds = seeds or [0, 1, 2]
    scns, source = propose(probe_results, n, use_llm)
    rates = _run([(spec, s.to_dict(), seeds) for s in scns], workers)
    return {"source": source, "attacks": [{"scenario": s.to_dict(), "pass_rate": r} for s, r in zip(scns, rates)]}
