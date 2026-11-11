"""Adversarial failure hunting: CEM search + a MAP-Elites failure atlas + shrinking.

1. **Search.** Cross-entropy method over a normalised stress vector. The objective is
   ``margin + λ·severity``: it rewards *failing* (negative margin) at *low severity*,
   so the hunter returns the cheapest ways to break the robot rather than absurd
   worlds where anything would fall.
2. **Atlas.** Every failure is filed in a MAP-Elites grid keyed by (dominant stressor,
   severity band), keeping the mildest failure per cell. This yields a *diverse*
   catalogue of distinct failure modes rather than 50 copies of "pushed too hard".
3. **Shrink.** Like property-based testing, each failure is shrunk toward the nominal
   world by bisection on an interpolation factor, giving a minimal counterexample.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .evaluate import rollout
from .policy import load_policy
from .scenario import Push, Scenario

# (name, lo, hi): each dimension of the normalised search vector in [0, 1]
DIMS: tuple[tuple[str, float, float], ...] = (
    ("friction", 1.0, 0.1),
    ("payload_kg", 0.0, 6.0),
    ("payload_offset_x", -0.12, 0.12),
    ("motor_strength", 1.0, 0.35),
    ("latency_steps", 0.0, 5.0),
    ("obs_noise", 0.0, 0.35),
    ("roughness", 0.0, 0.06),
    ("slope_deg", -15.0, 15.0),
    ("push_impulse", 0.0, 30.0),
    ("push_dir", -180.0, 180.0),
    ("push_t", 1.0, 4.0),
)
NEUTRAL = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5])
SEVERITY_WEIGHT = 0.8
SEVERITY_BANDS = (0.25, 0.5, 0.75, 1.0, 10.0)


def decode(x: np.ndarray, base: Scenario | None = None) -> Scenario:
    base = base or Scenario()
    x = np.clip(x, 0.0, 1.0)
    v = {name: lo + float(xi) * (hi - lo) for (name, lo, hi), xi in zip(DIMS, x)}
    impulse = v.pop("push_impulse")
    pushes = []
    if impulse > 0.5:
        pushes.append(Push(t=v["push_t"], force=impulse / 0.1, direction_deg=v["push_dir"], duration=0.1).__dict__)
    v.pop("push_dir")
    v.pop("push_t")
    v["latency_steps"] = round(v["latency_steps"])
    terrain = "rough" if v["roughness"] > 0.005 else ("slope" if abs(v["slope_deg"]) > 1 else "flat")
    return base.with_(terrain=terrain, pushes=pushes, duration_s=6.0, **v)


def _eval(args: tuple[str | None, list[float], int, dict]) -> dict:
    spec, x, seed, base_d = args
    scn = decode(np.asarray(x), Scenario.from_dict(base_d))
    out, _ = rollout(load_policy(spec), scn, seed=seed)
    sev = scn.severity()
    return {"x": list(map(float, x)), "scenario": scn.to_dict(), "outcome": out.to_dict(),
            "severity": round(sev, 4), "score": out.margin() + SEVERITY_WEIGHT * sev}


@dataclass
class HuntResult:
    evaluations: int
    failures_found: int
    atlas: dict[str, dict]
    history: list[dict]

    def to_dict(self) -> dict:
        return {"evaluations": self.evaluations, "failures_found": self.failures_found,
                "atlas": self.atlas, "history": self.history}


def _cell(scn: Scenario) -> str:
    sev = scn.severity()
    band = next(i for i, b in enumerate(SEVERITY_BANDS) if sev <= b)
    return f"{scn.dominant_stressor()}|{band}"


def hunt(policy_spec: str | None, generations: int = 6, population: int = 24, elite_frac: float = 0.25,
         seed: int = 0, workers: int = 3, base: Scenario | None = None, log=None) -> HuntResult:
    rng = np.random.default_rng(seed)
    base = base or Scenario()
    d = len(DIMS)
    mean = np.full(d, 0.35)
    std = np.full(d, 0.3)
    atlas: dict[str, dict] = {}
    history: list[dict] = []
    evals = 0
    fails = 0
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for g in range(generations):
            X = np.clip(mean + std * rng.standard_normal((population, d)), 0, 1)
            # keep some pure single-stressor probes so the atlas covers every mode
            for i in range(min(d, population // 4)):
                k = int(rng.integers(d))
                X[i] = NEUTRAL.copy()
                X[i, k] = rng.uniform(0.3, 1.0)
            jobs = [(policy_spec, list(x), int(rng.integers(1 << 20)), base.to_dict()) for x in X]
            res = list(pool.map(_eval, jobs)) if pool else [_eval(j) for j in jobs]
            evals += len(res)
            for job, r in zip(jobs, res):
                if not r["outcome"]["passed"]:
                    fails += 1
                    scn = Scenario.from_dict(r["scenario"])
                    cell = _cell(scn)
                    if cell not in atlas or r["severity"] < atlas[cell]["severity"]:
                        atlas[cell] = {**r, "cell": cell, "stressor": scn.dominant_stressor(),
                                       "seed": job[2]}
            scores = np.array([r["score"] for r in res])
            elite = X[np.argsort(scores)[: max(2, int(elite_frac * population))]]
            mean = 0.7 * elite.mean(0) + 0.3 * mean
            std = np.maximum(0.7 * elite.std(0) + 0.3 * std, 0.05)
            row = {"generation": g, "best_score": float(scores.min()), "fail_rate": float(np.mean(
                [not r["outcome"]["passed"] for r in res])), "atlas_cells": len(atlas)}
            history.append(row)
            if log:
                log(row)
    finally:
        if pool:
            pool.shutdown()
    return HuntResult(evals, fails, atlas, history)


def shrink(policy_spec: str | None, failing: Scenario, seed: int, iters: int = 6,
           base: Scenario | None = None) -> tuple[Scenario, float]:
    """Bisect between nominal and ``failing`` for the mildest still-failing world.

    Returns (minimal scenario, alpha) where alpha in (0, 1] is how much of the
    original stress was needed. Small alpha = alarmingly fragile.
    """
    nominal = (base or Scenario()).with_(duration_s=failing.duration_s)
    policy = load_policy(policy_spec)

    def blend(a: float) -> Scenario:
        fd, nd = failing.to_dict(), nominal.to_dict()
        out = dict(fd)
        for k, v in fd.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k in nd:
                out[k] = nd[k] + a * (v - nd[k])
        out["pushes"] = [{**p, "force": p["force"] * a} for p in fd["pushes"]]
        return Scenario.from_dict(out)

    if rollout(policy, failing, seed=seed)[0].passed:
        return failing, 1.0  # flaky: does not reproduce under this seed
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if rollout(policy, blend(mid), seed=seed)[0].passed:
            lo = mid
        else:
            hi = mid
    return blend(hi).with_(name=f"minimal {failing.dominant_stressor()} failure"), hi
