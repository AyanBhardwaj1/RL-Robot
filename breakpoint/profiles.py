"""Deployment profiles: failure probability under where the robot will actually work.

Margins say how far each stressor can be pushed; a deployment decision needs a
different number: "if we ship to a warehouse, how often will it fall?". A profile is a
distribution over scenario parameters. We sample worlds from it, roll out, and report
the failure rate with a 95% upper bound, plus the worst sampled worlds as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .evaluate import evaluate_many, failure_upper_bound
from .scenario import Push, Scenario

Range = tuple[float, float]


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    terrains: dict[str, float]  # terrain -> probability
    friction: Range = (0.6, 1.0)
    roughness: Range = (0.0, 0.0)
    slope_deg: Range = (0.0, 0.0)
    step_height: Range = (0.0, 0.0)
    payload_kg: Range = (0.0, 0.0)
    latency_steps: Range = (0.0, 1.0)
    obs_noise: Range = (0.0, 0.02)
    motor_strength: Range = (0.9, 1.0)
    push_prob: float = 0.0
    push_impulse: Range = (0.0, 0.0)  # N·s, 0.1 s pushes
    extra: dict = field(default_factory=dict)

    def sample(self, rng: np.random.Generator, duration_s: float = 6.0) -> Scenario:
        def u(r: Range) -> float:
            return float(rng.uniform(r[0], r[1])) if r[1] > r[0] else float(r[0])

        names = list(self.terrains)
        probs = np.array([self.terrains[n] for n in names], dtype=float)
        terrain = str(rng.choice(names, p=probs / probs.sum()))
        pushes = []
        if rng.random() < self.push_prob:
            pushes.append(Push(t=u((1.0, 4.0)), force=u(self.push_impulse) / 0.1,
                               direction_deg=u((-180.0, 180.0)), duration=0.1).__dict__)
        d = {
            "name": f"{self.name} sample", "terrain": terrain, "friction": u(self.friction),
            "roughness": u(self.roughness) if terrain == "rough" else 0.0,
            "slope_deg": u(self.slope_deg) if terrain == "slope" else 0.0,
            "step_height": u(self.step_height) if terrain == "stairs" else 0.0,
            "payload_kg": u(self.payload_kg), "latency_steps": round(u(self.latency_steps)),
            "obs_noise": u(self.obs_noise), "motor_strength": u(self.motor_strength),
            "duration_s": duration_s, "pushes": pushes, **self.extra,
        }
        return Scenario.from_dict(d)


PROFILES: dict[str, Profile] = {
    p.name: p
    for p in (
        Profile("warehouse", "Sealed concrete, occasional spills, carries totes, bumped by carts and people.",
                {"flat": 0.85, "slope": 0.15}, friction=(0.35, 1.0), slope_deg=(-4, 4), payload_kg=(0, 4),
                latency_steps=(0, 2), push_prob=0.35, push_impulse=(2, 7)),
        Profile("sidewalk", "Urban pavement, curbs, ramps, crowds, wet after rain.",
                {"flat": 0.4, "rough": 0.25, "slope": 0.2, "stairs": 0.15}, friction=(0.4, 1.0),
                roughness=(0.005, 0.02), slope_deg=(-8, 8), step_height=(0.02, 0.05), payload_kg=(0, 2),
                latency_steps=(0, 3), push_prob=0.3, push_impulse=(2, 6)),
        Profile("home", "Hardwood, rugs, a few steps, pets and kids.",
                {"flat": 0.7, "stairs": 0.2, "rough": 0.1}, friction=(0.3, 0.9), roughness=(0.005, 0.01),
                step_height=(0.02, 0.06), payload_kg=(0, 1), push_prob=0.4, push_impulse=(1, 5)),
        Profile("construction", "Gravel, mud, slopes, rubble, heavy tool payloads, weak comms.",
                {"rough": 0.6, "slope": 0.3, "flat": 0.1}, friction=(0.3, 0.9), roughness=(0.01, 0.05),
                slope_deg=(-12, 12), payload_kg=(0, 5), latency_steps=(0, 4), obs_noise=(0.0, 0.08),
                motor_strength=(0.75, 1.0), push_prob=0.2, push_impulse=(2, 8)),
    )
}


def deployment_check(policy_spec: str | None, profile: str, trials: int = 60, seed: int = 0,
                     workers: int = 3) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    prof = PROFILES[profile]
    rng = np.random.default_rng(seed)
    worlds = [prof.sample(rng) for _ in range(trials)]
    results = [r[0] for r in evaluate_many(policy_spec, worlds, seeds=[seed], workers=workers, distinct_seeds=True)]
    fails = [(w, r) for w, r in zip(worlds, results) if not r["passed"]]
    rate = len(fails) / trials
    by_cause: dict[str, int] = {}
    for w, _ in fails:
        by_cause[w.dominant_stressor()] = by_cause.get(w.dominant_stressor(), 0) + 1
    worst = sorted(fails, key=lambda f: f[1]["margin"])[:5]
    return {
        "profile": prof.name, "description": prof.description, "trials": trials,
        "failures": len(fails), "failure_rate": round(rate, 4),
        "failure_rate_upper_95": round(failure_upper_bound(trials, len(fails)), 4),
        "failures_by_dominant_stressor": dict(sorted(by_cause.items(), key=lambda kv: -kv[1])),
        "worst": [{"scenario": w.to_dict(), "outcome": r} for w, r in worst],
    }
