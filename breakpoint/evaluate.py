"""Rollouts and metrics. Every judgement in Breakpoint bottoms out in ``rollout``."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from .env import QuadrupedEnv
from .policy import Policy, load_policy
from .scenario import Scenario

MIN_PROGRESS_FRAC = 0.35  # must cover >= 35% of commanded distance to "pass"


@dataclass(frozen=True)
class Outcome:
    survived: bool
    fell: bool
    time_s: float
    distance: float
    min_clearance: float
    min_upright: float
    mean_speed: float
    progress_ok: bool

    @property
    def passed(self) -> bool:
        return self.survived and self.progress_ok

    def margin(self) -> float:
        """Continuous 'how close to failing' score: >0 passed, <0 failed.

        Gives the adversarial search a gradient-like signal instead of a 0/1 cliff.
        """
        upright_m = (self.min_upright - 0.42) / 0.58
        clear_m = (self.min_clearance - 0.11) / 0.15
        base = min(upright_m, clear_m)
        if not self.survived:
            # falling later is "less failed" than falling immediately
            return -1.0 + 0.5 * min(1.0, self.time_s / 10.0)
        if not self.progress_ok:
            return min(base, -0.1)
        return base

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        d["margin"] = round(self.margin(), 4)
        return d


def rollout(policy: Policy, scn: Scenario, seed: int = 0, record: bool = False,
            env: QuadrupedEnv | None = None) -> tuple[Outcome, list | None]:
    env = env or QuadrupedEnv(scn, seed=seed)
    obs = env.reset(scn, seed=seed)
    frames: list | None = [] if record else None
    min_clear, min_up = 1e9, 1.0
    done = False
    info = None
    while not done:
        obs, _, done, info = env.step(policy.act(obs))
        min_clear = min(min_clear, info.clearance)
        min_up = min(min_up, info.upright)
        if frames is not None:
            frames.append(env.body_poses())
    assert info is not None
    expected = scn.target_speed * scn.duration_s
    progress_ok = expected < 0.05 or info.x >= MIN_PROGRESS_FRAC * expected
    out = Outcome(
        survived=not info.fell,
        fell=info.fell,
        time_s=round(env.t, 3),
        distance=round(info.x, 3),
        min_clearance=round(min_clear, 4),
        min_upright=round(min_up, 4),
        mean_speed=round(info.x / max(env.t, 1e-6), 3),
        progress_ok=progress_ok,
    )
    return out, frames


def _job(args: tuple[str | None, dict, int]) -> dict:
    spec, scn_d, seed = args
    out, _ = rollout(load_policy(spec), Scenario.from_dict(scn_d), seed=seed)
    return out.to_dict()


def evaluate_many(policy_spec: str | None, scenarios: list[Scenario], seeds: list[int],
                  workers: int = 3) -> list[list[dict]]:
    """Evaluate every scenario under every seed. Returns [scenario][seed] -> outcome dict."""
    jobs = [(policy_spec, s.to_dict(), sd) for s in scenarios for sd in seeds]
    if workers <= 1 or len(jobs) < 4:
        flat = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            flat = list(ex.map(_job, jobs, chunksize=max(1, len(jobs) // (workers * 4))))
    k = len(seeds)
    return [flat[i * k:(i + 1) * k] for i in range(len(scenarios))]


def pass_rate(results: list[dict]) -> float:
    return float(np.mean([r["passed"] for r in results])) if results else 0.0


def failure_upper_bound(n_trials: int, n_failures: int = 0, conf: float = 0.95) -> float:
    """One-sided upper confidence bound on failure probability.

    With zero observed failures this is the exact binomial bound 1-(1-conf)^(1/n)
    (~3/n, the 'rule of three'); otherwise a Wilson score upper bound.
    """
    if n_trials <= 0:
        return 1.0
    if n_failures == 0:
        return float(1.0 - (1.0 - conf) ** (1.0 / n_trials))
    z = 1.6448536269514722 if conf == 0.95 else 2.326
    p = n_failures / n_trials
    denom = 1 + z * z / n_trials
    centre = p + z * z / (2 * n_trials)
    rad = z * np.sqrt(p * (1 - p) / n_trials + z * z / (4 * n_trials * n_trials))
    return float(min(1.0, (centre + rad) / denom))
