"""Design levers: which spec change buys the most push robustness, job by job?

A humanoid team trades between actuators, mechanics, controls and networking. This module answers the
question those trades turn on: if one spec improves by a realistic step, how many shifts that used to
end in a fall now finish, and how many that finished now fall? Every lever is evaluated on the *same*
sampled shifts as the baseline (common random numbers), so the comparison is paired: a shift either
flips from fail to pass ("fixed"), from pass to fail ("broken"), or stays the same. Paired counts are
far more sensitive than comparing two failure rates, and the discordant pairs give a sign test for free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from .humanoid import DIRECTIONS, JOBS, Conditions, Humanoid, breaking_point, job_rng, recover


@dataclass(frozen=True)
class Lever:
    key: str
    title: str
    owner: str  # the team that would own the change
    robot: Callable[[Humanoid], Humanoid] = lambda r: r
    cond: Callable[[Conditions], Conditions] = lambda c: c


LEVERS = (
    # A structural null in every job: even at the weakest sampled strength (0.8) and heaviest load the
    # ankle can hold the CoP past the foot edge, so the foot, not the ankle, limits it. Kept as a control.
    Lever("ankle_torque", "Ankle torque +25%", "Actuators",
          robot=lambda r: replace(r, ankle_torque=r.ankle_torque * 1.25)),
    Lever("swing_time", "Swing 15% faster", "Actuators and controls",
          robot=lambda r: replace(r, min_swing=r.min_swing * 0.85)),
    Lever("step_reach", "Leg reach +10 cm", "Mechanical",
          robot=lambda r: replace(r, max_step=r.max_step + 0.10)),
    Lever("lateral_reach", "Side-step reach +10 cm", "Mechanical and controls",
          robot=lambda r: replace(r, max_lateral_step=r.max_lateral_step + 0.10)),
    Lever("stance_width", "Stance 3 cm wider each side", "Controls",
          robot=lambda r: replace(r, stance_half_width=r.stance_half_width + 0.03)),
    Lever("foot_length", "Foot 4 cm longer", "Mechanical",
          robot=lambda r: replace(r, foot_half_length=r.foot_half_length + 0.02)),
    Lever("extra_step", "Allow a 4th recovery step", "Controls",
          robot=lambda r: replace(r, max_steps=r.max_steps + 1)),
    Lever("latency", "20 ms less control latency", "Embedded and networking",
          cond=lambda c: replace(c, latency_s=max(0.0, c.latency_s - 0.020))),
    Lever("grip", "Sole grip +15%", "Mechanical",
          cond=lambda c: replace(c, friction=c.friction * 1.15)),
)


def sign_test_p(fixed: int, broken: int) -> float:
    """Two-sided exact sign test on the discordant pairs (McNemar's exact test)."""
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _outcomes(job: str, robot: Humanoid, lever: Lever | None, trials: int, seed: int) -> np.ndarray:
    rng = job_rng(job, seed)  # same stream with and without the lever: paired
    ok = np.zeros(trials, dtype=bool)
    r = lever.robot(robot) if lever else robot
    for i in range(trials):
        cond, imp, ang = JOBS[job].sample(rng)
        if lever:
            cond = lever.cond(cond)
        ok[i] = recover(imp, ang, r, cond).recovered
    return ok


def lever_table(trials: int = 150, seed: int = 0, robot: Humanoid = Humanoid(),
                jobs: tuple[str, ...] | None = None, levers: tuple[Lever, ...] = LEVERS) -> dict:
    """Paired job outcomes for every lever, plus the push envelope each lever buys."""
    jobs = tuple(jobs or JOBS)
    base = {j: _outcomes(j, robot, None, trials, seed) for j in jobs}
    base_env = {d: breaking_point(DIRECTIONS[d], robot) for d in ("front", "left", "front-left")}
    rows = []
    for lv in levers:
        per_job = {}
        tot_fixed = tot_broken = 0
        for j in jobs:
            new = _outcomes(j, robot, lv, trials, seed)
            fixed = int(np.sum(~base[j] & new))
            broken = int(np.sum(base[j] & ~new))
            tot_fixed += fixed
            tot_broken += broken
            per_job[j] = {"failure_rate": round(float(np.mean(~new)), 4), "fixed": fixed, "broken": broken}
        env = {d: breaking_point(DIRECTIONS[d], lv.robot(robot), lv.cond(Conditions())) for d in base_env}
        rows.append({"key": lv.key, "title": lv.title, "owner": lv.owner, "jobs": per_job,
                     "fixed": tot_fixed, "broken": tot_broken, "net": tot_fixed - tot_broken,
                     "p_value": round(sign_test_p(tot_fixed, tot_broken), 4), "envelope": env})
    rows.sort(key=lambda r: (-r["net"], r["key"]))
    return {"trials_per_job": trials, "seed": seed, "jobs": list(jobs),
            "baseline": {"jobs": {j: {"failure_rate": round(float(np.mean(~base[j])), 4),
                                      "failures": int(np.sum(~base[j]))} for j in jobs},
                         "envelope": base_env},
            "levers": rows}


def to_text(table: dict) -> str:
    lines = [f"Design levers · {table['trials_per_job']} paired shifts per job, same samples as the baseline",
             "baseline envelope (N·s): " + "  ".join(f"{k} {v:.0f}" for k, v in table["baseline"]["envelope"].items()),
             "",
             f"  {'lever':30s} {'owner':24s} {'fixed':>5s} {'broke':>5s} {'net':>4s} {'p':>7s}   side / front N·s"]
    for r in table["levers"]:
        env = r["envelope"]
        lines.append(f"  {r['title']:30s} {r['owner']:24s} {r['fixed']:5d} {r['broken']:5d} {r['net']:+4d} {r['p_value']:7.4f}"
                     f"   {env['left']:.0f} / {env['front']:.0f}")
    lines += ["", "Per job, failure rate with the lever (baseline in brackets):"]
    for j in table["jobs"]:
        b = table["baseline"]["jobs"][j]["failure_rate"]
        if table["baseline"]["jobs"][j]["failures"] == 0:
            lines.append(f"  {j:30s} [{b:.1%}]  no failures to fix")
            continue
        best = max(table["levers"], key=lambda r: r["jobs"][j]["fixed"] - r["jobs"][j]["broken"])
        lines.append(f"  {j:30s} [{b:.1%}]  best: {best['title']} → {best['jobs'][j]['failure_rate']:.1%}")
    return "\n".join(lines)
