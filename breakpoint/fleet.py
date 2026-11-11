"""From fleet telemetry to deployment odds.

The job profiles in humanoid.py are hand-written guesses ("totes up to 25 kg, spills down to μ 0.3"). A fleet
knows better: every shift logs what the robot carried, how slippery the floor was estimated to be, how much
control lag it saw, the tallest step on its route and every external contact. This module fits a job profile
from those logs (robust quantiles, not min/max, so one bad sensor reading does not define the site) and
re-runs the odds for both bodies, so readiness tracks the site as it really is, night after night.

Log format: JSON Lines, one record per shift:
    {"site": "dc-austin-3", "payload_kg": 14.2, "friction": 0.55, "latency_ms": 22,
     "slope_deg": 1.1, "floor_step_m": 0.0, "contacts_ns": [18.5, 31.0]}
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .humanoid import JOBS, Humanoid, Job, WheeledBase, job_risk

FIELDS = ("payload_kg", "friction", "latency_ms", "slope_deg", "floor_step_m")
LOW_Q, HIGH_Q = 0.05, 0.95


def read_log(path: str | Path) -> list[dict]:
    rows = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i}: not valid JSON ({e.msg})") from e
        if not isinstance(rec, dict):
            raise ValueError(f"{path}:{i}: each line must be a JSON object")
        rows.append(rec)
    if not rows:
        raise ValueError(f"{path}: no shift records")
    return rows


def _col(records: Iterable[dict], key: str, default: float) -> np.ndarray:
    vals = []
    for r in records:
        v = r.get(key, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            vals.append(v)
    return np.asarray(vals if vals else [default], dtype=float)


def _range(a: np.ndarray, lo: float, hi: float, clip: tuple[float, float]) -> tuple[float, float]:
    q = np.clip(np.quantile(a, [lo, hi]), *clip)
    return float(round(q[0], 4)), float(round(max(q[1], q[0]), 4))


def fit_profile(records: list[dict], name: str = "fitted site") -> Job:
    """Fit a Job profile from per-shift telemetry using robust quantiles."""
    payload = _col(records, "payload_kg", 0.0)
    friction = _col(records, "friction", 0.8)
    latency = _col(records, "latency_ms", 0.0) / 1000.0
    slope = _col(records, "slope_deg", 0.0)
    step = _col(records, "floor_step_m", 0.0)
    contacts = [c for r in records for c in (r.get("contacts_ns") or []) if isinstance(c, (int, float)) and c > 0]
    shifts_with_contact = sum(1 for r in records if r.get("contacts_ns"))
    stepped = step[step > 0.005]
    return Job(
        name=name,
        description=f"Fitted from {len(records)} logged shifts.",
        payload=_range(payload, LOW_Q, HIGH_Q, (0.0, 25.0)),
        friction=_range(friction, LOW_Q, HIGH_Q, (0.05, 1.5)),
        latency=_range(latency, LOW_Q, HIGH_Q, (0.0, 0.5)),
        slope=_range(slope, LOW_Q, HIGH_Q, (-20.0, 20.0)),
        push_prob=round(shifts_with_contact / len(records), 4),
        push_impulse=_range(np.asarray(contacts or [0.0]), 0.10, HIGH_Q, (0.0, 200.0)),
        floor_step=_range(stepped if stepped.size else np.zeros(1), LOW_Q, HIGH_Q, (0.0, 0.3)),
        step_prob=round(float(np.mean(step > 0.005)), 4),
    )


def synthetic_log(job: str, shifts: int = 300, seed: int = 0, sensor_noise: float = 0.05) -> list[dict]:
    """Shift records drawn from a known profile, with sensor noise and the odd bad reading (for demos and tests)."""
    if job not in JOBS:
        raise ValueError(f"unknown job {job!r}; choose from {sorted(JOBS)}")
    rng = np.random.default_rng(seed)
    out = []
    for i in range(shifts):
        cond, imp, _ = JOBS[job].sample(rng)
        rec = {
            "site": job.replace(" ", "-"), "shift": i,
            "payload_kg": round(cond.payload_kg * (1 + sensor_noise * rng.standard_normal()), 2),
            "friction": round(max(0.05, cond.friction + sensor_noise * rng.standard_normal()), 3),
            "latency_ms": round(cond.latency_s * 1000, 1),
            "slope_deg": round(cond.floor_slope_deg, 2),
            "floor_step_m": round(cond.floor_step_m, 3),
            "contacts_ns": [round(imp, 1)] if imp > 0 else [],
        }
        if rng.random() < 0.01:  # a glitched sensor reading the quantiles must ignore
            rec["friction"] = 9.99
        out.append(rec)
    return out


def site_odds(records: list[dict], name: str = "fitted site", trials: int = 150, seed: int = 0) -> dict:
    """Fit the profile, register it temporarily and return odds for both bodies."""
    profile = fit_profile(records, name)
    key = f"__fleet__{name}"
    JOBS[key] = profile
    try:
        odds = {body: job_risk(key, robot, trials=trials, seed=seed)
                for body, robot in (("biped", Humanoid()), ("wheeled", WheeledBase()))}
    finally:
        JOBS.pop(key, None)
    return {"profile": profile.__dict__, "odds": odds}
