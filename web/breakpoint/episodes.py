"""Which teleoperated demonstrations should a policy learn from?

Demonstration data for robot foundation models is collected by people teleoperating the robot over a
network. Two things quietly poison it:

* **lag**: when the link stalls, the operator acts on a stale picture, so the recorded action no longer
  matches the recorded observation (a steady delay is learnable; a stall or a very slow link is not);
* **near-falls**: an episode where the robot was shoved close to its limit teaches a nominal policy to
  operate at the edge, and one that ended in a fall teaches it to fall.

This module tags each episode from its own telemetry. Every recorded contact is replayed through the
humanoid push-recovery model under that episode's conditions (payload, floor friction, slope, the
latency actually measured at that moment). A contact the model cannot recover from is a *fall*; one it
recovers from but would not survive 25% harder is a *near-fall*; a latency trace that stalls 100 ms above
its own median, or whose median exceeds 200 ms, is *lag*.
Falls are dropped, lag and near-falls are flagged (useful for recovery training, not as nominal demos),
the rest are clean.

Log format, JSON Lines, one episode per line:
    {"episode": "ep-0042", "latency_ms": [18, 22, 240, 31], "payload_kg": 6.0, "friction": 0.7,
     "slope_deg": 0.0, "contacts": [{"t_s": 2.0, "impulse_ns": 35.0, "direction_deg": -90.0}]}
`latency_ms` is sampled once per second; a contact uses the latency sample at its time.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .humanoid import JOBS, Conditions, Humanoid, recover

STALL_MS = 100.0  # a sample this far above the episode's own median is a stall
SLOW_MS = 200.0  # a median this high means the whole episode was driven blind
NEAR_FALL_MARGIN = 1.25  # recovered, but would fall 25% harder


@dataclass
class Verdict:
    episode: str
    tag: str  # clean | lag | near-fall | fall
    reasons: list[str] = field(default_factory=list)


def _num(v, what: str, lo: float | None = None, allow_none: bool = False) -> float:
    """A finite number (not a bool), optionally bounded below; anything else is a clear ValueError."""
    if v is None and allow_none:
        return math.nan
    try:
        finite = not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
    except OverflowError:  # an integer too large for a float, e.g. 10**400 in the JSON
        finite = False
    if not finite:
        raise ValueError(f"{what} must be a finite number, got {str(v)[:40]}")
    if lo is not None and v < lo:
        raise ValueError(f"{what} must be ≥ {lo:g}, got {v!r}")
    return float(v)


def _trace(ep: dict) -> list[float]:
    """Latency samples, one per second. Gaps (null) stay in place as NaN so later samples keep their time."""
    raw = ep.get("latency_ms")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"latency_ms must be a list of numbers (one per second), got {type(raw).__name__}")
    return [_num(x, f"latency_ms[{k}]", lo=0.0, allow_none=True) for k, x in enumerate(raw)]


def _latency_at(trace: list[float], t_s: float) -> tuple[float, str | None]:
    """The sample for the second containing t_s; a gap or a missing trace falls back with a stated reason."""
    finite = [x for x in trace if math.isfinite(x)]
    if not finite:
        return 0.0, "no latency samples: replayed at 0 ms, an optimistic assumption"
    x = trace[min(len(trace) - 1, max(0, int(t_s)))]
    if math.isfinite(x):
        return x, None
    return float(np.median(finite)), f"no latency sample at {t_s:.1f} s: used the episode median"


def tag_episode(ep: dict, robot: Humanoid = Humanoid()) -> Verdict:
    name = str(ep.get("episode", "?"))
    try:
        return _tag(ep, name, robot)
    except ValueError as e:
        raise ValueError(f"episode {name!r}: {e}") from None


def _tag(ep: dict, name: str, robot: Humanoid) -> Verdict:
    trace = _trace(ep)
    payload = _num(ep.get("payload_kg", 0.0), "payload_kg", lo=0.0)
    friction = _num(ep.get("friction", 0.8), "friction", lo=0.01)
    slope = _num(ep.get("slope_deg", 0.0), "slope_deg")
    contacts = ep.get("contacts") or []
    if not isinstance(contacts, list):
        raise ValueError(f"contacts must be a list, got {type(contacts).__name__}")
    reasons: list[str] = []
    fell = near = False
    for k, c in enumerate(contacts):
        if not isinstance(c, dict):
            raise ValueError(f"contacts[{k}] must be an object with t_s, impulse_ns and direction_deg")
        imp = _num(c.get("impulse_ns", 0.0), f"contacts[{k}].impulse_ns", lo=0.0)
        ang = _num(c.get("direction_deg", 0.0), f"contacts[{k}].direction_deg")
        t = _num(c.get("t_s", 0.0), f"contacts[{k}].t_s", lo=0.0)
        lat, note = _latency_at(trace, t)
        if note:
            reasons.append(note)
        cond = Conditions(payload_kg=payload, friction=friction, floor_slope_deg=slope, latency_s=lat / 1000.0)
        r = recover(imp, ang, robot, cond)
        if not r.recovered:
            fell = True
            reasons.append(f"fall at {t:.1f} s: {imp:.0f} N·s shove, {r.reason}")
        elif not recover(imp * NEAR_FALL_MARGIN, ang, robot, cond).recovered:
            near = True
            reasons.append(f"near-fall at {t:.1f} s: {imp:.0f} N·s shove recovered in {r.steps} step(s), "
                           f"{NEAR_FALL_MARGIN - 1:.0%} harder would fall")
    lag = False
    finite = [x for x in trace if math.isfinite(x)]
    if finite:
        peak, med = max(finite), float(np.median(finite))
        if peak - med > STALL_MS or med > SLOW_MS:
            lag = True
            reasons.append(f"lag: median {med:.0f} ms, peak {peak:.0f} ms")
    tag = "fall" if fell else "near-fall" if near else "lag" if lag else "clean"
    return Verdict(name, tag, reasons)


def filter_episodes(episodes: list[dict], robot: Humanoid = Humanoid()) -> dict:
    verdicts = []
    for i, ep in enumerate(episodes, 1):
        if not isinstance(ep, dict):
            raise ValueError(f"record {i}: each episode must be a JSON object")
        try:
            verdicts.append(tag_episode(ep, robot))
        except ValueError as e:
            raise ValueError(f"record {i}, {e}") from None
    counts = {k: sum(v.tag == k for v in verdicts) for k in ("clean", "lag", "near-fall", "fall")}
    n = max(len(verdicts), 1)
    return {"episodes": len(verdicts), "counts": counts,
            "keep_share": round(counts["clean"] / n, 4),
            "drop": [v.episode for v in verdicts if v.tag == "fall"],
            "flag": [{"episode": v.episode, "tag": v.tag, "reasons": v.reasons}
                     for v in sorted(verdicts, key=lambda v: v.tag != "near-fall") if v.tag in ("lag", "near-fall")],
            "verdicts": [v.__dict__ for v in verdicts]}


def read_episodes(path: str | Path) -> list[dict]:
    out = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i}: not valid JSON ({e.msg})") from e
        if not isinstance(rec, dict):
            raise ValueError(f"{path}:{i}: each line must be a JSON object")
        out.append(rec)
    if not out:
        raise ValueError(f"{path}: no episodes")
    return out


def synthetic_episodes(n: int = 200, seed: int = 0, seconds: int = 30) -> list[dict]:
    """Teleop episodes drawn from the teleoperated-data-collection job, with bursty network lag."""
    job = JOBS["teleoperated data collection"]
    rng = np.random.default_rng(seed)
    eps = []
    for i in range(n):
        cond, imp, ang = job.sample(rng)
        base = cond.latency_s * 1000.0
        trace = np.clip(base + rng.normal(0, 8, seconds), 5, None)
        if rng.random() < 0.15:  # a link stall
            k = int(rng.integers(0, seconds))
            trace[k:k + int(rng.integers(1, 4))] += rng.uniform(120, 400)
        t = float(rng.uniform(1, seconds - 1))
        contacts = [{"t_s": round(t, 1), "impulse_ns": round(imp, 1), "direction_deg": round(ang, 1)}] if imp > 0 else []
        eps.append({"episode": f"ep-{i:04d}", "latency_ms": [round(float(x), 1) for x in trace],
                    "payload_kg": round(cond.payload_kg, 2), "friction": round(cond.friction, 3),
                    "slope_deg": round(cond.floor_slope_deg, 2), "contacts": contacts})
    return eps


def to_text(report: dict, show: int = 8) -> str:
    c = report["counts"]
    lines = [(f"{report['episodes']} episodes: {c['clean']} clean, {c['lag']} lag, {c['near-fall']} near-fall, "
              f"{c['fall']} fall  ·  keep {report['keep_share']:.0%} as nominal demonstrations")]
    if report["drop"]:
        lines.append(f"drop ({len(report['drop'])}): " + ", ".join(report["drop"][:show]) + (" …" if len(report["drop"]) > show else ""))
    for f in report["flag"][:show]:
        lines.append(f"  {f['episode']}  {f['tag']:9s}  " + "; ".join(f["reasons"]))
    if len(report["flag"]) > show:
        lines.append(f"  … and {len(report['flag']) - show} more flagged")
    return "\n".join(lines)
