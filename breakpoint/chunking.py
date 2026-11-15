"""What action chunking costs a robot that gets shoved.

Physical-AI policies (vision-language-action models and diffusion or flow policies) usually output a chunk
of future actions and execute it open-loop. That buys smooth, temporally consistent motion, but a shove that
lands in the middle of a chunk cannot be answered until the policy acts on a new observation:

* **synchronous chunking**: observe, infer for d, execute the chunk (length H), repeat. A shove at a
  uniformly random moment in that H + d cycle is first seen at the next observation and answered d later:
  it waits ``phase * (H + d) + d``, with ``phase`` in [0, 1].
* **real-time chunking**: the next chunk is generated while the current one runs and the robot switches
  as soon as it is ready (Black, Galliker and Levine, "Real-Time Execution of Action Chunking Flow
  Policies", NeurIPS 2025, arXiv:2506.07339). The policy re-plans every ``d``, so the wait is
  ``phase * d + d``.

Here that wait is fed to the humanoid push-recovery model as control latency, so the cost of a chunking
scheme is read straight off the push envelope. Read it as: *if the chunked policy closes the balance loop
itself*, this is what the delay to its first reaction costs. Real humanoid stacks usually keep a fast
low-level balance controller under the policy, which a chunk does not delay. After that first reaction
the model re-plans every tick (optimistic), and while waiting it has no joint stiffness (pessimistic). It
says nothing about the smoothness at chunk boundaries that real-time chunking also fixes.
"""

from __future__ import annotations

from dataclasses import replace

from .humanoid import BODIES, Conditions, breaking_point

PHASES = {"median": 0.5, "worst": 1.0}


def reaction_delay_ms(chunk_ms: float, inference_ms: float, phase: float, realtime: bool) -> float:
    """Time from a shove to the first action that can respond to it."""
    if chunk_ms < 0 or inference_ms < 0 or not 0.0 <= phase <= 1.0:
        raise ValueError("chunk_ms and inference_ms must be ≥ 0 and phase in [0, 1]")
    cycle = inference_ms if realtime else chunk_ms + inference_ms
    return phase * cycle + inference_ms


def chunking_table(chunks_ms=(0, 50, 100, 200, 400), inference_ms: float = 50.0, body: str = "biped",
                   cond: Conditions = Conditions()) -> dict:
    """Largest survivable shove (front and side) for synchronous chunks of each length vs real-time chunking."""
    if body not in BODIES:
        raise ValueError(f"unknown body {body!r}; choose from {list(BODIES)}")
    robot = BODIES[body]

    def envelope_at(delay_ms: float) -> dict:
        c = replace(cond, latency_s=cond.latency_s + delay_ms / 1000.0)
        return {"front": breaking_point(180.0, robot, c), "side": breaking_point(-90.0, robot, c)}

    rows = []
    for h in chunks_ms:
        row = {"scheme": "synchronous", "chunk_ms": h}
        for name, ph in PHASES.items():
            d = reaction_delay_ms(h, inference_ms, ph, realtime=False)
            row[name] = {"delay_ms": d, **envelope_at(d)}
        rows.append(row)
    rt = {"scheme": "real-time", "chunk_ms": None}
    for name, ph in PHASES.items():
        d = reaction_delay_ms(0.0, inference_ms, ph, realtime=True)
        rt[name] = {"delay_ms": d, **envelope_at(d)}
    rows.append(rt)
    return {"body": body, "inference_ms": inference_ms, "no_delay": envelope_at(0.0), "rows": rows}


def chunking_tables(chunks_ms=(0, 50, 100, 200, 400), inference_ms: float = 50.0,
                    cond: Conditions = Conditions()) -> dict:
    """The chunking table for every body the lab compares, keyed by body name."""
    return {"inference_ms": inference_ms,
            "bodies": {name: chunking_table(chunks_ms, inference_ms, name, cond) for name in BODIES}}


def to_text(t: dict) -> str:
    nd = t["no_delay"]
    lines = [(f"Action chunking vs push recovery · {t['body']} · inference {t['inference_ms']:.0f} ms "
              f"(no delay at all: front {nd['front']:.0f}, side {nd['side']:.0f} N·s)"), "",
             f"  {'scheme':28s} {'median wait':>12s} {'front':>6s} {'side':>5s}   {'worst wait':>11s} {'front':>6s} {'side':>5s}"]
    for r in t["rows"]:
        label = "real-time chunking" if r["scheme"] == "real-time" else f"synchronous, {r['chunk_ms']:.0f} ms chunks"
        m, w = r["median"], r["worst"]
        lines.append(f"  {label:28s} {m['delay_ms']:>9.0f} ms {m['front']:>6.0f} {m['side']:>5.0f}   "
                     f"{w['delay_ms']:>8.0f} ms {w['front']:>6.0f} {w['side']:>5.0f}")
    return "\n".join(lines)
