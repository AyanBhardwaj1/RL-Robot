"""Plain-language explanations of a rollout outcome ("why did it fall?").

Deterministic and evidence-based: every clause is computed from the scenario and the
measured outcome, so the explanation cannot hallucinate a cause that wasn't present.
"""

from __future__ import annotations

from .scenario import BOUNDS, Scenario


def direction_name(deg: float) -> str:
    """Where a push comes *from*, given its force direction (0 = forward, +90 = toward the robot's left)."""
    from .probe import PUSH_DIRS

    def gap(a: float) -> float:
        return abs((deg - a + 180.0) % 360.0 - 180.0)

    return "the " + min(PUSH_DIRS, key=lambda n: gap(PUSH_DIRS[n]))


def contributing_factors(scn: Scenario) -> list[str]:
    f = []
    if scn.friction < 0.5:
        f.append(f"slippery ground (μ={scn.friction:.2f})")
    if scn.payload_kg >= 1.0:
        side = " front-heavy" if scn.payload_offset_x > 0.05 else (" rear-heavy" if scn.payload_offset_x < -0.05 else "")
        f.append(f"a {scn.payload_kg:.1f} kg{side} payload")
    if scn.motor_strength < 0.85:
        f.append(f"motors at {scn.motor_strength:.0%} torque")
    if scn.latency_steps >= 1:
        f.append(f"{int(scn.latency_steps) * 20} ms of control latency")
    if scn.obs_noise >= 0.05:
        f.append(f"noisy sensors (σ={scn.obs_noise:.2f})")
    if abs(scn.slope_deg) >= 3:
        f.append(f"a {abs(scn.slope_deg):.0f}° {'uphill' if scn.slope_deg > 0 else 'downhill'} slope")
    if scn.step_height > 0:
        f.append(f"{scn.step_height * 100:.0f} cm stairs")
    if scn.roughness > 0.005:
        f.append(f"rough ground (±{scn.roughness * 100:.1f} cm)")
    return f


def explain(scn: Scenario, outcome: dict) -> str:
    factors = contributing_factors(scn)
    ctx = (" with " + ", ".join(factors[:-1]) + (" and " if len(factors) > 1 else "") + factors[-1]) if factors else ""
    if outcome.get("passed"):
        took = [p for p in scn.pushes if p.t <= outcome["time_s"]]
        if took:
            p = max(took, key=lambda q: q.impulse)
            return (f"Took a {p.impulse:.1f} N·s shove from {direction_name(p.direction_deg)} at {p.t:.1f} s and kept "
                    f"walking: {outcome['distance']:.1f} m in {outcome['time_s']:.1f} s{ctx}.")
        return f"Walked {outcome['distance']:.1f} m and stayed upright for {outcome['time_s']:.1f} s{ctx}."
    if outcome.get("fell"):
        t = outcome["time_s"]
        recent = [p for p in scn.pushes if p.t <= t <= p.t + p.duration + 1.5]
        if recent:
            p = recent[-1]
            lead = (f"Fell {t - p.t:.1f} s after a {p.impulse:.1f} N·s shove from {direction_name(p.direction_deg)}")
        else:
            lead = f"Fell at {t:.1f} s with no shove nearby"
        how = "tipped over" if outcome.get("min_upright", 1) < 0.45 else "collapsed (body hit the ground)"
        return f"{lead}: it {how}{ctx}."
    expected = scn.target_speed * scn.duration_s
    sev = max(0.0, (1.0 - scn.motor_strength) / (1 - BOUNDS['motor_strength'][0]))
    reason = "too weak to make headway" if sev > 0.5 else "unable to make headway"
    return (f"Stayed upright but stalled: covered {outcome['distance']:.1f} m of {expected:.1f} m expected, "
            f"{reason}{ctx}.")
