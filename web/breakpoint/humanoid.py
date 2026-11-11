"""Humanoid push recovery: linear inverted pendulum + capture-point stepping.

A humanoid's balance problem reduced to its essentials. The centre of mass (CoM) moves like a
linear inverted pendulum (LIP) over the centre of pressure (CoP):

    x'' = w^2 (x - p),   w = sqrt(g / z0)

The *capture point* xi = x + x'/w is where the robot must place its CoP to come to rest
(Pratt et al. 2006; Koolen et al. 2012, "N-step capturability"). It diverges from the CoP:
xi' = w (xi - p). If xi is inside the support foot, ankle torque alone recovers (0 steps).
Otherwise the robot must step, and every limit of a real machine bounds that step:

* foot size bounds the CoP (ankle strategy),
* friction bounds the horizontal force, |x - p| <= mu * z0,
* actuator speed sets the minimum swing time,
* leg length sets the maximum step reach,
* sensing and control latency delays every decision,
* a tote held in front shifts the combined CoM forward and raises it.

`recover()` simulates the push and the stepping controller; `breaking_point()` bisects the
largest impulse it survives in a direction, exactly like the quadruped probe. Everything is
deterministic and dependency-free (numpy only), and is mirrored line for line in
site/assets/humanoid.js so the browser demo and this module agree (see tests).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

G = 9.81
DT = 0.002  # integration step (s)

# Robot frame: +x forward, +y left. A push "from the left" pushes toward -y.
DIRECTIONS = {"front": 180.0, "front-left": -135.0, "left": -90.0, "rear-left": -45.0,
              "rear": 0.0, "rear-right": 45.0, "right": 90.0, "front-right": 135.0}


@dataclass(frozen=True)
class Humanoid:
    """A generic full-size humanoid (not a model of any specific product)."""

    mass_kg: float = 72.0
    com_height: float = 0.95
    foot_half_length: float = 0.11
    foot_half_width: float = 0.05
    stance_half_width: float = 0.11  # lateral distance from body centre to each foot
    max_step: float = 0.55  # m, leg reach from the stance foot
    max_lateral_step: float = 0.40  # m sideways: the swing leg cannot cross the stance leg
    min_swing: float = 0.32  # s, fastest swing the actuators allow
    max_steps: int = 3
    max_step_up: float = 0.18  # m, tallest step the legs can climb
    ankle_torque: float = 150.0  # N·m per ankle: caps how far the CoP can move from the ankle
    swing_load_factor: float = 0.6  # swing slows by this fraction per body-mass of payload

    @property
    def omega(self) -> float:
        return math.sqrt(G / self.com_height)


@dataclass(frozen=True)
class Conditions:
    payload_kg: float = 0.0  # tote held in front of the body
    payload_reach: float = 0.35  # m in front of the CoM
    payload_height: float = 1.05  # m above the floor
    friction: float = 0.8
    latency_s: float = 0.0  # sensing + control delay before a step decision
    actuator_strength: float = 1.0  # < 1 slows the swing (weak or hot actuators)
    floor_slope_deg: float = 0.0  # adds a constant downhill CoM acceleration along +x
    floor_step_m: float = 0.0  # tallest curb, threshold or stair on the route

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Recovery:
    recovered: bool
    steps: int
    reason: str
    time_s: float
    com: list = field(default_factory=list)  # [(t, x, y)]
    capture: list = field(default_factory=list)  # [(t, xi_x, xi_y)]
    footsteps: list = field(default_factory=list)  # [(t, x, y, side)]

    def summary(self) -> dict:
        return {"recovered": self.recovered, "steps": self.steps, "reason": self.reason, "time_s": round(self.time_s, 3)}


def effective(robot: Humanoid, cond: Conditions) -> tuple[float, float, float]:
    """Combined CoM height, forward CoM offset and total mass with the payload."""
    m = robot.mass_kg + cond.payload_kg
    z = (robot.mass_kg * robot.com_height + cond.payload_kg * cond.payload_height) / m
    offset = cond.payload_kg * cond.payload_reach / m
    return z, offset, m


def _clamp_to_foot(p: np.ndarray, foot: np.ndarray, robot: Humanoid) -> np.ndarray:
    return np.array([np.clip(p[0], foot[0] - robot.foot_half_length, foot[0] + robot.foot_half_length),
                     np.clip(p[1], foot[1] - robot.foot_half_width, foot[1] + robot.foot_half_width)])


def _ticks(latency_s: float) -> int:
    """Latency in control ticks, rounding halves up (as Math.round and std::lround do), never negative."""
    return max(0, math.floor(latency_s / DT + 0.5))


def recover(impulse: float, direction_deg: float, robot: Humanoid = Humanoid(), cond: Conditions = Conditions(),
            record: bool = False, horizon_s: float = 5.0) -> Recovery:
    """Simulate a CoM impulse (N·s) and the capture-point stepping controller.

    The robot starts in double support; the push lands on the CoM. The controller:
      1. uses ankle torque: CoP goes to the point of the support nearest the capture point *as the
         controller sees it* (delayed by the control latency), limited by friction (|x - p| <= mu * z);
      2. if that delayed capture point leaves the support, it steps. While the swing foot is in the
         air only the stance foot carries weight. After the swing time it places the new foot where
         the capture point will be at touchdown (projected over swing time plus latency), limited by
         leg reach and lateral reach;
      3. it falls if the CoM leaves friction limits, the capture point outruns the reach, or
         it needs more than `max_steps` steps.
    """
    if cond.floor_step_m > robot.max_step_up:
        return Recovery(False, 0, f"blocked: a {cond.floor_step_m * 100:.0f} cm step is taller than it can climb", 0.0)
    z, offset, m = effective(robot, cond)
    w = math.sqrt(G / z)
    swing = robot.min_swing * (1 + robot.swing_load_factor * cond.payload_kg / robot.mass_kg) / max(cond.actuator_strength, 0.05)
    slope_acc = G * math.sin(math.radians(cond.floor_slope_deg))
    # A constant downhill pull is cancelled by leaning: the controller works with the capture point
    # shifted by the equilibrium CoP offset a/w^2 (the robot leans into the slope).
    bias = np.array([slope_acc / (w * w), 0.0])
    # Ankle torque bounds the CoP offset from the ankle: |p - ankle| <= tau / (m g). Conservatively the same
    # bound is used in double support (two ankles could share the load).
    cop_reach = robot.ankle_torque * max(cond.actuator_strength, 0.05) / (m * G)
    # Double support: treat the support as one wide foot centred under the body.
    stance = np.array([0.0, 0.0])
    support_half_w = robot.stance_half_width + robot.foot_half_width
    x = np.array([offset, 0.0])
    pre_push = x + bias  # capture point while standing, before the shove lands
    ang = math.radians(direction_deg)
    v = (impulse / m) * np.array([math.cos(ang), math.sin(ang)])
    t, steps = 0.0, 0
    double_support = True
    swing_until = -1.0
    planned = None
    latency_steps = _ticks(cond.latency_s)
    history: list[np.ndarray] = []
    rec = Recovery(False, 0, "", 0.0)
    max_offset = cond.friction * z  # friction cone: horizontal force <= mu * m * g

    def foot_box(center: np.ndarray, half_w: float) -> tuple[np.ndarray, np.ndarray]:
        hx = min(robot.foot_half_length, cop_reach)
        return (center - np.array([hx, half_w]), center + np.array([hx, half_w]))

    # Latency compensation (a Smith predictor): the controller measures the capture point `latency` late,
    # but knows the CoP it has commanded since, so it rolls the measurement forward with the exact
    # discrete LIP map xi' = p + (xi - p) e^(w dt). With S the running sum of the CoP terms over the last
    # `latency` ticks, the estimate of now is E^n * seen - S, updated in O(1) per tick. What latency costs
    # is then exactly what it should: the shove goes unseen for `latency` seconds.
    e_dt = math.exp(w * DT)
    e_n = e_dt ** latency_steps
    p0 = np.clip(pre_push, *foot_box(stance, support_half_w))  # CoP while standing before the shove
    cop_ring = [p0.copy() for _ in range(latency_steps)]
    cop_sum = (e_n - 1.0) * p0
    tick = 0
    swing_stance = stance.copy()  # the loaded foot during the first swing (set when the step is planned)

    while t < horizon_s:
        xi = x + v / w + bias
        history.append(xi.copy())
        # Until the delay has elapsed the controller still sees the robot standing still (pre-push state).
        seen = history[-1 - latency_steps] if len(history) > latency_steps else pre_push
        est = e_n * seen - cop_sum if latency_steps else seen
        half_w = support_half_w if double_support else robot.foot_half_width
        lo, hi = foot_box(stance, half_w)
        if planned is not None and double_support:
            # first swing: the stepping foot is in the air, so only the other foot can hold the CoP
            p = np.clip(est, *foot_box(swing_stance, robot.foot_half_width))
        else:
            p = np.clip(est, lo, hi)
        if latency_steps:
            j = tick % latency_steps
            cop_sum = e_dt * cop_sum + (e_dt - 1.0) * p - e_n * (e_dt - 1.0) * cop_ring[j]
            cop_ring[j] = p.copy()
        tick += 1
        # friction: the CoP may not sit further than mu*z from the CoM horizontally
        d = x - p
        dn = float(np.linalg.norm(d))
        if dn > max_offset:
            rec.reason = f"slipped: needed friction {dn / z:.2f} > {cond.friction:.2f}"
            break
        inside = bool(np.all(est >= lo - 1e-9) and np.all(est <= hi + 1e-9))
        if planned is None and not inside:
            if steps >= robot.max_steps:
                rec.reason = f"needed more than {robot.max_steps} steps"
                break
            # From double support the leg on the side of the step swings and the other one (the swing leg
            # cannot cross it) carries the weight, so that foot sets the CoP during the swing.
            if double_support:
                side = 1.0 if est[1] - stance[1] > 1e-9 else -1.0 if est[1] - stance[1] < -1e-9 else 1.0
                swing_stance = np.array([stance[0], stance[1] - side * robot.stance_half_width])
                p_swing = np.clip(est, *foot_box(swing_stance, robot.foot_half_width))
            else:
                p_swing = p
            # step where the (estimated) capture point will be at touchdown, given the CoP during the swing
            pred = p_swing + (est - p_swing) * math.exp(w * swing)
            step_vec = pred - stance
            reach = float(np.linalg.norm(step_vec))
            if reach > robot.max_step:
                step_vec = step_vec * (robot.max_step / reach)
            step_vec[1] = float(np.clip(step_vec[1], -robot.max_lateral_step, robot.max_lateral_step))
            planned = stance + step_vec
            swing_until = t + swing
        if planned is not None and t >= swing_until:
            stance = planned
            planned = None
            double_support = False
            steps += 1
            if record:
                rec.footsteps.append((round(t, 3), round(float(stance[0]), 4), round(float(stance[1]), 4), steps % 2))
        # integrate LIP (semi-implicit Euler)
        a = w * w * (x - p) + np.array([slope_acc, 0.0])
        v = v + a * DT
        x = x + v * DT
        t += DT
        if record and round(t / DT) % 10 == 0:
            rec.com.append((round(t, 3), round(float(x[0]), 4), round(float(x[1]), 4)))
            rec.capture.append((round(t, 3), round(float(xi[0]), 4), round(float(xi[1]), 4)))
        speed = float(np.linalg.norm(v))
        xi_now = x + v / w + bias
        rest_lo, rest_hi = foot_box(stance, support_half_w if double_support else robot.foot_half_width)
        if planned is None and speed < 0.02 and bool(np.all(xi_now >= rest_lo) and np.all(xi_now <= rest_hi)) and t > 0.3:
            rec.recovered, rec.reason = True, "came to rest"
            break
        if float(np.linalg.norm(xi_now - stance)) > robot.max_step * 3:
            rec.reason = "capture point ran away beyond leg reach"
            break
    else:
        rec.reason = rec.reason or "still moving at the end of the horizon"
        # settled if the capture point is inside the final support
        lo, hi = foot_box(stance, support_half_w if double_support else robot.foot_half_width)
        xi_end = x + v / w + bias
        rec.recovered = bool(np.all(xi_end >= lo) and np.all(xi_end <= hi))
        if rec.recovered:
            rec.reason = "came to rest"
    rec.steps, rec.time_s = steps, t
    return rec


@dataclass(frozen=True)
class WheeledBase:
    """An upper body on a statically stable wheeled base (no stepping; it drives instead)."""

    mass_kg: float = 90.0
    com_height: float = 1.0
    half_length: float = 0.30  # footprint half-size along x (m)
    half_width: float = 0.24
    max_accel: float = 2.5  # m/s^2 the drive can command
    max_speed: float = 1.5  # m/s
    reaction_s: float = 0.05  # drive reaction time on top of control latency
    max_step_up: float = 0.025  # m: wheels roll over a threshold, not a curb or stair


def recover_wheeled(impulse: float, direction_deg: float, base: WheeledBase = WheeledBase(),
                    cond: Conditions = Conditions(), record: bool = False, horizon_s: float = 5.0) -> Recovery:
    """Same push, but the support polygon is the base footprint and the base drives to catch the capture point.

    The CoP is clamped to the footprint; if the capture point leaves it, the base accelerates toward the
    capture point (bounded acceleration and speed) after the reaction delay. It tips over if the capture
    point gets further than the footprint plus what the drive can still close.
    """
    if cond.floor_step_m > base.max_step_up:
        return Recovery(False, 0, f"blocked: a {cond.floor_step_m * 100:.0f} cm step is taller than it can climb", 0.0)
    m = base.mass_kg + cond.payload_kg
    z = (base.mass_kg * base.com_height + cond.payload_kg * cond.payload_height) / m
    offset = cond.payload_kg * cond.payload_reach / m
    w = math.sqrt(G / z)
    accel = base.max_accel * max(cond.actuator_strength, 0.05) * min(1.0, 2 * cond.friction)  # traction-limited
    slope_acc = G * math.sin(math.radians(cond.floor_slope_deg))
    bias = np.array([slope_acc / (w * w), 0.0])
    x = np.array([offset, 0.0])
    pre_push = x + bias  # relative to the base, which starts at the origin
    b = np.zeros(2)
    bv = np.zeros(2)
    ang = math.radians(direction_deg)
    v = (impulse / m) * np.array([math.cos(ang), math.sin(ang)])
    half = np.array([base.half_length, base.half_width])
    delay = _ticks(cond.latency_s + base.reaction_s)
    history: list[np.ndarray] = []
    rec = Recovery(False, 0, "", 0.0)
    t = 0.0
    while t < horizon_s:
        xi = x + v / w + bias
        history.append(xi - b)
        seen = history[-1 - delay] if len(history) > delay else pre_push
        p = np.clip(xi, b - half, b + half)
        if float(np.linalg.norm(x - p)) > cond.friction * z:
            rec.reason = f"slipped: needed friction {float(np.linalg.norm(x - p)) / z:.2f} > {cond.friction:.2f}"
            break
        # drive: chase the part of the capture point that lies outside the footprint, then settle
        err = seen - np.clip(seen, -half * 0.5, half * 0.5)
        desired = np.clip(err * 4.0, -base.max_speed, base.max_speed) if np.any(np.abs(err) > 1e-6) else np.zeros(2)
        dv = desired - bv
        n = float(np.linalg.norm(dv))
        if n > accel * DT:
            dv = dv * (accel * DT / n)
        # reaction force of accelerating the base acts on the body like an extra CoP shift
        bv = bv + dv
        b = b + bv * DT
        a = w * w * (x - p) + np.array([slope_acc, 0.0])
        v = v + a * DT
        x = x + v * DT
        t += DT
        if record and round(t / DT) % 10 == 0:
            rec.com.append((round(t, 3), round(float(x[0]), 4), round(float(x[1]), 4)))
            rec.capture.append((round(t, 3), round(float(xi[0]), 4), round(float(xi[1]), 4)))
            rec.footsteps.append((round(t, 3), round(float(b[0]), 4), round(float(b[1]), 4), 2))
        xi_rel = x + v / w + bias - b
        if np.any(np.abs(xi_rel) > half + base.max_speed ** 2 / (2 * accel) + 0.25):
            rec.reason = "tipped over: the base could not drive under the capture point"
            break
        if float(np.linalg.norm(v)) < 0.02 and float(np.linalg.norm(bv)) < 0.02 and np.all(np.abs(xi_rel) <= half) and t > 0.3:
            rec.recovered, rec.reason = True, "came to rest"
            break
    else:
        xi_rel = x + v / w + bias - b
        rec.recovered = bool(np.all(np.abs(xi_rel) <= half))
        rec.reason = "came to rest" if rec.recovered else "still moving at the end of the horizon"
    rec.time_s = t
    return rec


def _bisect(ok, max_impulse: float, iters: int) -> float:
    if not ok(0.0):
        return 0.0
    if ok(max_impulse):
        return max_impulse
    lo, hi = 0.0, max_impulse
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return round(lo, 2)


def breaking_point(direction_deg: float, robot: Humanoid | WheeledBase = Humanoid(), cond: Conditions = Conditions(),
                   max_impulse: float = 200.0, iters: int = 12) -> float:
    """Largest impulse (N·s) recovered in this direction, by bisection (biped or wheeled base)."""
    sim = recover_wheeled if isinstance(robot, WheeledBase) else recover
    return _bisect(lambda j: sim(j, direction_deg, robot, cond).recovered, max_impulse, iters)


def envelope(robot: Humanoid | WheeledBase = Humanoid(), cond: Conditions = Conditions()) -> dict[str, float]:
    return {name: breaking_point(deg, robot, cond) for name, deg in DIRECTIONS.items()}


def zero_step_limit(direction_deg: float, robot: Humanoid = Humanoid(), cond: Conditions = Conditions()) -> float:
    """Analytic ankle-only limit at zero latency: the impulse that puts the capture point on the support edge.

    Uses the same support as the simulation: the foot length capped by ankle torque, and the capture
    point's standing position shifted by the slope lean.
    """
    z, offset, m = effective(robot, cond)
    w = math.sqrt(G / z)
    ang = math.radians(direction_deg)
    c, s = math.cos(ang), math.sin(ang)
    cop_reach = robot.ankle_torque * max(cond.actuator_strength, 0.05) / (m * G)
    half_x = min(robot.foot_half_length, cop_reach)
    half_y = robot.stance_half_width + robot.foot_half_width
    x0 = offset + G * math.sin(math.radians(cond.floor_slope_deg)) / (w * w)  # standing capture point
    if abs(x0) > half_x:
        return 0.0
    # distance from the standing capture point to the support edge along the push direction
    tx = ((half_x - x0) / c) if c > 1e-9 else ((-half_x - x0) / c if c < -1e-9 else math.inf)
    ty = (half_y / abs(s)) if abs(s) > 1e-9 else math.inf
    reach = max(0.0, min(tx, ty))
    return m * w * reach


def capturability_map(robot: Humanoid | WheeledBase = Humanoid(), cond: Conditions = Conditions(), n: int = 41,
                      v_max: float = 2.0) -> list[list[int]]:
    """Steps needed to recover from an initial CoM velocity on an n x n grid (row = forward speed, col = sideways).

    Cell values: 0..max_steps = steps taken to recover, -1 = fell or blocked. Row 0 is +v_max forward,
    column 0 is +v_max to the robot's left (screen left), so the grid reads like the top-view lab.
    """
    m = robot.mass_kg + cond.payload_kg
    sim = recover_wheeled if isinstance(robot, WheeledBase) else recover
    grid = []
    for i in range(n):
        vx = v_max - 2 * v_max * i / (n - 1)
        row = []
        for j in range(n):
            vy = v_max - 2 * v_max * j / (n - 1)
            speed = math.hypot(vx, vy)
            r = sim(speed * m, math.degrees(math.atan2(vy, vx)), robot, cond)
            row.append(r.steps if r.recovered else -1)
        grid.append(row)
    return grid


# ------------------------------------------------------------------ deployment profiles
@dataclass(frozen=True)
class Job:
    name: str
    description: str
    payload: tuple[float, float]
    friction: tuple[float, float]
    latency: tuple[float, float]
    slope: tuple[float, float]
    push_prob: float
    push_impulse: tuple[float, float]  # N·s, contact with people, carts, totes
    strength: tuple[float, float] = (0.9, 1.0)
    floor_step: tuple[float, float] = (0.0, 0.0)
    step_prob: float = 0.0  # share of routes with a curb, threshold or stair

    def sample(self, rng: np.random.Generator) -> tuple[Conditions, float, float]:
        def u(r):
            return float(rng.uniform(*r)) if r[1] > r[0] else float(r[0])

        step = u(self.floor_step) if rng.random() < self.step_prob else 0.0
        cond = Conditions(payload_kg=u(self.payload), friction=u(self.friction), latency_s=u(self.latency),
                          floor_slope_deg=u(self.slope), actuator_strength=u(self.strength), floor_step_m=step)
        pushed = rng.random() < self.push_prob
        return cond, (u(self.push_impulse) if pushed else 0.0), float(rng.uniform(-180, 180))


JOBS = {j.name: j for j in (
    Job("line-side kitting", "Automotive plant: carry parts totes to the line on sealed concrete, people nearby.",
        payload=(5, 16), friction=(0.6, 0.9), latency=(0.0, 0.03), slope=(0, 0), push_prob=0.3, push_impulse=(10, 55)),
    Job("tote handling", "3PL warehouse: move totes between conveyors and racks, occasional spills, carts.",
        payload=(5, 25), friction=(0.3, 0.9), latency=(0.0, 0.04), slope=(0, 1.5), push_prob=0.35, push_impulse=(10, 60)),
    Job("trailer unloading", "Dock and trailer floors: ramps, debris, heavier cases.",
        payload=(10, 25), friction=(0.3, 0.8), latency=(0.0, 0.05), slope=(-5, 5), push_prob=0.3, push_impulse=(15, 70),
        strength=(0.8, 1.0), floor_step=(0.01, 0.06), step_prob=0.35),
    Job("home care", "Homes and care facilities: thresholds, a few stairs, rugs, light objects, people close by.",
        payload=(0, 5), friction=(0.4, 0.9), latency=(0.0, 0.03), slope=(0, 0), push_prob=0.3, push_impulse=(5, 35),
        floor_step=(0.01, 0.18), step_prob=0.5),
    Job("teleoperated data collection", "A remote operator drives the robot to record demonstrations over a network.",
        payload=(0, 10), friction=(0.6, 0.9), latency=(0.05, 0.25), slope=(0, 0), push_prob=0.3, push_impulse=(10, 50)),
)}


def job_risk(job: str, robot: Humanoid | WheeledBase = Humanoid(), trials: int = 200, seed: int = 0) -> dict:
    from .evaluate import failure_upper_bound

    if job not in JOBS:
        raise ValueError(f"unknown job {job!r}; choose from {sorted(JOBS)}")
    rng = np.random.default_rng(seed)
    fails, causes = 0, {}
    for _ in range(trials):
        cond, imp, ang = JOBS[job].sample(rng)
        r = recover_wheeled(imp, ang, robot, cond) if isinstance(robot, WheeledBase) else recover(imp, ang, robot, cond)
        if not r.recovered:
            fails += 1
            key = r.reason.split(":")[0]
            causes[key] = causes.get(key, 0) + 1
    return {"job": job, "trials": trials, "failures": fails, "failure_rate": fails / trials,
            "failure_rate_upper_95": round(failure_upper_bound(trials, fails), 4), "causes": causes}


def with_(obj, **kw):
    return replace(obj, **kw)
