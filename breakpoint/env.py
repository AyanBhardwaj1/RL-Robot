"""Gym-style environment that applies a Scenario's stressors to the quadruped.

Control: 50 Hz policy, 200 Hz physics. The policy outputs a 12-D *residual* on top of
a trotting central pattern generator (CPG); PD control converts joint targets into
torques, which are clipped by the scenario's motor strength.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from .scenario import Scenario
from .world import (
    JOINT_RANGE,
    JOINTS_PER_LEG,
    LEGS,
    NOMINAL,
    TORQUE_LIMIT,
    build_xml,
    heightfield,
    height_at,
)

CONTROL_DT = 0.02
SUBSTEPS = 4
KP = 22.0
KD = 0.6
RESIDUAL_SCALE = 0.35
GAIT_FREQ = 2.5  # Hz
OBS_DIM = 3 + 3 + 3 + 12 + 12 + 12 + 3
ACT_DIM = 12
FALL_TILT_COS = np.cos(np.radians(65))
FALL_CLEARANCE = 0.11

# Trot: diagonal pairs move together.
LEG_PHASE = {"FL": 0.0, "RR": 0.0, "FR": np.pi, "RL": np.pi}
CPG_HIP_AMP = -0.2
CPG_KNEE_LIFT = 0.4
SWING_SIGN = 1.0


def cpg_targets(phase: float, speed: float) -> np.ndarray:
    """Joint targets for a trot at the given phase. Stride scales with speed."""
    gain = np.clip(speed / 0.5, 0.0, 2.0)
    out = []
    for leg in LEGS:
        p = phase + LEG_PHASE[leg]
        hip = NOMINAL["hip"] - CPG_HIP_AMP * gain * np.sin(p)
        swing = max(0.0, SWING_SIGN * np.cos(p))
        knee = NOMINAL["knee"] - CPG_KNEE_LIFT * swing * min(1.0, 0.4 + gain)
        hip += 0.5 * CPG_KNEE_LIFT * swing * min(1.0, 0.4 + gain)
        out.extend([NOMINAL["abd"], hip, knee])
    return np.asarray(out)


@dataclass(frozen=True)
class StepInfo:
    x: float
    clearance: float
    upright: float
    fell: bool


class QuadrupedEnv:
    """One simulated world. Rebuild with ``reset(scenario)`` to change worlds."""

    def __init__(self, scenario: Scenario | None = None, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.scenario = scenario or Scenario()
        self._build(self.scenario)

    # ----------------------------------------------------------- building
    def _build(self, scn: Scenario) -> None:
        xml, data = build_xml(scn, seed=self.seed)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.model.hfield_data[:] = data.ravel()
        self.data = mujoco.MjData(self.model)
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        names = [f"{leg}_{j}" for leg in LEGS for j in JOINTS_PER_LEG]
        self.qadr = np.array(
            [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
        )
        self.dadr = np.array(
            [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
        )
        lo = np.array([JOINT_RANGE[j][0] for _ in LEGS for j in JOINTS_PER_LEG])
        hi = np.array([JOINT_RANGE[j][1] for _ in LEGS for j in JOINTS_PER_LEG])
        self.q_lo, self.q_hi = lo, hi
        self._hfield = heightfield(scn, self.seed)  # cached: clearance is checked every step

    def reset(self, scenario: Scenario | None = None, seed: int | None = None,
              terrain_seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rebuild = scenario is not None and scenario != self.scenario
        if terrain_seed is not None and terrain_seed != self.seed:
            self.seed = terrain_seed
            rebuild = True
        if rebuild:
            self.scenario = scenario or self.scenario
            self._build(self.scenario)
        mujoco.mj_resetData(self.model, self.data)
        nominal = np.array([NOMINAL[j] for _ in LEGS for j in JOINTS_PER_LEG])
        self.data.qpos[self.qadr] = nominal
        self.data.qpos[0] = 0.0
        self.data.qpos[3:7] = [0.0, 0.0, 0.0, 1.0]  # body faces world +x (gait runs along body -x)
        self.data.qpos[2] = height_at(self._hfield, 0.0, 0.0) + 0.28
        mujoco.mj_forward(self.model, self.data)
        self.t = 0.0
        rand_phase = float(self.rng.uniform(0, 2 * np.pi))  # always draw: keeps seeds reproducible
        self.phase = np.radians(self.scenario.phase0_deg) if self.scenario.phase0_deg >= 0 else rand_phase
        self.prev_action = np.zeros(ACT_DIM)
        latency = int(self.scenario.latency_steps)
        self.action_queue: deque[np.ndarray] = deque(
            [np.zeros(ACT_DIM)] * latency, maxlen=latency + 1
        )
        self.push_log: list[tuple[float, float]] = []
        return self._obs()

    # --------------------------------------------------------------- core
    def _torso_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        quat = self.data.qpos[3:7]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        R = rot.reshape(3, 3)
        lin_body = R.T @ self.data.qvel[0:3]
        ang_body = self.data.qvel[3:6]  # free-joint angular velocity is in body frame
        grav_body = R.T @ np.array([0.0, 0.0, -1.0])
        return R, lin_body, ang_body, grav_body

    def _obs(self) -> np.ndarray:
        _, lin, ang, grav = self._torso_state()
        q = self.data.qpos[self.qadr]
        qd = self.data.qvel[self.dadr]
        nominal = np.array([NOMINAL[j] for _ in LEGS for j in JOINTS_PER_LEG])
        obs = np.concatenate([
            grav, ang * 0.25, lin * 2.0, q - nominal, qd * 0.05, self.prev_action,
            [np.sin(self.phase), np.cos(self.phase), self.scenario.target_speed * 2.0],
        ])
        if self.scenario.obs_noise > 0:
            obs = obs + self.rng.normal(0.0, self.scenario.obs_noise, size=obs.shape)
        return obs.astype(np.float64)

    def _apply_pushes(self) -> None:
        f = np.zeros(6)
        for p in self.scenario.pushes:
            if p.t <= self.t < p.t + p.duration:
                ang = np.radians(p.direction_deg)
                f[0] += p.force * np.cos(ang)
                f[1] += p.force * np.sin(ang)
        self.data.xfrc_applied[self.torso_id] = f
        if f[0] or f[1]:
            self.push_log.append((self.t, float(np.hypot(f[0], f[1]))))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, StepInfo]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.action_queue.append(action)
        applied = self.action_queue.popleft() if len(self.action_queue) > int(self.scenario.latency_steps) else action
        target = cpg_targets(self.phase, self.scenario.target_speed) + RESIDUAL_SCALE * applied
        target = np.clip(target, self.q_lo, self.q_hi)
        limit = TORQUE_LIMIT * self.scenario.motor_strength
        x_before = float(self.data.qpos[0])
        energy = 0.0
        for _ in range(SUBSTEPS):
            self._apply_pushes()
            q = self.data.qpos[self.qadr]
            qd = self.data.qvel[self.dadr]
            tau = np.clip(KP * (target - q) - KD * qd, -limit, limit)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)
            energy += float(np.sum(np.abs(tau * qd))) * self.model.opt.timestep
            self.t += self.model.opt.timestep
        self.phase = (self.phase + 2 * np.pi * GAIT_FREQ * CONTROL_DT) % (2 * np.pi)

        R, lin, ang, grav = self._torso_state()
        x, y, z = self.data.qpos[0:3]
        clearance = float(z - height_at(self._hfield, float(x), float(y)))
        upright = float(R[2, 2])
        fell = bool(upright < FALL_TILT_COS or clearance < FALL_CLEARANCE or not np.isfinite(z))

        v_fwd = (float(x) - x_before) / CONTROL_DT
        tgt = self.scenario.target_speed
        reward = (
            1.5 * np.exp(-((v_fwd - tgt) ** 2) / 0.08)
            - 0.5 * abs(float(lin[1]))
            - 0.05 * float(ang[2] ** 2)
            + 0.5 * upright
            - 0.002 * energy / CONTROL_DT
            - 0.02 * float(np.sum((action - self.prev_action) ** 2))
            - 0.4 * abs(float(y)) * 0.25
        )
        if fell:
            reward -= 10.0
        self.prev_action = action
        done = fell or self.t >= self.scenario.duration_s - 1e-9
        return self._obs(), float(reward), done, StepInfo(float(x), clearance, upright, fell)

    # ------------------------------------------------------------ viewer
    def body_poses(self) -> list[dict]:
        """Geom-level poses for the browser viewer (MuJoCo is Z-up)."""
        out = []
        for g in range(self.model.ngeom):
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD:
                continue
            pos = self.data.geom_xpos[g]
            mat = self.data.geom_xmat[g]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, mat)
            out.append([round(float(v), 4) for v in (*pos, *quat)])
        return out

    def geom_specs(self) -> list[dict]:
        types = {
            int(mujoco.mjtGeom.mjGEOM_BOX): "box",
            int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
        }
        specs = []
        for g in range(self.model.ngeom):
            t = int(self.model.geom_type[g])
            if t == int(mujoco.mjtGeom.mjGEOM_HFIELD):
                continue
            specs.append({
                "type": types.get(t, "sphere"),
                "size": [round(float(s), 4) for s in self.model.geom_size[g]],
                "rgba": [round(float(c), 3) for c in self.model.geom_rgba[g]],
            })
        return specs
