"""Procedural MuJoCo world: a 12-DoF quadruped on scenario-defined terrain."""

from __future__ import annotations

import numpy as np

from .scenario import Scenario

HFIELD_N = 160  # samples per side
HFIELD_HALF_X = 8.0  # metres
HFIELD_HALF_Y = 3.0
TERRAIN_START_X = 0.6  # flat run-up before hazards begin
BASE_HEIGHT = 0.06  # hfield base so negative slopes still have ground

LEGS = ("FL", "FR", "RL", "RR")
LEG_POS = {"FL": (0.17, 0.10), "FR": (0.17, -0.10), "RL": (-0.17, 0.10), "RR": (-0.17, -0.10)}
JOINTS_PER_LEG = ("abd", "hip", "knee")
JOINT_RANGE = {"abd": (-0.6, 0.6), "hip": (-1.2, 1.6), "knee": (-2.4, -0.2)}
NOMINAL = {"abd": 0.0, "hip": 0.6, "knee": -1.2}
TORQUE_LIMIT = 9.0  # N·m before motor_strength scaling
THIGH = 0.15
SHIN = 0.15
SPAWN_Z = 0.36


def heightfield(scn: Scenario, seed: int = 0) -> np.ndarray:
    """Return an (N, N) array of absolute heights in metres (row = y, col = x)."""
    xs = np.linspace(-HFIELD_HALF_X, HFIELD_HALF_X, HFIELD_N)
    ys = np.linspace(-HFIELD_HALF_Y, HFIELD_HALF_Y, HFIELD_N)
    X, _ = np.meshgrid(xs, ys)
    h = np.zeros_like(X)
    ahead = np.clip(X - TERRAIN_START_X, 0.0, None)
    if scn.terrain == "slope" or abs(scn.slope_deg) > 1e-6:
        h += np.tan(np.radians(scn.slope_deg)) * ahead
    if scn.terrain == "stairs" or scn.step_height > 0:
        riser = scn.step_height if scn.step_height > 0 else 0.05
        h += np.floor(ahead / 0.35) * riser * (ahead > 0)
    if scn.terrain == "rough" or scn.roughness > 0:
        amp = scn.roughness if scn.roughness > 0 else 0.03
        rng = np.random.default_rng(seed)
        noise = rng.uniform(-1.0, 1.0, size=X.shape)
        # cheap smoothing to make bumps foot-sized rather than per-sample spikes
        for _ in range(2):
            noise = (noise + np.roll(noise, 1, 0) + np.roll(noise, 1, 1)
                     + np.roll(noise, -1, 0) + np.roll(noise, -1, 1)) / 5.0
        noise /= max(1e-6, np.abs(noise).max())
        h += amp * noise * (X > TERRAIN_START_X - 0.2)
    return h


def _leg_xml(leg: str, friction: float) -> str:
    x, y = LEG_POS[leg]
    side = 1 if y > 0 else -1
    return f"""
      <body name="{leg}_hip" pos="{x} {y} 0">
        <joint name="{leg}_abd" type="hinge" axis="1 0 0" range="{JOINT_RANGE['abd'][0]} {JOINT_RANGE['abd'][1]}" damping="0.05"/>
        <geom type="sphere" size="0.025" mass="0.15" rgba="0.15 0.15 0.18 1"/>
        <body name="{leg}_thigh" pos="0 {0.03 * side} 0">
          <joint name="{leg}_hip" type="hinge" axis="0 1 0" range="{JOINT_RANGE['hip'][0]} {JOINT_RANGE['hip'][1]}" damping="0.05"/>
          <geom type="capsule" fromto="0 0 0 0 0 -{THIGH}" size="0.018" mass="0.35" rgba="0.95 0.45 0.1 1"/>
          <body name="{leg}_shin" pos="0 0 -{THIGH}">
            <joint name="{leg}_knee" type="hinge" axis="0 1 0" range="{JOINT_RANGE['knee'][0]} {JOINT_RANGE['knee'][1]}" damping="0.05"/>
            <geom type="capsule" fromto="0 0 0 0 0 -{SHIN}" size="0.014" mass="0.15" rgba="0.2 0.2 0.24 1"/>
            <geom name="{leg}_foot" type="sphere" pos="0 0 -{SHIN}" size="0.022" mass="0.05"
                  friction="{friction} 0.02 0.002" rgba="0.1 0.1 0.1 1"/>
          </body>
        </body>
      </body>"""


def build_xml(scn: Scenario, seed: int = 0) -> tuple[str, np.ndarray]:
    """Return (MJCF string, normalised hfield data) for a scenario."""
    h = heightfield(scn, seed) + BASE_HEIGHT
    lo = min(0.0, float(h.min()))
    h = h - lo
    top = max(float(h.max()), 1e-3)
    data = (h / top).astype(np.float64)
    payload = ""
    if scn.payload_kg > 0:
        payload = (
            f'<geom name="payload" type="box" size="0.08 0.07 0.035" '
            f'pos="{-scn.payload_offset_x} 0 0.075" mass="{scn.payload_kg}" rgba="0.3 0.55 0.95 1"/>'
        )
    legs = "".join(_leg_xml(leg, scn.friction) for leg in LEGS)
    xml = f"""
<mujoco model="breakpoint-quadruped">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" integrator="implicitfast"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <default><geom contype="2" conaffinity="1"/></default>
  <asset>
    <hfield name="terrain" nrow="{HFIELD_N}" ncol="{HFIELD_N}"
            size="{HFIELD_HALF_X} {HFIELD_HALF_Y} {top} 0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="hfield" hfield="terrain" pos="0 0 {lo}" contype="1" conaffinity="2"
          friction="{scn.friction} 0.02 0.002" rgba="0.55 0.6 0.55 1"/>
    <body name="torso" pos="0 0 {SPAWN_Z + BASE_HEIGHT}">
      <freejoint name="root"/>
      <geom name="torso" type="box" size="0.21 0.075 0.04" mass="3.0" rgba="0.92 0.92 0.95 1"/>
      <geom name="head" type="box" size="0.03 0.05 0.025" pos="-0.23 0 0.01" mass="0.1" rgba="0.95 0.45 0.1 1"/>
      {payload}
      <site name="imu" pos="0 0 0"/>
      {legs}
    </body>
  </worldbody>
  <actuator>
    {''.join(f'<motor name="{leg}_{j}_m" joint="{leg}_{j}" ctrlrange="-{TORQUE_LIMIT * 1.5} {TORQUE_LIMIT * 1.5}"/>' for leg in LEGS for j in JOINTS_PER_LEG)}
  </actuator>
</mujoco>"""
    return xml, data


def height_at(h: np.ndarray, x: float, y: float) -> float:
    """Conservative ground height (m) under (x, y): max over the enclosing grid cell.

    Taking the cell max (not the lower corner) means a torso resting on a stair edge
    is measured against the *upper* step, so such falls are not missed.
    """
    fi = float(np.clip((y + HFIELD_HALF_Y) / (2 * HFIELD_HALF_Y) * (HFIELD_N - 1), 0, HFIELD_N - 1))
    fj = float(np.clip((x + HFIELD_HALF_X) / (2 * HFIELD_HALF_X) * (HFIELD_N - 1), 0, HFIELD_N - 1))
    i0, j0 = int(fi), int(fj)
    i1, j1 = min(i0 + 1, HFIELD_N - 1), min(j0 + 1, HFIELD_N - 1)
    return float(max(h[i0, j0], h[i0, j1], h[i1, j0], h[i1, j1])) + BASE_HEIGHT


def ground_height(scn: Scenario, x: float, y: float, seed: int = 0) -> float:
    """Terrain height (m) at world (x, y), used to measure body clearance."""
    return height_at(heightfield(scn, seed), x, y)
