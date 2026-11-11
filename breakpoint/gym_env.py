"""Gymnasium adapter so any RL library (SB3, CleanRL, ...) can train in Breakpoint worlds.

    import gymnasium as gym
    from breakpoint.gym_env import BreakpointEnv
    env = BreakpointEnv(scenario={"terrain": "rough", "roughness": 0.03})
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from .env import ACT_DIM, OBS_DIM, QuadrupedEnv
from .scenario import Scenario


class BreakpointEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, scenario: dict | Scenario | None = None, sampler=None):
        super().__init__()
        self.scenario = scenario if isinstance(scenario, Scenario) else Scenario.from_dict(scenario or {})
        self.sampler = sampler  # optional ScenarioSampler for domain randomisation
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float64)
        self._env = QuadrupedEnv(self.scenario)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        scn = self.sampler.sample(self.scenario.duration_s) if self.sampler else self.scenario
        if options and "scenario" in options:
            scn = Scenario.from_dict(options["scenario"])
        obs = self._env.reset(scn, seed=int(self.np_random.integers(1 << 30)))
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(np.asarray(action, dtype=np.float64))
        terminated = bool(info.fell)
        truncated = bool(done and not info.fell)
        return obs, reward, terminated, truncated, {"x": info.x, "clearance": info.clearance, "upright": info.upright}
