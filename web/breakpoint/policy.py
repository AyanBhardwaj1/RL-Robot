"""Policies: the bare CPG gait and a learned residual policy on top of it."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .env import ACT_DIM, OBS_DIM
from .nn import MLP, RunningNorm

HIDDEN = [64, 64]


class Policy:
    name = "policy"

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        raise NotImplementedError


class CPGPolicy(Policy):
    """Zero residual: the open-loop trot alone. The baseline every report compares to."""

    name = "cpg-baseline"

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return np.zeros(ACT_DIM)


class ResidualPolicy(Policy):
    def __init__(self, seed: int = 0, name: str = "residual-ppo"):
        rng = np.random.default_rng(seed)
        self.pi = MLP([OBS_DIM, *HIDDEN, ACT_DIM], rng, out_scale=0.01)
        self.vf = MLP([OBS_DIM, *HIDDEN, 1], rng, out_scale=1.0)
        self.log_std = np.full(ACT_DIM, -1.0)
        self.norm = RunningNorm(OBS_DIM)
        self.name = name
        self.meta: dict = {}

    def mean(self, obs: np.ndarray) -> np.ndarray:
        return self.pi(self.norm(obs))

    def act(self, obs: np.ndarray, deterministic: bool = True, rng: np.random.Generator | None = None) -> np.ndarray:
        mu = self.mean(np.atleast_2d(obs))[0]
        if deterministic:
            return np.clip(mu, -1, 1)
        rng = rng or np.random.default_rng()
        return mu + np.exp(self.log_std) * rng.standard_normal(ACT_DIM)

    def save(self, path: str | Path) -> None:
        d = {f"pi_{k}": v for k, v in self.pi.state().items()}
        d.update({f"vf_{k}": v for k, v in self.vf.state().items()})
        d.update(log_std=self.log_std, n_mean=self.norm.mean, n_var=self.norm.var,
                 n_count=np.array(self.norm.count), name=np.array(self.name))
        np.savez(Path(path), **d)

    @classmethod
    def load(cls, path: str | Path) -> ResidualPolicy:
        z = np.load(Path(path), allow_pickle=False)
        p = cls(name=str(z["name"]))
        p.pi.load({k[3:]: z[k] for k in z.files if k.startswith("pi_")})
        p.vf.load({k[3:]: z[k] for k in z.files if k.startswith("vf_")})
        p.log_std = np.array(z["log_std"])
        p.norm.mean, p.norm.var, p.norm.count = np.array(z["n_mean"]), np.array(z["n_var"]), float(z["n_count"])
        return p


class CallablePolicy(Policy):
    """Wraps any ``obs -> action`` callable (an SB3 model, an ONNX session, a hand-written controller)."""

    def __init__(self, fn, name: str = "external"):
        self.fn = fn
        self.name = name

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return np.clip(np.asarray(self.fn(obs), dtype=np.float64).reshape(ACT_DIM), -1.0, 1.0)


class LatencyGatedPolicy(Policy):
    """Fades a learned residual out as measured link latency grows.

    The open-loop CPG never reads observations, so control latency cannot hurt it; a
    closed-loop residual acting on stale state can. Deployment stacks know their link
    latency (timestamps), so the residual runs at full strength up to ``full_steps`` of
    delay and falls back linearly to the latency-immune gait by ``zero_steps``.
    """

    def __init__(self, inner: Policy, full_steps: float = 2.0, zero_steps: float = 4.0):
        if zero_steps <= full_steps:
            raise ValueError("zero_steps must exceed full_steps")
        self.inner, self.full, self.zero = inner, full_steps, zero_steps
        self.latency_steps = 0.0
        self.name = f"gated-{getattr(inner, 'name', 'policy')}"

    def set_link_latency(self, steps: float) -> None:
        self.latency_steps = float(steps)

    @property
    def gain(self) -> float:
        return float(np.clip((self.zero - self.latency_steps) / (self.zero - self.full), 0.0, 1.0))

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        g = self.gain
        return g * self.inner.act(obs) if g > 0 else np.zeros(ACT_DIM)


def load_policy(spec: str | None) -> Policy:
    """Resolve a policy spec.

    * ``None`` / ``"cpg"``          -> the open-loop CPG baseline
    * ``path/to/policy.npz``        -> a saved residual PPO policy
    * ``gated:<spec>``              -> that policy behind a latency-adaptive gate
    * ``py:package.module:factory`` -> ``factory()`` returns an ``obs -> action`` callable or a Policy
      (bring your own controller; only for trusted local code, never exposed by the web server)
    """
    if not spec or spec == "cpg":
        return CPGPolicy()
    if spec.startswith("gated:"):
        return LatencyGatedPolicy(load_policy(spec[6:]))
    if spec.startswith("py:"):
        import importlib

        mod_name, _, attr = spec[3:].partition(":")
        if not mod_name or not attr:
            raise ValueError("py: policy spec must look like py:package.module:factory")
        obj = getattr(importlib.import_module(mod_name), attr)()
        return obj if isinstance(obj, Policy) else CallablePolicy(obj, name=attr)
    path = Path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"policy file not found: {spec}")
    return ResidualPolicy.load(path)
