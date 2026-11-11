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


def load_policy(spec: str | None) -> Policy:
    """'cpg' / None -> baseline, otherwise a path to a saved .npz residual policy."""
    if not spec or spec == "cpg":
        return CPGPolicy()
    path = Path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"policy file not found: {spec}")
    return ResidualPolicy.load(path)
