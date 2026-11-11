"""PPO (clipped surrogate, GAE) over the residual policy, in pure numpy.

Training draws each episode's world from a ``ScenarioSampler``: a mix of nominal
domain randomisation and replayed failures (the hardening loop). The mix follows the
research recommendation of ~50% randomisation / 30% replayed failures / 20% widened.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .env import ACT_DIM, QuadrupedEnv
from .nn import Adam, clip_grads
from .policy import ResidualPolicy
from .scenario import Push, Scenario

LOG2PI = np.log(2 * np.pi)
TERRAIN_VARIANTS = 16  # distinct random heightfields seen during training


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 60_000
    rollout_steps: int = 2048
    epochs: int = 5
    minibatch: int = 256
    gamma: float = 0.97
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float = 0.03
    episode_s: float = 6.0


class ScenarioSampler:
    """Draws training worlds: randomised nominal worlds plus replayed failures."""

    def __init__(self, base: Scenario | None = None, failures: list[Scenario] | None = None,
                 replay_frac: float = 0.3, widen_frac: float = 0.2, seed: int = 0):
        self.base = base or Scenario()
        self.failures = list(failures or [])
        self.replay_frac = replay_frac if self.failures else 0.0
        self.widen_frac = widen_frac
        self.rng = np.random.default_rng(seed)

    def _randomised(self, widen: float) -> Scenario:
        r = self.rng
        pushes = []
        if r.random() < 0.6:
            pushes.append(Push(t=float(r.uniform(1.0, 4.0)), force=float(r.uniform(20, 60 + 80 * widen)),
                               direction_deg=float(r.uniform(-180, 180)), duration=0.1))
        return self.base.with_(
            terrain=str(r.choice(["flat", "rough"], p=[0.6, 0.4])) if widen else self.base.terrain,
            roughness=float(r.uniform(0, 0.02 + 0.03 * widen)),
            friction=float(r.uniform(0.5 - 0.25 * widen, 1.2)),
            payload_kg=float(r.uniform(0, 1.0 + 2.0 * widen)),
            motor_strength=float(r.uniform(0.85 - 0.25 * widen, 1.1)),
            latency_steps=float(r.integers(0, 2 + int(2 * widen))),
            obs_noise=float(r.uniform(0, 0.02 + 0.05 * widen)),
            pushes=[p.__dict__ for p in pushes],
        )

    def sample(self, episode_s: float) -> Scenario:
        u = self.rng.random()
        if u < self.replay_frac:
            scn = self.failures[int(self.rng.integers(len(self.failures)))]
        elif u < self.replay_frac + self.widen_frac:
            scn = self._randomised(widen=1.0)
        else:
            scn = self._randomised(widen=0.0)
        return scn.with_(duration_s=episode_s)


def _logp(mu: np.ndarray, log_std: np.ndarray, a: np.ndarray) -> np.ndarray:
    z = (a - mu) / np.exp(log_std)
    return -0.5 * np.sum(z * z, axis=1) - np.sum(log_std) - 0.5 * ACT_DIM * LOG2PI


EVAL_WORLDS = (
    Scenario(duration_s=6.0),
    Scenario(duration_s=6.0, terrain="rough", roughness=0.02, friction=0.6, payload_kg=1.0),
    Scenario(duration_s=6.0, pushes=(Push(t=2.0, force=50.0, direction_deg=90.0, duration=0.1),)),
    Scenario(duration_s=6.0, terrain="slope", slope_deg=5.0, latency_steps=1.0),
)


def deterministic_score(policy: ResidualPolicy, seeds: tuple[int, ...] = (0, 1)) -> float:
    """Model-selection score with the *deterministic* policy (what actually ships).

    Training curves use stochastic actions, which can hide a mean policy that stalls;
    this score is pass rate plus a small progress bonus over fixed evaluation worlds.
    """
    from .evaluate import rollout  # local import: evaluate imports env, keep ppo light

    total = 0.0
    n = 0
    for scn in EVAL_WORLDS:
        for sd in seeds:
            out, _ = rollout(policy, scn, seed=sd)
            expected = scn.target_speed * scn.duration_s
            total += float(out.passed) + 0.25 * float(np.clip(out.distance / expected, 0.0, 1.0))
            n += 1
    return total / n


def train(policy: ResidualPolicy, sampler: ScenarioSampler, cfg: PPOConfig = PPOConfig(),
          seed: int = 0, log: Callable[[dict], None] | None = None,
          checkpoint: Callable[[ResidualPolicy, dict], None] | None = None, eval_every: int = 10) -> list[dict]:
    rng = np.random.default_rng(seed)
    env = QuadrupedEnv(sampler.sample(cfg.episode_s), seed=seed)
    obs = env.reset(sampler.sample(cfg.episode_s), seed=int(rng.integers(1 << 30)))
    pi_opt = Adam([*policy.pi.params(), policy.log_std], lr=cfg.lr)
    vf_opt = Adam(policy.vf.params(), lr=cfg.lr * 3)
    history: list[dict] = []
    steps = 0
    ep_ret, ep_len = 0.0, 0
    finished: list[tuple[float, int, float, bool]] = []
    t0 = time.time()
    while steps < cfg.total_steps:
        n = cfg.rollout_steps
        O = np.zeros((n, obs.shape[0]))
        A = np.zeros((n, ACT_DIM))
        R = np.zeros(n)
        D = np.zeros(n)
        V = np.zeros(n + 1)
        LP = np.zeros(n)
        raw_obs = np.zeros((n, obs.shape[0]))
        for i in range(n):
            raw_obs[i] = obs
            o = policy.norm(obs)
            mu = policy.pi(o[None])[0]
            a = mu + np.exp(policy.log_std) * rng.standard_normal(ACT_DIM)
            O[i], A[i] = o, a
            LP[i] = _logp(mu[None], policy.log_std, a[None])[0]
            V[i] = policy.vf(o[None])[0, 0]
            obs, r, done, info = env.step(a)
            R[i], D[i] = r, float(done)
            ep_ret += r
            ep_len += 1
            if done:
                if not info.fell:
                    # time-limit truncation is not a real terminal: bootstrap from V(s_T)
                    R[i] += cfg.gamma * float(policy.vf(policy.norm(obs)[None])[0, 0])
                finished.append((ep_ret, ep_len, info.x, info.fell))
                ep_ret, ep_len = 0.0, 0
                obs = env.reset(sampler.sample(cfg.episode_s), seed=int(rng.integers(1 << 30)),
                                terrain_seed=int(rng.integers(TERRAIN_VARIANTS)))
        V[n] = policy.vf(policy.norm(obs)[None])[0, 0]
        policy.norm.update(raw_obs)
        steps += n

        adv = np.zeros(n)
        last = 0.0
        for i in reversed(range(n)):
            nonterm = 1.0 - D[i]
            delta = R[i] + cfg.gamma * V[i + 1] * nonterm - V[i]
            last = delta + cfg.gamma * cfg.lam * nonterm * last
            adv[i] = last
        ret = adv + V[:n]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        kl = 0.0
        for _ in range(cfg.epochs):
            idx = rng.permutation(n)
            for s in range(0, n, cfg.minibatch):
                b = idx[s:s + cfg.minibatch]
                mu, acts = policy.pi.forward(O[b])
                std = np.exp(policy.log_std)
                lp = _logp(mu, policy.log_std, A[b])
                ratio = np.exp(lp - LP[b])
                ab = adv[b]
                active = np.where(ab >= 0, ratio <= 1 + cfg.clip, ratio >= 1 - cfg.clip)
                dlp = -(ab * ratio * active) / len(b)  # dLoss/dlogp
                z = (A[b] - mu) / std
                g_mu = dlp[:, None] * (z / std)
                g_logstd = np.sum(dlp[:, None] * (z * z - 1.0), axis=0) - cfg.ent_coef
                grads = policy.pi.backward(acts, g_mu)
                grads = clip_grads([*grads, g_logstd], cfg.max_grad_norm)
                pi_opt.step(grads)
                np.clip(policy.log_std, -2.5, 0.0, out=policy.log_std)

                v, vacts = policy.vf.forward(O[b])
                g_v = cfg.vf_coef * (v[:, 0] - ret[b])[:, None] / len(b)
                vf_opt.step(clip_grads(policy.vf.backward(vacts, g_v), cfg.max_grad_norm))
                kl = float(np.mean(LP[b] - lp))
            if kl > cfg.target_kl:
                break

        recent = finished[-20:]
        row = {
            "steps": steps,
            "episodes": len(finished),
            "return": float(np.mean([e[0] for e in recent])) if recent else 0.0,
            "distance": float(np.mean([e[2] for e in recent])) if recent else 0.0,
            "fall_rate": float(np.mean([e[3] for e in recent])) if recent else 0.0,
            "kl": kl,
            "std": float(np.exp(policy.log_std).mean()),
            "sps": steps / max(1e-6, time.time() - t0),
        }
        if checkpoint and (len(history) % eval_every == eval_every - 1 or steps >= cfg.total_steps):
            row["det_score"] = deterministic_score(policy)
            checkpoint(policy, row)
        history.append(row)
        if log:
            log(row)
    return history
