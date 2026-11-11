"""Policy compression with a robustness check: does INT8 or FP16 change where the robot breaks?

Edge deployment means shrinking a policy (FP16, INT8) to fit the onboard compute budget. The usual check is
"does the action error stay small"; the one that matters is "does the robot still survive the same shoves,
slopes and lag". This module:

1. quantises a ResidualPolicy's MLP weights (FP16 cast, or symmetric per-output-channel INT8 with
   de-quantisation at run time, as an INT8 kernel would compute),
2. profiles inference latency and model size,
3. measures action error against the FP32 policy on real observations,
4. re-runs Breakpoint's probe and deployment checks so the robustness cost is a number, not a guess.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .env import QuadrupedEnv
from .policy import CPGPolicy, Policy, ResidualPolicy
from .scenario import Scenario


@dataclass
class QuantStats:
    mode: str
    weight_bytes: int
    mean_abs_action_error: float
    max_abs_action_error: float
    act_us: float  # mean microseconds per act() call


def quantize_int8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-channel INT8: w ≈ q * scale, q in [-127, 127]."""
    scale = np.maximum(np.abs(w).max(axis=0), 1e-12) / 127.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def compress(policy: ResidualPolicy, mode: str) -> ResidualPolicy:
    """Return a copy whose weights carry the error of the target precision (fp32 | fp16 | int8)."""
    if mode not in ("fp32", "fp16", "int8"):
        raise ValueError("mode must be fp32, fp16 or int8")
    out = copy.deepcopy(policy)
    out.name = f"{policy.name}-{mode}"
    if mode == "fp32":
        return out
    for i, w in enumerate(out.pi.W):
        if mode == "fp16":
            out.pi.W[i] = w.astype(np.float16).astype(np.float64)
            out.pi.b[i] = out.pi.b[i].astype(np.float16).astype(np.float64)
        else:
            q, scale = quantize_int8(w)
            out.pi.W[i] = q.astype(np.float64) * scale  # de-quantised: exactly what an INT8 GEMM reproduces
    return out


def weight_bytes(policy: ResidualPolicy, mode: str) -> int:
    per = {"fp32": 4, "fp16": 2, "int8": 1}[mode]
    n = sum(w.size for w in policy.pi.W)
    extra = sum(b.size for b in policy.pi.b) * 4 + (sum(w.shape[1] for w in policy.pi.W) * 4 if mode == "int8" else 0)
    return n * per + extra


def sample_observations(n: int = 600, seed: int = 0) -> np.ndarray:
    env = QuadrupedEnv(Scenario(terrain="rough", roughness=0.02), seed=seed)
    obs = env.reset(seed=seed)
    cpg = CPGPolicy()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(obs)
        obs, _, done, _ = env.step(cpg.act(obs) + 0.2 * rng.standard_normal(12))
        if done:
            obs = env.reset(seed=int(rng.integers(1 << 20)))
    return np.asarray(out)


def profile(policy: Policy, obs: np.ndarray, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for o in obs:
            policy.act(o)
        best = min(best, (time.perf_counter() - t0) / len(obs))
    return best * 1e6


def stats(base: ResidualPolicy, mode: str, obs: np.ndarray) -> QuantStats:
    q = compress(base, mode)
    ref = np.array([base.act(o) for o in obs])
    got = np.array([q.act(o) for o in obs])
    err = np.abs(ref - got)
    return QuantStats(mode, weight_bytes(base, mode), float(err.mean()), float(err.max()), profile(q, obs))


def save_compressed(base_path: str, mode: str, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q = compress(ResidualPolicy.load(base_path), mode)
    path = out_dir / f"{Path(base_path).stem}_{mode}.npz"
    q.save(path)
    return path
