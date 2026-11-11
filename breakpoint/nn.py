"""A dependency-free MLP + Adam in numpy, with hand-written backprop.

Keeping the learner in numpy means Breakpoint trains anywhere MuJoCo runs: no GPU,
no PyTorch wheel, no CUDA. Networks are small (2x64), so CPU throughput is dominated
by physics, not by the learner.
"""

from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, sizes: list[int], rng: np.random.Generator, out_scale: float = 1.0):
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            gain = out_scale if i == len(sizes) - 2 else np.sqrt(2.0)
            w = rng.standard_normal((n_in, n_out))
            q, _ = np.linalg.qr(w if n_in >= n_out else w.T)
            q = q if n_in >= n_out else q.T
            self.W.append(gain * q[:n_in, :n_out])
            self.b.append(np.zeros(n_out))

    def params(self) -> list[np.ndarray]:
        return [*self.W, *self.b]

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        acts = [x]
        h = x
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            h = h @ w + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
            acts.append(h)
        return h, acts

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    def backward(self, acts: list[np.ndarray], grad_out: np.ndarray) -> list[np.ndarray]:
        """Return grads matching ``params()`` order for dLoss/dOutput = grad_out."""
        gW: list[np.ndarray] = [np.empty(0)] * len(self.W)
        gb: list[np.ndarray] = [np.empty(0)] * len(self.b)
        g = grad_out
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:
                g = g * (1.0 - acts[i + 1] ** 2)
            gW[i] = acts[i].T @ g
            gb[i] = g.sum(axis=0)
            g = g @ self.W[i].T
        return [*gW, *gb]

    def state(self) -> dict[str, np.ndarray]:
        d = {f"W{i}": w for i, w in enumerate(self.W)}
        d.update({f"b{i}": b for i, b in enumerate(self.b)})
        return d

    def load(self, d: dict[str, np.ndarray]) -> None:
        for i in range(len(self.W)):
            self.W[i] = np.array(d[f"W{i}"])
            self.b[i] = np.array(d[f"b{i}"])


class Adam:
    def __init__(self, params: list[np.ndarray], lr: float = 3e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1**self.t)
            vh = v / (1 - self.b2**self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)  # in-place: params are shared refs


def clip_grads(grads: list[np.ndarray], max_norm: float) -> list[np.ndarray]:
    norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
    if norm <= max_norm or norm == 0.0:
        return grads
    return [g * (max_norm / norm) for g in grads]


class RunningNorm:
    """Welford running mean/variance for observation normalisation."""

    def __init__(self, dim: int, clip: float = 5.0):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 1e-4
        self.clip = clip

    def update(self, x: np.ndarray) -> None:
        x = np.atleast_2d(x)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta**2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip)
