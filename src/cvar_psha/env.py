"""PSHA logic-tree GMM environment with three discrete epistemic arms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Arm:
    name: str
    mu: float
    sigma: float
    weight: float


DEFAULT_ARMS = (
    Arm("optimistic", mu=-2.0, sigma=0.6, weight=0.4),
    Arm("average", mu=-1.0, sigma=0.6, weight=0.4),
    Arm("pessimistic", mu=0.0, sigma=0.6, weight=0.2),
)


class LogicTreeEnv:
    """Single site / single source; one GMM epistemic node with K arms."""

    def __init__(self, arms: tuple[Arm, ...] | None = None, rng: np.random.Generator | None = None):
        self.arms = arms or DEFAULT_ARMS
        self.n_arms = len(self.arms)
        self.weights = np.asarray([a.weight for a in self.arms], dtype=float)
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError(f"Prior weights must sum to 1, got {self.weights.sum()}")
        self.mus = np.asarray([a.mu for a in self.arms], dtype=float)
        self.sigmas = np.asarray([a.sigma for a in self.arms], dtype=float)
        self.rng = rng or np.random.default_rng()

    def sample_arm(self, q: np.ndarray | None = None) -> int:
        probs = self.weights if q is None else np.asarray(q, dtype=float)
        probs = probs / probs.sum()
        return int(self.rng.choice(self.n_arms, p=probs))

    def sample_ground_motion(self, arm: int) -> float:
        """Draw PGA Y where ln(Y) ~ N(mu_i, sigma_i)."""
        ln_y = self.rng.normal(self.mus[arm], self.sigmas[arm])
        return float(np.exp(ln_y))

    def step(self, q: np.ndarray | None = None) -> tuple[int, float]:
        arm = self.sample_arm(q)
        y = self.sample_ground_motion(arm)
        return arm, y

    def sample_batch(
        self,
        n: int,
        q: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        arms = np.empty(n, dtype=int)
        ys = np.empty(n, dtype=float)
        for t in range(n):
            arms[t], ys[t] = self.step(q)
        return arms, ys
