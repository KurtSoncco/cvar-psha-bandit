"""Shared policy-gradient helpers used by every categorical / tree method."""

from __future__ import annotations

from itertools import product

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


def softmax_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(float(temperature), 1e-4)
    z = (logits - np.max(logits)) / t
    e = np.exp(z)
    return e / e.sum()


def categorical_score(q: np.ndarray, action: int) -> np.ndarray:
    """∇_logits log π(action) for a softmax policy: 𝟙_action − q."""
    grad = -np.asarray(q, dtype=float).copy()
    grad[action] += 1.0
    return grad


class RewardScaler:
    """Online max-abs scaling so heavy-tailed tail rewards stay O(1)."""

    def __init__(self, floor: float = 1e-8) -> None:
        self.floor = floor
        self.r_max = floor

    def scale(self, r: float) -> float:
        self.r_max = max(self.r_max, abs(r), self.floor)
        return r / self.r_max


class EMABaseline:
    """Exponential moving-average baseline for REINFORCE-style advantages."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = float(alpha)
        self.value = 0.0

    def advantage(self, reward: float) -> float:
        adv = reward - self.value
        self.value = (1.0 - self.alpha) * self.value + self.alpha * reward
        return adv


class TabularTreePolicy:
    """One softmax table per (depth, path-prefix), shared by 3-node and spatial trees."""

    def __init__(
        self,
        n_actions: list[int],
        prior_fn,
        logit_floor: float = 1e-8,
    ) -> None:
        self.n_actions = list(n_actions)
        self.depth = len(self.n_actions)
        self.logit_floor = logit_floor
        self.logits: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}
        for d in range(self.depth):
            prefixes: list[tuple[int, ...]]
            if d == 0:
                prefixes = [()]
            else:
                ranges = [range(n) for n in self.n_actions[:d]]
                prefixes = [tuple(p) for p in product(*ranges)]
            for pref in prefixes:
                prior = np.asarray(prior_fn(pref), dtype=float)
                self.logits[(d, pref)] = np.log(np.clip(prior, logit_floor, None)).copy()

    def probs(self, state: tuple[int, ...], temperature: float = 1.0) -> np.ndarray:
        key = (len(state), state)
        if temperature == 1.0:
            return softmax(self.logits[key])
        return softmax_temperature(self.logits[key], temperature)

    def sample(
        self,
        state: tuple[int, ...],
        rng: np.random.Generator,
        temperature: float = 1.0,
    ) -> tuple[int, float, np.ndarray]:
        q = self.probs(state, temperature=temperature)
        a = int(rng.choice(len(q), p=q))
        return a, float(q[a]), q

    def update_step(
        self,
        state: tuple[int, ...],
        action: int,
        advantage: float,
        learning_rate: float,
        entropy_coef: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        key = (len(state), state)
        q = softmax_temperature(self.logits[key], temperature) if temperature != 1.0 else softmax(
            self.logits[key]
        )
        grad = categorical_score(q, action)
        if entropy_coef > 0:
            ent_grad = -np.log(np.clip(q, 1e-12, None)) - 1.0
            ent_grad -= np.sum(q * ent_grad)
            grad = grad + entropy_coef * ent_grad
        self.logits[key] = self.logits[key] + learning_rate * advantage * grad

    def path_distribution(
        self,
        paths: list[tuple[int, ...]],
        temperature: float = 1.0,
        fallback: np.ndarray | None = None,
    ) -> np.ndarray:
        q_path = np.zeros(len(paths), dtype=float)
        for i, path in enumerate(paths):
            p = 1.0
            state: tuple[int, ...] = ()
            for a in path:
                p *= float(self.probs(state, temperature=temperature)[a])
                state = state + (a,)
            q_path[i] = p
        s = q_path.sum()
        if s > 0:
            return q_path / s
        if fallback is not None:
            return np.asarray(fallback, dtype=float).copy()
        return q_path
