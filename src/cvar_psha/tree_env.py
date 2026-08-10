"""3-node PSHA logic-tree MDP with delayed leaf rewards.

Epistemic chain: Source -> Magnitude -> GMM.
State is the path prefix; intermediate rewards are 0; the leaf returns y_T.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Branch:
    name: str
    weight: float
    delta_mu: float = 0.0
    mu: float | None = None
    sigma: float | None = None


@dataclass(frozen=True)
class NodeSpec:
    name: str
    branches: tuple[Branch, ...]

    @property
    def weights(self) -> np.ndarray:
        w = np.asarray([b.weight for b in self.branches], dtype=float)
        return w / w.sum()


def default_nodes() -> tuple[NodeSpec, ...]:
    return (
        NodeSpec(
            "source",
            (
                Branch("softer", weight=0.5, delta_mu=-0.3),
                Branch("harder", weight=0.5, delta_mu=0.3),
            ),
        ),
        NodeSpec(
            "magnitude",
            (
                Branch("low", weight=0.6, delta_mu=-0.2),
                Branch("high", weight=0.4, delta_mu=0.2),
            ),
        ),
        NodeSpec(
            "gmm",
            (
                Branch("optimistic", weight=0.4, mu=-2.0, sigma=0.6),
                Branch("average", weight=0.4, mu=-1.0, sigma=0.6),
                Branch("pessimistic", weight=0.2, mu=0.0, sigma=0.6),
            ),
        ),
    )


class TreeLogicEnv:
    """Delayed-reward MDP over a depth-T categorical logic tree."""

    def __init__(
        self,
        nodes: tuple[NodeSpec, ...] | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.nodes = nodes or default_nodes()
        self.depth = len(self.nodes)
        self.rng = rng or np.random.default_rng()
        self.n_actions = [len(n.branches) for n in self.nodes]
        self.paths = self._enumerate_paths()
        self.n_paths = len(self.paths)
        self.path_priors = np.asarray(
            [self.path_prior_prob(p) for p in self.paths], dtype=float
        )
        self.path_index = {p: i for i, p in enumerate(self.paths)}

    def _enumerate_paths(self) -> list[tuple[int, ...]]:
        ranges = [range(n) for n in self.n_actions]
        return [tuple(p) for p in product(*ranges)]

    def reset(self) -> tuple[tuple[int, ...], dict[str, Any]]:
        return (), {"depth": 0}

    def depth_of(self, state: tuple[int, ...]) -> int:
        return len(state)

    def available_actions(self, state: tuple[int, ...]) -> np.ndarray:
        d = self.depth_of(state)
        if d >= self.depth:
            return np.array([], dtype=int)
        return np.arange(self.n_actions[d], dtype=int)

    def prior(self, state: tuple[int, ...]) -> np.ndarray:
        d = self.depth_of(state)
        return self.nodes[d].weights.copy()

    def prior_action(self, state: tuple[int, ...], action: int) -> float:
        return float(self.prior(state)[action])

    def path_prior_prob(self, path: tuple[int, ...]) -> float:
        p = 1.0
        state: tuple[int, ...] = ()
        for a in path:
            p *= self.prior_action(state, a)
            state = state + (a,)
        return p

    def leaf_params(self, path: tuple[int, ...]) -> tuple[float, float]:
        """Return (mu, sigma) for ln Y under a full path."""
        if len(path) != self.depth:
            raise ValueError(f"Expected path length {self.depth}, got {len(path)}")
        delta = 0.0
        mu = 0.0
        sigma = 0.6
        for d, a in enumerate(path):
            br = self.nodes[d].branches[a]
            delta += br.delta_mu
            if br.mu is not None:
                mu = br.mu
            if br.sigma is not None:
                sigma = br.sigma
        return mu + delta, sigma

    def sample_y(self, path: tuple[int, ...]) -> float:
        mu, sigma = self.leaf_params(path)
        ln_y = self.rng.normal(mu, sigma)
        return float(np.exp(ln_y))

    def step(
        self,
        state: tuple[int, ...],
        action: int,
    ) -> tuple[tuple[int, ...], float, bool, dict[str, Any]]:
        """Environment step; intermediate reward is always 0.

        Leaf info includes y. Callers attach the path IW product to form r_T.
        """
        d = self.depth_of(state)
        if d >= self.depth:
            raise RuntimeError("Cannot step from a terminal state")
        if action < 0 or action >= self.n_actions[d]:
            raise ValueError(f"Invalid action {action} at depth {d}")

        nxt = state + (action,)
        done = len(nxt) == self.depth
        info: dict[str, Any] = {
            "depth": len(nxt),
            "prior_w": self.prior_action(state, action),
            "action": action,
            "state": state,
        }
        if done:
            y = self.sample_y(nxt)
            info["y"] = y
            info["path"] = nxt
            info["path_idx"] = self.path_index[nxt]
            return nxt, 0.0, True, info
        return nxt, 0.0, False, info

    def sample_path_from_flat_q(
        self, q: np.ndarray
    ) -> tuple[tuple[int, ...], float, float]:
        """Sample a full path directly from a path-level categorical q,
        for oracle IS baselines (e.g. q_star / q_disagg). Mirrors
        SpatialPortfolioEnv.sample_path_from_flat_q."""
        q = np.asarray(q, dtype=float)
        q = q / q.sum()
        idx = int(self.rng.choice(self.n_paths, p=q))
        path = self.paths[idx]
        y = self.sample_y(path)
        iw = self.path_priors[idx] / max(q[idx], 1e-12)
        return path, y, iw

    def rollout_with_policy(
        self,
        action_fn,
    ) -> tuple[tuple[int, ...], float, float, list[dict[str, Any]]]:
        """Roll one full path. action_fn(state, prior) -> (action, q_prob)."""
        state, _ = self.reset()
        iw = 1.0
        traj: list[dict[str, Any]] = []
        done = False
        info: dict[str, Any] = {}
        while not done:
            prior = self.prior(state)
            action, q_prob = action_fn(state, prior)
            q_prob = max(float(q_prob), 1e-12)
            nxt, reward, done, info = self.step(state, action)
            step_w = info["prior_w"] / q_prob
            iw *= step_w
            traj.append(
                {
                    "state": state,
                    "action": action,
                    "prior_w": info["prior_w"],
                    "q_prob": q_prob,
                    "step_iw": step_w,
                    "depth": len(state),
                }
            )
            state = nxt
        y = float(info["y"])
        return state, y, iw, traj
