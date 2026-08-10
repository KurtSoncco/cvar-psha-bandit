"""Hierarchical REINFORCE: separate q(a|s) per node/state with delayed leaf reward."""

from __future__ import annotations

from itertools import product

import numpy as np

from cvar_psha.estimators import OnlineCVaRTracker, path_tail_reward
from cvar_psha.methods import MethodResult
from cvar_psha.methods.tree_common import softmax
from cvar_psha.tree_env import TreeLogicEnv


class HierarchicalPolicy:
    """Tabular softmax policies keyed by (depth, state_prefix)."""

    def __init__(self, env: TreeLogicEnv):
        self.env = env
        self.logits: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}
        for d, node in enumerate(env.nodes):
            if d == 0:
                prefixes: list[tuple[int, ...]] = [()]
            else:
                ranges = [range(n) for n in env.n_actions[:d]]
                prefixes = [tuple(p) for p in product(*ranges)]
            for pref in prefixes:
                prior = node.weights
                self.logits[(d, pref)] = np.log(np.clip(prior, 1e-8, None)).copy()

    def probs(self, state: tuple[int, ...]) -> np.ndarray:
        d = len(state)
        return softmax(self.logits[(d, state)])

    def sample(self, state: tuple[int, ...], rng: np.random.Generator) -> tuple[int, float]:
        q = self.probs(state)
        a = int(rng.choice(len(q), p=q))
        return a, float(q[a])

    def update(
        self,
        traj: list[dict],
        advantage: float,
        learning_rate: float,
    ) -> None:
        for step in traj:
            state = step["state"]
            action = step["action"]
            d = step["depth"]
            key = (d, state)
            q = softmax(self.logits[key])
            grad = -q
            grad[action] += 1.0
            self.logits[key] = self.logits[key] + learning_rate * advantage * grad

    def path_distribution(self) -> np.ndarray:
        """Marginal probability of each full path under the hierarchical policy."""
        q_path = np.zeros(self.env.n_paths, dtype=float)
        for i, path in enumerate(self.env.paths):
            p = 1.0
            state: tuple[int, ...] = ()
            for a in path:
                p *= float(self.probs(state)[a])
                state = state + (a,)
            q_path[i] = p
        return q_path / max(q_path.sum(), 1e-12)


def run_hierarchical(
    env: TreeLogicEnv,
    v95: float,
    budget: int,
    learning_rate: float = 0.05,
    baseline_alpha: float = 0.1,
    eval_every: int = 200,
) -> MethodResult:
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    policy = HierarchicalPolicy(env)
    baseline = 0.0
    r_max = 1e-8

    for _ in range(budget):
        def action_fn(state, prior):
            return policy.sample(state, env.rng)

        path, y, iw, traj = env.rollout_with_policy(action_fn)
        tracker.update(y, iw)

        r = path_tail_reward(y, iw, v95)
        r_max = max(r_max, abs(r), 1e-8)
        r_scaled = r / r_max
        advantage = r_scaled - baseline
        baseline = (1.0 - baseline_alpha) * baseline + baseline_alpha * r_scaled
        policy.update(traj, advantage, learning_rate)

    return MethodResult(
        name="Hierarchical",
        metrics=tracker.finalize(),
        final_q=policy.path_distribution(),
        extras={"node_logits": {str(k): v.copy() for k, v in policy.logits.items()}},
    )
