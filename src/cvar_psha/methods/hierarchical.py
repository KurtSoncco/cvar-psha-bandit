"""Hierarchical REINFORCE: separate q(a|s) per node/state with delayed leaf reward."""

from __future__ import annotations

import numpy as np

from cvar_psha.core.estimators import OnlineCVaRTracker, path_tail_reward
from cvar_psha.core.policy import EMABaseline, RewardScaler, TabularTreePolicy
from cvar_psha.core.result import MethodResult
from cvar_psha.tree_env import TreeLogicEnv


class HierarchicalPolicy:
    """Tabular softmax policies keyed by (depth, state_prefix)."""

    def __init__(self, env: TreeLogicEnv):
        self.env = env
        self._table = TabularTreePolicy(
            n_actions=env.n_actions,
            prior_fn=lambda pref: env.nodes[len(pref)].weights,
        )

    @property
    def logits(self):
        return self._table.logits

    def probs(self, state: tuple[int, ...]) -> np.ndarray:
        return self._table.probs(state)

    def sample(self, state: tuple[int, ...], rng: np.random.Generator) -> tuple[int, float]:
        a, q_prob, _q = self._table.sample(state, rng)
        return a, q_prob

    def update(self, traj: list[dict], advantage: float, learning_rate: float) -> None:
        for step in traj:
            self._table.update_step(
                step["state"], step["action"], advantage, learning_rate
            )

    def path_distribution(self) -> np.ndarray:
        return self._table.path_distribution(self.env.paths, fallback=self.env.path_priors)


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
    baseline = EMABaseline(baseline_alpha)
    scaler = RewardScaler()

    for _ in range(budget):
        def action_fn(state, prior):
            return policy.sample(state, env.rng)

        _path, y, iw, traj = env.rollout_with_policy(action_fn)
        tracker.update(y, iw)
        advantage = baseline.advantage(scaler.scale(path_tail_reward(y, iw, v95)))
        policy.update(traj, advantage, learning_rate)

    return MethodResult(
        name="Hierarchical",
        metrics=tracker.finalize(),
        final_q=policy.path_distribution(),
        extras={"node_logits": {str(k): v.copy() for k, v in policy.logits.items()}},
    )
