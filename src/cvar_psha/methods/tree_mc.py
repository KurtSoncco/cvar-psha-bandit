"""Naive Monte Carlo over the 3-node tree (sample each node from prior)."""

from __future__ import annotations

import numpy as np

from cvar_psha.estimators import OnlineCVaRTracker
from cvar_psha.methods import MethodResult
from cvar_psha.methods.tree_common import prior_rollout
from cvar_psha.tree_env import TreeLogicEnv


def run_tree_mc(
    env: TreeLogicEnv,
    v95: float,
    budget: int,
    eval_every: int = 200,
) -> MethodResult:
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    for _ in range(budget):
        _path, y, iw = prior_rollout(env)
        tracker.update(y, iw)
    return MethodResult(
        name="Naive MC",
        metrics=tracker.finalize(),
        final_q=env.path_priors.copy(),
    )


def run_tree_oracle(
    env: TreeLogicEnv,
    v95: float,
    budget: int,
    q: np.ndarray,
    name: str = "q* oracle",
    eval_every: int = 200,
) -> MethodResult:
    """Sample full paths directly from a fixed reference distribution q
    (e.g. the closed-form q_star or q_disagg from tree_ground_truth.py) --
    a static, non-learning IS baseline."""
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    q = np.asarray(q, dtype=float)
    q = q / q.sum()
    for _ in range(budget):
        _path, y, iw = env.sample_path_from_flat_q(q)
        tracker.update(y, iw)
    return MethodResult(name=name, metrics=tracker.finalize(), final_q=q.copy())
