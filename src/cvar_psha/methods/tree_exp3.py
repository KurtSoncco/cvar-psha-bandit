"""Flat Exp3 over full logic-tree paths (12 arms)."""

from __future__ import annotations

import numpy as np

from cvar_psha.estimators import OnlineCVaRTracker, path_tail_reward
from cvar_psha.methods import MethodResult
from cvar_psha.methods.tree_common import sample_path_from_flat_q
from cvar_psha.tree_env import TreeLogicEnv


def run_tree_exp3(
    env: TreeLogicEnv,
    v95: float,
    budget: int,
    gamma: float = 0.05,
    eval_every: int = 200,
) -> MethodResult:
    k = env.n_paths
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    weights = np.ones(k, dtype=float)
    q_hist = []
    r_max = 1e-8

    for _ in range(budget):
        wsum = weights.sum()
        q = (1.0 - gamma) * (weights / wsum) + gamma / k
        q_hist.append(q.copy())

        path, y, iw = sample_path_from_flat_q(env, q)
        idx = env.path_index[path]
        tracker.update(y, iw)

        r = path_tail_reward(y, iw, v95)
        r_max = max(r_max, abs(r))
        r_hat = r / r_max
        weights[idx] *= np.exp(gamma * (r_hat / q[idx]) / k)
        weights /= weights.max()

    return MethodResult(
        name="Flat Exp3",
        metrics=tracker.finalize(),
        final_q=q_hist[-1] if q_hist else env.path_priors.copy(),
        extras={"q_history": np.asarray(q_hist)},
    )
