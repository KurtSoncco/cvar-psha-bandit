"""Flat REINFORCE over full logic-tree paths."""

from __future__ import annotations

import numpy as np

from cvar_psha.estimators import OnlineCVaRTracker, path_tail_reward
from cvar_psha.methods import MethodResult
from cvar_psha.methods.tree_common import sample_path_from_flat_q, softmax
from cvar_psha.tree_env import TreeLogicEnv


def run_tree_reinforce(
    env: TreeLogicEnv,
    v95: float,
    budget: int,
    learning_rate: float = 0.05,
    baseline_alpha: float = 0.1,
    eval_every: int = 200,
) -> MethodResult:
    k = env.n_paths
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    logits = np.log(np.clip(env.path_priors, 1e-8, None))
    baseline = 0.0
    q_hist = []
    r_max = 1e-8

    for _ in range(budget):
        q = softmax(logits)
        q_hist.append(q.copy())

        path, y, iw = sample_path_from_flat_q(env, q)
        idx = env.path_index[path]
        tracker.update(y, iw)

        r = path_tail_reward(y, iw, v95)
        r_max = max(r_max, abs(r), 1e-8)
        r_scaled = r / r_max
        advantage = r_scaled - baseline
        baseline = (1.0 - baseline_alpha) * baseline + baseline_alpha * r_scaled

        grad = -q
        grad[idx] += 1.0
        logits += learning_rate * advantage * grad

    return MethodResult(
        name="Flat REINFORCE",
        metrics=tracker.finalize(),
        final_q=q_hist[-1] if q_hist else env.path_priors.copy(),
        extras={"q_history": np.asarray(q_hist)},
    )
