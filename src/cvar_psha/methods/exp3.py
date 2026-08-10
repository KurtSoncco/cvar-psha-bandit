"""Exp3 bandit that learns an IS proposal from scalar tail rewards."""

from __future__ import annotations

import numpy as np

from cvar_psha.env import LogicTreeEnv
from cvar_psha.estimators import OnlineCVaRTracker, importance_weight, tail_reward
from cvar_psha.methods import MethodResult


def run_exp3(
    env: LogicTreeEnv,
    v95: float,
    budget: int,
    gamma: float = 0.05,
    eval_every: int = 200,
) -> MethodResult:
    """Exp3 on K arms; reward = IW tail contribution (0 below VaR)."""
    k = env.n_arms
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    weights = np.ones(k, dtype=float)
    q_hist = []

    # Online normalization scale for unbounded rewards.
    r_max = 1e-8

    for _ in range(budget):
        wsum = weights.sum()
        q = (1.0 - gamma) * (weights / wsum) + gamma / k
        q_hist.append(q.copy())

        arm, y = env.step(q)
        iw = importance_weight(env.weights[arm], q[arm])
        tracker.update(y, iw)

        r = tail_reward(y, iw, v95)
        r_max = max(r_max, abs(r))
        r_hat = r / r_max  # scale into roughly [-1, 1] / [0, 1]
        # Unbiased Exp3 estimate for the sampled arm.
        est = r_hat / q[arm]
        weights[arm] *= np.exp(gamma * est / k)
        # Numerical stability.
        weights /= weights.max()

    final_q = q_hist[-1] if q_hist else env.weights.copy()
    return MethodResult(
        name="Exp3",
        metrics=tracker.finalize(),
        final_q=final_q,
        extras={"q_history": np.asarray(q_hist)},
    )
