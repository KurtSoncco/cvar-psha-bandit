"""REINFORCE policy-gradient learner for categorical IS proposals."""

from __future__ import annotations

import numpy as np

from cvar_psha.env import LogicTreeEnv
from cvar_psha.estimators import OnlineCVaRTracker, importance_weight, tail_reward
from cvar_psha.methods import MethodResult


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def run_reinforce(
    env: LogicTreeEnv,
    v95: float,
    budget: int,
    learning_rate: float = 0.05,
    baseline_alpha: float = 0.1,
    eval_every: int = 200,
) -> MethodResult:
    """Categorical REINFORCE with exponential moving-average baseline."""
    k = env.n_arms
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    logits = np.zeros(k, dtype=float)
    baseline = 0.0
    q_hist = []
    r_max = 1e-8

    for _ in range(budget):
        q = softmax(logits)
        q_hist.append(q.copy())

        arm, y = env.step(q)
        iw = importance_weight(env.weights[arm], q[arm])
        tracker.update(y, iw)

        r = tail_reward(y, iw, v95)
        r_max = max(r_max, abs(r), 1e-8)
        r_scaled = r / r_max
        advantage = r_scaled - baseline
        baseline = (1.0 - baseline_alpha) * baseline + baseline_alpha * r_scaled

        # ∇ log π(a) = 1_a - π
        grad = -q
        grad[arm] += 1.0
        logits += learning_rate * advantage * grad

    final_q = q_hist[-1] if q_hist else env.weights.copy()
    return MethodResult(
        name="REINFORCE",
        metrics=tracker.finalize(),
        final_q=final_q,
        extras={"q_history": np.asarray(q_hist)},
    )
