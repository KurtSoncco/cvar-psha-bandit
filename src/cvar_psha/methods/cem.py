"""Cross-Entropy Method importance sampling over categorical GMM arms."""

from __future__ import annotations

import numpy as np

from cvar_psha.env import LogicTreeEnv
from cvar_psha.estimators import OnlineCVaRTracker, importance_weight
from cvar_psha.methods import MethodResult


def run_cem(
    env: LogicTreeEnv,
    v95: float,
    budget: int,
    batch_size: int = 500,
    elite_frac: float = 0.05,
    smoothing: float = 0.7,
    eval_every: int = 200,
    q_init: np.ndarray | None = None,
) -> MethodResult:
    """Batched CEM: fit Q on elite (high-Y / exceedance-heavy) samples."""
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    q = env.weights.copy() if q_init is None else np.asarray(q_init, dtype=float)
    q = q / q.sum()
    n_done = 0
    history_q = [q.copy()]

    while n_done < budget:
        m = min(batch_size, budget - n_done)
        arms = np.empty(m, dtype=int)
        ys = np.empty(m, dtype=float)
        for t in range(m):
            arms[t], ys[t] = env.step(q)
            w = importance_weight(env.weights[arms[t]], q[arms[t]])
            tracker.update(ys[t], w)
        n_done += m

        # Elite = top elite_frac by Y (preferring the upper tail).
        k = max(1, int(np.ceil(elite_frac * m)))
        elite_idx = np.argpartition(ys, -k)[-k:]
        elite_arms = arms[elite_idx]
        counts = np.bincount(elite_arms, minlength=env.n_arms).astype(float)
        if counts.sum() > 0:
            q_hat = counts / counts.sum()
            q = smoothing * q_hat + (1.0 - smoothing) * q
            q = np.clip(q, 1e-6, None)
            q = q / q.sum()
        history_q.append(q.copy())

    return MethodResult(
        name="CEM-IS",
        metrics=tracker.finalize(),
        final_q=q,
        extras={"q_history": np.asarray(history_q)},
    )
