"""CEM importance sampling over full logic-tree paths."""

from __future__ import annotations

import numpy as np

from cvar_psha.estimators import OnlineCVaRTracker
from cvar_psha.methods import MethodResult
from cvar_psha.methods.tree_common import sample_path_from_flat_q
from cvar_psha.tree_env import TreeLogicEnv


def run_tree_cem(
    env: TreeLogicEnv,
    v95: float,
    budget: int,
    batch_size: int = 500,
    elite_frac: float = 0.05,
    smoothing: float = 0.7,
    eval_every: int = 200,
) -> MethodResult:
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    q = env.path_priors.copy()
    q = q / q.sum()
    n_done = 0
    history = [q.copy()]

    while n_done < budget:
        m = min(batch_size, budget - n_done)
        path_ids = np.empty(m, dtype=int)
        ys = np.empty(m, dtype=float)
        for t in range(m):
            path, y, iw = sample_path_from_flat_q(env, q)
            path_ids[t] = env.path_index[path]
            ys[t] = y
            tracker.update(y, iw)
        n_done += m

        k = max(1, int(np.ceil(elite_frac * m)))
        elite = path_ids[np.argpartition(ys, -k)[-k:]]
        counts = np.bincount(elite, minlength=env.n_paths).astype(float)
        if counts.sum() > 0:
            q_hat = counts / counts.sum()
            q = smoothing * q_hat + (1.0 - smoothing) * q
            q = np.clip(q, 1e-8, None)
            q = q / q.sum()
        history.append(q.copy())

    return MethodResult(
        name="CEM-IS",
        metrics=tracker.finalize(),
        final_q=q,
        extras={"q_history": np.asarray(history)},
    )
