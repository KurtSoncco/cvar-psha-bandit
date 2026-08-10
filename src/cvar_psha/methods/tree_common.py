"""Shared helpers for full-path / flat-path tree methods."""

from __future__ import annotations

import numpy as np

from cvar_psha.tree_env import TreeLogicEnv


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


def sample_path_from_flat_q(
    env: TreeLogicEnv,
    q: np.ndarray,
) -> tuple[tuple[int, ...], float, float]:
    """Sample a full path from categorical q over paths; return path, y, IW."""
    q = np.asarray(q, dtype=float)
    q = q / q.sum()
    idx = int(env.rng.choice(env.n_paths, p=q))
    path = env.paths[idx]
    y = env.sample_y(path)
    iw = env.path_priors[idx] / max(q[idx], 1e-12)
    return path, y, iw


def prior_rollout(env: TreeLogicEnv) -> tuple[tuple[int, ...], float, float]:
    """Sample path under epistemic priors (IW = 1)."""

    def action_fn(state, prior):
        a = int(env.rng.choice(len(prior), p=prior))
        return a, float(prior[a])

    path, y, iw, _ = env.rollout_with_policy(action_fn)
    return path, y, iw
