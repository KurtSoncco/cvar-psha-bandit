"""Shared helpers for full-path / flat-path tree methods."""

from __future__ import annotations

from cvar_psha.core.policy import softmax
from cvar_psha.tree_env import TreeLogicEnv

__all__ = ["softmax", "sample_path_from_flat_q", "prior_rollout"]


def sample_path_from_flat_q(env: TreeLogicEnv, q):
    return env.sample_path_from_flat_q(q)


def prior_rollout(env: TreeLogicEnv):
    """Sample path under epistemic priors (IW = 1)."""

    def action_fn(state, prior):
        a = int(env.rng.choice(len(prior), p=prior))
        return a, float(prior[a])

    path, y, iw, _ = env.rollout_with_policy(action_fn)
    return path, y, iw
