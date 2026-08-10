"""Large-sample ground truth for the 3-node logic-tree MDP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cvar_psha.tree_env import TreeLogicEnv


@dataclass(frozen=True)
class TreeGroundTruth:
    v95: float
    cvar: float
    n: int
    percentile: float
    q_star: np.ndarray  # categorical over full paths
    path_labels: list[str]


def compute_tree_ground_truth(
    env: TreeLogicEnv,
    n: int = 1_000_000,
    percentile: float = 0.95,
) -> TreeGroundTruth:
    """Estimate VaR/CVaR under path priors and a path-level IS reference q*."""
    path_ids = env.rng.choice(env.n_paths, size=n, p=env.path_priors)
    mus = np.empty(n, dtype=float)
    sigmas = np.empty(n, dtype=float)
    for i, path in enumerate(env.paths):
        mu, sigma = env.leaf_params(path)
        mask = path_ids == i
        mus[mask] = mu
        sigmas[mask] = sigma
    ys = np.exp(env.rng.normal(mus, sigmas))

    v = float(np.quantile(ys, percentile))
    tail = ys[ys > v]
    if tail.size == 0:
        raise RuntimeError("No exceedances in tree ground-truth sample; increase n.")
    cvar = float(tail.mean())

    # Path contributions ~ prior(path) * P(Y>v|path) * E[Y | Y>v, path]
    contributions = np.zeros(env.n_paths, dtype=float)
    m = max(n // env.n_paths, 10_000)
    for i, path in enumerate(env.paths):
        mu, sigma = env.leaf_params(path)
        ln_y = env.rng.normal(mu, sigma, size=m)
        y = np.exp(ln_y)
        exceed = y > v
        if np.any(exceed):
            contributions[i] = (
                env.path_priors[i] * float(exceed.mean()) * float(y[exceed].mean())
            )
    if contributions.sum() <= 0:
        q_star = env.path_priors.copy()
    else:
        q_star = contributions / contributions.sum()

    labels = [
        "/".join(env.nodes[d].branches[a].name for d, a in enumerate(path))
        for path in env.paths
    ]
    return TreeGroundTruth(
        v95=v,
        cvar=cvar,
        n=n,
        percentile=percentile,
        q_star=q_star,
        path_labels=labels,
    )
