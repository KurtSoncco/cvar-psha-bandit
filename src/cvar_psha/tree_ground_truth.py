"""Exact (closed-form) ground truth for the 3-node logic-tree MDP.

See `disaggregation.py` for the underlying closed-form lognormal identities
and the theoretical grounding (Houng & Ceferino, 2025,
https://doi.org/10.1785/0120240153): the optimal IS distribution for
exceedance-probability estimation equals the hazard disaggregation
distribution, q_disagg(path) prop to prior(path) * P(Y>v|path). We also
report our CVaR (severity-weighted) extension, q_star.

Each full path through the tree pins down a single leaf lognormal
(ln Y | path ~ N(mu_path, sigma_path)), so VaR/CVaR/disaggregation over the
full path-mixture are closed-form (up to a 1-D root-find for VaR) -- no
Monte Carlo sampling noise, unlike the previous nested-MC estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cvar_psha.core.disaggregation import closed_form_targets
from cvar_psha.tree_env import TreeLogicEnv


@dataclass(frozen=True)
class TreeGroundTruth:
    v95: float
    cvar: float
    n: int
    percentile: float
    q_star: np.ndarray  # categorical over full paths (CVaR-optimal, ours)
    q_disagg: np.ndarray  # categorical over full paths (paper-1 proven-optimal)
    path_labels: list[str]


def _path_leaf_params(env: TreeLogicEnv) -> tuple[np.ndarray, np.ndarray]:
    mus = np.empty(env.n_paths, dtype=float)
    sigmas = np.empty(env.n_paths, dtype=float)
    for i, path in enumerate(env.paths):
        mus[i], sigmas[i] = env.leaf_params(path)
    return mus, sigmas


def compute_tree_ground_truth(
    env: TreeLogicEnv,
    n: int = 1_000_000,
    percentile: float = 0.95,
) -> TreeGroundTruth:
    """Exact VaR/CVaR/q_star/q_disagg under path priors (closed-form;
    `n` is unused, kept for API compatibility)."""
    mus, sigmas = _path_leaf_params(env)
    v, cvar, q_star, q_disagg = closed_form_targets(
        env.path_priors, mus, sigmas, percentile
    )

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
        q_disagg=q_disagg,
        path_labels=labels,
    )
