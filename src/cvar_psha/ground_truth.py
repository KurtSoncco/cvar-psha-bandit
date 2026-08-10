"""Large-sample prior Monte Carlo ground truth for VaR and CVaR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cvar_psha.env import LogicTreeEnv


@dataclass(frozen=True)
class GroundTruth:
    v95: float
    cvar: float
    n: int
    percentile: float
    # Soft-optimal categorical IS reference: arm mass proportional to
    # E[Y * 1{Y>v95} | arm] * prior (proxy for CVaR IS target).
    q_star: np.ndarray


def _mixture_samples(env: LogicTreeEnv, n: int) -> np.ndarray:
    arms = env.rng.choice(env.n_arms, size=n, p=env.weights)
    ln_y = env.rng.normal(env.mus[arms], env.sigmas[arms])
    return np.exp(ln_y)


def compute_ground_truth(
    env: LogicTreeEnv,
    n: int = 1_000_000,
    percentile: float = 0.95,
) -> GroundTruth:
    """Estimate v_alpha and CVaR_alpha under the epistemic prior mixture."""
    ys = _mixture_samples(env, n)
    v = float(np.quantile(ys, percentile))
    tail = ys[ys > v]
    if tail.size == 0:
        raise RuntimeError("No exceedances in ground-truth sample; increase n.")
    cvar = float(tail.mean())

    # Per-arm contribution to the importance-sampling objective.
    contributions = np.zeros(env.n_arms, dtype=float)
    for i in range(env.n_arms):
        ln_y = env.rng.normal(env.mus[i], env.sigmas[i], size=n)
        y_i = np.exp(ln_y)
        exceed = y_i > v
        if np.any(exceed):
            contributions[i] = env.weights[i] * float(y_i[exceed].mean()) * exceed.mean()
        else:
            contributions[i] = 0.0

    if contributions.sum() <= 0:
        q_star = env.weights.copy()
    else:
        q_star = contributions / contributions.sum()

    return GroundTruth(v95=v, cvar=cvar, n=n, percentile=percentile, q_star=q_star)
