"""Mean and fractile hazard curves for the continuous-epistemic environment.

Mirrors the actual deliverable of Houng, Ceferino & Abrahamson (2025),
https://doi.org/10.1785/0120250205: a single adaptive-IS run over the
continuous epistemic parameters theta gives efficient access to both the
*mean* hazard curve H(y) = E_theta[P(Y>y|theta)] and the *fractile*
hazard curves (the distribution, over epistemic realizations, of
P(Y>y|theta) at each y) -- not just a single VaR/CVaR summary.

Two ways to compute both:
  - `exact_hazard_curve`: closed-form via the same Gauss-Hermite quadrature
    grid used elsewhere in this project (continuous_ground_truth.py) --
    deterministic, no Monte Carlo noise.
  - `hazard_curve_from_samples`: reconstructed post-hoc from a single
    logged (theta, y, iw) run of any IS method (see `log_samples=True` on
    run_gpmc_ais / run_jepa_cvar_v2) -- mean via a plain (non-self-
    normalized) importance-sampling average of 1{y>threshold}, fractiles
    via a weighted quantile of the closed-form conditional hazard
    P(Y>threshold|theta) evaluated at the sampled theta, weighted by iw.
    This is the standard trick that makes a single AIS run, tuned toward
    one (the most extreme) threshold, sufficient for an entire curve: the
    same weighted sample answers the query at every less-extreme
    threshold too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cvar_psha.continuous_env import ContinuousEpistemicEnv
from cvar_psha.continuous_ground_truth import quadrature_grid
from cvar_psha.disaggregation import lognormal_exceedance_prob

DEFAULT_FRACTILES = (0.05, 0.5, 0.95)


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray | tuple[float, ...]
) -> np.ndarray:
    """Weighted quantiles of `values` under (unnormalized) `weights`, via
    linear interpolation on the weighted empirical CDF."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w) - 0.5 * w
    cum = cum / w.sum()
    return np.interp(quantiles, cum, v)


@dataclass(frozen=True)
class HazardCurveResult:
    y_grid: np.ndarray
    mean: np.ndarray
    fractiles: dict[float, np.ndarray]  # quantile -> array over y_grid


def exact_hazard_curve(
    env: ContinuousEpistemicEnv,
    y_grid: np.ndarray,
    deg: int = 40,
    fractiles: tuple[float, ...] = DEFAULT_FRACTILES,
) -> HazardCurveResult:
    nodes, weights = quadrature_grid(env.spec.tau_mu, env.spec.tau_sigma, deg=deg)
    mus, sigmas = env.leaf_params(nodes)

    mean = np.empty(len(y_grid))
    fractile_arrays = {q: np.empty(len(y_grid)) for q in fractiles}
    for i, y in enumerate(y_grid):
        hazard_node = lognormal_exceedance_prob(mus, sigmas, y)  # P(Y>y|theta) per node
        mean[i] = float(np.sum(weights * hazard_node))
        qs = weighted_quantile(hazard_node, weights, fractiles)
        for q, val in zip(fractiles, qs):
            fractile_arrays[q][i] = val

    return HazardCurveResult(y_grid=np.asarray(y_grid), mean=mean, fractiles=fractile_arrays)


def hazard_curve_from_samples(
    env: ContinuousEpistemicEnv,
    thetas: np.ndarray,
    ys: np.ndarray,
    iws: np.ndarray,
    y_grid: np.ndarray,
    fractiles: tuple[float, ...] = DEFAULT_FRACTILES,
) -> HazardCurveResult:
    n = ys.size
    mean = np.array(
        [float(np.sum(iws * (ys > y)) / n) for y in y_grid]
    )

    mu_all, sigma_all = env.leaf_params(thetas)
    fractile_arrays = {q: np.empty(len(y_grid)) for q in fractiles}
    for i, y in enumerate(y_grid):
        hazard_theta = lognormal_exceedance_prob(mu_all, sigma_all, y)  # P(Y>y|theta_i)
        qs = weighted_quantile(hazard_theta, iws, fractiles)
        for q, val in zip(fractiles, qs):
            fractile_arrays[q][i] = val

    return HazardCurveResult(y_grid=np.asarray(y_grid), mean=mean, fractiles=fractile_arrays)
