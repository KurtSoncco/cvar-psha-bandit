"""Closed-form lognormal disaggregation utilities.

Implements the proven-optimal importance-sampling result of:

    Houng, S. E., & Ceferino, L. (2025). Fast probabilistic seismic hazard
    analysis through adaptive importance sampling. Bulletin of the
    Seismological Society of America, 115(2), 646-662.
    https://doi.org/10.1785/0120240153

Their central theorem: for a discrete scenario variable S (e.g. a logic-tree
path -- source x magnitude x GMM branch combination) with prior p(S), and a
continuous ground-motion variable Y | S with known conditional law, the
zero-variance-optimal importance-sampling density for estimating an
exceedance probability / hazard-curve ordinate P(Y > v) = E[1{Y > v}] is

    q*(S, Y) = p(S) p(Y | S) 1{Y > v} / P(Y > v)

whose S-marginal is exactly the classical PSHA *hazard disaggregation*
distribution:

    q_disagg(S) = p(S) P(Y > v | S) / P(Y > v)                        (paper 1, Eq. for optimal IS density)

This module computes q_disagg in closed form for lognormal leaves
(ln Y | S ~ N(mu_S, sigma_S)) using standard truncated-lognormal identities,
replacing the nested Monte Carlo estimate used previously in
`ground_truth.py` / `tree_ground_truth.py`. Because it is analytic (up to a
1-D root-find for VaR), it removes the simulation noise that a per-leaf MC
estimate of P(Y>v|S) would otherwise inject into the reference distribution
we benchmark policies against.

We additionally derive the natural CVaR extension used elsewhere in this
project as `q_star`: replace the indicator 1{Y>v} with the loss Y*1{Y>v},
the correct zero-variance IS target for the *numerator* of
CVaR = E[Y|Y>v] = E[Y 1{Y>v}] / P(Y>v):

    q_star(S) prop to p(S) * E[Y 1{Y>v} | S]

This severity-weighted disaggregation is NOT the result proven in
Houng & Ceferino (2025) -- that paper addresses hazard-curve /
exceedance-probability estimation, not tail-mean (CVaR) estimation. We
document it here as our own generalization, distinguished from the proven
`q_disagg` result so that empirical comparisons against the paper are not
conflated with our own extension.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import brentq

__all__ = [
    "lognormal_exceedance_prob",
    "lognormal_tail_mean_unnormalized",
    "lognormal_tail_conditional_mean",
    "mixture_cdf",
    "solve_mixture_var",
    "disaggregation_weights",
    "cvar_disaggregation_weights",
    "exact_var_cvar",
]


def lognormal_exceedance_prob(
    mu: np.ndarray | float, sigma: np.ndarray | float, v: float
) -> np.ndarray:
    """P(Y > v) for ln Y ~ N(mu, sigma), vectorized over mu/sigma."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z0 = np.log(v)
    return stats.norm.cdf((mu - z0) / sigma)


def lognormal_tail_mean_unnormalized(
    mu: np.ndarray | float, sigma: np.ndarray | float, v: float
) -> np.ndarray:
    """E[Y * 1{Y > v}] for ln Y ~ N(mu, sigma) -- closed form.

    Derivation: complete the square in the truncated Gaussian integral
    (equivalently, the truncated-MGF / exponential-tilting identity):

        E[e^Z 1{Z > z0}] = exp(mu + sigma^2/2) * Phi((mu + sigma^2 - z0) / sigma)
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z0 = np.log(v)
    return np.exp(mu + 0.5 * sigma**2) * stats.norm.cdf((mu + sigma**2 - z0) / sigma)


def lognormal_tail_conditional_mean(
    mu: np.ndarray | float, sigma: np.ndarray | float, v: float
) -> np.ndarray:
    """E[Y | Y > v] for ln Y ~ N(mu, sigma)."""
    p = lognormal_exceedance_prob(mu, sigma, v)
    num = lognormal_tail_mean_unnormalized(mu, sigma, v)
    return np.divide(num, p, out=np.zeros_like(np.asarray(p, dtype=float)), where=p > 0)


def mixture_cdf(
    v: float, priors: np.ndarray, mus: np.ndarray, sigmas: np.ndarray
) -> float:
    """F(v) = sum_s prior_s * P(Y <= v | s) for a discrete lognormal mixture."""
    priors = np.asarray(priors, dtype=float)
    mus = np.asarray(mus, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    z0 = np.log(v)
    return float(np.sum(priors * stats.norm.cdf((z0 - mus) / sigmas)))


def solve_mixture_var(
    percentile: float,
    priors: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
) -> float:
    """Exact VaR_percentile of a discrete lognormal mixture via 1-D root-find.

    F is continuous and strictly increasing in v (a mixture of lognormal
    CDFs), so bisection/Brent's method converges to machine precision --
    no Monte Carlo sampling noise, unlike `np.quantile` on a simulated batch.
    """
    f = lambda v: mixture_cdf(v, priors, mus, sigmas) - percentile

    lo, hi = 1e-6, 1.0
    n_tries = 0
    while f(hi) < 0:
        hi *= 10.0
        n_tries += 1
        if n_tries > 400:
            raise RuntimeError("Failed to bracket mixture VaR root (hi)")
    n_tries = 0
    while f(lo) > 0:
        lo /= 10.0
        n_tries += 1
        if n_tries > 400:
            raise RuntimeError("Failed to bracket mixture VaR root (lo)")
    return brentq(f, lo, hi, xtol=1e-12, rtol=1e-14)


def disaggregation_weights(
    priors: np.ndarray, mus: np.ndarray, sigmas: np.ndarray, v: float
) -> np.ndarray:
    """q_disagg(s) prop to prior_s * P(Y>v|s).

    This is the paper-1 proven-optimal IS target for hazard-curve /
    exceedance-probability estimation -- the classical PSHA disaggregation
    distribution.
    """
    priors = np.asarray(priors, dtype=float)
    p_exceed = lognormal_exceedance_prob(mus, sigmas, v)
    contrib = priors * p_exceed
    total = contrib.sum()
    if total <= 0:
        return priors.copy()
    return contrib / total


def cvar_disaggregation_weights(
    priors: np.ndarray, mus: np.ndarray, sigmas: np.ndarray, v: float
) -> np.ndarray:
    """q_star(s) prop to prior_s * E[Y * 1{Y>v} | s].

    Severity-weighted extension of the disaggregation result, appropriate
    as the IS target for CVaR (tail-mean) rather than exceedance-probability
    estimation. Not proven optimal in Houng & Ceferino (2025); see module
    docstring.
    """
    priors = np.asarray(priors, dtype=float)
    tail = lognormal_tail_mean_unnormalized(mus, sigmas, v)
    contrib = priors * tail
    total = contrib.sum()
    if total <= 0:
        return priors.copy()
    return contrib / total


def exact_var_cvar(
    percentile: float, priors: np.ndarray, mus: np.ndarray, sigmas: np.ndarray
) -> tuple[float, float]:
    """Closed-form (up to 1-D root-finding) (VaR_alpha, CVaR_alpha) for a
    discrete lognormal mixture Y = exp(mu_S + sigma_S Z), S ~ priors."""
    priors = np.asarray(priors, dtype=float)
    v = solve_mixture_var(percentile, priors, mus, sigmas)
    tail_num = float(np.sum(priors * lognormal_tail_mean_unnormalized(mus, sigmas, v)))
    cvar = tail_num / (1.0 - percentile)
    return v, cvar
