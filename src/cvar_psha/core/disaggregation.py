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

    q_disagg(S) = p(S) P(Y > v | S) / P(Y > v)

This module computes q_disagg in closed form for lognormal leaves
(ln Y | S ~ N(mu_S, sigma_S)) using standard truncated-lognormal identities.

We additionally derive the natural CVaR extension used elsewhere in this
project as `q_star`: replace the indicator 1{Y>v} with the loss Y*1{Y>v},
the correct zero-variance IS target for the *numerator* of
CVaR = E[Y|Y>v] = E[Y 1{Y>v}] / P(Y>v):

    q_star(S) prop to p(S) * E[Y 1{Y>v} | S]

This severity-weighted disaggregation is NOT the result proven in
Houng & Ceferino (2025). It is kept distinct from the proven `q_disagg`.
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
    "closed_form_targets",
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
    """E[Y * 1{Y > v}] for ln Y ~ N(mu, sigma) -- closed form."""
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
    """Exact VaR_percentile of a discrete lognormal mixture via 1-D root-find."""
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
    """q_disagg(s) prop to prior_s * P(Y>v|s)."""
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
    """q_star(s) prop to prior_s * E[Y * 1{Y>v} | s]."""
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
    """Closed-form (up to 1-D root-finding) (VaR_alpha, CVaR_alpha)."""
    priors = np.asarray(priors, dtype=float)
    v = solve_mixture_var(percentile, priors, mus, sigmas)
    tail_num = float(np.sum(priors * lognormal_tail_mean_unnormalized(mus, sigmas, v)))
    cvar = tail_num / (1.0 - percentile)
    return v, cvar


def closed_form_targets(
    priors: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    percentile: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return (VaR, CVaR, q_star, q_disagg) for a discrete lognormal mixture."""
    v, cvar = exact_var_cvar(percentile, priors, mus, sigmas)
    q_star = cvar_disaggregation_weights(priors, mus, sigmas, v)
    q_disagg = disaggregation_weights(priors, mus, sigmas, v)
    return v, cvar, q_star, q_disagg
