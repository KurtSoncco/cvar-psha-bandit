"""Pareto-Smoothed Importance Sampling (PSIS).

Vehtari, A., Gelman, A., & Gabry, J. (2024). Pareto smoothed importance
sampling. Journal of Machine Learning Research, 25(72), 1-58 (originally
arXiv:1507.02646, 2015). Standard practice in the Stan/ArviZ/`loo`
ecosystem for stabilizing self-normalized importance-sampling estimators
and diagnosing when they cannot be trusted.

Algorithm: fit a Generalized Pareto Distribution (GPD) to the M largest
importance weights (the tail above a threshold u), then replace those M
weights with the expected order statistics implied by the fitted GPD --
smoothing the tail instead of hard-clipping it. The fitted shape
parameter `k_hat` is itself a reliability diagnostic, independent of ESS:
k_hat < 0.5 is reliable, 0.5-0.7 usable with caution, > 0.7 means the
tail is too heavy for the sample size to characterize (the importance-
sampling estimate should not be trusted regardless of what ESS says).

We use scipy.stats.genpareto's maximum-likelihood fit for the GPD
parameters rather than hand-implementing the original paper's Zhang &
Stephens (2009) empirical-Bayes estimator (a small-sample refinement over
plain MLE). This is a deliberate simplification, not an attempt to
reproduce the `loo` package's exact reference implementation -- documented
here rather than silently claimed. Validated against synthetic GPD
samples with known (k, sigma) before use in the CE/PMC refit (jepa_cvar.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PSISResult:
    weights: np.ndarray
    k_hat: float
    n_tail: int
    reliable: bool


def psis_smooth(raw_weights: np.ndarray, min_tail: int = 5) -> PSISResult:
    """Pareto-smooth the upper tail of a set of nonnegative importance weights."""
    w = np.asarray(raw_weights, dtype=float).copy()
    s = w.size
    if s < 20 or not np.any(w > 0):
        return PSISResult(weights=w, k_hat=float("nan"), n_tail=0, reliable=True)

    m = int(min(np.ceil(0.2 * s), np.ceil(3 * np.sqrt(s))))
    if m < min_tail:
        return PSISResult(weights=w, k_hat=float("nan"), n_tail=0, reliable=True)

    order = np.argsort(w)
    tail_idx = order[-m:]
    sorted_tail = w[tail_idx]
    u = w[order[-m - 1]] if s - m - 1 >= 0 else 0.0
    exceedances = np.clip(sorted_tail - u, 1e-12, None)

    if np.all(exceedances <= 1e-12) or np.allclose(exceedances, exceedances[0]):
        return PSISResult(weights=w, k_hat=float("nan"), n_tail=0, reliable=True)

    try:
        k_hat, _loc, sigma_hat = stats.genpareto.fit(exceedances, floc=0.0)
    except Exception:
        return PSISResult(weights=w, k_hat=float("nan"), n_tail=0, reliable=False)

    probs = (np.arange(1, m + 1) - 0.5) / m
    smoothed_exceedances = stats.genpareto.ppf(
        probs, k_hat, loc=0.0, scale=max(sigma_hat, 1e-12)
    )
    smoothed_tail = np.minimum(u + smoothed_exceedances, w.max())
    w[tail_idx] = smoothed_tail
    return PSISResult(weights=w, k_hat=float(k_hat), n_tail=m, reliable=bool(k_hat < 0.7))
