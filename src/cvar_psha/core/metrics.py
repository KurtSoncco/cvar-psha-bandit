"""Shared distributional-distance diagnostics.

`ks_statistic` is the categorical analog of the Kolmogorov-Smirnov distance
reported by Houng & Ceferino (2025, https://doi.org/10.1785/0120240153;
KS-D in [0.017, 0.113]) and Houng, Ceferino & Abrahamson (2025,
https://doi.org/10.1785/0120250205; KS-D < 5%) when they compare their
adaptive-IS-approximated distribution against the true disaggregation /
fractile-hazard distribution. Their KS-D is computed on a continuous
hazard curve; ours is computed on a discrete categorical distribution over
enumerable logic-tree paths (or, for the continuous-epistemic environment,
over a fine grid), using a fixed shared category order so it reduces to
the same max-CDF-gap definition. It is directly comparable in spirit, not
identical in construction -- kept distinct from KL/TV so both are visible.
"""

from __future__ import annotations

import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, None)
    q = np.clip(np.asarray(q, dtype=float), eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return 0.5 * float(np.abs(p - q).sum())


def ks_statistic(p: np.ndarray, q: np.ndarray, order: np.ndarray | None = None) -> float:
    """Max CDF gap between two categorical distributions under a shared
    category order (defaults to the given index order, e.g. path index).

    Pass `order` (an argsort of indices, e.g. np.argsort(q_disagg)[::-1])
    to compare mass accumulation in a scenario-relevance order instead of
    raw enumeration order.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    if order is not None:
        p = p[order]
        q = q[order]
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.max(np.abs(cdf_p - cdf_q)))
