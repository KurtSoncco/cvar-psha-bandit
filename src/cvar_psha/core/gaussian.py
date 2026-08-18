"""Weighted Gaussian moment matching used by G-PMC, JEPA v2, and QR-SRM."""

from __future__ import annotations

import numpy as np


def weighted_gaussian_moments(
    samples: np.ndarray,
    weights: np.ndarray,
    *,
    cov_inflation: float = 1.0,
    cov_floor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (mean, cov) from self-normalized weights, or None if all mass is 0."""
    w = np.asarray(weights, dtype=float)
    wsum = float(w.sum())
    if wsum <= 0:
        return None
    w_norm = w / wsum
    x = np.asarray(samples, dtype=float)
    mean = (w_norm[:, None] * x).sum(axis=0)
    diff = x - mean[None, :]
    cov = (w_norm[:, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0)
    cov = cov * cov_inflation
    if cov_floor is not None:
        cov = cov + cov_floor
    return mean, cov
