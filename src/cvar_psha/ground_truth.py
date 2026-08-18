"""Exact (closed-form) ground truth for VaR, CVaR, and optimal IS targets.

Previously this module estimated VaR/CVaR and the IS reference distribution
from nested Monte Carlo batches, which injects simulation noise into the
very quantity every method is benchmarked against. Since each arm's
ln Y ~ N(mu_i, sigma_i) is lognormal, VaR/CVaR/disaggregation all have
closed forms (up to a 1-D root-find for VaR) -- see `disaggregation.py`,
which implements the proven-optimal-IS-equals-hazard-disaggregation result
of Houng & Ceferino (2025, https://doi.org/10.1785/0120240153).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cvar_psha.core.disaggregation import closed_form_targets
from cvar_psha.env import LogicTreeEnv


@dataclass(frozen=True)
class GroundTruth:
    v95: float
    cvar: float
    n: int
    percentile: float
    # CVaR-optimal (severity-weighted) IS reference: arm mass proportional to
    # E[Y * 1{Y>v95} | arm] * prior -- our extension of the disaggregation
    # result to tail-mean (CVaR) estimation. See disaggregation.py.
    q_star: np.ndarray
    # Paper-1 proven-optimal IS reference for exceedance-probability /
    # hazard-curve estimation: arm mass proportional to
    # P(Y>v95 | arm) * prior -- the classical hazard disaggregation.
    q_disagg: np.ndarray


def compute_ground_truth(
    env: LogicTreeEnv,
    n: int = 1_000_000,
    percentile: float = 0.95,
) -> GroundTruth:
    """Exact VaR_alpha, CVaR_alpha, q_star, q_disagg under the epistemic
    prior mixture (closed-form; `n` is unused, kept for API compatibility).
    """
    v, cvar, q_star, q_disagg = closed_form_targets(
        env.weights, env.mus, env.sigmas, percentile
    )

    return GroundTruth(
        v95=v,
        cvar=cvar,
        n=n,
        percentile=percentile,
        q_star=q_star,
        q_disagg=q_disagg,
    )
