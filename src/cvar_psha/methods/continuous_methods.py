"""Naive MC and static oracle baselines for the continuous-epistemic env."""

from __future__ import annotations

import numpy as np

from cvar_psha.continuous_env import ContinuousEpistemicEnv, GaussianProposal
from cvar_psha.core.estimators import OnlineCVaRTracker
from cvar_psha.core.gaussian import weighted_gaussian_moments
from cvar_psha.core.result import MethodResult


def run_continuous_mc(
    env: ContinuousEpistemicEnv,
    v95: float,
    budget: int,
    eval_every: int = 200,
) -> MethodResult:
    """Sample theta directly from the nominal epistemic prior (iw == 1)."""
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    for _ in range(budget):
        theta, y, iw = env.rollout_with_proposal(env.prior)
        tracker.update(y, iw)
    return MethodResult(
        name="Naive MC",
        metrics=tracker.finalize(),
        final_q=None,
        extras={"proposal_mean": env.prior.mean.copy(), "proposal_cov": env.prior.cov.copy()},
    )


def run_continuous_oracle(
    env: ContinuousEpistemicEnv,
    v95: float,
    budget: int,
    proposal: GaussianProposal,
    name: str = "Disagg-IS oracle",
    eval_every: int = 200,
) -> MethodResult:
    """Sample theta from a fixed reference proposal (e.g. a Gaussian moment-
    matched to the closed-form q_disagg grid) -- a static, non-learning
    upper-baseline, analogous to the discrete-tree oracle runners."""
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    for _ in range(budget):
        theta, y, iw = env.rollout_with_proposal(proposal)
        tracker.update(y, iw)
    return MethodResult(
        name=name,
        metrics=tracker.finalize(),
        final_q=None,
        extras={"proposal_mean": proposal.mean.copy(), "proposal_cov": proposal.cov.copy()},
    )


def grid_moment_match(nodes: np.ndarray, mass: np.ndarray, inflation: float = 1.1) -> GaussianProposal:
    """Fit a Gaussian to a categorical (grid-node, mass) distribution by
    moment matching -- used to turn q_disagg's grid representation into a
    sampleable proposal for the oracle baseline."""
    moments = weighted_gaussian_moments(nodes, mass, cov_inflation=inflation)
    if moments is None:
        raise ValueError("grid_moment_match received zero mass")
    return GaussianProposal(*moments)
