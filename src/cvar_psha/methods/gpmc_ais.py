"""Gaussian Population Monte Carlo Adaptive Importance Sampling (G-PMC AIS).

Non-learning baseline for the continuous-epistemic environment, implementing
the batched, moment-matching adaptive importance sampling scheme of
Houng, Ceferino & Abrahamson (2025), https://doi.org/10.1785/0120250205.
"""

from __future__ import annotations

import numpy as np

from cvar_psha.continuous_env import ContinuousEpistemicEnv, GaussianProposal
from cvar_psha.core.estimators import OnlineCVaRTracker
from cvar_psha.core.gaussian import weighted_gaussian_moments
from cvar_psha.core.result import MethodResult


def run_gpmc_ais(
    env: ContinuousEpistemicEnv,
    v95: float,
    budget: int,
    batch_size: int = 300,
    smoothing: float = 0.5,
    cov_inflation: float = 1.15,
    cov_floor_scale: float = 0.05,
    defensive_eps: float = 0.1,
    target: str = "disagg",
    eval_every: int = 200,
    log_samples: bool = False,
) -> MethodResult:
    """`log_samples=True` additionally stores every (theta, y, iw) drawn
    across the whole run in extras -- needed to reconstruct hazard curves
    (mean and fractile, at any threshold) post-hoc from a single run
    instead of re-running per threshold; see hazard_curve.py."""
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    prior = env.prior
    dim = env.theta_dim
    mean = env.spec.prior_means.copy()
    cov = np.diag(env.spec.prior_scales**2)
    cov_floor = cov_floor_scale * cov
    proposal = GaussianProposal(mean, cov)
    mean_history = [mean.copy()]
    log_thetas: list = []
    log_ys: list = []
    log_iws: list = []

    # Defensive prior mixture uses a Gaussian anchored at prior means.
    prior_gauss = GaussianProposal.from_spec(env.spec)

    n_done = 0
    while n_done < budget:
        m = min(batch_size, budget - n_done)

        use_prior_mask = env.rng.random(m) < defensive_eps
        thetas = np.empty((m, dim))
        n_prior = int(use_prior_mask.sum())
        n_prop = m - n_prior
        if n_prop > 0:
            thetas[~use_prior_mask] = env._clip_theta(proposal.sample(env.rng, n=n_prop))
        if n_prior > 0:
            thetas[use_prior_mask] = prior.sample(env.rng, n=n_prior)

        q_mix_pdf = (1 - defensive_eps) * proposal.pdf(thetas) + defensive_eps * prior_gauss.pdf(
            thetas
        )
        prior_pdf = prior.pdf(thetas)

        ys = env.sample_y(thetas)
        iw = prior_pdf / np.clip(q_mix_pdf, 1e-300, None)
        for y_i, w_i in zip(ys, iw):
            tracker.update(float(y_i), float(w_i))
        if log_samples:
            log_thetas.append(thetas.copy())
            log_ys.append(ys.copy())
            log_iws.append(iw.copy())
        n_done += m

        if target == "cvar":
            relevance = env.tail_mean(thetas, v95)
        else:
            relevance = env.exceedance_prob(thetas, v95)
        target_unnorm = prior_pdf * relevance
        pmc_w = target_unnorm / np.clip(q_mix_pdf, 1e-300, None)
        moments = weighted_gaussian_moments(
            thetas, pmc_w, cov_inflation=cov_inflation, cov_floor=cov_floor
        )
        if moments is not None:
            mean_hat, cov_hat = moments
            mean = smoothing * mean_hat + (1.0 - smoothing) * mean
            cov = smoothing * cov_hat + (1.0 - smoothing) * cov
            proposal = GaussianProposal(mean, cov)
        mean_history.append(mean.copy())

    name = "G-PMC AIS (CVaR)" if target == "cvar" else "G-PMC AIS"
    extras = {
        "final_mean": mean.copy(),
        "final_cov": cov.copy(),
        "proposal_mean": mean.copy(),
        "proposal_cov": cov.copy(),
        "mean_history": np.asarray(mean_history),
    }
    if log_samples:
        extras["thetas"] = np.concatenate(log_thetas, axis=0)
        extras["ys"] = np.concatenate(log_ys, axis=0)
        extras["iws"] = np.concatenate(log_iws, axis=0)
    return MethodResult(
        name=name,
        metrics=tracker.finalize(),
        final_q=None,
        extras=extras,
    )
