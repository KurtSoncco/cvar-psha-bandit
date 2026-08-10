"""Exact (Gauss-Hermite quadrature) ground truth for the continuous-
epistemic environment.

Because theta = (theta_mu, theta_sigma) has an independent bivariate
Gaussian prior, its expectation integrals can be evaluated to near machine
precision with a tensor-product Gauss-Hermite quadrature rule -- no Monte
Carlo sampling noise, continuing the same rigor as the discrete-tree
closed-form ground truth (`disaggregation.py`).

Gauss-Hermite quadrature nodes/weights for integrating against a Gaussian
measure are themselves a discrete (node, probability-mass) representation
of the continuous prior -- i.e. exactly the "priors, mus, sigmas" input
shape `disaggregation.py` already expects for the discrete-tree case. So
the continuous-epistemic ground truth (VaR, CVaR, q_disagg, q_star) is
computed by *reusing* `disaggregation.py` unchanged, with the quadrature
grid standing in for the "paths."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cvar_psha.continuous_env import ContinuousEpistemicEnv, GaussianProposal
from cvar_psha.disaggregation import (
    cvar_disaggregation_weights,
    disaggregation_weights,
    exact_var_cvar,
)


def quadrature_grid(
    tau_mu: float, tau_sigma: float, deg: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Tensor-product Gauss-Hermite grid for theta ~ N(0, diag(tau_mu^2, tau_sigma^2)).

    Returns (nodes (deg*deg, 2), weights (deg*deg,)) with weights summing to 1
    and standing in for the prior probability mass at each node.
    """
    x, w = np.polynomial.hermite.hermgauss(deg)
    # E_{N(0,tau^2)}[g] = (1/sqrt(pi)) sum_i w_i g(sqrt(2) tau x_i)
    pts = np.sqrt(2.0) * x
    wts = w / np.sqrt(np.pi)

    mu_pts = tau_mu * pts
    sigma_pts = tau_sigma * pts
    MU, SIGMA = np.meshgrid(mu_pts, sigma_pts, indexing="ij")
    WM, WS = np.meshgrid(wts, wts, indexing="ij")

    nodes = np.stack([MU.ravel(), SIGMA.ravel()], axis=-1)
    weights = (WM * WS).ravel()
    weights = weights / weights.sum()
    return nodes, weights


@dataclass(frozen=True)
class ContinuousGroundTruth:
    v95: float
    cvar: float
    percentile: float
    nodes: np.ndarray  # (N, 2) theta grid
    node_priors: np.ndarray  # (N,) prior mass per node (quadrature weights)
    node_mus: np.ndarray  # (N,) mu(theta) per node
    node_sigmas: np.ndarray  # (N,) sigma(theta) per node
    q_star: np.ndarray  # (N,) CVaR-optimal disaggregation mass (ours)
    q_disagg: np.ndarray  # (N,) paper-2 proven-optimal disaggregation mass


def compute_continuous_ground_truth(
    env: ContinuousEpistemicEnv,
    deg: int = 40,
    percentile: float = 0.95,
) -> ContinuousGroundTruth:
    nodes, weights = quadrature_grid(env.spec.tau_mu, env.spec.tau_sigma, deg=deg)
    mus, sigmas = env.leaf_params(nodes)

    v, cvar = exact_var_cvar(percentile, weights, mus, sigmas)
    q_star = cvar_disaggregation_weights(weights, mus, sigmas, v)
    q_disagg = disaggregation_weights(weights, mus, sigmas, v)

    return ContinuousGroundTruth(
        v95=v,
        cvar=cvar,
        percentile=percentile,
        nodes=nodes,
        node_priors=weights,
        node_mus=mus,
        node_sigmas=sigmas,
        q_star=q_star,
        q_disagg=q_disagg,
    )


def policy_grid_mass(
    gt: ContinuousGroundTruth,
    prior: GaussianProposal,
    policy_pdf,
) -> np.ndarray:
    """Represent an arbitrary continuous proposal (e.g. a G-PMC AIS mixture
    or the JEPA-CVaR policy) as a categorical distribution over the SAME
    quadrature nodes used for q_disagg/q_star, via importance reweighting
    of the prior's quadrature mass by the proposal/prior density ratio:

        mass(node_k) prop to prior_weight_k * policy_pdf(node_k) / prior_pdf(node_k)

    This makes the learned proposal directly comparable to q_disagg/q_star
    with the existing KL/TV/KS utilities (metrics.py), without needing a
    second, independently-noisy quadrature scheme per method.
    """
    prior_pdf = prior.pdf(gt.nodes)
    ratio = np.asarray(policy_pdf(gt.nodes), dtype=float) / np.clip(prior_pdf, 1e-300, None)
    mass = gt.node_priors * ratio
    total = mass.sum()
    if total <= 0:
        return gt.node_priors.copy()
    return mass / total
