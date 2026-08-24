"""Exact (product-quadrature) ground truth for the Houng PSHA continuous env.

Epistemic theta = (b, m_max, dmu, dsigma) has a product prior. We integrate
with Gauss-Legendre on truncated (b, m_max) and Gauss-Hermite on (dmu,
dsigma). For each theta node, magnitudes are discretized and folded into a
lognormal mixture so `disaggregation.py` can compute VaR/CVaR and IS targets.

Hazard curves lambda(a) = E_theta[lambda(a|theta)] and fractiles are computed
on the same quadrature grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import brentq

from cvar_psha.continuous_env import (
    ContinuousEpistemicEnv,
    ContinuousEpistemicSpec,
    EpistemicPrior,
    GaussianProposal,
)
from cvar_psha.core.disaggregation import (
    lognormal_exceedance_prob,
    lognormal_tail_mean_unnormalized,
)


def _gauss_legendre_truncated_normal(
    mean: float,
    std: float,
    lo: float,
    hi: float,
    deg: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and normalized weights for TN(mean, std^2) on [lo, hi]."""
    x, w = np.polynomial.legendre.leggauss(deg)
    # Map [-1, 1] -> [lo, hi]
    pts = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
    pdf_vals = stats.truncnorm.pdf(
        pts,
        (lo - mean) / std,
        (hi - mean) / std,
        loc=mean,
        scale=std,
    )
    weights = w * pdf_vals
    weights = weights / weights.sum()
    return pts, weights


def _gauss_hermite_normal(std: float, deg: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for N(0, std^2) via Hermite quadrature."""
    x, w = np.polynomial.hermite.hermgauss(deg)
    pts = np.sqrt(2.0) * std * x
    weights = w / np.sqrt(np.pi)
    weights = weights / weights.sum()
    return pts, weights


def quadrature_grid(
    spec: ContinuousEpistemicSpec,
    deg_trunc: int = 8,
    deg_normal: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Product quadrature for the four-dimensional epistemic prior.

    Returns (nodes (N, 4), weights (N,)) summing to 1.
    """
    b_pts, b_w = _gauss_legendre_truncated_normal(
        spec.b_mean, spec.b_std, spec.b_lo, spec.b_hi, deg_trunc
    )
    m_pts, m_w = _gauss_legendre_truncated_normal(
        spec.mmax_mean, spec.mmax_std, spec.mmax_lo, spec.mmax_hi, deg_trunc
    )
    dmu_pts, dmu_w = _gauss_hermite_normal(spec.dmu_std, deg_normal)
    ds_pts, ds_w = _gauss_hermite_normal(spec.dsigma_std, deg_normal)

    grids = np.meshgrid(b_pts, m_pts, dmu_pts, ds_pts, indexing="ij")
    w_grids = np.meshgrid(b_w, m_w, dmu_w, ds_w, indexing="ij")

    nodes = np.stack([g.ravel() for g in grids], axis=-1)
    weights = np.prod([w.ravel() for w in w_grids], axis=0)
    weights = weights / weights.sum()
    return nodes, weights


def flatten_theta_mag_mixture(
    env: ContinuousEpistemicEnv,
    theta_nodes: np.ndarray,
    theta_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand each theta node into (theta, M) lognormal leaves.

    Returns (leaf_priors, leaf_mus, leaf_sigmas) for the flattened mixture
    and (theta_indices,) mapping each leaf back to its theta node index.
    """
    priors: list[float] = []
    mus: list[float] = []
    sigmas: list[float] = []
    theta_idx: list[int] = []

    for i, (theta, w_theta) in enumerate(zip(theta_nodes, theta_weights)):
        theta_row = theta.reshape(1, -1)
        m_max = theta[1]
        mags = np.arange(env.spec.m_min, m_max + 1e-9, env.spec.m_step)
        w_mag = env.gr_weights(theta_row[:, 0], theta_row[:, 1], mags[None, :])[0]
        mu_row, sigma_row = env._gmm_params(theta_row, mags[None, :])
        for j, m in enumerate(mags):
            priors.append(w_theta * w_mag[j])
            mus.append(mu_row[0, j])
            sigmas.append(sigma_row[0, j])
            theta_idx.append(i)

    leaf_priors = np.asarray(priors)
    leaf_priors = leaf_priors / leaf_priors.sum()
    return (
        leaf_priors,
        np.asarray(mus),
        np.asarray(sigmas),
        np.asarray(theta_idx, dtype=int),
    )


def marginalize_to_theta(
    theta_weights: np.ndarray,
    leaf_priors: np.ndarray,
    theta_idx: np.ndarray,
    leaf_values: np.ndarray,
) -> np.ndarray:
    """Sum leaf contributions back to theta nodes."""
    n_theta = theta_weights.size
    out = np.zeros(n_theta)
    for k, i in enumerate(theta_idx):
        out[i] += leaf_priors[k] * leaf_values[k]
    total = out.sum()
    if total <= 0:
        return theta_weights / theta_weights.sum()
    return out / total


def solve_pga_for_rate(
    env: ContinuousEpistemicEnv,
    theta_nodes: np.ndarray,
    theta_weights: np.ndarray,
    target_rate: float,
    a_lo: float = 0.01,
    a_hi: float = 5.0,
) -> float:
    """Find PGA a such that mean hazard lambda(a) = target_rate."""

    def mean_hazard(a: float) -> float:
        lam = env.conditional_hazard(theta_nodes, a)
        return float(np.sum(theta_weights * lam))

    f = lambda a: mean_hazard(a) - target_rate

    lo = a_lo
    while f(lo) < 0 and lo > 1e-5:
        lo *= 0.5
    if f(lo) < 0:
        raise RuntimeError(
            f"Target rate {target_rate} below hazard at minimum PGA={lo:g} g"
        )

    hi = a_hi
    while f(hi) > 0 and hi < 50.0:
        hi *= 2.0
    if f(hi) > 0:
        raise RuntimeError(
            f"Target rate {target_rate} above hazard at maximum PGA={hi:g} g"
        )
    return float(brentq(f, lo, hi, xtol=1e-10, rtol=1e-12))


def compute_hazard_curves(
    env: ContinuousEpistemicEnv,
    theta_nodes: np.ndarray,
    theta_weights: np.ndarray,
    pga_grid: np.ndarray,
    fractiles: tuple[float, ...] = (16.0, 50.0, 84.0),
) -> dict[str, np.ndarray]:
    """Mean and fractile hazard curves on a PGA grid."""
    n = pga_grid.size
    lam_individual = np.zeros((theta_nodes.shape[0], n))
    for j, a in enumerate(pga_grid):
        lam_individual[:, j] = env.conditional_hazard(theta_nodes, float(a))

    lam_mean = theta_weights @ lam_individual

    curves: dict[str, np.ndarray] = {"mean": lam_mean.copy()}
    for pct in fractiles:
        p = pct / 100.0
        fract = np.empty(n)
        for j in range(n):
            col = lam_individual[:, j]
            order = np.argsort(col)
            cumw = np.cumsum(theta_weights[order])
            idx = min(int(np.searchsorted(cumw, p)), len(order) - 1)
            fract[j] = col[order[idx]]
        curves[f"p{pct:.0f}"] = fract
    return curves


@dataclass(frozen=True)
class ContinuousGroundTruth:
    v95: float  # PGA threshold at target annual rate (engineering VaR)
    cvar: float
    percentile: float
    target_rate: float
    nodes: np.ndarray
    node_priors: np.ndarray
    node_mus: np.ndarray  # kept for API compat; not used directly
    node_sigmas: np.ndarray
    q_star: np.ndarray  # (n_theta,) marginalized from leaves
    q_disagg: np.ndarray
    leaf_priors: np.ndarray
    leaf_mus: np.ndarray
    leaf_sigmas: np.ndarray
    theta_idx: np.ndarray
    pga_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    hazard_curves: dict[str, np.ndarray] = field(default_factory=dict)


def compute_continuous_ground_truth(
    env: ContinuousEpistemicEnv,
    deg_trunc: int = 8,
    deg_normal: int = 8,
    percentile: float = 0.95,
    target_rate: float = 1e-4,
    pga_grid: np.ndarray | None = None,
) -> ContinuousGroundTruth:
    nodes, weights = quadrature_grid(env.spec, deg_trunc=deg_trunc, deg_normal=deg_normal)

    leaf_priors, leaf_mus, leaf_sigmas, theta_idx = flatten_theta_mag_mixture(
        env, nodes, weights
    )

    v = solve_pga_for_rate(env, nodes, weights, target_rate)
    p_exceed_total = float(
        np.sum(leaf_priors * lognormal_exceedance_prob(leaf_mus, leaf_sigmas, v))
    )
    tail_total = float(
        np.sum(leaf_priors * lognormal_tail_mean_unnormalized(leaf_mus, leaf_sigmas, v))
    )
    cvar = tail_total / max(p_exceed_total, 1e-300)

    p_exceed_theta = env.exceedance_prob(nodes, v)
    q_disagg_theta = weights * p_exceed_theta
    q_disagg_theta = q_disagg_theta / max(q_disagg_theta.sum(), 1e-300)

    tail_theta = env.tail_mean(nodes, v)
    q_star_theta = weights * tail_theta
    q_star_theta = q_star_theta / max(q_star_theta.sum(), 1e-300)

    if pga_grid is None:
        pga_grid = np.logspace(np.log10(0.05), np.log10(2.0), 40)
    hazard_curves = compute_hazard_curves(env, nodes, weights, pga_grid)

    return ContinuousGroundTruth(
        v95=v,
        cvar=cvar,
        percentile=percentile,
        target_rate=target_rate,
        nodes=nodes,
        node_priors=weights,
        node_mus=leaf_mus[: nodes.shape[0]] if leaf_mus.size >= nodes.shape[0] else np.zeros(nodes.shape[0]),
        node_sigmas=leaf_sigmas[: nodes.shape[0]] if leaf_sigmas.size >= nodes.shape[0] else np.full(nodes.shape[0], env.spec.sigma0),
        q_star=q_star_theta,
        q_disagg=q_disagg_theta,
        leaf_priors=leaf_priors,
        leaf_mus=leaf_mus,
        leaf_sigmas=leaf_sigmas,
        theta_idx=theta_idx,
        pga_grid=pga_grid,
        hazard_curves=hazard_curves,
    )


def policy_grid_mass(
    gt: ContinuousGroundTruth,
    prior: EpistemicPrior,
    policy_pdf,
) -> np.ndarray:
    """Map a continuous proposal to categorical mass on theta quadrature nodes."""
    prior_pdf = prior.pdf(gt.nodes)
    ratio = np.asarray(policy_pdf(gt.nodes), dtype=float) / np.clip(prior_pdf, 1e-300, None)
    mass = gt.node_priors * ratio
    total = mass.sum()
    if total <= 0:
        return gt.node_priors.copy()
    return mass / total
