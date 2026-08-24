"""Continuous-epistemic PSHA environment (Houng, Ceferino & Abrahamson 2025).

Mirrors the four-parameter epistemic model from Houng, Ceferino &
Abrahamson (2025), BSSA 116(4):1709, https://doi.org/10.1785/0120250205:

    theta = (b, m_max, Delta_mu, Delta_sigma)

with truncated-normal priors on b and m_max and Gaussian priors on the GMM
offsets. Aleatory uncertainty integrates magnitude (truncated Gutenberg-
Richter / exponential recurrence) and a Sadigh et al. (1997) strike-slip
rock GMPE at fixed distance R = 35 km (their Fig. 1 geometry).

Conditional hazard at intensity a:

    lambda(a | theta) = nu * P(PGA > a | theta)

where nu is the annual occurrence rate for M > m_min (fixed at 0.01 /yr).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from cvar_psha.core.disaggregation import (
    lognormal_exceedance_prob,
    lognormal_tail_mean_unnormalized,
)

THETA_NAMES = ("b", "m_max", "dmu", "dsigma")


NEAR_FIELD_MAG = 6.5

# Sadigh et al. (1997) rock PGA coefficients (OpenQuake table 2).
_SADIGH_ROCK_LOW = (-0.624, 1.0, -2.100, 1.29649, 0.250)
_SADIGH_ROCK_HI = (-1.274, 1.1, -2.100, -0.48451, 0.524)


def sadigh_ln_pga_median(m: np.ndarray | float, r_km: float) -> np.ndarray:
    """Sadigh et al. (1997) strike-slip rock median ln(PGA) [g], vectorized."""
    m = np.clip(np.asarray(m, dtype=float), None, 8.5)
    out = np.empty_like(m)
    low = m <= NEAR_FIELD_MAG
    hi = ~low
    if np.any(low):
        ml = m[low]
        c1, c2, c4, c5, c6 = _SADIGH_ROCK_LOW
        out[low] = c1 + c2 * ml + c4 * np.log(r_km + np.exp(c5 + c6 * ml))
    if np.any(hi):
        mh = m[hi]
        c1, c2, c4, c5, c6 = _SADIGH_ROCK_HI
        out[hi] = c1 + c2 * mh + c4 * np.log(r_km + np.exp(c5 + c6 * mh))
    return out


def sadigh_ln_pga_sigma(m: np.ndarray | float, sigma0: float = 1.39, mag_factor: float = -0.14) -> np.ndarray:
    """Sadigh et al. (1997) total sigma for rock PGA (table 3), capped at M=7.21."""
    m = np.asarray(m, dtype=float)
    raw = sigma0 + mag_factor * m
    return np.clip(raw, 0.38, None)


def truncated_normal_pdf(
    x: np.ndarray,
    mean: float,
    std: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """PDF of N(mean, std^2) truncated to [lo, hi]."""
    x = np.asarray(x, dtype=float)
    a, b = (lo - mean) / std, (hi - mean) / std
    return stats.truncnorm.pdf(x, a, b, loc=mean, scale=std)


def truncated_normal_sample(
    rng: np.random.Generator,
    n: int,
    mean: float,
    std: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    a, b = (lo - mean) / std, (hi - mean) / std
    return stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=n, random_state=rng)


@dataclass(frozen=True)
class ContinuousEpistemicSpec:
    """Houng et al. (2025) four-parameter epistemic specification."""

    nu: float = 0.01
    m_min: float = 5.0
    m_step: float = 0.1

    b_mean: float = 1.0
    b_std: float = 0.1
    b_lo: float = 0.7
    b_hi: float = 1.1

    mmax_mean: float = 7.0
    mmax_std: float = 0.3
    mmax_lo: float = 5.9
    mmax_hi: float = 7.1

    dmu_std: float = 0.1
    dsigma_std: float = 0.05

    sigma0: float = 1.39
    sigma_mag_factor: float = -0.14
    distance_km: float = 35.0

    @property
    def theta_dim(self) -> int:
        return 4

    @property
    def prior_means(self) -> np.ndarray:
        return np.array([self.b_mean, self.mmax_mean, 0.0, 0.0], dtype=float)

    @property
    def prior_scales(self) -> np.ndarray:
        return np.array([self.b_std, self.mmax_std, self.dmu_std, self.dsigma_std], dtype=float)

    @property
    def prior_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([self.b_lo, self.mmax_lo, -np.inf, -np.inf], dtype=float)
        hi = np.array([self.b_hi, self.mmax_hi, np.inf, np.inf], dtype=float)
        return lo, hi


class EpistemicPrior:
    """Product prior: truncated normal (b, m_max) x normal (dmu, dsigma)."""

    def __init__(self, spec: ContinuousEpistemicSpec):
        self.spec = spec
        self.dim = spec.theta_dim

    def pdf(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        s = self.spec
        p = (
            truncated_normal_pdf(theta[:, 0], s.b_mean, s.b_std, s.b_lo, s.b_hi)
            * truncated_normal_pdf(theta[:, 1], s.mmax_mean, s.mmax_std, s.mmax_lo, s.mmax_hi)
            * stats.norm.pdf(theta[:, 2], loc=0.0, scale=s.dmu_std)
            * stats.norm.pdf(theta[:, 3], loc=0.0, scale=s.dsigma_std)
        )
        return p

    def logpdf(self, theta: np.ndarray) -> np.ndarray:
        return np.log(np.clip(self.pdf(theta), 1e-300, None))

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        s = self.spec
        out = np.empty((n, self.dim))
        out[:, 0] = truncated_normal_sample(rng, n, s.b_mean, s.b_std, s.b_lo, s.b_hi)
        out[:, 1] = truncated_normal_sample(rng, n, s.mmax_mean, s.mmax_std, s.mmax_lo, s.mmax_hi)
        out[:, 2] = rng.normal(0.0, s.dmu_std, size=n)
        out[:, 3] = rng.normal(0.0, s.dsigma_std, size=n)
        return out

    @property
    def mean(self) -> np.ndarray:
        return self.spec.prior_means.copy()

    @property
    def cov(self) -> np.ndarray:
        return np.diag(self.spec.prior_scales ** 2)


class GaussianProposal:
    """n-D Gaussian IS proposal over epistemic theta."""

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = np.asarray(mean, dtype=float).reshape(-1)
        self.dim = self.mean.size
        self.cov = np.asarray(cov, dtype=float).reshape(self.dim, self.dim)
        self._chol = np.linalg.cholesky(self.cov + 1e-12 * np.eye(self.dim))
        self._prec = np.linalg.inv(self.cov + 1e-12 * np.eye(self.dim))
        self._logdet = float(np.linalg.slogdet(self.cov + 1e-12 * np.eye(self.dim))[1])

    @classmethod
    def from_spec(cls, spec: ContinuousEpistemicSpec) -> "GaussianProposal":
        scales = spec.prior_scales
        return cls(mean=spec.prior_means.copy(), cov=np.diag(scales**2))

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        z = rng.standard_normal((n, self.dim))
        return self.mean[None, :] + z @ self._chol.T

    def logpdf(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        d = theta - self.mean[None, :]
        quad = np.einsum("ni,ij,nj->n", d, self._prec, d)
        return -0.5 * (self.dim * np.log(2 * np.pi) + self._logdet + quad)

    def pdf(self, theta: np.ndarray) -> np.ndarray:
        return np.exp(self.logpdf(theta))


@dataclass
class ContinuousEpistemicEnv:
    """Single-site PGA under Houng-style continuous epistemic uncertainty."""

    spec: ContinuousEpistemicSpec = field(default_factory=ContinuousEpistemicSpec)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        self.prior = EpistemicPrior(self.spec)
        self.theta_dim = self.spec.theta_dim
        self.theta_names = THETA_NAMES
        self.prior_scales = self.spec.prior_scales.copy()
        self.rng = self.rng

    def _effective_sigma(self, m: np.ndarray, dsigma: np.ndarray) -> np.ndarray:
        """Aleatory sigma with epistemic offset, floored for positivity."""
        m = np.asarray(m, dtype=float)
        base = sadigh_ln_pga_sigma(m, self.spec.sigma0, self.spec.sigma_mag_factor)
        ds = np.asarray(dsigma, dtype=float).reshape(-1)
        if ds.size == 1:
            ds = np.full(np.shape(m), ds[0])
        else:
            ds = np.broadcast_to(ds, np.shape(m))
        return np.maximum(base + ds, 0.08)

    def _clip_theta(self, theta: np.ndarray) -> np.ndarray:
        """Clip truncated dimensions to prior support."""
        theta = np.atleast_2d(theta).copy()
        s = self.spec
        theta[:, 0] = np.clip(theta[:, 0], s.b_lo, s.b_hi)
        theta[:, 1] = np.clip(theta[:, 1], s.mmax_lo, s.mmax_hi)
        return theta

    def mag_grid(self, m_max: np.ndarray | float) -> np.ndarray:
        """Magnitude bin centers up to m_max (vectorized over m_max rows)."""
        m_max = np.atleast_1d(m_max)
        n = m_max.size
        base = np.arange(self.spec.m_min, self.spec.mmax_hi + 1e-9, self.spec.m_step)
        grids = []
        for mm in m_max:
            grids.append(base[base <= mm + 1e-9])
        max_len = max(len(g) for g in grids)
        out = np.full((n, max_len), np.nan)
        for i, g in enumerate(grids):
            out[i, : len(g)] = g
        return out

    def gr_weights(self, b: np.ndarray, m_max: np.ndarray, mags: np.ndarray) -> np.ndarray:
        """Truncated GR mass on magnitude bins, shape (n_theta, n_mag).

        mags may contain NaN padding for variable-length grids; masked out.
        """
        b = np.asarray(b, dtype=float).reshape(-1, 1)
        m_max = np.asarray(m_max, dtype=float).reshape(-1, 1)
        mags = np.asarray(mags, dtype=float)
        beta = b * np.log(10.0)
        valid = np.isfinite(mags) & (mags <= m_max + 1e-9)
        w = np.exp(-beta * (mags - self.spec.m_min))
        w = np.where(valid, w, 0.0)
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum = np.where(w_sum > 0, w_sum, 1.0)
        return w / w_sum

    def _gmm_params(
        self, theta: np.ndarray, mags: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-(theta, mag) lognormal params: mu, sigma arrays (n_theta, n_mag)."""
        theta = np.atleast_2d(theta)
        n = theta.shape[0]
        n_mag = mags.shape[1]
        mu = np.zeros((n, n_mag))
        sigma = np.zeros((n, n_mag))
        r = self.spec.distance_km
        for i in range(n):
            valid = np.isfinite(mags[i])
            mu[i, valid] = sadigh_ln_pga_median(mags[i, valid], r) + theta[i, 2]
            sigma[i, valid] = self._effective_sigma(mags[i, valid], theta[i, 3])
        return mu, sigma

    def exceedance_prob(self, theta: np.ndarray, a: float) -> np.ndarray:
        """P(PGA > a | theta) via magnitude Riemann sum."""
        theta = self._clip_theta(theta)
        m_max = theta[:, 1]
        mags = self.mag_grid(m_max)
        w_mag = self.gr_weights(theta[:, 0], m_max, mags)
        mu, sigma = self._gmm_params(theta, mags)
        p_bin = lognormal_exceedance_prob(mu, sigma, a)
        p_bin = np.where(np.isfinite(mags), p_bin, 0.0)
        return (w_mag * p_bin).sum(axis=1)

    def tail_mean(self, theta: np.ndarray, a: float) -> np.ndarray:
        """E[PGA * 1{PGA > a} | theta]."""
        theta = self._clip_theta(theta)
        m_max = theta[:, 1]
        mags = self.mag_grid(m_max)
        w_mag = self.gr_weights(theta[:, 0], m_max, mags)
        mu, sigma = self._gmm_params(theta, mags)
        tail = lognormal_tail_mean_unnormalized(mu, sigma, a)
        tail = np.where(np.isfinite(mags), tail, 0.0)
        return (w_mag * tail).sum(axis=1)

    def conditional_hazard(self, theta: np.ndarray, a: float) -> np.ndarray:
        """lambda(a | theta) = nu * P(PGA > a | theta)."""
        return self.spec.nu * self.exceedance_prob(theta, a)

    def leaf_params(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """GR-weighted mixture mean of (ln PGA mu, sigma) given theta."""
        theta = self._clip_theta(theta)
        m_max = theta[:, 1]
        mags = self.mag_grid(m_max)
        w = self.gr_weights(theta[:, 0], m_max, mags)
        mu, sigma = self._gmm_params(theta, mags)
        mu = np.where(np.isfinite(mags), mu, 0.0)
        sigma = np.where(np.isfinite(mags), sigma, 0.0)
        mu_eff = (w * mu).sum(axis=1)
        sig_eff = np.maximum((w * sigma).sum(axis=1), 0.08)
        return mu_eff, sig_eff

    def sample_magnitude(self, theta: np.ndarray) -> np.ndarray:
        """Draw one M ~ truncated GR per theta row."""
        theta = self._clip_theta(theta)
        n = theta.shape[0]
        m_out = np.empty(n)
        for i in range(n):
            mm = theta[i, 1]
            mags = np.arange(self.spec.m_min, mm + 1e-9, self.spec.m_step)
            w = self.gr_weights(theta[i:i + 1, 0], theta[i:i + 1, 1], mags[None, :])[0]
            m_out[i] = self.rng.choice(mags, p=w)
        return m_out

    def sample_y(self, theta: np.ndarray) -> np.ndarray:
        """Draw PGA given theta (integrates aleatory M and epsilon)."""
        theta = self._clip_theta(theta)
        m = self.sample_magnitude(theta)
        mu = sadigh_ln_pga_median(m, self.spec.distance_km) + theta[:, 2]
        sigma = self._effective_sigma(m, theta[:, 3])
        ln_y = self.rng.normal(mu, sigma)
        return np.exp(ln_y)

    def leaf_params(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """GR-weighted mixture mean of (ln PGA mu, sigma) given theta."""
        theta = self._clip_theta(theta)
        m_max = theta[:, 1]
        mags = self.mag_grid(m_max)
        w = self.gr_weights(theta[:, 0], m_max, mags)
        mu, sigma = self._gmm_params(theta, mags)
        mu = np.where(np.isfinite(mags), mu, 0.0)
        sigma = np.where(np.isfinite(mags), sigma, 0.0)
        mu_eff = (w * mu).sum(axis=1)
        sig_eff = np.maximum((w * sigma).sum(axis=1), 0.08)
        return mu_eff, sig_eff

    def rollout_nominal(self) -> tuple[np.ndarray, float, float]:
        """Draw theta ~ prior, y | theta; importance weight is 1."""
        theta = self.prior.sample(self.rng, n=1)
        y = float(self.sample_y(theta)[0])
        return theta[0], y, 1.0

    def rollout_with_proposal(
        self, proposal: GaussianProposal
    ) -> tuple[np.ndarray, float, float]:
        """Draw theta ~ proposal, y | theta, return IS weight p(theta)/q(theta)."""
        theta = self._clip_theta(proposal.sample(self.rng, n=1))
        y = float(self.sample_y(theta)[0])
        q = max(float(proposal.pdf(theta)[0]), 1e-300)
        p = max(float(self.prior.pdf(theta)[0]), 1e-300)
        return theta[0], y, p / q
