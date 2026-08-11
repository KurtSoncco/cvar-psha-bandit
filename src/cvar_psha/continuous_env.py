"""Continuous-epistemic-space PSHA environment (paper 2 style).

Mirrors Houng, Ceferino & Abrahamson (2025), "Fast propagation of epistemic
uncertainty in seismic hazard via adaptive importance sampling," BSSA
116(4):1709, https://doi.org/10.1785/0120250205: epistemic uncertainty in
the median GMPE and its aleatory sigma -- the two axes real logic trees
most commonly branch on for GMPE selection -- is represented as a
*continuous* Gaussian prior over parameters theta, instead of a handful of
discrete logic-tree branches (the discrete case is `tree_env.py`).

theta = (theta_mu, theta_sigma), independent epistemic priors:
    theta_mu    ~ N(0, tau_mu^2)      -- median-GMPE epistemic offset
    theta_sigma ~ N(0, tau_sigma^2)   -- aleatory-sigma epistemic scaling (log-scale)

Given theta, the aleatory ground-motion law is
    ln Y | theta ~ N(mu0 + theta_mu, sigma0 * exp(s_max * tanh(theta_sigma / tau_sigma))).

theta_sigma enters nonlinearly (through the aleatory sigma), so, unlike a
purely mean-shift epistemic model, the tail is genuinely two-dimensional:
disaggregation mass does not collapse onto a single linear combination of
the two epistemic axes.

The sigma link saturates (tanh) rather than growing as a bare exp(theta_sigma):
with an *unbounded* Gaussian theta_sigma and sigma(theta) = sigma0*exp(theta_sigma),
E[Y] is genuinely infinite (sigma^2 grows doubly-exponentially in theta_sigma,
which beats the Gaussian prior's tail decay -- confirmed numerically: the
quadrature ground truth diverged before this was caught). Saturating the link
keeps sigma(theta) in the bounded range sigma0 * exp([-s_max, s_max]) for any
theta_sigma, which guarantees every moment used here (VaR, CVaR) is finite,
while still behaving like the original multiplicative model for
|theta_sigma| << tau_sigma. This is also a more realistic model of epistemic
sigma-uncertainty in practice (a bounded set of published sigma models), not
just a numerical patch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContinuousEpistemicSpec:
    mu0: float = -1.0
    sigma0: float = 0.6
    tau_mu: float = 0.5
    tau_sigma: float = 0.35
    s_max: float = 0.5  # saturation bound on log-sigma epistemic multiplier


class GaussianProposal:
    """A 2-D Gaussian IS proposal over theta = (theta_mu, theta_sigma).

    Used both as the nominal prior (mean 0, diag(tau_mu^2, tau_sigma^2)) and
    as the trainable/adaptive proposal for G-PMC AIS and the JEPA-CVaR
    policy -- anything that can report (mean, cov) can be scored on the
    same quadrature grid as the ground truth.
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = np.asarray(mean, dtype=float).reshape(2)
        self.cov = np.asarray(cov, dtype=float).reshape(2, 2)
        self._chol = np.linalg.cholesky(self.cov + 1e-12 * np.eye(2))
        self._prec = np.linalg.inv(self.cov + 1e-12 * np.eye(2))
        self._logdet = float(np.linalg.slogdet(self.cov + 1e-12 * np.eye(2))[1])

    @classmethod
    def from_spec(cls, spec: ContinuousEpistemicSpec) -> "GaussianProposal":
        return cls(mean=np.zeros(2), cov=np.diag([spec.tau_mu**2, spec.tau_sigma**2]))

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        z = rng.standard_normal((n, 2))
        return self.mean[None, :] + z @ self._chol.T

    def logpdf(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        d = theta - self.mean[None, :]
        quad = np.einsum("ni,ij,nj->n", d, self._prec, d)
        return -0.5 * (2 * np.log(2 * np.pi) + self._logdet + quad)

    def pdf(self, theta: np.ndarray) -> np.ndarray:
        return np.exp(self.logpdf(theta))


class ContinuousEpistemicEnv:
    """Single-site PGA under continuous epistemic uncertainty in (mu, sigma)."""

    def __init__(
        self,
        spec: ContinuousEpistemicSpec | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.spec = spec or ContinuousEpistemicSpec()
        self.rng = rng or np.random.default_rng()
        self.prior = GaussianProposal.from_spec(self.spec)

    def leaf_params(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(mu, sigma) of ln Y given theta (vectorized, theta shape (n,2))."""
        theta = np.atleast_2d(theta)
        s = self.spec
        mu = s.mu0 + theta[:, 0]
        sigma = s.sigma0 * np.exp(s.s_max * np.tanh(theta[:, 1] / s.tau_sigma))
        return mu, sigma

    def sample_y(self, theta: np.ndarray) -> np.ndarray:
        mu, sigma = self.leaf_params(theta)
        ln_y = self.rng.normal(mu, sigma)
        return np.exp(ln_y)

    def rollout_with_proposal(self, proposal: GaussianProposal) -> tuple[np.ndarray, float, float]:
        """Draw one theta ~ proposal, one y | theta ~ nominal aleatory law;
        return (theta, y, importance_weight = prior_pdf(theta)/proposal_pdf(theta)).

        Matches the papers' IS scope: only the epistemic marginal is
        importance-sampled; the aleatory conditional is drawn from its true
        (nominal) law given theta, exactly as in hazard disaggregation.
        """
        theta = proposal.sample(self.rng, n=1)
        y = float(self.sample_y(theta)[0])
        q = max(float(proposal.pdf(theta)[0]), 1e-300)
        p = float(self.prior.pdf(theta)[0])
        iw = p / q
        return theta[0], y, iw
