"""QR-SRM AIS: Moghimi & Ku (ICML 2025) static Spectral Risk Measures, for
continuous-epistemic PSHA importance sampling.

Source paper
------------
Moghimi, M. & Ku, H. (2025). Beyond CVaR: Leveraging Static Spectral Risk
Measures for Enhanced Decision-Making in Distributional Reinforcement
Learning. ICML 2025. https://arxiv.org/abs/2501.02087
https://openreview.net/forum?id=WeMpvGxXMn

That paper is distributional RL on sequential MDPs, not PSHA: they
maximize a *static* Spectral Risk Measure (SRM) of the discounted return
by alternating (their Algorithm 1)

  1. Outer: closed-form update of a concave utility ``h`` from a quantile
     representation of the current return law (their Eq. 6).
  2. Inner: quantile-regression DRL (QR-DQN-style pinball loss, Algorithm 2)
     with greedy action ``argmax E[h(s + c G)]`` on an augmented state.

SRM (Acerbi 2002) is ``∫ F^{-1}(u) φ(u) du`` for a spectrum ``φ``; CVaR_α
is the special case with all mass on one tail. Equivalently (Kusuoka 2001)
an SRM is a convex combination of CVaRs at several levels — the paper's
reason for going "beyond CVaR", including Mean-CVaR
``λ E[Z] + (1-λ) CVaR_α(Z)``.

What transfers here, and what does not
--------------------------------------
Used, faithfully:
  * Quantile representation of the outcome law (N atoms at mid-probabilities
    ``τ̂_i``), updated by IS-weighted pinball / QR (Algorithm 2's loss,
    without a sequential Bellman backup).
  * Alternating outer/inner (Algorithm 1): each batch, update the quantile
    grid (outer ``h``), then refit the IS proposal (inner ``π``) toward
    that ``h``.
  * SRM as a convex combination of CVaRs (Kusuoka). Default here is
    ``srm_mix`` — several upper-CVaR levels — because Mean-CVaR with a
    large mean weight leaves too much body mass for rare-event IS.
    Mean-CVaR and single-level CVaR remain available.

Not used (does not map):
  * Sequential MDP, discounted returns, or the ``(x, s, c)`` state
    augmentation ``s ← s + c r``, ``c ← γ c``. Our environment is a
    one-step draw ``θ → Y``; there is no later-stage risk preference to
    decompose (their Theorem 5.1 / Pflug–Pichler dual).
  * Greedy discrete action selection over a Q table. The inner step here
    is the same Gaussian CE/PMC moment-match the rest of this repo uses
    for continuous ``θ``, with relevance the variable part of ``h``
    (upper-tail ``(y − q)_+``, mixed across CVaR levels). Constants in
    the paper's ``h`` drop out of normalized CE weights.

Sign convention: the paper's CVaR/SRM is on the *lower* tail of returns
(risk of small G). Our Y is a severity we want the *upper* tail of, so
the spectrum puts mass on high quantiles (equivalently, their SRM on
``−Y``). Privilege: **never** given ``P(Y>v|θ)`` — only realized
``(θ, y, iw)``, same as JEPA-CVaR v2.
"""

from __future__ import annotations

import numpy as np

from cvar_psha.continuous_env import ContinuousEpistemicEnv, GaussianProposal
from cvar_psha.estimators import OnlineCVaRTracker
from cvar_psha.methods import MethodResult

# Default Kusuoka mix (paper Eq. 4): more weight on the experiment's CVaR
# level, leftover mass on milder tails so the proposal still covers them.
_DEFAULT_MIX_ALPHAS = (0.05, 0.10, 0.25)
_DEFAULT_MIX_MU = (0.50, 0.30, 0.20)


def _quantile_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """N quantile midpoints τ̂_i = (τ_{i-1} + τ_i)/2 as in QR-DQN / QR-SRM."""
    tau = np.linspace(0.0, 1.0, n + 1)
    tau_hat = 0.5 * (tau[:-1] + tau[1:])
    return tau, tau_hat


def srm_quantile_weights(
    n: int,
    spectrum: str,
    alpha: float,
    lambda_mean: float,
    mix_alphas: tuple[float, ...] = _DEFAULT_MIX_ALPHAS,
    mix_mu: tuple[float, ...] = _DEFAULT_MIX_MU,
) -> tuple[np.ndarray, np.ndarray]:
    """Discrete spectrum weights on the N quantile atoms, summing to 1.

    Upper-tail: more mass on high ``τ̂``. ``mean_cvar`` is the paper's
    highlighted special case; ``srm_mix`` is a Kusuoka combination of
    several upper CVaRs; ``cvar`` recovers a single upper CVaR_α.
    """
    _, tau_hat = _quantile_grid(n)
    spectrum = str(spectrum).lower()
    if spectrum == "cvar":
        w = (tau_hat >= (1.0 - alpha)).astype(float)
    elif spectrum == "mean_cvar":
        w = np.full(n, lambda_mean / n)
        tail = tau_hat >= (1.0 - alpha)
        if tail.any():
            w[tail] += (1.0 - lambda_mean) / float(tail.sum())
    elif spectrum == "srm_mix":
        w = np.zeros(n)
        for a, m in zip(mix_alphas, mix_mu):
            tail = tau_hat >= (1.0 - float(a))
            if tail.any():
                w[tail] += float(m) / float(tail.sum())
    else:
        raise ValueError(f"unknown spectrum={spectrum!r}; use cvar|mean_cvar|srm_mix")
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0:
        w = np.full(n, 1.0 / n)
    else:
        w = w / s
    return w, tau_hat


def _interp_quantile(q_atoms: np.ndarray, tau_hat: np.ndarray, level: float) -> float:
    """Quantile of Y at probability `level`, from the QR atom grid."""
    level = float(np.clip(level, float(tau_hat[0]), float(tau_hat[-1])))
    return float(np.interp(level, tau_hat, q_atoms))


def _srm_h(
    y: np.ndarray,
    q_atoms: np.ndarray,
    tau_hat: np.ndarray,
    spectrum: str,
    alpha: float,
    lambda_mean: float,
    mix_alphas: tuple[float, ...],
    mix_mu: tuple[float, ...],
) -> np.ndarray:
    """CE kernel for the inner step: variable part of paper Eq. 6, upper-tail.

    The paper's ``h(z)`` includes a quantile offset ``q`` that does not
    depend on the sample. CE/PMC uses *normalized* weights, so that
    offset is dropped. The remaining upper-CVaR piece is ``(y − q)_+``
    (analog of their ``[z − q]_-``), mixed with ``y`` itself for
    Mean-CVaR. Using ``max(y, q)`` would leave a near-constant body
    weight and starve the tail signal.
    """
    y = np.asarray(y, dtype=float)
    spectrum = str(spectrum).lower()
    if spectrum == "cvar":
        q = _interp_quantile(q_atoms, tau_hat, 1.0 - alpha)
        return np.maximum(y - q, 0.0)
    if spectrum == "mean_cvar":
        q = _interp_quantile(q_atoms, tau_hat, 1.0 - alpha)
        return lambda_mean * y + (1.0 - lambda_mean) * np.maximum(y - q, 0.0)
    h = np.zeros_like(y)
    for a, m in zip(mix_alphas, mix_mu):
        q = _interp_quantile(q_atoms, tau_hat, 1.0 - float(a))
        h = h + float(m) * np.maximum(y - q, 0.0)
    return h


def _pinball_update(
    q_atoms: np.ndarray,
    tau_hat: np.ndarray,
    y: np.ndarray,
    iw: np.ndarray,
    lr: float,
) -> np.ndarray:
    """IS-weighted quantile-regression step (QR-DQN pinball gradient)."""
    y = np.asarray(y, dtype=float)
    iw = np.asarray(iw, dtype=float)
    # (m, n): τ̂ − 1{y < q}
    ind = (y[:, None] < q_atoms[None, :]).astype(float)
    grad = (tau_hat[None, :] - ind) * iw[:, None]
    wsum = float(iw.sum())
    if wsum <= 0:
        return q_atoms
    q_new = q_atoms + lr * (grad.sum(axis=0) / wsum)
    q_new = np.sort(q_new)
    return np.maximum.accumulate(q_new)


def run_qr_srm_ais(
    env: ContinuousEpistemicEnv,
    v95: float,
    budget: int,
    batch_size: int = 300,
    smoothing: float = 0.5,
    cov_inflation: float = 1.15,
    cov_floor_scale: float = 0.05,
    defensive_eps: float = 0.1,
    n_quantiles: int = 32,
    spectrum: str = "srm_mix",
    lambda_mean: float = 0.10,
    alpha: float | None = None,
    qr_lr: float = 0.08,
    mix_alphas: tuple[float, ...] = _DEFAULT_MIX_ALPHAS,
    mix_mu: tuple[float, ...] = _DEFAULT_MIX_MU,
    eval_every: int = 200,
) -> MethodResult:
    """QR-SRM AIS: quantile grid (outer h) + CE/PMC proposal (inner π).

    ``alpha`` defaults to the experiment tail mass implied by ``v95``'s
    percentile if the caller passes it; otherwise 0.05. The reported
    estimator is still the repo's IS CVaR at ``v95`` — SRM is the
    *training* objective, not a replacement metric.
    """
    if alpha is None:
        alpha = 0.05
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    prior = env.prior
    mean = prior.mean.copy()
    cov = prior.cov.copy()
    cov_floor = cov_floor_scale * prior.cov
    proposal = GaussianProposal(mean, cov)
    mean_history = [mean.copy()]

    spec_w, tau_hat = srm_quantile_weights(
        n_quantiles, spectrum, alpha, lambda_mean, mix_alphas, mix_mu
    )
    # Outer h is defined from the current policy's return law (Algorithm 1).
    # Unset until the first batch supplies an IS-weighted empirical quantile
    # grid — a log-spaced prior sitting above typical Y would make h(y)≈const
    # and kill the inner CE signal.
    q_atoms: np.ndarray | None = None

    n_done = 0
    srm_history: list[float] = []
    while n_done < budget:
        m = min(batch_size, budget - n_done)

        use_prior_mask = env.rng.random(m) < defensive_eps
        thetas = np.empty((m, 2))
        n_prior = int(use_prior_mask.sum())
        n_prop = m - n_prior
        if n_prop > 0:
            thetas[~use_prior_mask] = proposal.sample(env.rng, n=n_prop)
        if n_prior > 0:
            thetas[use_prior_mask] = prior.sample(env.rng, n=n_prior)

        q_mix_pdf = (1.0 - defensive_eps) * proposal.pdf(thetas) + defensive_eps * prior.pdf(thetas)
        prior_pdf = prior.pdf(thetas)
        mu_leaf, sigma_leaf = env.leaf_params(thetas)
        ys = np.exp(env.rng.normal(mu_leaf, sigma_leaf))
        iw = prior_pdf / np.clip(q_mix_pdf, 1e-300, None)
        for y_i, w_i in zip(ys, iw):
            tracker.update(float(y_i), float(w_i))
        n_done += m

        # Outer: update h from the quantile representation of this batch.
        if q_atoms is None:
            order = np.argsort(ys)
            w_ord = iw[order]
            wsum_q = float(w_ord.sum())
            if wsum_q > 0:
                cdf = np.cumsum(w_ord) / wsum_q
                q_atoms = np.interp(tau_hat, cdf, ys[order])
            else:
                q_atoms = np.quantile(ys, tau_hat)
        else:
            q_atoms = _pinball_update(q_atoms, tau_hat, ys, iw, qr_lr)
        srm_hat = float(np.dot(spec_w, q_atoms))
        srm_history.append(srm_hat)

        # Inner: CE/PMC refit toward H(y) = h(y) (paper J(π, h) = E[h(G)]).
        h_y = _srm_h(
            ys, q_atoms, tau_hat, spectrum, alpha, lambda_mean, mix_alphas, mix_mu
        )
        pmc_w = (prior_pdf / np.clip(q_mix_pdf, 1e-300, None)) * h_y
        wsum = float(pmc_w.sum())
        if wsum > 0:
            w_norm = pmc_w / wsum
            mean_hat = (w_norm[:, None] * thetas).sum(axis=0)
            diff = thetas - mean_hat[None, :]
            cov_hat = (w_norm[:, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0)
            cov_hat = cov_hat * cov_inflation + cov_floor
            mean = smoothing * mean_hat + (1.0 - smoothing) * mean
            cov = smoothing * cov_hat + (1.0 - smoothing) * cov
            proposal = GaussianProposal(mean, cov)
        mean_history.append(mean.copy())

    return MethodResult(
        name="QR-SRM AIS",
        metrics=tracker.finalize(),
        final_q=None,
        extras={
            "final_mean": mean.copy(),
            "final_cov": cov.copy(),
            "proposal_mean": mean.copy(),
            "proposal_cov": cov.copy(),
            "mean_history": np.asarray(mean_history),
            "q_atoms": (q_atoms.copy() if q_atoms is not None else np.array([])),
            "srm_history": np.asarray(srm_history, dtype=float),
            "spectrum": spectrum,
            "lambda_mean": float(lambda_mean),
            "alpha": float(alpha),
            "final_srm": float(srm_history[-1]) if srm_history else float("nan"),
        },
    )
