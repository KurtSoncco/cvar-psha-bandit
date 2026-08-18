"""CVaR-BF AIS: Wang et al. (2026) risk-adaptive CVaR barrier filter, for
continuous-epistemic PSHA importance sampling.

Source paper
------------
Wang, Kim, Hoxha, Fainekos & Panagou (2026). Reinforcement Learning for
Risk Adaptation via Differentiable CVaR Barrier Functions.
https://arxiv.org/abs/2605.21257

That paper is crowd-navigation, not PSHA: an RL actor outputs a nominal
control ``u_nom``, a risk level ``β``, and a safety margin ``ΔR``; a
differentiable QP then projects ``u_nom`` onto the set of inputs that
satisfy a closed-form Gaussian CVaR barrier (their Eq. 21–22), so the
applied action is probabilistically safe under GMM obstacle uncertainty.

What transfers here, and what does not
--------------------------------------
Used, faithfully:
  * Closed-form Gaussian CVaR (paper Eq. 21):
        CVaR_β(H) = μ − φ(Φ^{-1}(β))/β · σ    (lower tail)
        CVaR^upper_β(H) = μ + φ(Φ^{-1}(1−β))/β · σ
    ``ln Y | θ`` is exactly Gaussian, so this formula is native to this
    environment, not an extra assumption.
  * The architecture: an RL actor proposes a *nominal* proposal mean
    ``μ_nom`` plus learned ``(β, Δ)``; a QP-style projection produces the
    *applied* sampling mean ``μ*`` that actually draws ``θ``.
  * Joint learning of risk level and margin from the same scalar tail
    reward every other method in this repo sees.

Not used (does not map):
  * Robot dynamics, Control Barrier Function Lie derivatives, crowd GMM
    obstacle modes, or a multi-obstacle union bound. Given ``θ``, the
    aleatory law is unimodal Gaussian, so the paper's mode-wise GMM lemma
    (Lemma 1) is vacuous here — one mode, one constraint.
  * Their ``β`` is a *safety-violation* probability with ``β_max = 0.5``.
    Ours is the complementary chance level of the event ``Y > v`` at the
    proposal mean: the QP enforces ``P(Y > v | θ = μ*) ≥ 1 − β``, and
    ``β`` is initialized at the experiment's tail probability so that
    ``1 − β`` starts at ``1 − percentile`` (0.05 at the 95th percentile).
    Small ``β`` here means a more demanding tail-focusing filter (the
    analog of their conservative setting), not a smaller collision
    probability on a robot.

The QP constraint is the paper's probabilistic CBF (their Eq. 9), which
for ``ln Y | θ ~ N(μ(θ), σ(θ)²)`` is the halfspace

    μ(θ)_0  ≥  ln(v) + σ(θ) · Φ^{-1}(1 − β) + Δ

linear in the median-GMPE coordinate once ``σ`` is frozen at the current
``θ_σ`` — the same "σ independent of u" structure that made their QP a
linear inequality. Projection onto that halfspace is closed form (clip
``θ_μ``); no quadratic-program solver is required in 2-D.

Privilege, honestly: the *filter* uses the closed-form Gaussian / lognormal
leaf (``env.leaf_params`` + ``Φ``), same family of access G-PMC AIS has.
The *actor* is still trained only on ``estimators.tail_reward``. This is
therefore a privileged-filter + scalar-reward-actor method, not a
JEPA-style "never sees P(Y>v|θ)" agent and not a G-PMC-style "refit q
directly to the closed-form target density" method.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from cvar_psha.continuous_env import ContinuousEpistemicEnv
from cvar_psha.estimators import OnlineCVaRTracker, importance_weight, tail_reward
from cvar_psha.methods import MethodResult


def gaussian_cvar_lower(mu: np.ndarray | float, sigma: np.ndarray | float, beta: float) -> np.ndarray:
    """Paper Eq. 21: CVaR_β of N(μ, σ²) on the *lower* tail.

    CVaR_β(H) = E[H | H ≤ VaR_β(H)] = μ − φ(Φ^{-1}(β))/β · σ.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    beta = float(np.clip(beta, 1e-12, 1.0 - 1e-12))
    z = stats.norm.ppf(beta)
    return mu - (stats.norm.pdf(z) / beta) * sigma


def gaussian_cvar_upper(mu: np.ndarray | float, sigma: np.ndarray | float, beta: float) -> np.ndarray:
    """Upper-tail counterpart of paper Eq. 21: E[H | H ≥ VaR_{1−β}(H)]."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    beta = float(np.clip(beta, 1e-12, 1.0 - 1e-12))
    z = stats.norm.ppf(1.0 - beta)
    return mu + (stats.norm.pdf(z) / beta) * sigma


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0))))


def _project_chance_qp(
    mu_nom: np.ndarray,
    sigma_leaf: float,
    spec_mu0: float,
    v95: float,
    beta: float,
    delta: float,
) -> tuple[np.ndarray, bool, float]:
    """Project μ_nom onto P(Y > v | θ = μ) ≥ 1 − β, plus margin Δ.

    For ln Y ~ N(μ0 + θ_μ, σ²) this is the halfspace
        θ_μ ≥ ln(v) − μ0 + σ Φ^{-1}(1 − β) + Δ
    (σ frozen at the current θ_σ, matching the paper's "σ independent of
    u" linear CVaR-BF inequality). Returns (μ*, active, θ_μ_floor).
    """
    beta = float(np.clip(beta, 1e-12, 1.0 - 1e-12))
    z = float(stats.norm.ppf(1.0 - beta))
    floor = float(np.log(v95) - spec_mu0 + sigma_leaf * z + delta)
    mu_star = mu_nom.copy()
    active = mu_star[0] < floor
    if active:
        mu_star[0] = floor
    return mu_star, active, floor


def run_cvar_bf_ais(
    env: ContinuousEpistemicEnv,
    v95: float,
    budget: int,
    learning_rate: float = 0.05,
    beta_lr: float = 0.03,
    delta_lr: float = 0.03,
    baseline_alpha: float = 0.1,
    std_start: float = 0.9,
    std_end: float = 0.35,
    beta_init: float | None = None,
    beta_min: float = 0.50,
    beta_max: float = 0.99,
    delta_max: float = 0.5,
    defensive_eps: float = 0.1,
    eval_every: int = 200,
    lr_decay_horizon_frac: float = 0.5,
    tail_avg_frac: float = 0.3,
) -> MethodResult:
    """RL actor + CVaR-BF chance-constraint projection over continuous θ.

    Actor outputs (μ_nom, β, Δ). Each episode the QP filter produces μ*,
    θ is drawn from a defensive mixture of N(μ*, std² I) and the prior,
    Y is drawn from the true aleatory law, and (μ_nom, β, Δ) are updated
    by REINFORCE on the scalar tail reward. Gradients through an *active*
    projection also flow into β and Δ (the filter is the applied policy).
    """
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    theta_dim = 2
    spec = env.spec
    prior = env.prior

    mu_nom = np.zeros(theta_dim)
    # β ∈ [beta_min, beta_max] via a sigmoid; default init so 1−β equals
    # the experiment tail mass P(Y > v95) ≈ 0.05 when percentile=0.95.
    if beta_init is None:
        beta_init = 0.95
    beta_init = float(np.clip(beta_init, beta_min + 1e-6, beta_max - 1e-6))
    span_b = beta_max - beta_min
    logit_beta = float(np.log((beta_init - beta_min) / (beta_max - beta_init)))
    logit_delta = -2.0  # start near Δ ≈ 0 (filter margin off)

    baseline = 0.0
    r_max = 1e-8
    mean_history: list[np.ndarray] = []
    beta_history: list[float] = []
    n_active = 0
    lr_decay_horizon = max(1.0, lr_decay_horizon_frac * budget)

    for t in range(budget):
        frac = t / max(budget - 1, 1)
        std = std_start + (std_end - std_start) * frac
        cov_diag = np.full(theta_dim, std * std)
        lr_scale = 1.0 / (1.0 + t / lr_decay_horizon)

        beta = beta_min + span_b * _sigmoid(logit_beta)
        delta = delta_max * _sigmoid(logit_delta)

        _, sigma_at_nom = env.leaf_params(mu_nom[None, :])
        mu_star, active, floor = _project_chance_qp(
            mu_nom,
            float(sigma_at_nom[0]),
            spec.mu0,
            v95,
            beta,
            delta,
        )
        if active:
            n_active += 1
        mean_history.append(mu_star.copy())
        beta_history.append(beta)

        use_prior = env.rng.random() < defensive_eps
        if use_prior:
            theta = prior.sample(env.rng, n=1)[0]
        else:
            theta = mu_star + std * env.rng.standard_normal(theta_dim)

        log_q_prop = -0.5 * np.sum(((theta - mu_star) ** 2) / cov_diag) - 0.5 * np.sum(
            np.log(2.0 * np.pi * cov_diag)
        )
        q_prop = max(float(np.exp(log_q_prop)), 1e-300)
        prior_pdf = float(prior.pdf(theta[None, :])[0])
        q_pdf = (1.0 - defensive_eps) * q_prop + defensive_eps * prior_pdf
        y = float(env.sample_y(theta[None, :])[0])
        iw = importance_weight(prior_pdf, max(q_pdf, 1e-300))
        tracker.update(y, iw)

        r = tail_reward(y, iw, v95)
        r_max = max(r_max, abs(r), 1e-8)
        r_scaled = r / r_max
        advantage = r_scaled - baseline
        baseline = (1.0 - baseline_alpha) * baseline + baseline_alpha * r_scaled

        # Mixture score ∇_μ log q_mix = [(1-ε) q_prop / q_mix] ∇_μ log q_prop.
        gate = (1.0 - defensive_eps) * q_prop / max(q_pdf, 1e-300)
        score = gate * (theta - mu_star) / cov_diag
        mu_nom = mu_nom + lr_scale * learning_rate * advantage * score

        # Implicit filter Jacobian: when the halfspace is active,
        # μ*_0 = floor(β, Δ, σ(μ_nom_1)), so β and Δ get credit.
        if active:
            z = float(stats.norm.ppf(np.clip(1.0 - beta, 1e-12, 1.0 - 1e-12)))
            phi_z = max(float(stats.norm.pdf(z)), 1e-12)
            d_floor_dbeta = -float(sigma_at_nom[0]) / phi_z
            d_floor_ddelta = 1.0
            sigmoid_b = _sigmoid(logit_beta)
            sigmoid_d = _sigmoid(logit_delta)
            d_beta_dlogit = span_b * sigmoid_b * (1.0 - sigmoid_b)
            d_delta_dlogit = delta_max * sigmoid_d * (1.0 - sigmoid_d)
            logit_beta = (
                logit_beta
                + lr_scale * beta_lr * advantage * score[0] * d_floor_dbeta * d_beta_dlogit
            )
            logit_delta = (
                logit_delta
                + lr_scale * delta_lr * advantage * score[0] * d_floor_ddelta * d_delta_dlogit
            )

        # σ depends on θ_σ; a small score-driven nudge of μ_nom[1] already
        # covers that via the Gaussian score above. Clip logits for stability.
        logit_beta = float(np.clip(logit_beta, -8.0, 8.0))
        logit_delta = float(np.clip(logit_delta, -8.0, 8.0))

    mean_history_arr = np.asarray(mean_history)
    n_tail = max(1, int(tail_avg_frac * budget))
    tail_avg_mean = mean_history_arr[-n_tail:].mean(axis=0)
    final_std = std_end
    final_cov = np.diag([final_std * final_std, final_std * final_std])
    final_beta = beta_min + span_b * _sigmoid(logit_beta)
    final_delta = delta_max * _sigmoid(logit_delta)
    mu_leaf_f, sig_leaf_f = env.leaf_params(tail_avg_mean[None, :])
    tail_mass = max(1.0 - (beta_init if beta_init is not None else 0.95), 1e-6)

    return MethodResult(
        name="CVaR-BF AIS",
        metrics=tracker.finalize(),
        final_q=None,
        extras={
            "final_mean": tail_avg_mean.copy(),
            "final_cov": final_cov,
            "proposal_mean": tail_avg_mean.copy(),
            "proposal_cov": final_cov,
            "tail_avg_mean": tail_avg_mean.copy(),
            "final_std": final_std,
            "final_beta": final_beta,
            "final_delta": final_delta,
            "frac_qp_active": n_active / max(budget, 1),
            "mean_history": mean_history_arr,
            "beta_history": np.asarray(beta_history, dtype=float),
            "closed_form_cvar_lnY": float(
                gaussian_cvar_upper(mu_leaf_f[0], sig_leaf_f[0], tail_mass)
            ),
            "closed_form_cvar_lower_lnY": float(
                gaussian_cvar_lower(mu_leaf_f[0], sig_leaf_f[0], final_beta)
            ),
        },
    )
