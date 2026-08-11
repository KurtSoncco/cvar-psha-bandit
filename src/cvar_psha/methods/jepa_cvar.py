"""Hierarchical JEPA-CVaR: a learned IS proposal over continuous epistemic
parameters, driven only by a scalar CVaR-tail reward.

This is the agent side of the "meet or surpass" comparison against
G-PMC AIS (`gpmc_ais.py`), which is *given* the closed-form conditional
hazard P(Y>v|theta). This method never sees that formula.

Hierarchical structure (mirrors `methods/hierarchical.py`'s Manager/Worker
split, adapted to continuous theta):
  - Manager: a slowly-updated coarse Gaussian anchor mu_m (theta_dim,).
  - Worker: a fast-updated linear readout Ww, bw of the JEPA context
    embedding z_c = jepa.encode_context(mu_m) -- i.e. the fine-grained
    correction is conditioned on a *learned representation* of the
    Manager's current coarse operating point, not on the raw scalar.
  - Sampling distribution: theta ~ N(mu_m + Ww @ z_c + bw, diag(std^2)),
    with std annealed by a fixed schedule (as in the spatial Hierarchical
    method) rather than learned, for stability.

Both the Manager mean and the Worker readout are trained by REINFORCE
(Gaussian score function) on a CVaR-CPO-style advantage: the scalar
tail-exceedance reward r = (p(theta)/q(theta)) * y * 1{y>v95}, shaped by a
dual variable that penalizes drift of the running CVaR estimate from a
target (same mechanism as `methods/cvar_cpo.py`, adapted to continuous
actions). The JEPA (`jepa.py`) is trained online, in parallel, purely from
realized rollout outcomes -- it never receives P(Y>v|theta) either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cvar_psha.continuous_env import ContinuousEpistemicEnv, GaussianProposal
from cvar_psha.estimators import OnlineCVaRTracker, importance_weight, tail_reward
from cvar_psha.jepa import LightweightJEPA, build_target_features
from cvar_psha.methods import MethodResult


@dataclass
class _RolloutBuffer:
    thetas: list = field(default_factory=list)
    feats: list = field(default_factory=list)

    def add(self, theta: np.ndarray, feat: np.ndarray) -> None:
        self.thetas.append(theta)
        self.feats.append(feat)

    def flush(self) -> tuple[np.ndarray, np.ndarray]:
        thetas = np.asarray(self.thetas)
        feats = np.asarray(self.feats)
        self.thetas.clear()
        self.feats.clear()
        return thetas, feats


def run_jepa_cvar(
    env: ContinuousEpistemicEnv,
    v95: float,
    budget: int,
    manager_lr: float = 0.02,
    worker_lr: float = 0.08,
    baseline_alpha: float = 0.1,
    std_start: float = 0.9,
    std_end: float = 0.35,
    dual_lr: float = 0.05,
    kl_coef_unused: float = 0.0,  # reserved; continuous trust-region kept implicit via std floor
    cvar_tol: float = 0.15,
    target_ema: float = 0.05,
    jepa_batch: int = 16,
    jepa_train_every: int = 16,
    embed_dim: int = 4,
    true_cvar: float | None = None,
    eval_every: int = 200,
    seed: int = 0,
    lr_decay_horizon_frac: float = 0.5,
    tail_avg_frac: float = 0.3,
) -> MethodResult:
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    theta_dim = 2
    jepa = LightweightJEPA(theta_dim=theta_dim, feat_dim=3, embed_dim=embed_dim, seed=seed, lr=0.08)
    buf = _RolloutBuffer()

    mu_m = np.zeros(theta_dim)
    Ww = np.zeros((theta_dim, embed_dim))
    bw = np.zeros(theta_dim)

    baseline = 0.0
    r_max = 1e-8
    lam = 0.0
    target = true_cvar if true_cvar is not None else 0.0
    has_target = true_cvar is not None

    mean_history = []

    lr_decay_horizon = max(1.0, lr_decay_horizon_frac * budget)

    for t in range(budget):
        frac = t / max(budget - 1, 1)
        std = std_start + (std_end - std_start) * frac
        cov_diag = np.full(theta_dim, std * std)
        # Robbins-Monro-style step-size decay: raw REINFORCE with a constant
        # step size does not converge (confirmed empirically -- the mean
        # reaches the right neighborhood quickly, then drifts indefinitely
        # under gradient noise from the heavy-tailed reward). Decaying the
        # step size lets the iterate settle instead of random-walking.
        lr_scale = 1.0 / (1.0 + t / lr_decay_horizon)

        z_c = jepa.encode_context(mu_m)[0]  # (embed_dim,)
        mean_total = mu_m + Ww @ z_c + bw
        mean_history.append(mean_total.copy())

        eps = env.rng.standard_normal(theta_dim)
        theta = mean_total + std * eps  # ~ N(mean_total, diag(std^2))

        y = float(env.sample_y(theta[None, :])[0])
        prior_pdf = float(env.prior.pdf(theta[None, :])[0])
        # q(theta) = N(theta; mean_total, diag(std^2)) -- closed form, exact
        # (no marginalization needed: theta is a single Gaussian draw).
        log_q = -0.5 * np.sum(((theta - mean_total) ** 2) / cov_diag) - 0.5 * np.sum(
            np.log(2 * np.pi * cov_diag)
        )
        q_pdf = max(float(np.exp(log_q)), 1e-300)
        iw = importance_weight(prior_pdf, q_pdf)
        tracker.update(y, iw)

        r = tail_reward(y, iw, v95)
        r_max = max(r_max, abs(r), 1e-8)
        r_scaled = r / r_max

        cvar_hat = tracker.current_cvar()
        if np.isfinite(cvar_hat):
            if not has_target:
                target = cvar_hat if target == 0.0 else (1 - target_ema) * target + target_ema * cvar_hat
            violation = abs(cvar_hat - target) - cvar_tol * max(abs(target), 1e-6)
            lam = max(0.0, lam + dual_lr * violation)
        soft_penalty = abs(cvar_hat - target) / max(abs(target), 1e-6) if np.isfinite(cvar_hat) and target != 0.0 else 0.0

        advantage = (r_scaled - lam * soft_penalty) - baseline
        baseline = (1 - baseline_alpha) * baseline + baseline_alpha * r_scaled

        # Gaussian score function w.r.t. mean_total.
        score = (theta - mean_total) / cov_diag  # (theta_dim,)
        mu_m = mu_m + lr_scale * manager_lr * advantage * score
        Ww = Ww + lr_scale * worker_lr * advantage * np.outer(score, z_c)
        bw = bw + lr_scale * worker_lr * advantage * score

        # Online JEPA training from purely empirical rollout data.
        feat = build_target_features(y, r_scaled, iw)
        buf.add(theta, feat)
        if (t + 1) % jepa_train_every == 0 and len(buf.thetas) >= min(jepa_batch, jepa_train_every):
            th_b, ft_b = buf.flush()
            jepa.train_step(th_b, ft_b)

    mean_history = np.asarray(mean_history)
    n_tail = max(1, int(tail_avg_frac * budget))
    tail_avg_mean = mean_history[-n_tail:].mean(axis=0)

    return MethodResult(
        name="Hierarchical JEPA-CVaR",
        metrics=tracker.finalize(),
        final_q=None,
        extras={
            "mu_m": mu_m.copy(),
            "Ww": Ww.copy(),
            "bw": bw.copy(),
            "final_std": std,
            "jepa": jepa,
            "mean_history": mean_history,
            "tail_avg_mean": tail_avg_mean,
            "final_lambda": lam,
            "target": target,
        },
    )


# ============================================================================
# v2: Cross-Entropy / Population-Monte-Carlo empirical-weight refit.
#
# v1 (above) uses single-sample score-function REINFORCE with an ad-hoc
# dual-Lagrangian CVaR-drift penalty -- a heuristic, and one that needed a
# real stability patch (LR annealing) to stop it random-walking after
# converging. v2 replaces that update rule entirely.
#
# Reframing: our actual objective is not "safe RL over an agent's own
# returns" (that's what Tamar et al. 2015 / Chow & Ghavamzadeh's CVaR
# policy-gradient / CPO papers solve, and what v1's naming/citation
# incorrectly gestured at). It is *adaptive importance sampling to minimize
# the variance of a CVaR estimator of a fixed external Y under a fixed
# nominal prior p* -- the classical Cross-Entropy / Population Monte Carlo
# literature (de Boer, Kroese, Mannor & Rubinstein 2005; Cappe, Guillin,
# Marin & Robert 2004), which `gpmc_ais.py` already implements correctly.
#
# The CE method fits q_phi by minimizing KL(q*, q_phi) where
# q*(x) ~ p(x) H(x). For exponential-family q_phi this is a weighted
# moment-matching update with weights w(x) = p(x) H(x) / q_ref(x).
# G-PMC AIS uses the closed-form H(theta) = P(Y>v|theta). The *only* thing
# that requires closed-form access is the choice of H -- the refit
# machinery doesn't. Substituting the single-rollout empirical outcome
# H(theta, y) = y * 1{y>v} is a textbook-legitimate CE variant (noisier,
# unbiased, zero analytic access needed) -- and p(theta) H(theta, y) / q(theta)
# is exactly `estimators.tail_reward`, already used everywhere in this repo
# as the reward signal. So Stage A below is "the same algorithm as
# G-PMC AIS, with an empirical H instead of a closed-form one" -- not new
# machinery.
#
# Stage B adds a joint (theta, y) tilt: the Worker also proposes a mean
# shift to ln Y | theta, refit by weighted least squares against the same
# empirical weights. Stage C adds a second, coarser JEPA whose embedding
# conditions the fine JEPA, so the representation is genuinely two-level.
# All three stages are controlled by flags below so they can be validated
# incrementally (see scripts run during development, not committed).
# ============================================================================


def _weighted_moment_match(
    x: np.ndarray, w: np.ndarray, prior_cov: np.ndarray, cov_inflation: float, cov_floor_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """CE/PMC weighted mean + covariance refit (same formula as gpmc_ais.py)."""
    wsum = w.sum()
    if wsum <= 0:
        return x.mean(axis=0), prior_cov.copy()
    w_norm = w / wsum
    mean_hat = (w_norm[:, None] * x).sum(axis=0)
    diff = x - mean_hat[None, :]
    cov_hat = (w_norm[:, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0)
    cov_hat = cov_hat * cov_inflation + cov_floor_scale * prior_cov
    return mean_hat, cov_hat


def _weighted_linear_fit(
    z: np.ndarray, target: np.ndarray, w: np.ndarray, ridge: float = 1e-4
) -> tuple[np.ndarray, float]:
    """Weighted ridge least squares: target ~ z @ a + b, weights w.
    Returns (a, b). Verified against synthetic recovery (max err ~3e-4)."""
    n, d = z.shape
    X = np.concatenate([z, np.ones((n, 1))], axis=1)
    XtWX = X.T @ (w[:, None] * X) + ridge * np.eye(d + 1)
    XtWy = X.T @ (w * target)
    beta = np.linalg.solve(XtWX, XtWy)
    return beta[:-1], float(beta[-1])


def _exp_tilt_ratio(z: np.ndarray, mu: np.ndarray, delta: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """p(z)/q(z) for z ~ N(mu, sigma) vs q = N(mu+delta, sigma) -- exact
    closed-form Gaussian-mean-shift likelihood ratio (Girsanov / exponential
    tilting), avoiding division of two possibly-tiny pdf values. Verified
    against direct scipy.stats.norm.pdf ratio to 1e-16."""
    u = z - mu
    return np.exp(delta * (delta / 2.0 - u) / (sigma**2))


def run_jepa_cvar_v2(
    env: ContinuousEpistemicEnv,
    v95: float,
    budget: int,
    batch_size: int = 300,
    smoothing: float = 0.5,
    cov_inflation: float = 1.15,
    cov_floor_scale: float = 0.05,
    defensive_eps: float = 0.1,
    embed_dim: int = 4,
    jepa_lr: float = 0.08,
    jepa_train_every: int = 16,
    enable_joint_tilt: bool = True,
    enable_hierarchical: bool = True,
    coarse_embed_dim: int = 3,
    eval_every: int = 200,
    seed: int = 0,
) -> MethodResult:
    """Stage A (+B +C) JEPA-CVaR: CE/PMC empirical-weight refit, optional
    joint (theta,y) tilt, optional two-level hierarchical JEPA. Never given
    P(Y>v|theta) -- the empirical weight is exactly `tail_reward`."""
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    theta_dim = 2
    prior = env.prior
    mean_theta = prior.mean.copy()
    cov_theta = prior.cov.copy()
    proposal = GaussianProposal(mean_theta, cov_theta)

    fine_input_dim = theta_dim + (coarse_embed_dim if enable_hierarchical else 0)
    jepa_fine = LightweightJEPA(
        theta_dim=fine_input_dim, feat_dim=3, embed_dim=embed_dim, seed=seed, lr=jepa_lr
    )
    jepa_coarse = None
    coarse_buf_ctx: list = []
    coarse_buf_feat: list = []
    if enable_hierarchical:
        jepa_coarse = LightweightJEPA(
            theta_dim=theta_dim, feat_dim=3, embed_dim=coarse_embed_dim, seed=seed + 1, lr=jepa_lr
        )

    Wy = np.zeros(embed_dim)  # readout: delta(theta) = z_fine @ Wy + by, a scalar mean-shift on ln Y
    by = 0.0

    fine_buf_ctx: list = []
    fine_buf_feat: list = []

    n_done = 0
    batch_r_history: list = []  # for coarse JEPA's batch-aggregated target

    while n_done < budget:
        m = min(batch_size, budget - n_done)

        # Coarse embedding for this batch: describes the current operating
        # region (the proposal mean), shared across the batch -- refit once
        # per batch as the coarse "Manager" signal.
        if enable_hierarchical:
            z_coarse_batch = jepa_coarse.encode_context(mean_theta)[0]  # (coarse_embed_dim,)
        else:
            z_coarse_batch = None

        # --- sample a population (defensive mixture, as in gpmc_ais.py) ---
        use_prior_mask = env.rng.random(m) < defensive_eps
        thetas = np.empty((m, theta_dim))
        n_prior = int(use_prior_mask.sum())
        n_prop = m - n_prior
        if n_prop > 0:
            thetas[~use_prior_mask] = proposal.sample(env.rng, n=n_prop)
        if n_prior > 0:
            thetas[use_prior_mask] = prior.sample(env.rng, n=n_prior)

        q_theta_mix_pdf = (1 - defensive_eps) * proposal.pdf(thetas) + defensive_eps * prior.pdf(thetas)
        p_theta_pdf = prior.pdf(thetas)

        mu_leaf, sigma_leaf = env.leaf_params(thetas)

        if enable_hierarchical:
            fine_input = np.concatenate([thetas, np.tile(z_coarse_batch, (m, 1))], axis=1)
        else:
            fine_input = thetas
        z_fine = jepa_fine.encode_context(fine_input)  # (m, embed_dim)

        delta_y = (z_fine @ Wy + by) if enable_joint_tilt else np.zeros(m)

        ln_y = env.rng.normal(mu_leaf + delta_y, sigma_leaf)
        y = np.exp(ln_y)

        iw_theta = p_theta_pdf / np.clip(q_theta_mix_pdf, 1e-300, None)
        iw_y = _exp_tilt_ratio(ln_y, mu_leaf, delta_y, sigma_leaf) if enable_joint_tilt else np.ones(m)
        iw = iw_theta * iw_y

        r = np.where(y > v95, iw * y, 0.0)  # == tail_reward(y_i, iw_i, v95), vectorized
        for y_i, w_i in zip(y, iw):
            tracker.update(float(y_i), float(w_i))
        n_done += m

        # --- CE/PMC refit using the empirical weight r (no closed form) ---
        mean_theta, cov_theta = _weighted_moment_match(thetas, r, prior.cov, cov_inflation, cov_floor_scale)
        proposal = GaussianProposal(mean_theta, cov_theta)

        if enable_joint_tilt and r.sum() > 0:
            residual = ln_y - mu_leaf  # target for the tilt readout
            Wy, by = _weighted_linear_fit(z_fine, residual, r)

        # --- online JEPA training, purely from empirical rollout data ---
        r_max = max(1e-8, float(np.max(np.abs(r))))
        r_scaled = r / r_max
        for i in range(m):
            feat = build_target_features(float(y[i]), float(r_scaled[i]), float(iw[i]))
            fine_buf_ctx.append(fine_input[i])
            fine_buf_feat.append(feat)
        if len(fine_buf_ctx) >= jepa_train_every:
            ctx_b = np.asarray(fine_buf_ctx)
            feat_b = np.asarray(fine_buf_feat)
            jepa_fine.train_step(ctx_b, feat_b)
            fine_buf_ctx.clear()
            fine_buf_feat.clear()

        if enable_hierarchical:
            coarse_feat = np.array(
                [np.tanh(r_scaled.mean()), np.tanh((y > v95).mean() * 3.0), np.tanh(0.3 * np.log1p(y.mean()))],
                dtype=float,
            )
            coarse_buf_ctx.append(mean_theta.copy())
            coarse_buf_feat.append(coarse_feat)
            if len(coarse_buf_ctx) >= 4:  # small batches; coarse signal changes slowly
                jepa_coarse.train_step(np.asarray(coarse_buf_ctx), np.asarray(coarse_buf_feat))
                coarse_buf_ctx.clear()
                coarse_buf_feat.clear()

    return MethodResult(
        name="Hierarchical JEPA-CVaR v2",
        metrics=tracker.finalize(),
        final_q=None,
        extras={
            "final_mean": mean_theta.copy(),
            "final_cov": cov_theta.copy(),
            "Wy": Wy.copy(),
            "by": by,
            "jepa_fine": jepa_fine,
            "jepa_coarse": jepa_coarse,
            "enable_joint_tilt": enable_joint_tilt,
            "enable_hierarchical": enable_hierarchical,
        },
    )
