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

from cvar_psha.continuous_env import ContinuousEpistemicEnv
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
