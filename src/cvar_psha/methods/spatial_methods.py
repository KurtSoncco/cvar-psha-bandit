"""Spatial multi-site portfolio IS methods."""

from __future__ import annotations

import numpy as np

from cvar_psha.core.categorical import FlatPathProblem
from cvar_psha.core.categorical import run_cvar_cpo as _run_cvar_cpo
from cvar_psha.core.categorical import run_mc as _run_mc
from cvar_psha.core.categorical import run_oracle as _run_oracle
from cvar_psha.core.categorical import run_reinforce as _run_reinforce
from cvar_psha.core.estimators import OnlineCVaRTracker, path_tail_reward
from cvar_psha.core.metrics import kl_divergence, tv_distance
from cvar_psha.core.policy import TabularTreePolicy
from cvar_psha.core.result import MethodResult
from cvar_psha.spatial_env import MANAGER_DEPTHS, SpatialPortfolioEnv


def run_spatial_mc(
    env: SpatialPortfolioEnv,
    v95: float,
    budget: int,
    eval_every: int = 500,
) -> MethodResult:
    return _run_mc(FlatPathProblem(env), v95, budget, eval_every=eval_every)


def run_spatial_qstar(
    env: SpatialPortfolioEnv,
    v95: float,
    budget: int,
    q_star: np.ndarray,
    eval_every: int = 500,
) -> MethodResult:
    return _run_oracle(
        FlatPathProblem(env), v95, budget, q_star, name="q* oracle", eval_every=eval_every
    )


def run_spatial_flat_reinforce(
    env: SpatialPortfolioEnv,
    v95: float,
    budget: int,
    learning_rate: float = 0.03,
    baseline_alpha: float = 0.1,
    eval_every: int = 500,
) -> MethodResult:
    return _run_reinforce(
        FlatPathProblem(env),
        v95,
        budget,
        learning_rate=learning_rate,
        baseline_alpha=baseline_alpha,
        eval_every=eval_every,
        q_init="prior",
        name="Flat REINFORCE",
    )


def run_spatial_cvar_cpo(
    env: SpatialPortfolioEnv,
    v95: float,
    budget: int,
    learning_rate: float = 0.03,
    dual_lr: float = 0.05,
    kl_coef: float = 0.15,
    prior_kl_coef: float = 0.25,
    cvar_tol: float = 0.15,
    target_ema: float = 0.05,
    eval_every: int = 500,
    true_cvar: float | None = None,
) -> MethodResult:
    return _run_cvar_cpo(
        FlatPathProblem(env),
        v95,
        budget,
        learning_rate=learning_rate,
        dual_lr=dual_lr,
        kl_coef=kl_coef,
        prior_kl_coef=prior_kl_coef,
        cvar_tol=cvar_tol,
        target_ema=target_ema,
        eval_every=eval_every,
        true_cvar=true_cvar,
    )


class ManagerWorkerPolicy:
    """Hierarchical policies: Manager (geom, GMM), Worker (mag, rupture | state)."""

    def __init__(self, env: SpatialPortfolioEnv):
        self.env = env
        self._table = TabularTreePolicy(
            n_actions=env.n_actions,
            prior_fn=env.prior,
            logit_floor=1e-12,
        )

    @property
    def logits(self):
        return self._table.logits

    def probs(self, state: tuple[int, ...], temperature: float = 1.0) -> np.ndarray:
        return self._table.probs(state, temperature=temperature)

    def sample(
        self,
        state: tuple[int, ...],
        rng: np.random.Generator,
        temperature: float = 1.0,
    ) -> tuple[int, float, np.ndarray]:
        return self._table.sample(state, rng, temperature=temperature)

    def update_step(
        self,
        state: tuple[int, ...],
        action: int,
        advantage: float,
        learning_rate: float,
        entropy_coef: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        self._table.update_step(
            state,
            action,
            advantage,
            learning_rate,
            entropy_coef=entropy_coef,
            temperature=temperature,
        )

    def path_distribution(self, temperature: float = 1.0) -> np.ndarray:
        return self._table.path_distribution(
            self.env.paths, temperature=temperature, fallback=self.env.path_priors
        )


def _gae_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 1.0,
    lam: float = 0.9,
) -> np.ndarray:
    """GAE(lambda) for a short finite episode (bootstrap value after last = 0)."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=float)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages


def run_spatial_hierarchical(
    env: SpatialPortfolioEnv,
    v95: float,
    budget: int,
    learning_rate: float = 0.05,
    baseline_alpha: float = 0.1,
    eval_every: int = 500,
    manager_lr_scale: float = 0.6,
    worker_lr_scale: float = 1.0,
    temp_start: float = 1.25,
    temp_end: float = 0.95,
    entropy_start: float = 0.05,
    entropy_end: float = 0.01,
    gae_lambda: float = 0.9,
    soft_tau_start: float = 0.01,
    soft_tau_end: float = 0.002,
    soft_mix_start: float = 0.0,
    soft_mix_end: float = 0.0,
    prior_mix: float = 0.0,
    q_star: np.ndarray | None = None,
) -> MethodResult:
    """Hierarchical with per-depth baselines, mild temp/entropy, two-timescale LRs, GAE."""
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    policy = ManagerWorkerPolicy(env)
    depth_baselines = np.zeros(env.depth, dtype=float)
    global_baseline = 0.0
    r_max = 1e-8
    kl_hist: list[float] = []
    tv_hist: list[float] = []

    log_priors = {
        (d, pref): np.log(np.clip(env.prior(pref), 1e-12, None))
        for (d, pref) in policy.logits
    }

    for t in range(budget):
        frac = t / max(budget - 1, 1)
        temperature = temp_start + (temp_end - temp_start) * frac
        entropy_coef = entropy_start + (entropy_end - entropy_start) * frac

        def action_fn(state, prior):
            a, q_prob, _q = policy.sample(state, env.rng, temperature=temperature)
            return a, q_prob

        _path, L, iw, traj, _pga = env.rollout_with_policy(action_fn)
        tracker.update(L, iw)

        hard_r = path_tail_reward(L, iw, v95)
        r_max = max(r_max, abs(hard_r), 1e-8)
        r_scaled = hard_r / r_max

        # GAE only used to shape leaf credit across depths; fall back to shared adv.
        rewards = np.zeros(env.depth, dtype=float)
        rewards[-1] = r_scaled
        advantages = _gae_advantages(
            rewards, depth_baselines.copy(), gamma=1.0, lam=gae_lambda
        )
        # Blend with classic shared advantage (stabilizes sparse zero-reward updates).
        shared = r_scaled - global_baseline
        advantages = 0.5 * advantages + 0.5 * shared

        for step, adv in zip(traj, advantages):
            d = step["depth"]
            lr = learning_rate * (
                manager_lr_scale if d in MANAGER_DEPTHS else worker_lr_scale
            )
            policy.update_step(
                step["state"],
                step["action"],
                float(adv),
                learning_rate=lr,
                entropy_coef=entropy_coef,
                temperature=1.0,  # update in logit space without extra sharpening
            )

        global_baseline = (1.0 - baseline_alpha) * global_baseline + baseline_alpha * r_scaled
        for d in range(env.depth):
            depth_baselines[d] = (
                (1.0 - baseline_alpha) * depth_baselines[d] + baseline_alpha * r_scaled
            )

        if prior_mix > 0 and (t + 1) % 50 == 0:
            for key, logits in policy.logits.items():
                policy.logits[key] = (1.0 - prior_mix) * logits + prior_mix * log_priors[key]

        if q_star is not None and (t + 1) % eval_every == 0:
            q_now = policy.path_distribution(temperature=1.0)
            kl_hist.append(kl_divergence(q_now, q_star))
            tv_hist.append(tv_distance(q_now, q_star))

    final_q = policy.path_distribution(temperature=1.0)
    extras: dict = {
        "depth_baselines": depth_baselines.tolist(),
        "kl_to_qstar": kl_hist,
        "tv_to_qstar": tv_hist,
    }
    if q_star is not None:
        extras["final_kl_to_qstar"] = kl_divergence(final_q, q_star)
        extras["final_tv_to_qstar"] = tv_distance(final_q, q_star)
        top20 = np.argsort(q_star)[::-1][:20]
        extras["mass_on_top20_qstar"] = float(final_q[top20].sum())

    return MethodResult(
        name="Hierarchical",
        metrics=tracker.finalize(),
        final_q=final_q,
        extras=extras,
    )
