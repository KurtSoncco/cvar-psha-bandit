"""Spatial multi-site portfolio IS methods."""

from __future__ import annotations

from itertools import product

import numpy as np

from cvar_psha.estimators import OnlineCVaRTracker, path_tail_reward
from cvar_psha.methods import MethodResult
from cvar_psha.methods.tree_common import softmax
from cvar_psha.metrics import kl_divergence, ks_statistic, tv_distance  # noqa: F401  (re-exported)
from cvar_psha.spatial_env import MANAGER_DEPTHS, SpatialPortfolioEnv


def softmax_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    t = max(float(temperature), 1e-4)
    z = (logits - np.max(logits)) / t
    e = np.exp(z)
    return e / e.sum()


def run_spatial_mc(
    env: SpatialPortfolioEnv,
    v95: float,
    budget: int,
    eval_every: int = 500,
) -> MethodResult:
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)

    def action_fn(state, prior):
        a = int(env.rng.choice(len(prior), p=prior))
        return a, float(prior[a])

    for _ in range(budget):
        _path, L, iw, _traj, _pga = env.rollout_with_policy(action_fn)
        tracker.update(L, iw)
    return MethodResult(
        name="Naive MC",
        metrics=tracker.finalize(),
        final_q=env.path_priors.copy(),
    )


def run_spatial_qstar(
    env: SpatialPortfolioEnv,
    v95: float,
    budget: int,
    q_star: np.ndarray,
    eval_every: int = 500,
) -> MethodResult:
    """Oracle IS: sample full paths from reference q* (mathematical upper baseline)."""
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    q = np.asarray(q_star, dtype=float)
    q = q / q.sum()
    for _ in range(budget):
        _path, L, iw, _pga = env.sample_path_from_flat_q(q)
        tracker.update(L, iw)
    return MethodResult(
        name="q* oracle",
        metrics=tracker.finalize(),
        final_q=q.copy(),
    )


def run_spatial_flat_reinforce(
    env: SpatialPortfolioEnv,
    v95: float,
    budget: int,
    learning_rate: float = 0.03,
    baseline_alpha: float = 0.1,
    eval_every: int = 500,
) -> MethodResult:
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    logits = np.log(np.clip(env.path_priors, 1e-12, None))
    baseline = 0.0
    r_max = 1e-8
    q_last = env.path_priors.copy()

    for _ in range(budget):
        q = softmax(logits)
        q_last = q
        path, L, iw, _pga = env.sample_path_from_flat_q(q)
        idx = env.path_index[path]
        tracker.update(L, iw)

        r = path_tail_reward(L, iw, v95)
        r_max = max(r_max, abs(r), 1e-8)
        r_scaled = r / r_max
        advantage = r_scaled - baseline
        baseline = (1.0 - baseline_alpha) * baseline + baseline_alpha * r_scaled
        grad = -q
        grad[idx] += 1.0
        logits += learning_rate * advantage * grad

    return MethodResult(
        name="Flat REINFORCE",
        metrics=tracker.finalize(),
        final_q=q_last,
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
    """Flat CVaR-CPO with KL damping to previous q and pull toward path prior."""
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    prior = env.path_priors / env.path_priors.sum()
    logits = np.log(np.clip(prior, 1e-12, None))
    lam = 0.0
    target = true_cvar if true_cvar is not None else 0.0
    has_target = true_cvar is not None
    r_max = 1e-8
    q_last = prior.copy()

    for _ in range(budget):
        q_old = softmax(logits)
        q = q_old
        q_last = q
        path, L, iw, _pga = env.sample_path_from_flat_q(q)
        idx = env.path_index[path]
        tracker.update(L, iw)

        cvar_hat = tracker.current_cvar()
        if np.isfinite(cvar_hat):
            if not has_target:
                if target == 0.0:
                    target = cvar_hat
                else:
                    target = (1.0 - target_ema) * target + target_ema * cvar_hat
            violation = abs(cvar_hat - target) - cvar_tol * max(abs(target), 1e-6)
            lam = max(0.0, lam + dual_lr * violation)

        r = path_tail_reward(L, iw, v95)
        r_max = max(r_max, abs(r), 1e-8)
        r_scaled = r / r_max
        soft_penalty = 0.0
        if np.isfinite(cvar_hat) and target != 0.0:
            soft_penalty = abs(cvar_hat - target) / max(abs(target), 1e-6)
        advantage = r_scaled - lam * soft_penalty

        grad = -q
        grad[idx] += 1.0
        logits = logits + learning_rate * advantage * grad

        q_new = softmax(logits)
        q_damped = (
            (1.0 - kl_coef - prior_kl_coef) * q_new
            + kl_coef * q_old
            + prior_kl_coef * prior
        )
        q_damped = np.clip(q_damped, 1e-12, None)
        q_damped /= q_damped.sum()
        logits = np.log(q_damped)

    return MethodResult(
        name="CVaR-CPO",
        metrics=tracker.finalize(),
        final_q=q_last,
        extras={"final_lambda": lam, "target": target},
    )


class ManagerWorkerPolicy:
    """Hierarchical policies: Manager (geom, GMM), Worker (mag, rupture | state)."""

    def __init__(self, env: SpatialPortfolioEnv):
        self.env = env
        self.logits: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}
        for d in range(env.depth):
            if d == 0:
                prefixes: list[tuple[int, ...]] = [()]
            else:
                ranges = [range(n) for n in env.n_actions[:d]]
                prefixes = [tuple(p) for p in product(*ranges)]
            for pref in prefixes:
                prior = env.prior(pref)
                self.logits[(d, pref)] = np.log(np.clip(prior, 1e-12, None)).copy()

    def probs(self, state: tuple[int, ...], temperature: float = 1.0) -> np.ndarray:
        return softmax_temperature(self.logits[(len(state), state)], temperature)

    def sample(
        self,
        state: tuple[int, ...],
        rng: np.random.Generator,
        temperature: float = 1.0,
    ) -> tuple[int, float, np.ndarray]:
        q = self.probs(state, temperature=temperature)
        a = int(rng.choice(len(q), p=q))
        return a, float(q[a]), q

    def update_step(
        self,
        state: tuple[int, ...],
        action: int,
        advantage: float,
        learning_rate: float,
        entropy_coef: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        d = len(state)
        key = (d, state)
        q = softmax_temperature(self.logits[key], temperature)
        grad = -q
        grad[action] += 1.0
        # Entropy bonus gradient (encourage higher-entropy categorical).
        if entropy_coef > 0:
            ent_grad = -np.log(np.clip(q, 1e-12, None)) - 1.0
            ent_grad -= np.sum(q * ent_grad)  # center in probability space (approx)
            grad = grad + entropy_coef * ent_grad
        lr = learning_rate
        if d in MANAGER_DEPTHS:
            lr *= 1.0  # manager scale applied by caller
        self.logits[key] = self.logits[key] + lr * advantage * grad

    def path_distribution(self, temperature: float = 1.0) -> np.ndarray:
        q_path = np.zeros(self.env.n_paths, dtype=float)
        for i, path in enumerate(self.env.paths):
            p = 1.0
            state: tuple[int, ...] = ()
            for a in path:
                p *= float(self.probs(state, temperature=temperature)[a])
                state = state + (a,)
            q_path[i] = p
        s = q_path.sum()
        return q_path / s if s > 0 else self.env.path_priors.copy()


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
