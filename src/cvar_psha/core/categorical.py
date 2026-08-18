"""Generic categorical importance-sampling algorithms.

1-node GMM arms, 3-node full paths, and spatial full paths all reduce to:
sample an index from a categorical `q`, observe a scalar outcome `y`, and
form the importance weight `prior[i] / q[i]`. The runners below take a
`CategoricalISProblem` adapter; thin wrappers in `methods/` keep the old
`run_mc` / `run_tree_mc` / `run_spatial_mc` names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from cvar_psha.core.estimators import OnlineCVaRTracker, importance_weight, tail_reward
from cvar_psha.core.policy import EMABaseline, RewardScaler, categorical_score, softmax
from cvar_psha.core.result import MethodResult


class CategoricalISProblem(Protocol):
    """Minimal interface for a discrete IS environment."""

    rng: np.random.Generator

    @property
    def n_actions(self) -> int: ...

    @property
    def prior_mass(self) -> np.ndarray: ...

    def sample_from_q(self, q: np.ndarray) -> tuple[int, float, float]:
        """Return (index, outcome y, importance weight)."""
        ...


@dataclass
class BanditProblem:
    """1-node GMM arms."""

    env: Any

    @property
    def rng(self) -> np.random.Generator:
        return self.env.rng

    @property
    def n_actions(self) -> int:
        return int(self.env.n_arms)

    @property
    def prior_mass(self) -> np.ndarray:
        return self.env.weights

    def sample_from_q(self, q: np.ndarray) -> tuple[int, float, float]:
        arm, y = self.env.step(q)
        w = importance_weight(self.env.weights[arm], q[arm])
        return arm, y, w


@dataclass
class FlatPathProblem:
    """Full-path categorical over a tree (3-node or spatial)."""

    env: Any

    @property
    def rng(self) -> np.random.Generator:
        return self.env.rng

    @property
    def n_actions(self) -> int:
        return int(self.env.n_paths)

    @property
    def prior_mass(self) -> np.ndarray:
        return self.env.path_priors

    def sample_from_q(self, q: np.ndarray) -> tuple[int, float, float]:
        out = self.env.sample_path_from_flat_q(q)
        path, y, iw = out[0], out[1], out[2]
        return self.env.path_index[path], y, iw


def run_mc(
    problem: CategoricalISProblem,
    v95: float,
    budget: int,
    eval_every: int = 200,
    name: str = "Naive MC",
) -> MethodResult:
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    q = np.asarray(problem.prior_mass, dtype=float)
    q = q / q.sum()
    for _ in range(budget):
        _idx, y, w = problem.sample_from_q(q)
        tracker.update(y, w)
    return MethodResult(name=name, metrics=tracker.finalize(), final_q=q.copy())


def run_oracle(
    problem: CategoricalISProblem,
    v95: float,
    budget: int,
    q: np.ndarray,
    name: str = "q* oracle",
    eval_every: int = 200,
) -> MethodResult:
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    q = np.asarray(q, dtype=float)
    q = q / q.sum()
    for _ in range(budget):
        _idx, y, w = problem.sample_from_q(q)
        tracker.update(y, w)
    return MethodResult(name=name, metrics=tracker.finalize(), final_q=q.copy())


def run_exp3(
    problem: CategoricalISProblem,
    v95: float,
    budget: int,
    gamma: float = 0.05,
    eval_every: int = 200,
    name: str = "Exp3",
) -> MethodResult:
    k = problem.n_actions
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    weights = np.ones(k, dtype=float)
    q_hist: list[np.ndarray] = []
    scaler = RewardScaler()

    for _ in range(budget):
        wsum = weights.sum()
        q = (1.0 - gamma) * (weights / wsum) + gamma / k
        q_hist.append(q.copy())
        idx, y, iw = problem.sample_from_q(q)
        tracker.update(y, iw)
        r_hat = scaler.scale(tail_reward(y, iw, v95))
        weights[idx] *= np.exp(gamma * (r_hat / q[idx]) / k)
        weights /= weights.max()

    return MethodResult(
        name=name,
        metrics=tracker.finalize(),
        final_q=q_hist[-1] if q_hist else problem.prior_mass.copy(),
        extras={"q_history": np.asarray(q_hist)},
    )


def run_reinforce(
    problem: CategoricalISProblem,
    v95: float,
    budget: int,
    learning_rate: float = 0.05,
    baseline_alpha: float = 0.1,
    eval_every: int = 200,
    q_init: np.ndarray | str | None = None,
    name: str = "REINFORCE",
) -> MethodResult:
    k = problem.n_actions
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    if q_init is None:
        logits = np.zeros(k, dtype=float)
    elif isinstance(q_init, str) and q_init == "prior":
        logits = np.log(np.clip(problem.prior_mass, 1e-8, None))
    else:
        logits = np.log(np.clip(np.asarray(q_init, dtype=float), 1e-8, None))
    baseline = EMABaseline(baseline_alpha)
    scaler = RewardScaler()
    q_hist: list[np.ndarray] = []

    for _ in range(budget):
        q = softmax(logits)
        q_hist.append(q.copy())
        idx, y, iw = problem.sample_from_q(q)
        tracker.update(y, iw)
        advantage = baseline.advantage(scaler.scale(tail_reward(y, iw, v95)))
        logits += learning_rate * advantage * categorical_score(q, idx)

    return MethodResult(
        name=name,
        metrics=tracker.finalize(),
        final_q=q_hist[-1] if q_hist else problem.prior_mass.copy(),
        extras={"q_history": np.asarray(q_hist)},
    )


def run_cem(
    problem: CategoricalISProblem,
    v95: float,
    budget: int,
    batch_size: int = 500,
    elite_frac: float = 0.05,
    smoothing: float = 0.7,
    eval_every: int = 200,
    q_init: np.ndarray | None = None,
    name: str = "CEM-IS",
) -> MethodResult:
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    q = problem.prior_mass.copy() if q_init is None else np.asarray(q_init, dtype=float)
    q = q / q.sum()
    n_done = 0
    history = [q.copy()]
    clip_floor = 1e-8

    while n_done < budget:
        m = min(batch_size, budget - n_done)
        ids = np.empty(m, dtype=int)
        ys = np.empty(m, dtype=float)
        for t in range(m):
            ids[t], ys[t], iw = problem.sample_from_q(q)
            tracker.update(ys[t], iw)
        n_done += m
        k_elite = max(1, int(np.ceil(elite_frac * m)))
        elite = ids[np.argpartition(ys, -k_elite)[-k_elite:]]
        counts = np.bincount(elite, minlength=problem.n_actions).astype(float)
        if counts.sum() > 0:
            q_hat = counts / counts.sum()
            q = smoothing * q_hat + (1.0 - smoothing) * q
            q = np.clip(q, clip_floor, None)
            q = q / q.sum()
        history.append(q.copy())

    return MethodResult(
        name=name,
        metrics=tracker.finalize(),
        final_q=q,
        extras={"q_history": np.asarray(history)},
    )


def run_cvar_cpo(
    problem: CategoricalISProblem,
    v95: float,
    budget: int,
    learning_rate: float = 0.05,
    dual_lr: float = 0.05,
    kl_coef: float = 0.1,
    prior_kl_coef: float = 0.0,
    cvar_tol: float = 0.15,
    target_ema: float = 0.05,
    eval_every: int = 200,
    true_cvar: float | None = None,
    name: str = "CVaR-CPO",
) -> MethodResult:
    """Flat softmax CPO: maximize tail IS reward with a dual CVaR-drift penalty."""
    k = problem.n_actions
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    prior = np.asarray(problem.prior_mass, dtype=float)
    prior = prior / prior.sum()
    logits = np.log(np.clip(prior, 1e-12, None))
    lam = 0.0
    target = true_cvar if true_cvar is not None else 0.0
    has_target = true_cvar is not None
    scaler = RewardScaler()
    q_hist: list[np.ndarray] = []

    for _ in range(budget):
        q_old = softmax(logits)
        q_hist.append(q_old.copy())
        idx, y, iw = problem.sample_from_q(q_old)
        tracker.update(y, iw)

        cvar_hat = tracker.current_cvar()
        if np.isfinite(cvar_hat):
            if not has_target:
                if target == 0.0:
                    target = cvar_hat
                else:
                    target = (1.0 - target_ema) * target + target_ema * cvar_hat
            violation = abs(cvar_hat - target) - cvar_tol * max(abs(target), 1e-6)
            lam = max(0.0, lam + dual_lr * violation)

        r_scaled = scaler.scale(tail_reward(y, iw, v95))
        soft_penalty = 0.0
        if np.isfinite(cvar_hat) and target != 0.0:
            soft_penalty = abs(cvar_hat - target) / max(abs(target), 1e-6)
        advantage = r_scaled - lam * soft_penalty
        logits = logits + learning_rate * advantage * categorical_score(q_old, idx)

        q_new = softmax(logits)
        mix_old = kl_coef
        mix_prior = prior_kl_coef
        mix_new = 1.0 - mix_old - mix_prior
        q_damped = mix_new * q_new + mix_old * q_old + mix_prior * prior
        q_damped = np.clip(q_damped, 1e-12, None)
        q_damped = q_damped / q_damped.sum()
        logits = np.log(q_damped)

    return MethodResult(
        name=name,
        metrics=tracker.finalize(),
        final_q=q_hist[-1] if q_hist else prior.copy(),
        extras={"q_history": np.asarray(q_hist), "final_lambda": lam, "target": target},
    )
