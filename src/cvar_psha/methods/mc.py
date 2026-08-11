"""Naive Monte Carlo using the epistemic prior weights."""

from __future__ import annotations

import numpy as np

from cvar_psha.env import LogicTreeEnv
from cvar_psha.estimators import OnlineCVaRTracker, importance_weight
from cvar_psha.methods import MethodResult


def run_mc(
    env: LogicTreeEnv,
    v95: float,
    budget: int,
    eval_every: int = 200,
) -> MethodResult:
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    q = env.weights.copy()

    for _ in range(budget):
        arm, y = env.step(q)
        w = importance_weight(env.weights[arm], q[arm])
        tracker.update(y, w)

    return MethodResult(name="Naive MC", metrics=tracker.finalize(), final_q=q)


def run_oracle(
    env: LogicTreeEnv,
    v95: float,
    budget: int,
    q: np.ndarray,
    name: str = "q* oracle",
    eval_every: int = 200,
) -> MethodResult:
    """Sample arms directly from a fixed reference distribution q (e.g. the
    closed-form q_star or q_disagg from ground_truth.py) -- a static,
    non-learning IS baseline analogous to the papers' (adaptive but
    ultimately static-per-run) disaggregation-informed proposal."""
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    q = np.asarray(q, dtype=float)
    q = q / q.sum()
    for _ in range(budget):
        arm = int(env.rng.choice(env.n_arms, p=q))
        y = env.sample_ground_motion(arm)
        w = importance_weight(env.weights[arm], q[arm])
        tracker.update(y, w)
    return MethodResult(name=name, metrics=tracker.finalize(), final_q=q.copy())
