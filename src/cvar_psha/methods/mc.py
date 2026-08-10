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
