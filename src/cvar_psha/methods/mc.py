"""1-node GMM bandit methods (thin wrappers around ``core.categorical``)."""

from __future__ import annotations

import numpy as np

from cvar_psha.core.categorical import BanditProblem, run_cem as _run_cem
from cvar_psha.core.categorical import run_exp3 as _run_exp3
from cvar_psha.core.categorical import run_mc as _run_mc
from cvar_psha.core.categorical import run_oracle as _run_oracle
from cvar_psha.core.categorical import run_reinforce as _run_reinforce
from cvar_psha.core.result import MethodResult
from cvar_psha.env import LogicTreeEnv


def run_mc(
    env: LogicTreeEnv,
    v95: float,
    budget: int,
    eval_every: int = 200,
) -> MethodResult:
    return _run_mc(BanditProblem(env), v95, budget, eval_every=eval_every)


def run_oracle(
    env: LogicTreeEnv,
    v95: float,
    budget: int,
    q: np.ndarray,
    name: str = "q* oracle",
    eval_every: int = 200,
) -> MethodResult:
    return _run_oracle(BanditProblem(env), v95, budget, q, name=name, eval_every=eval_every)


def run_exp3(
    env: LogicTreeEnv,
    v95: float,
    budget: int,
    gamma: float = 0.05,
    eval_every: int = 200,
) -> MethodResult:
    return _run_exp3(BanditProblem(env), v95, budget, gamma=gamma, eval_every=eval_every)


def run_reinforce(
    env: LogicTreeEnv,
    v95: float,
    budget: int,
    learning_rate: float = 0.05,
    baseline_alpha: float = 0.1,
    eval_every: int = 200,
) -> MethodResult:
    return _run_reinforce(
        BanditProblem(env),
        v95,
        budget,
        learning_rate=learning_rate,
        baseline_alpha=baseline_alpha,
        eval_every=eval_every,
    )


def run_cem(
    env: LogicTreeEnv,
    v95: float,
    budget: int,
    batch_size: int = 500,
    elite_frac: float = 0.05,
    smoothing: float = 0.7,
    eval_every: int = 200,
    q_init: np.ndarray | None = None,
) -> MethodResult:
    return _run_cem(
        BanditProblem(env),
        v95,
        budget,
        batch_size=batch_size,
        elite_frac=elite_frac,
        smoothing=smoothing,
        eval_every=eval_every,
        q_init=q_init,
    )
