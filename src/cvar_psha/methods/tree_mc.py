"""3-node / spatial full-path methods (wrappers around ``core.categorical``)."""

from __future__ import annotations

import numpy as np

from cvar_psha.core.categorical import FlatPathProblem, run_cem as _run_cem
from cvar_psha.core.categorical import run_cvar_cpo as _run_cvar_cpo
from cvar_psha.core.categorical import run_exp3 as _run_exp3
from cvar_psha.core.categorical import run_mc as _run_mc
from cvar_psha.core.categorical import run_oracle as _run_oracle
from cvar_psha.core.categorical import run_reinforce as _run_reinforce
from cvar_psha.core.result import MethodResult
from cvar_psha.tree_env import TreeLogicEnv


def run_tree_mc(
    env: TreeLogicEnv,
    v95: float,
    budget: int,
    eval_every: int = 200,
) -> MethodResult:
    return _run_mc(FlatPathProblem(env), v95, budget, eval_every=eval_every)


def run_tree_oracle(
    env: TreeLogicEnv,
    v95: float,
    budget: int,
    q: np.ndarray,
    name: str = "q* oracle",
    eval_every: int = 200,
) -> MethodResult:
    return _run_oracle(FlatPathProblem(env), v95, budget, q, name=name, eval_every=eval_every)


def run_tree_exp3(
    env: TreeLogicEnv,
    v95: float,
    budget: int,
    gamma: float = 0.05,
    eval_every: int = 200,
) -> MethodResult:
    return _run_exp3(
        FlatPathProblem(env), v95, budget, gamma=gamma, eval_every=eval_every, name="Flat Exp3"
    )


def run_tree_reinforce(
    env: TreeLogicEnv,
    v95: float,
    budget: int,
    learning_rate: float = 0.05,
    baseline_alpha: float = 0.1,
    eval_every: int = 200,
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


def run_tree_cem(
    env: TreeLogicEnv,
    v95: float,
    budget: int,
    batch_size: int = 500,
    elite_frac: float = 0.05,
    smoothing: float = 0.7,
    eval_every: int = 200,
) -> MethodResult:
    return _run_cem(
        FlatPathProblem(env),
        v95,
        budget,
        batch_size=batch_size,
        elite_frac=elite_frac,
        smoothing=smoothing,
        eval_every=eval_every,
    )


def run_cvar_cpo(
    env: TreeLogicEnv,
    v95: float,
    budget: int,
    learning_rate: float = 0.05,
    dual_lr: float = 0.05,
    kl_coef: float = 0.1,
    cvar_tol: float = 0.15,
    target_ema: float = 0.05,
    eval_every: int = 200,
    true_cvar: float | None = None,
) -> MethodResult:
    return _run_cvar_cpo(
        FlatPathProblem(env),
        v95,
        budget,
        learning_rate=learning_rate,
        dual_lr=dual_lr,
        kl_coef=kl_coef,
        cvar_tol=cvar_tol,
        target_ema=target_ema,
        eval_every=eval_every,
        true_cvar=true_cvar,
    )
