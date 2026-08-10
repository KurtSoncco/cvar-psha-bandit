"""Tabular CVaR-CPO: maximize leaf IS reward with a CVaR tracking constraint.

Uses a flat softmax over paths, Lagrangian dual ascent on a soft CVaR-tolerance
constraint, and a KL-damped policy step (lightweight discrete CPO).
"""

from __future__ import annotations

import numpy as np

from cvar_psha.estimators import OnlineCVaRTracker, path_tail_reward
from cvar_psha.methods import MethodResult
from cvar_psha.methods.tree_common import sample_path_from_flat_q, softmax
from cvar_psha.tree_env import TreeLogicEnv


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
    """Maximize E[r_T] s.t. |CVaR_hat - target| soft-bounded via dual variable.

    target tracks an EMA of the running CVaR estimate (or true_cvar if given for
    the constraint center). Trust-region flavour: KL penalty vs previous q.
    """
    k = env.n_paths
    tracker = OnlineCVaRTracker(v95=v95, eval_every=eval_every)
    logits = np.log(np.clip(env.path_priors, 1e-8, None))
    lam = 0.0  # dual variable for soft CVaR constraint
    target = true_cvar if true_cvar is not None else 0.0
    has_target = true_cvar is not None
    r_max = 1e-8
    q_hist = []

    for _ in range(budget):
        q_old = softmax(logits)
        q = q_old.copy()
        q_hist.append(q.copy())

        path, y, iw = sample_path_from_flat_q(env, q)
        idx = env.path_index[path]
        tracker.update(y, iw)

        cvar_hat = tracker.current_cvar()
        if np.isfinite(cvar_hat):
            if not has_target:
                if target == 0.0:
                    target = cvar_hat
                else:
                    target = (1.0 - target_ema) * target + target_ema * cvar_hat
            # Constraint violation: estimate too far from target.
            violation = abs(cvar_hat - target) - cvar_tol * max(abs(target), 1e-6)
            lam = max(0.0, lam + dual_lr * violation)

        r = path_tail_reward(y, iw, v95)
        r_max = max(r_max, abs(r), 1e-8)
        r_scaled = r / r_max

        # Lagrangian surrogate reward: maximize r, penalize CVaR constraint via
        # redirecting probability mass (advantage shaped by dual).
        # If violated, boost updates only when sample is a tail exceedance that
        # improves local alignment (proxy: use r_scaled - lam * soft_penalty).
        soft_penalty = 0.0
        if np.isfinite(cvar_hat) and target != 0.0:
            soft_penalty = abs(cvar_hat - target) / max(abs(target), 1e-6)
        advantage = r_scaled - lam * soft_penalty

        grad = -q
        grad[idx] += 1.0
        logits = logits + learning_rate * advantage * grad

        # KL / trust-region damping toward previous policy.
        q_new = softmax(logits)
        # Natural-ish step: mix back toward q_old.
        q_damped = (1.0 - kl_coef) * q_new + kl_coef * q_old
        q_damped = np.clip(q_damped, 1e-8, None)
        q_damped = q_damped / q_damped.sum()
        logits = np.log(q_damped)

    return MethodResult(
        name="CVaR-CPO",
        metrics=tracker.finalize(),
        final_q=q_hist[-1] if q_hist else env.path_priors.copy(),
        extras={"q_history": np.asarray(q_hist), "final_lambda": lam, "target": target},
    )
