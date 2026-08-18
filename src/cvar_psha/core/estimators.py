"""Importance-weighted CVaR estimators, rolling variance, and ESS."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def importance_weight(prior_w: float, q_i: float, eps: float = 1e-12) -> float:
    return prior_w / max(q_i, eps)


def tail_reward(y: float, weight: float, v95: float) -> float:
    """Scalar bandit reward: IW tail contribution (0 below VaR)."""
    if y > v95:
        return weight * y
    return 0.0


def path_tail_reward(y: float, path_iw: float, v95: float) -> float:
    """Leaf reward for delayed-reward tree MDPs: (prod w/q) * y * 1{y>v95}."""
    return tail_reward(y, path_iw, v95)


def ess(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return 0.0
    s = w.sum()
    if s == 0:
        return 0.0
    return float((s * s) / np.square(w).sum())


@dataclass
class OnlineCVaRTracker:
    """Track IS-weighted CVaR (mean of Y | Y > v95 with likelihood ratios).

    Uses self-normalized importance sampling on exceedances:
        CVaR ≈ sum(W_t * Y_t * 1_tail) / sum(W_t * 1_tail)
    and reports rolling sample variance of the running CVaR trajectory.
    """

    v95: float
    eval_every: int = 200
    budgets: list[int] = field(default_factory=list)
    cvar_estimates: list[float] = field(default_factory=list)
    rolling_vars: list[float] = field(default_factory=list)
    ess_values: list[float] = field(default_factory=list)

    _sum_wy: float = 0.0
    _sum_w: float = 0.0
    _all_w: list[float] = field(default_factory=list)
    _history: list[float] = field(default_factory=list)
    _n: int = 0

    def update(self, y: float, w: float) -> None:
        self._n += 1
        self._all_w.append(w)
        if y > self.v95:
            self._sum_wy += w * y
            self._sum_w += w

        estimate = self.current_cvar()
        self._history.append(estimate)

        if self._n % self.eval_every == 0:
            self.budgets.append(self._n)
            self.cvar_estimates.append(estimate)
            self.rolling_vars.append(self._rolling_variance())
            self.ess_values.append(ess(np.asarray(self._all_w, dtype=float)))

    def current_cvar(self) -> float:
        if self._sum_w <= 0:
            return float("nan")
        return self._sum_wy / self._sum_w

    def _rolling_variance(self) -> float:
        hist = np.asarray(self._history, dtype=float)
        hist = hist[np.isfinite(hist)]
        if hist.size < 2:
            return float("nan")
        window = hist[-min(len(hist), max(self.eval_every, 50)) :]
        return float(np.var(window, ddof=1)) if window.size >= 2 else float("nan")

    def finalize(self) -> dict[str, np.ndarray]:
        if self._n > 0 and (not self.budgets or self.budgets[-1] != self._n):
            self.budgets.append(self._n)
            self.cvar_estimates.append(self.current_cvar())
            self.rolling_vars.append(self._rolling_variance())
            self.ess_values.append(ess(np.asarray(self._all_w, dtype=float)))
        return {
            "budget": np.asarray(self.budgets, dtype=int),
            "cvar": np.asarray(self.cvar_estimates, dtype=float),
            "variance": np.asarray(self.rolling_vars, dtype=float),
            "ess": np.asarray(self.ess_values, dtype=float),
        }
