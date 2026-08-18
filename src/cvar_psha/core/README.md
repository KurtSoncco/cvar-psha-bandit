# Core

Shared math used by every environment and method. Nothing here knows about YAML configs or result folders.

| File | Role |
|---|---|
| `estimators.py` | Importance weights, `tail_reward`, ESS, `OnlineCVaRTracker` |
| `metrics.py` | KL, total variation, Kolmogorov–Smirnov (categorical) |
| `policy.py` | `softmax`, categorical score, EMA baseline, reward scaler, `TabularTreePolicy` |
| `categorical.py` | Generic MC / oracle / Exp3 / REINFORCE / CEM / CVaR-CPO over a `CategoricalISProblem` |
| `gaussian.py` | Weighted 2-D Gaussian moment matching (G-PMC, JEPA v2, QR-SRM) |
| `disaggregation.py` | Closed-form lognormal VaR, CVaR, `q_disagg`, `q_star` |
| `psis.py` | Pareto-smoothed importance sampling |
| `result.py` | `MethodResult` dataclass |

Compatibility shims: `cvar_psha.estimators`, `cvar_psha.metrics`, `cvar_psha.disaggregation`, `cvar_psha.psis`.
