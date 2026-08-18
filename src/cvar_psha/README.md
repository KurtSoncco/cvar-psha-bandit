# Library (`src/cvar_psha`)

Installable package `cvar_psha`. Environments, closed-form ground truth, IS/RL methods, and experiment runners live here. YAML configs, plots, and writeups do **not**.

## Layout

| Module / package | Contents |
|---|---|
| [`core/`](core/README.md) | Estimators, KL/TV/KS, softmax / REINFORCE helpers, generic categorical IS, PSIS, lognormal disaggregation |
| [`envs/`](envs/README.md) | Public re-exports of the four environments |
| [`experiment/`](experiment/README.md) | CLI and per-mode runners |
| [`methods/`](methods/README.md) | Algorithms (wrappers around `core.categorical` plus specialized methods) |
| `env.py`, `tree_env.py`, `continuous_env.py`, `spatial_env.py` | Environment implementations |
| `ground_truth.py`, `tree_ground_truth.py`, `continuous_ground_truth.py`, `spatial_ground_truth.py` | Benchmark targets |
| `hazard_curve.py` | Mean + fractile hazard from quadrature or logged IS samples |
| `jepa.py` | Lightweight numpy JEPA used by hierarchical JEPA-CVaR |
| `plot.py` | CVaR/ESS/policy figures |

Old import paths (`cvar_psha.estimators`, `cvar_psha.disaggregation`, `cvar_psha.run_experiment`, …) still work via compatibility shims.

## Shared abstractions

- **`CategoricalISProblem`** (`core/categorical.py`) — 1-node arms, 3-node paths, and spatial paths all expose `(index, y, importance_weight)`. One implementation of MC, oracle, Exp3, REINFORCE, CEM, and CVaR-CPO.
- **`TabularTreePolicy`** (`core/policy.py`) — one softmax table per `(depth, prefix)`; used by 3-node hierarchical REINFORCE and spatial Manager/Worker.
- **`closed_form_targets` / `weighted_gaussian_moments`** — discrete GT and Gaussian PMC refits share one formula each.
