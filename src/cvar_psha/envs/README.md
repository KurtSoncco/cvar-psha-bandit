# Environments

Four sampling environments, increasing in structure. Implementations stay in the parent package (`env.py`, `tree_env.py`, `continuous_env.py`, `spatial_env.py`); this package re-exports them.

| Environment | Module | State | Outcome |
|---|---|---|---|
| 1-node GMM bandit | `env.py` (`LogicTreeEnv`) | Choose one of K arms | `Y` lognormal given arm |
| 3-node tree | `tree_env.py` (`TreeLogicEnv`) | Source → Magnitude → GMM | Delayed `Y` at the leaf |
| Continuous epistemic | `continuous_env.py` | `θ = (θ_μ, θ_σ)` | `Y \| θ` from the true aleatory law |
| Spatial portfolio | `spatial_env.py` | Geometry → GMM → mag → rupture | `L = Σ_k PGA_k` over 10 sites |

Discrete environments implement `sample_path_from_flat_q` (or `step` for 1-node) so `core.categorical.BanditProblem` / `FlatPathProblem` can run the same IS algorithms on all three.
