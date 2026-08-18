# Methods

Sampling / learning algorithms. Discrete methods that only differ by “arm vs full path” are thin wrappers around [`core.categorical`](../core/categorical.py).

## Discrete (1-node, 3-node, spatial flat)

| Name | Module | Mechanism |
|---|---|---|
| Naive MC | `mc.py`, `tree_mc.py`, `spatial_methods.py` | Sample from the prior; no learning |
| `q*` / Disagg-IS oracle | same | Sample from closed-form `q_star` or `q_disagg` |
| CEM-IS | `cem.py` / `tree_cem.py` | Refit `q` to elite high-`Y` counts |
| Exp3 | `exp3.py` / `tree_exp3.py` | Multiplicative weights on tail reward |
| REINFORCE | `reinforce.py` / `tree_reinforce.py` | Softmax policy gradient |
| CVaR-CPO | `cvar_cpo.py` | Dual penalty on CVaR drift + KL damping |
| CO-STC | `sto_assign.py` | PPO assignment of a sampled fan onto fixed leaves (Pavirani et al. 2026) |

## Tree-structured

| Name | Module | Mechanism |
|---|---|---|
| Hierarchical | `hierarchical.py` | One softmax per node/history; leaf credit to every step |
| Spatial hierarchical | `spatial_methods.py` | Manager (geometry, GMM) / Worker (mag, rupture), GAE, temperature anneal |

## Continuous `θ`

| Name | Module | Privileged closed form? |
|---|---|---|
| G-PMC AIS | `gpmc_ais.py` | Yes — `P(Y>v\|θ)` or `E[Y 1{Y>v}\|θ]` |
| Hierarchical JEPA-CVaR v1/v2 | `jepa_cvar.py` + `jepa.py` | No — scalar reward only |
| CVaR-BF AIS | `cvar_bf.py` | Filter uses Gaussian leaf; actor does not |
| QR-SRM AIS | `qr_srm.py` | No |

The scalar reward every *learning* method sees is `tail_reward = w·y` on exceedances, else 0. See the repo README for VaR/CVaR/IS definitions.
