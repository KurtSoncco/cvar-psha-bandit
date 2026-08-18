# Continuous-epistemic results

From `python scripts/run.py --config continuous` (budget 20k). Hierarchical JEPA-CVaR v2 defaults are PSIS + `mixture_window=6` + `tilt_cap=1.0`. G-PMC AIS is given the closed-form hazard; JEPA and QR-SRM are not.

**Current defaults vs references** (v2 as shipped):

| Method | ESS (of 20k) | CVaR RMSE | KS vs `q_disagg` |
|---|---|---|---|
| Naive MC | 20,000 | 0.0520 | 0.503 |
| Hierarchical JEPA-CVaR v1 | 2,577 ± 739 | 0.0240 | 0.092 ± 0.015 |
| **Hierarchical JEPA-CVaR v2** | **2,480 ± 162** | **0.0114** | 0.120 ± 0.064 |
| G-PMC AIS (given closed form) | 6,089 ± 74 | 0.0327 | **0.038 ± 0.007** |

**Same protocol, 5-rep run including CVaR-BF and QR-SRM** (true CVaR 2.322):

| Method | ESS (of 20k) | CVaR mean | KS vs `q_disagg` |
|---|---|---|---|
| Naive MC | 20,000 | 2.328 | 0.503 |
| Hierarchical JEPA-CVaR v1 | 2,720 | 2.306 | 0.077 |
| Hierarchical JEPA-CVaR v2 | 2,411 | 2.316 | 0.054 |
| G-PMC AIS | 6,086 | 2.334 | **0.032** |
| CVaR-BF AIS | **7,210** | **2.324** | 0.104 |
| QR-SRM AIS | 6,000 | 2.320 | 0.052 |

v2 has the best CVaR RMSE among learned methods and a trustworthy PSIS `k_hat` once the y-tilt is capped; G-PMC still wins density match (KS) and raw ESS. CVaR-BF has the highest adaptive ESS in the 5-rep table. QR-SRM matches G-PMC ESS with no closed-form leaf.

Expected files: `cvar_convergence.png`, `ess_comparison.png`, `continuous_theta_comparison.png`, `summary.json`.
