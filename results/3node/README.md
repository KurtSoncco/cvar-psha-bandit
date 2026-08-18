# 3-node results

From `python scripts/run.py --config 3node` (budget 20k, 5 replications). **CO-STC** reaches KS = 0.029 vs `q_disagg` (inside the paper KS-D band 0.017–0.113), ahead of Hierarchical (0.046) on this run. CVaR mean is on truth. CEM-IS one-hots the top path and ESS collapses.

| Method | ESS (of 20k) | CVaR mean (true 2.559) | KS vs `q_disagg` |
|---|---|---|---|
| Naive MC | 20,000 | 2.559 | 0.523 |
| CEM-IS | 23 | 2.728 | 0.599 |
| Flat Exp3 | 7,591 | 2.563 | 0.065 |
| Flat REINFORCE | 8,890 | 2.565 | 0.060 |
| Hierarchical | 8,190 | 2.545 | 0.046 |
| CVaR-CPO | 8,945 | 2.561 | 0.064 |
| **CO-STC** | 5,165 | **2.559** | **0.029** |

Expected files: `cvar_convergence.png`, `ess_comparison.png`, `learning_curves.png`, `final_policies.png`, `summary.json`.
