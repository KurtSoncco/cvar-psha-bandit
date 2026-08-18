# 1-node results

From `python scripts/run.py --config 1node` (budget 20k, 5 replications). Exp3 is the KS winner on this run. CO-STC's CVaR mean sits on truth; mass is on the pessimistic arm but under-peaks relative to Exp3.

| Method | CVaR mean (true 2.279) | KS vs `q_disagg` |
|---|---|---|
| Naive MC | 2.285 | 0.733 |
| CEM-IS | 2.305 | 0.067 |
| **Exp3** | **2.279** | **0.021** |
| REINFORCE | 2.284 | 0.034 |
| CO-STC | 2.280 | 0.137 |

Expected files: `learning_curves.png`, `cvar_convergence.png`, `ess_comparison.png`, `final_policies.png`, `summary.json`.
