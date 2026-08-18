# Spatial multi-site portfolio

Fault-line ruptures, K=10 sites, Gutenberg–Richter magnitude bins, epistemic geometry + GMM (Manager) and magnitude + rupture location (Worker). About 1,920 complete paths. Leaf loss `L = Σ_k PGA_k`.

```powershell
python scripts/run.py --config spatial
```

Config: [`config.yaml`](config.yaml). Ground truth is large-N Monte Carlo (the portfolio sum is not a single lognormal).

Methods: Naive MC, `q*` oracle, Flat REINFORCE, CVaR-CPO, Hierarchical.

Outputs: [`results/spatial/`](../../results/spatial/README.md).
