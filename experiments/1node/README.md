# 1-node GMM bandit

One epistemic node: which Ground Motion Model is correct. Three arms, each `ln Y ~ N(μ, σ)`.

```powershell
python scripts/run.py --config 1node
```

Config: [`config.yaml`](config.yaml). Ground truth is closed-form (`v95`, CVaR, `q_star`, `q_disagg`).

Methods: Naive MC, `q*` oracle, Disagg-IS oracle, CEM-IS, Exp3, REINFORCE, CO-STC.

Outputs: [`results/1node/`](../../results/1node/README.md).
