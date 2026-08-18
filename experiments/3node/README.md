# 3-node delayed-reward tree

Epistemic chain **Source → Magnitude → GMM** (2 × 2 × 3 = 12 paths). Reward is delayed until the leaf.

```powershell
python scripts/run.py --config 3node
```

Config: [`config.yaml`](config.yaml). Closed-form path-mixture VaR/CVaR/disaggregation.

Methods: Naive MC, oracles, CEM-IS, Flat Exp3, Flat REINFORCE, Hierarchical, CVaR-CPO, CO-STC.

Outputs: [`results/3node/`](../../results/3node/README.md).
