# Continuous epistemic space

Median-GMPE offset and aleatory-sigma scaling as `θ = (θ_μ, θ_σ)` with a bivariate Gaussian prior (`continuous_env.py`). Ground truth is Gauss-Hermite quadrature.

```powershell
python scripts/run.py --config continuous
```

Config: [`config.yaml`](config.yaml).

Methods: Naive MC, Disagg-IS oracle, G-PMC AIS (given closed-form hazard), Hierarchical JEPA-CVaR v1/v2 (scalar reward only), CVaR-BF AIS, QR-SRM AIS.

Outputs: [`results/continuous/`](../../results/continuous/README.md).
