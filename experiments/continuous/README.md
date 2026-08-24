# Continuous epistemic PSHA (Houng et al. 2025)

Four continuous epistemic variables: Gutenberg–Richter **b-value** and **m_max**, plus GMM offsets **Δμ** and **Δσ**. Aleatory uncertainty integrates truncated GR magnitudes and a Sadigh et al. (1997) rock GMPE at R = 35 km. The CVaR threshold is the PGA at a target annual exceedance rate (default 10⁻⁴ /yr).

```powershell
python scripts/run.py --config continuous
python scripts/plot_hazard_curve.py --config continuous
```

Config: [`config.yaml`](config.yaml).

Methods: Naive MC, Disagg-IS oracle, G-PMC AIS (given closed-form hazard), Hierarchical JEPA-CVaR v1/v2 (scalar reward only), CVaR-BF AIS, QR-SRM AIS.

Outputs: [`results/continuous/`](../../results/continuous/README.md).
