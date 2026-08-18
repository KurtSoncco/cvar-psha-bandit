# Results

Generated figures and `summary.json` from experiment runs. **Markdown writeups are tracked; PNG/PDF plots are gitignored.**

Each subfolder matches an experiment in [`experiments/`](../experiments/README.md). Re-run the matching `python scripts/run.py --config …` command to regenerate plots.

| Folder | Experiment | Typical artifacts |
|---|---|---|
| [`1node/`](1node/README.md) | GMM bandit | `learning_curves.png`, `cvar_convergence.png`, `ess_comparison.png`, `summary.json` |
| [`3node/`](3node/README.md) | Logic tree | same + `final_policies.png` |
| [`continuous/`](continuous/README.md) | Continuous `θ` | CVaR/ESS plots + `continuous_theta_comparison.png` |
| [`spatial/`](spatial/README.md) | Portfolio | CVaR/ESS + `fault_mag_mass.png` |
| [`hazard_curve_example/`](hazard_curve_example/README.md) | Mean + fractile curves | `SUMMARY.md`, hazard-curve PNGs |

Verdict-panel figures (optional): `python scripts/plot_verdict_panels.py` writes `results/verdict_panels/`.
