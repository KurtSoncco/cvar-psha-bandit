# Spatial portfolio results

From `python scripts/run.py --config spatial`. The ~1920-path tree is the setting that still fails to match `q*` on mag × rupture mass.

Expected files: `cvar_convergence.png`, `ess_comparison.png`, `fault_mag_mass.png` (magnitude × rupture marginal vs `q*`), `summary.json`.

Re-run to refresh numbers; `summary.json` records CVaR, ESS, KL/TV to `q_star`, and mass on the top-20 `q*` paths.
