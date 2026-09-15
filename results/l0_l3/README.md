# L0–L3 build & validation results

Curated outputs from the first end-to-end L0–L3 build (see the main
[README](../../README.md#l0l3-build--validation-branch-filippo-l0-l3-validated) and
[`docs/l0_l3/`](../../docs/l0_l3/) for the full writeup).

## Contents

| Path | What it is |
|---|---|
| `figures/fig_final_pipeline_validation.png` | Dispersion collapse across the two catalog-bug fixes, L0 rate before/after correction, and final N-test pass rates across all four model variants |
| `figures/floatcsep_l0_results_corrected.png` | 3-panel floatCSEP L1 scoreboard on the truly-corrected catalog (consistency pass rates, forecast rate vs. observed count, comparative skill over time) |
| `figures/catalog_declustering_corrected.png` | Corrected Gardner-Knopoff declustering map + Gutenberg-Richter b-value plot |
| `figures/fig_gmpe_validation.png` | L2 physics-based ground-motion model residual vs. NGA-West2 empirical variance |
| `figures/fig_sdof_ida_validation.png` | L3 SDOF closed-form linear-limit check + IDA collapse-onset behavior |
| `figures/fig1_sdof_tail.png` | Pilot: linear vs. degrading-hysteresis SDOF loss-exceedance and CVaR sensitivity |
| `floatcsep_final_results_long.csv` | Full parsed floatCSEP evaluation results (long format), all 4 model variants x 71 windows x test types |
| `floatcsep_pass_rates_corrected_summary.csv`, `floatcsep_ttest_verdicts_summary.csv`, `l0_correction_ttest_comparison.csv` | Summary tables behind the headline pass-rate and T-test numbers |
| `floatcsep_config/{config,models,tests}.yml` | floatCSEP experiment configuration for the final validated run (`filippo` branch code + vetted-declustering catalog) |
| `data/california_catalog_declustered_vetted.csv` | Final corrected catalog: independent-event flags from the vetted `seismostats`/OpenQuake-HMTK Gardner-Knopoff implementation |
| `data/catalog_independent_floatcsep_format.csv` | Same independent-events catalog, formatted for floatCSEP ingestion |
| `model/eq_jepa_model.pt`, `model/eq_jepa_config.yaml`, `model/eq_jepa_training_history.json` | Trained L0 checkpoint, its config, and training-loss history |

## Deliberately not checked in

The raw floatCSEP experiment directories (`floatcsep_exp*/`, ~25-60 MB each) —
per-window `.dat` forecast files and per-test-type result JSONs — are **regenerable**
from `floatcsep_config/` + the vetted catalog + `eq-jepa --config ...`, and were left
out to avoid bloating repo history with reproducible binary/JSON output. Superseded
catalog stages (raw, aftershock-contaminated, and the first-pass buggy-declustered
catalog) and the SDOF pilot sweep (`pilot.npz`, `pilot_sweep.csv`) were likewise
excluded as either superseded or out of scope for this build's headline results. Ask
if you want any of these packaged separately (e.g. as a GitHub release asset or via
git-lfs) rather than committed to the tree.

## Reproducing

```bash
pip install -e .
eq-jepa --config <path-to-trained-model-config>  # or reuse model/eq_jepa_config.yaml
cd results/l0_l3/floatcsep_config
floatcsep run config.yml
```
