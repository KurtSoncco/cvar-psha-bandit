# Data

This project does not ship external catalogs, waveforms, or site files. Every environment is a **synthetic data-generating process** coded in `src/cvar_psha/` and parameterized by YAML in [`experiments/`](../experiments/README.md).

| What you might look for | Where it actually lives |
|---|---|
| GMM arm means/sigmas/weights | `experiments/1node/config.yaml` |
| Logic-tree branch weights | `experiments/3node/config.yaml` |
| Continuous `θ` prior (`τ_μ`, `τ_σ`) | `experiments/continuous/config.yaml` |
| Fault geometry, sites, GR bins | `experiments/spatial/config.yaml` |
| Closed-form / quadrature “truth” | computed at run time (`disaggregation.py`, `*_ground_truth.py`) |

If a real PSHA inventory is added later, put raw files here and keep experiment YAML as the only place that *selects* a dataset.
