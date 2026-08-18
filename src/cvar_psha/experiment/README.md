# Experiment runners

CLI and per-environment loops. Config YAML lives in [`experiments/`](../../../experiments/README.md); this package only reads it.

| File | Role |
|---|---|
| `cli.py` | `python scripts/run.py --config 1node` (aliases or YAML path) |
| `io.py` | Load YAML, resolve `output_dir` from repo root, env builders, distance reporting |
| `run_1node.py` | GMM bandit |
| `run_3node.py` | Source → Magnitude → GMM |
| `run_continuous.py` | Continuous `θ` |
| `run_spatial.py` | Multi-site portfolio |

`cvar_psha.run_experiment` remains as a compatibility shim.
