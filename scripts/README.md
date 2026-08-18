# Scripts

Thin entry points. Library code stays in `src/cvar_psha/`.

| Script | Purpose |
|---|---|
| `run.py` | Main experiment CLI (`--config 1node\|3node\|continuous\|spatial` or a YAML path) |
| `hazard_curve_example.py` | Mean + fractile hazard curves from one G-PMC and one JEPA-CVaR v2 run |
| `plot_verdict_panels.py` | KS / policy summary figures from existing `results/*/summary.json` |

```powershell
python scripts/run.py --config 1node
python scripts/hazard_curve_example.py
python scripts/plot_verdict_panels.py
```
