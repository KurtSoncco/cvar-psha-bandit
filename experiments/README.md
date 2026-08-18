# Experiments

One folder per environment. Each folder has `config.yaml` (budget, seeds, method hyperparameters) and a README. The library in `src/` does not depend on these files except at CLI time.

```powershell
python scripts/run.py --config 1node
python scripts/run.py --config 3node
python scripts/run.py --config continuous
python scripts/run.py --config spatial
```

| Folder | Environment | Output |
|---|---|---|
| [`1node/`](1node/README.md) | 3 GMM arms | `results/1node/` |
| [`3node/`](3node/README.md) | 12-path logic tree | `results/3node/` |
| [`continuous/`](continuous/README.md) | 2-D Gaussian `θ` | `results/continuous/` |
| [`spatial/`](spatial/README.md) | ~1920-path portfolio | `results/spatial/` |

Legacy filenames (`config.yaml`, `config_3node.yaml`, `config_continuous.yaml`, `config_spatial.yaml`) in this directory still work.
