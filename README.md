# CVaR PSHA Bandit

Earthquake engineers need the rare, severe end of ground shaking — **Probabilistic Seismic Hazard Analysis (PSHA)** — not just the average case. Brute-force simulation is expensive because severe events are rare. This repo asks whether an agent that sees only a **scalar reward** after each simulated earthquake can learn an **importance-sampling** proposal that focuses effort on the tail of the risk distribution.

**Importance sampling (IS).** Estimate a quantity under a nominal distribution `p` by sampling from a proposal `q` and reweighting by `w = p/q`. A good `q` cuts variance; a bad `q` can make estimates worse.

**VaR / CVaR.** For outcome `Y` and small `α`, `VaR_α` is the threshold exceeded with probability `α`. `CVaR_α = E[Y | Y > VaR_α]` is the mean of that tail — the quantity this project estimates.

**Logic tree.** Alternative scientific models (source, magnitude, GMM) with branch weights. Uncertainty over *which model is correct* is **epistemic**; run-to-run shaking given a model is **aleatory**.

Empirical claims are benchmarked against closed-form / quadrature targets from:

- Houng & Ceferino (2025), BSSA — zero-variance-optimal IS for exceedance probability equals hazard **disaggregation** `q_disagg(s) ∝ p(s)·P(Y>v|s)`.
- Houng, Ceferino & Abrahamson (2025), BSSA — continuous epistemic uncertainty via G-PMC AIS.

Neither paper addresses CVaR, portfolio risk, or reward-driven agents. The CVaR target `q_star(s) ∝ p(s)·E[Y·1{Y>v}|s]` is this project's extension and is kept distinct from `q_disagg`.

## Repository layout

| Path | Role |
|---|---|
| [`src/cvar_psha/`](src/cvar_psha/README.md) | Installable library: environments, ground truth, methods, experiment runners |
| [`experiments/`](experiments/README.md) | YAML configs, one folder per environment |
| [`results/`](results/README.md) | Figures, `summary.json`, and writeups (markdown tracked; plots gitignored) |
| [`data/`](data/README.md) | No external datasets; synthetic DGP notes |
| [`scripts/`](scripts/README.md) | CLI wrappers (`run.py`, hazard-curve example, verdict panels) |

Library internals:

- [`src/cvar_psha/core/`](src/cvar_psha/core/README.md) — shared estimators, distances, policy helpers, generic categorical IS
- [`src/cvar_psha/envs/`](src/cvar_psha/envs/README.md) — 1-node, 3-node, continuous, spatial environments
- [`src/cvar_psha/methods/`](src/cvar_psha/methods/README.md) — learning / IS algorithms
- [`src/cvar_psha/experiment/`](src/cvar_psha/experiment/README.md) — per-mode runners and CLI

## Setup

```powershell
cd C:\Users\kurt-\Projects\cvar-psha-bandit
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## EQ-JEPA earthquake forecasting

`cvar_psha.eq_jepa` is a separate, prospective forecasting module. It retains
the existing lightweight NumPy JEPA for CVaR importance sampling and adds a
PyTorch hierarchical JEPA for catalogue histories. Its rate head guarantees
that the regional expected rate equals the sum over spatial/magnitude cells;
`forecast.dat` is compatible with floatCSEP's gridded forecast format.

```bash
pip install -e .
eq-jepa --config experiments/eq_jepa/config.yaml
docker build -t eq-jepa .
docker run --rm -v "$PWD/results:/app/results" eq-jepa --config experiments/eq_jepa/config.yaml
```

Catalog source adapters are explicit: `iris` and `epos` query standard FDSN
event endpoints; `kiknet` consumes an approved local/exported CSV with
`time,latitude,longitude,depth_km,magnitude[,event_id]`. KiK-net waveform
archive access is not scraped. The provided `HistoricalRateBaseline` and
`ETASLiteBaseline` are transparent CSEP-ready baselines; the latter is a
simple triggering benchmark, not a substitute for calibrated ETAS software.

## Run experiments

```powershell
python scripts/run.py --config 1node
python scripts/run.py --config 3node
python scripts/run.py --config continuous
python scripts/run.py --config spatial
```

Aliases resolve to `experiments/<name>/config.yaml`. A YAML path still works (`python scripts/run.py --config experiments/1node/config.yaml`). Legacy filenames under `experiments/config*.yaml` remain.

Hazard-curve worked example (mean + fractile curves from one AIS run):

```powershell
python scripts/hazard_curve_example.py
python scripts/plot_hazard_curve.py --config continuous
```

Writeup: [`results/hazard_curve_example/SUMMARY.md`](results/hazard_curve_example/SUMMARY.md). Ground-truth mean/fractile curves for the four-parameter model also go to `results/continuous/hazard_curves.png`.

## What each experiment is

| Mode | Environment | Config | Results |
|---|---|---|---|
| 1-node | K GMM arms, one site | [`experiments/1node/`](experiments/1node/README.md) | [`results/1node/`](results/1node/README.md) |
| 3-node | Source → Magnitude → GMM tree | [`experiments/3node/`](experiments/3node/README.md) | [`results/3node/`](results/3node/README.md) |
| Continuous | Houng et al. 2025 four-parameter θ = (b, m_max, Δμ, Δσ) | [`experiments/continuous/`](experiments/continuous/README.md) | [`results/continuous/`](results/continuous/README.md) |
| Spatial | 10-site portfolio, ~1920 paths | [`experiments/spatial/`](experiments/spatial/README.md) | [`results/spatial/`](results/spatial/README.md) |

Every learning method sees only `tail_reward(y, w, v95) = w·y` if `y > v95`, else `0`. Closed-form `P(Y>v|·)` is given only to privileged baselines (oracles, G-PMC AIS, the CVaR-BF *filter*).

## Metrics

- **CVaR** — self-normalized IS tail mean (`OnlineCVaRTracker` in `core/estimators.py`).
- **ESS** — `(Σ W)² / Σ W²`; collapse diagnostic for the proposal.
- **KL / TV / KS** — distance of the learned `q` to `q_star` and `q_disagg` (`core/metrics.py`). KS is the categorical analog of the papers' KS-D.

## Ground truth

For lognormal leaves, VaR, CVaR, `q_disagg`, and `q_star` are closed-form (1-D root-find for VaR) in `core/disaggregation.py`. The continuous Houng environment uses product quadrature over θ plus magnitude bins, and ties the CVaR threshold to a target annual exceedance rate rather than an event-PGA percentile. Spatial portfolio CVaR is estimated by large-N Monte Carlo (the leaf is a multi-site sum, not a single lognormal).
