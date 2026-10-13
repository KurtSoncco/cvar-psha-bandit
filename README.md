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

The encoder's latent is horizon-agnostic; horizon conditioning happens only at
decode time (`EQJEPA.forecast_rates(batch, horizon_days=...)`), so one trained
model can be queried at any requested horizon post-hoc -- e.g. writing a 1-day
and a 30-day forecast from the same forward pass (`train.write_forecast`
does exactly this, one `forecast_h{H}d.dat` file per horizon). The count head
is Negative-Binomial, not Poisson: earthquake counts are over-dispersed from
spatial-temporal clustering, so `forecast.dat` carries an `nb_dispersion`
column alongside `rate`.

```bash
pip install -e .
eq-jepa --config experiments/eq_jepa/config.yaml
docker build -t eq-jepa .
docker run --rm -v "$PWD/results:/app/results" eq-jepa --config experiments/eq_jepa/config.yaml
```

Catalog source adapters are explicit: `usgs` and `epos` query standard FDSN
event endpoints (`service.iris.edu/fdsnws/event` was retired by EarthScope in
2023 and is no longer supported; use `usgs` instead); `kiknet` consumes an
approved local/exported CSV with
`time,latitude,longitude,depth_km,magnitude[,event_id]`. KiK-net waveform
archive access is not scraped. The provided `HistoricalRateBaseline` and
`ETASLiteBaseline` are transparent CSEP-ready baselines; the latter is a
simple triggering benchmark, not a substitute for calibrated ETAS software.

### L0–L3 build & validation (branch `filippo-l0-l3-validated`)

A first end-to-end pass through four of the six planned layers — L0 (Hierarchical
JEPA seismicity forecaster), L1 (floatCSEP evaluation), L2 (physics-based ground-motion
scenario generator), L3 (nonlinear SDOF response) — was built and scored against real
data: USGS/ANSS California catalog, M2.8+, 2019–2024, 71 rolling monthly floatCSEP
windows. Full writeup: [`docs/l0_l3/L0_L3_verification_report.md`](docs/l0_l3/L0_L3_verification_report.md);
the layered L0→L5 framework connecting this build to floatCSEP, CVaR, and the
constrained bandit/MPC decision layer is in
[`docs/l0_l3/EQ-JEPA_sequential_definition.md`](docs/l0_l3/EQ-JEPA_sequential_definition.md);
slide deck: [`docs/l0_l3/EQ-JEPA_L0-L3_presentation.pptx`](docs/l0_l3/EQ-JEPA_L0-L3_presentation.pptx).

**Two catalog bugs were found and fixed during L1 validation**, both upstream of any
model:

1. *Aftershock contamination* — the independent-events filter was computed but never
   applied before scoring, leaving floatCSEP scored against the raw clustered catalog
   (monthly count dispersion 379.5x Poisson).
2. *Declustering formula error* — a hand-coded Gardner–Knopoff window had swapped
   space/time coefficients for M≥6.5, fabricating multi-year gaps that read as real
   seismic quiescence (residual dispersion 10.5x Poisson even after "fixing" bug 1).

Re-declustering with the vetted `seismostats` port of OpenQuake HMTK's Gardner–Knopoff
implementation collapsed dispersion to **0.94x Poisson** — effectively the Poisson
regime L0's count head already assumes. Corrected catalog: 19,152 events, 5,491
independent mainshocks, 13,661 aftershock/cluster-flagged; Gutenberg–Richter b-value
0.93 (raw) → 0.84 (declustered).

**With the catalog fixed**, the previously "universal" N-test failure turned out to be
a genuine mean-rate deficit shared by all three models (observed mean 16.8
events/window vs. L0's 4.8, Historical-GR's 5.2, ETAS-lite's 7.5) — not overdispersion,
and not something a negative-binomial count head alone would fix. A minimal 2-parameter
Hawkes-like multiplicative rate correction on L0 (amplitude 0.20, decay tau=5.1 days,
fit by Poisson-likelihood MLE against recent seismicity) raised L0's N-test pass rate
from 5.6% to 53.5%, and flipped its comparative T-test standing from losing to both
baselines in the majority of windows to beating both in roughly a third of them
(24 better / 45 tied / 2 worse vs. ETAS-lite; 25/42/4 vs. Historical-GR). M-test
(spatial/magnitude shape) passed in 73–97% of windows throughout — relative
spatial-magnitude structure was never the problem.

Curated code, docs, results, and the trained checkpoint from this build live under
[`results/l0_l3/`](results/l0_l3/README.md) and
[`docs/l0_l3/`](docs/l0_l3/L0_L3_verification_report.md); see that directory's README
for what is (and, deliberately, is not) checked in.

**Open caveat carried forward**: the T-test comparisons above apply the rate
correction to L0 only, against *uncorrected* GR/ETAS-lite baselines. Whether the
baselines' own rate deficits are fixable artifacts (in which case the comparison
should be redone symmetrically) or a genuine skill difference is unresolved.

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
