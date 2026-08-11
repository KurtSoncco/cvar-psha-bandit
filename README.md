# CVaR PSHA Bandit

Prove that an autonomous agent can learn the theoretically optimal Importance Sampling (IS) distribution for Conditional Value-at-Risk (CVaR) in a Probabilistic Seismic Hazard Analysis (PSHA) logic tree — using only a scalar reward.

## Theoretical grounding

This project's empirical claims are benchmarked against two proven results from:

- Houng, S. E., & Ceferino, L. (2025). Fast probabilistic seismic hazard analysis through adaptive importance sampling. *Bulletin of the Seismological Society of America*, 115(2), 646–662. https://doi.org/10.1785/0120240153
  **Proven result used here:** the zero-variance-optimal IS distribution for hazard-curve / exceedance-probability estimation equals the classical PSHA *hazard disaggregation* distribution, `q_disagg(s) ∝ p(s) · P(Y>v | s)`.
- Houng, S. E., Ceferino, L., & Abrahamson, N. (2025). Fast propagation of epistemic uncertainty in seismic hazard via adaptive importance sampling. *Bulletin of the Seismological Society of America*, 116(4), 1709. https://doi.org/10.1785/0120250205
  **Approach mirrored here:** propagating epistemic uncertainty as a *continuous* distribution over model parameters (Gaussian Population Monte Carlo AIS) rather than a small set of discrete logic-tree branches.

Neither paper addresses CVaR (tail-mean) estimation, portfolio risk, or reward-driven agents — that is this project's own extension, kept clearly distinguished from their proven results wherever it's used (see `disaggregation.py`).

**What is and isn't proven:** we do *not* have a theorem showing an optimal policy can represent the disaggregation-optimal IS distribution for an arbitrary logic tree. What's below is empirical: RL/bandit agents, given only a scalar tail-reward, are benchmarked against the papers' closed-form/proven-optimal targets and against their own reported accuracy (KS-D ranges, speedups), on environments of increasing structure (a flat GMM node → an explicit 3-node logic tree → a continuous epistemic space).

## Experiments

### 1-node (GMM only)

One epistemic GMM node with three arms (`ln Y ~ N(μ, σ)`).

```powershell
python scripts/run.py --config experiments/config.yaml
```

Ground truth (`v95`, `CVaR`, `q_star`, `q_disagg`) is now **closed-form** (see "Ground truth & disaggregation" below), not simulated. Baselines include a `q* oracle` (samples from our CVaR-optimal target) and a `Disagg-IS oracle` (samples from the paper-1 proven-optimal target), alongside CEM-IS, Exp3, and REINFORCE.

Outputs: `results/`

### 3-node delayed-reward MDP

Classic epistemic chain **Source → Magnitude → GMM** — this is the environment that most directly matches a real PSHA logic tree.

```powershell
python scripts/run.py --config experiments/config_3node.yaml
```

Outputs: `results/3node/` (`cvar_convergence.png`, `ess_comparison.png`)

**Result (full budget, 20k samples/rep, 5 reps):** the best learning method (Hierarchical) reaches **KS = 0.046** against `q_disagg`, inside the paper's own reported KS-D range (0.017–0.113) — using only a scalar reward, never given the disaggregation formula.

### Continuous epistemic space (paper 2 style)

Median-GMPE offset and aleatory-sigma scaling as a **continuous** bivariate Gaussian prior `θ = (θ_μ, θ_σ)` instead of discrete branches (`continuous_env.py`). Ground truth is computed by Gauss-Hermite quadrature (`continuous_ground_truth.py`) — exact up to quadrature order, no Monte Carlo noise, cross-checked against 20M-sample brute-force MC (~3e-4 relative error).

```powershell
python scripts/run.py --config experiments/config_continuous.yaml
```

Methods compared:
- **Naive MC** — samples the nominal prior.
- **Disagg-IS oracle** — samples a Gaussian moment-matched to the exact `q_disagg` grid (static upper baseline).
- **G-PMC AIS** — Gaussian Population Monte Carlo adaptive importance sampling (`methods/gpmc_ais.py`), our implementation of the paper 2 approach. *Given* the closed-form conditional hazard `P(Y>v|θ)` directly.
- **Hierarchical JEPA-CVaR** — our agent (`methods/jepa_cvar.py` + `jepa.py`), described precisely below. **Never sees `P(Y>v|θ)`.**

**What "Hierarchical JEPA-CVaR" actually is (and isn't).** This is a lightweight numpy analog, not a reproduction of published architectures — stated plainly so the results below aren't over-read:
- *JEPA:* real self-supervised joint-embedding mechanics (separate context/target encoders, EMA target network with stop-gradient, a predictor trained to match embeddings rather than raw rollout values), gradient-checked against numerical differentiation (1e-9 agreement) — but linear (no hidden layer) encoders, `embed_dim=4`, 2-D input. Not the architecture of the I-JEPA/V-JEPA papers.
- *Hierarchical:* a 2-level Gaussian mean composition (a slow Manager anchor + a fast JEPA-latent-conditioned Worker correction), named after this repo's existing `hierarchical.py` pattern — **not** LeCun's proposed Hierarchical-JEPA (a stack of JEPA modules predicting at multiple temporal/spatial abstraction levels).
- *CVaR:* the dual variable `lam` penalizes drift of the running CVaR *estimate* from a target — the same heuristic mechanism as this repo's `cvar_cpo.py` (itself documented as "lightweight discrete CPO," not a rigorous CVaR policy gradient). It also only tilts θ (the epistemic marginal), leaving `y|θ` at its nominal conditional law — by construction this **cannot** reach the true zero-variance CVaR target, which requires tilting the joint `p(θ)p(y|θ) → q(θ,y) ∝ p(θ)p(y|θ)·y·1{y>v}`. See "Toward a real JEPA-CVaR algorithm" below for what closing this gap would take.

**Result (budget=20k, 10 replications — not a single point estimate):**

| Method | KS vs `q_disagg` (mean ± std) | CVaR estimate mean ± std | \|bias\| | RMSE |
|---|---|---|---|---|
| Naive MC | 0.503 ± 0.000 | 2.339 ± 0.049 | 0.75% | 0.0520 |
| G-PMC AIS (given closed form) | **0.038 ± 0.007** | 2.321 ± 0.033 | **0.03%** | 0.0327 |
| Hierarchical JEPA-CVaR (scalar reward only) | 0.092 ± 0.015 | 2.306 ± 0.018 | 0.67% | **0.0240** |

(true CVaR = 2.3219)

Direct answers, since the honest picture is mixed and metric-dependent:
- **Density match vs G-PMC AIS: loses, clearly.** ~2.4× worse KS (0.092 vs 0.038), non-overlapping across 10 reps. It beats Naive MC (0.503) but does not beat G-PMC AIS.
- **CVaR error vs G-PMC AIS: depends on the metric.** G-PMC AIS has essentially zero mean bias (0.03%) and wins on bias. Hierarchical JEPA-CVaR is consistently biased ~0.7% low, but its variance across replications is under half of G-PMC AIS's and less than half of Naive MC's — low enough that its **RMSE ends up lowest of the three** (0.0240 vs 0.0327 vs 0.0520). Plausible explanation: the dual variable directly targets CVaR-tracking rather than density matching, at the cost of a small systematic bias worth investigating further, not a clean unqualified win.

Outputs: `results/continuous/` (`cvar_convergence.png`, `ess_comparison.png`, `continuous_theta_comparison.png`)

### Toward a real JEPA-CVaR algorithm

The above is a scoped, honestly-labeled analog. Closing the gap to something that deserves the name without qualifiers would need, roughly in order of expected impact:

1. **Tilt the joint, not just θ.** Let the Worker also propose an aleatory correction to `y | θ` (e.g. a learned shift/scale on the sampling of `ln Y`), with the importance weight computed over the full joint `q(θ, y)`. This is the only way to approach the true zero-variance CVaR target rather than the θ-marginal-only target every method here (including G-PMC AIS) is limited to.
2. **A real CVaR policy gradient**, replacing the drift-penalizing dual heuristic — e.g. Tamar, Glynn & Mannor (2015)'s CVaR policy gradient theorem, or a properly derived CPO with trust-region KKT conditions (Chow & Ghavamzadeh), rather than a hand-tuned Lagrangian proxy.
3. **A genuinely hierarchical (multi-scale) JEPA**, predicting at more than one level of abstraction (e.g. a coarse "region of θ-space" predictor feeding a fine "exact θ" predictor, each with its own target encoder/EMA pair), rather than the current single-level context/target pair.
4. **A deeper encoder** (hidden layers, larger `embed_dim`) once (1)–(3) justify the added capacity — right now the bottleneck is the algorithm, not encoder width.
5. Re-run the RMSE/KS comparison at several budgets and seeds-per-budget (10 reps is a first read, not a settled result) to see whether the low-bias/low-variance trade this version shows is a real property of CVaR-shaped objectives or an artifact of this budget/config.

### 2D multi-site portfolio hazard MDP

Fault-line ruptures, \(K=10\) sites, Gutenberg–Richter magnitude bins, epistemic geometry + GMM (Manager) and mag + rupture location (Worker).

Portfolio loss \(L = \sum_k \mathrm{PGA}_k\) under a distance-attenuated GMPE footprint. Leaf reward:

```text
r_T = (prod_t w/q) * L * 1{L > v95}
```

Methods: Naive MC, Flat REINFORCE, **CVaR-CPO**, **Hierarchical** (~1920 joint paths).

```powershell
python scripts/run.py --config experiments/config_spatial.yaml
```

Outputs: `results/spatial/`

- `cvar_convergence.png`, `ess_comparison.png`
- `fault_mag_mass.png` — mag × rupture marginal mass vs \(q^*\)

## Ground truth & disaggregation (`disaggregation.py`, `continuous_ground_truth.py`)

For lognormal leaves (`ln Y | s ~ N(μ_s, σ_s)`), VaR, CVaR, and both IS targets are **closed-form**:

- `q_disagg(s) ∝ p(s) · P(Y>v | s)` — Houng & Ceferino (2025)'s proven-optimal target for exceedance-probability estimation (the classical hazard disaggregation distribution).
- `q_star(s) ∝ p(s) · E[Y · 1{Y>v} | s]` — our severity-weighted extension, the correct IS target for CVaR's numerator. **Not** proven in the source papers; kept distinguished from `q_disagg` throughout.

VaR is solved exactly via 1-D root-finding on the closed-form mixture CDF (no `np.quantile` sampling noise). For the continuous environment, Gauss-Hermite quadrature nodes/weights stand in for "priors, mus, sigmas," so the same closed-form machinery is reused unchanged. All closed forms were cross-checked against large-N brute-force Monte Carlo before use (see git history / `disaggregation.py`, `continuous_ground_truth.py` docstrings).

## Setup

```powershell
cd C:\Users\kurt-\Projects\cvar-psha-bandit
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Metrics

- **CVaR convergence** toward the large-sample / closed-form true portfolio CVaR
- **ESS** = `(Σ W)² / Σ W²` to detect path-policy collapse
- **KL / TV / KS** of a method's final sampling distribution against `q_star` and `q_disagg` (`metrics.py`) — KS is the categorical analog of the KS-D metric both source papers report, for direct comparability.
