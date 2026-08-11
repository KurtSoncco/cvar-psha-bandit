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
- **Hierarchical JEPA-CVaR** — our agent (`methods/jepa_cvar.py`): a numpy joint-embedding predictive architecture (`jepa.py`) trained only on realized rollout outcomes, feeding a hierarchical (Manager anchor + JEPA-latent-conditioned Worker correction) Gaussian sampling policy, trained by a CVaR-CPO-style dual-constrained REINFORCE. **Never sees `P(Y>v|θ)`.**

**Result (full budget, 20k samples/rep, 5 reps), KS against `q_disagg`:**

| Method | Given closed-form target? | KS | CVaR estimate (true = 2.3219) |
|---|---|---|---|
| Naive MC | no | 0.503 | 2.328 |
| Disagg-IS oracle | yes (exact) | 0.023 | 2.347 |
| G-PMC AIS | yes | 0.032 | 2.334 |
| **Hierarchical JEPA-CVaR** | **no — scalar reward only** | **0.077** | **2.306** |

The agent doesn't quite reach G-PMC AIS's density-matching accuracy at this budget, but its CVaR point estimate is competitive (and its own KS is within paper 1's reported 0.017–0.113 band even though it's compared against paper 2's stricter <5% target). Reported honestly, not cherry-picked — see `src/cvar_psha/methods/jepa_cvar.py` docstring for the known REINFORCE step-size instability this required fixing (LR annealing + Polyak tail-averaging).

Outputs: `results/continuous/` (`cvar_convergence.png`, `ess_comparison.png`, `continuous_theta_comparison.png`)

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
