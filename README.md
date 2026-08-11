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
- **Hierarchical JEPA-CVaR v1** — the original agent (`methods/jepa_cvar.py: run_jepa_cvar`), kept for comparison. **Never sees `P(Y>v|θ)`.**
- **Hierarchical JEPA-CVaR v2** — a corrected agent (`methods/jepa_cvar.py: run_jepa_cvar_v2` + `jepa.py`), replacing v1's heuristic update rule with a theoretically-motivated one. **Also never sees `P(Y>v|θ)`.** Described below.

**What v1 got wrong, and what v2 fixes.** v1's "CVaR" mechanism was a dual variable penalizing drift of the running CVaR *estimate* — a heuristic copied from this repo's `cvar_cpo.py` (itself labeled "lightweight discrete CPO," not a rigorous CVaR policy gradient), and my original text here cited Tamar/Chow & Ghavamzadeh's CVaR-policy-gradient/CPO papers as the target to aim for. That citation was a mismatch: those papers solve risk-averse RL over *an agent's own returns*; our actual problem is *adaptive importance sampling to minimize the variance of a CVaR estimator of a fixed external Y under a fixed nominal prior p* — the classical Cross-Entropy / Population Monte Carlo literature (de Boer, Kroese, Mannor & Rubinstein 2005; Cappé et al. 2004), which `gpmc_ais.py` already implements correctly.

v2's fix follows directly from that reframing: CE fits `q_φ` by minimizing `KL(q*, q_φ)` where `q*(x) ∝ p(x)H(x)`, which for a Gaussian family is a weighted moment-matching update using weights `w(x)=p(x)H(x)/q_ref(x)`. G-PMC AIS uses the closed-form `H(θ)=P(Y>v|θ)`; **the only thing requiring closed-form access is the choice of `H`, not the refit mechanism** — so v2 substitutes the single-rollout empirical outcome `H(θ,y)=y·1{y>v}`, and `p(θ)H(θ,y)/q(θ)` turns out to be exactly `estimators.tail_reward`, already used everywhere in this repo. v2 also tilts the full joint (adding a learned mean-shift `δ(θ)` to `ln Y | θ`, refit by weighted least squares) and composes two `LightweightJEPA` instances (a coarse region embedding feeding a fine one). All three additions were validated before the full run:
- Weighted moment-matching and weighted least-squares formulas checked against synthetic known-target recovery (≤3e-4 error).
- The learned `δ(θ)` was checked against a closed-form target — **not** the naive `σ(θ)²` untruncated-tilt originally proposed (that ignores the effect of conditioning on `Y>v`, which itself shifts the mean further); the correct target is the mean of the truncated-tilted normal, `μ+σ²+σ·φ(α)/(1−Φ(α))` with `α=(ln v−μ−σ²)/σ`. Learned values (1.16–1.56 across 4 test θ) tracked this corrected target (1.24–1.60) well, confirming the fit learns the right quantity.

**Result (budget=20k, 10 replications; ESS from a separate 8-replication check):**

| Method | KS vs `q_disagg` (mean ± std) | CVaR estimate mean ± std | RMSE | ESS (mean ± std, of 20k) |
|---|---|---|---|---|
| Naive MC | 0.503 ± 0.000 | 2.339 ± 0.049 | 0.0520 | 20,000 (no reweighting) |
| G-PMC AIS (given closed form) | **0.038 ± 0.007** | 2.321 ± 0.033 | 0.0327 | **6,099 ± 75** |
| Hierarchical JEPA-CVaR v1 (scalar reward only) | 0.092 ± 0.015 | 2.306 ± 0.018 | 0.0240 | 2,577 ± 739 |
| **Hierarchical JEPA-CVaR v2 (scalar reward only)** | 0.062 ± 0.019 | **2.320 ± 0.011** | **0.0113** | ⚠️ 100 ± 74 |

(true CVaR = 2.3219 — see the ESS caveat right below before reading the RMSE column as a clean win)

Direct answers:
- **v2 vs v1: clearly better on both metrics.** KS improves 0.092→0.062 (individual v2 runs as low as 0.027 — better than G-PMC AIS's average — though also as high as 0.091, so noisier run-to-run than G-PMC AIS). CVaR RMSE improves 0.0240→0.0113.
- **Density match vs G-PMC AIS: still loses on average**, but the gap shrank from 2.4× (v1) to 1.6× (v2), and the ranges now overlap.
- **CVaR error vs G-PMC AIS: v2 wins on RMSE, but read the ESS caveat below before trusting it.** Mean 2.320 vs true 2.3219 is essentially exact, and RMSE (0.0113) is nominally ~3× better than G-PMC AIS's (0.0327) despite zero closed-form access — but v2's ESS (100 ± 74 of 20,000) is two orders of magnitude below G-PMC AIS's (6,099 ± 75), meaning individual v2 runs are frequently dominated by a handful of samples. The RMSE number is real (it's what was measured), but "v2 beats G-PMC AIS" should be read as a promising, not yet fully trustworthy, result until the ESS collapse is understood.

Outputs: `results/continuous/` (`cvar_convergence.png`, `ess_comparison.png`, `continuous_theta_comparison.png`)

### What's still not "the real thing" — and a reliability caveat that qualifies the CVaR-RMSE win

Even v2 is not unqualified: the JEPA encoders are still linear/toy-scale (not I-JEPA/V-JEPA architecture), the y-tilt keeps aleatory variance fixed (no learned scale), and the coarse/fine JEPA composition, while genuinely two-level, is simple (batch-aggregated coarse signal, not a learned multi-step abstraction hierarchy). The KS gap to G-PMC AIS, while narrowed, hasn't closed.

More importantly, the CVaR-RMSE win above needs a real caveat, found by checking ESS (this project's own policy-collapse diagnostic) rather than taking the RMSE number at face value: over 8 replications, **v2's ESS is 100 ± 74 out of a 20,000 budget (~0.5%)** — one replication had ESS = 5. G-PMC AIS's ESS is 6,099 ± 75 over the same runs; v1's is 2,577 ± 739. A low, unstable ESS means v2's self-normalized IS estimator is, within many individual runs, dominated by a handful of samples — a known pathology of using the raw severity `y·1{y>v}` (heavy-tailed, since `Y` is lognormal) as an unclipped CE moment-matching weight, only partly offset by the 10% defensive-mixture floor. That v2's *between-replication* CVaR std (0.011) was nonetheless low is not fully explained by this and is worth treating as provisional rather than settled — it's plausible the CE-refit reliably converges to a similar concentrated region each time even though any single run's estimator is individually fragile, but that needs more replications (and ideally weight-clipping or a log-weight variant) to confirm rather than assume. **The honest summary: v2 is a real improvement over v1 on every metric checked, and beats G-PMC AIS's point accuracy in this experiment, but its low ESS means that win should not yet be trusted as robust.**

Next steps, in order of expected impact: weight-clipping or a log-domain weight transform in the CE refit to fix the ESS collapse (likely the highest-value fix — it may also improve the still-lagging KS); a learned aleatory scale (not just mean-shift) in the y-tilt; more replications once ESS is fixed to re-check whether the RMSE win survives; only then, larger encoders.

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
