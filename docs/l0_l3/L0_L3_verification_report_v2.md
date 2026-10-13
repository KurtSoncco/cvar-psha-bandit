# L0–L3 verification report, v2 — the next iteration

Supersedes `L0_L3_verification_report.md`. Same pipeline, same corrected catalog, same 71 monthly
test windows (2019-01 to 2024-11, California M2.8+), but forecasts regenerated per window and every
correction fitted on the 108 pre-2019 windows and frozen before scoring. Full narrative and revised
hypotheses: §10 of `EQ-JEPA_sequential_definition.md`.

## Headline

The previous report's two central numbers were artifacts of the forecast writer and of a
train/score catalog mismatch, and correcting both reveals a different and sharper picture: the
encoder had collapsed, the input embedding was the cause, and the target itself carries no
predictable monthly signal in any of four regions tested.

## What changed, and why

| previous claim | status | what it actually was |
|---|---|---|
| L0 under-forecasts the rate 3.5x (4.8 vs 16.8 events/window) | **superseded** | `write_forecast` averaged rates over a shuffled-`DataLoader` batch mixing 1/7/30-day horizons. Decoding per window at the window's own horizon gives 58.8 vs 17.1 — an over-forecast of 3.4x |
| a 2-parameter Hawkes-like correction raises N-test pass 5.6% → 53.5% | **superseded** | with a constant scale admitted, the excitation amplitude fits to zero for all three models and the likelihood is identical to scale-only. The correction was a constant: `c0 = 0.326 ≈ 5491/19152`, the declustering ratio |
| L0's T-test standing improves after correction | **closed, negatively** | applied symmetrically and fit prospectively, L0 beats HistoricalRateGR in 0 of 71 windows |
| M-test passes 76–85%, so the magnitude structure is right | **holds, but incomplete** | the M-test passes for every model while the implied moment rate is 4–60x the observed release |

## Results by layer

**L0.** The trained encoder is directionally collapsed: participation ratio 1.09 of 128, PC1 = 95.5%
of variance, mean pairwise cosine 0.958 between issue times, 85% of sketched projections rejecting
Gaussianity, and no dead units (so a per-dimension-std check would have missed it). Replacing the
latent with a constant matches or beats the trained encoder on all four CSEP tests. A 2x2 ablation
shows the cause is the unnormalized 5-feature input map, not the missing isotropy term: the fixed
embedding (normalisation + Fourier for lat/lon/depth + Time2Vec for `dt`) lifts the participation
ratio to 22.4 and the CL pass rate from 49% to 76%, while a LeJEPA-style SIGReg penalty at
gradient-norm-matched weight cannot rescue the original embedding (1.09 → 1.18) and adds nothing
once the embedding is fixed.

**L0′ (point process).** `model_pp.py` emits a continuous marked intensity with exactly normalized
closed-form factors and trains on the marked-Poisson likelihood of real event coordinates. It
self-calibrates (18.0 vs 17.1 events/window) and makes the moment budget well-posed for the first
time (4.14e18 N·m/yr, 0.38x observed, against 4–60x for every gridded form). Its spatial factor
fails: S-test 9.9% gridded, catalog-track spatial 31% with calibration quantile 2e-46, because
axis-aligned mixture components cannot represent fault-aligned lineaments.

**L1.** Scored with `csep.core.poisson_evaluations` and `csep.core.catalog_evaluations` directly
rather than through floatCSEP's run directories — the same test functions, without the `.dat`
round-trip. Two new tests were written on pycsep's simulate-and-rank pattern: an **E-test** (energy,
low discriminating power by construction) and a **D-test** (depth, which the point-process L0 passes
in 91.5% of windows). pycsep 0.8.0's `_simulate_catalog` needed a one-line patch: its cumulative
sampling weights can end a few ulp below 1.0 on peaked rate maps, which raises `IndexError`.

**L2.** The within-scenario `sigma_ln(Sa)` of 0.331 is unchanged, and reaching the empirical
NGA-West2 total range 0.5–0.7 requires a missing 0.375–0.617 in quadrature from terms the physics
randomisation does not touch. That shortfall is worth **+46%** on CVaR_0.99. The GMM code itself was
never committed and is not recoverable from the repo; the saved scenario ensemble refits exactly as
`log Sa = a + b·M + c·log R + d·R` (R² = 1.000 at T = 0.2 s) and was used as the surrogate.

**L3 / L4.** Depth is a material lever: real hypocentral distances cut CVaR_0.99 by 33% relative to
epicentral distances (equivalently, surface-source treatment inflates the tail 49%), and the effect
is entirely near-field. Count dispersion is not a lever: CVaR moves 0.97–1.07x across NB `r` from
0.25 to infinity, because the tail is mark-driven. These are lower bounds — the ductility map is the
linear limit `mu_d = Sa/eta`.

## Hypotheses

| | statement | verdict |
|---|---|---|
| H1 | the JEPA representation beats statistical baselines on L1 consistency | **refuted on this target, and unfalsifiable there** |
| H6 | a continuous intensity passes the catalog track where the gridded form cannot | **not supported as implemented** (spatial factor) |
| H7 | real depth closes part of the L2 sigma gap | **partially**: depth matters (49%) but acts through distance, not sigma |
| H9 | on the undeclustered catalog at sub-week horizons a JEPA latent beats a fitted excitation feature | **new, now falsifiable** — baselines measured per region |
| H10 | an anisotropic spatial decoder recovers the gridded S-test while keeping well-posed moment content | **new** |

## The binding constraint is the target, not the architecture

Monthly counts of declustered M2.8+ California events have lag-1 autocorrelation −0.08 and
variance/mean 0.92, and no model correlates positively with them. Across California, Japan, Chile and
Italy–Greece at two thresholds each, an exponential-decay excitation feature buys 0.017–0.22 nats per
event out of sample on the **raw** catalogs and 0.0001–0.006 after declustering. Declustering removes
precisely the signal L0 is being asked to learn. Any further work on the encoder should follow, not
precede, moving the target (M11).

## Caveats carried forward

1. The ductility map in every CVaR number here is the linear limit; §4.1 found a degrading oscillator
   roughly doubles CVaR_0.99, so all tail sensitivities are lower bounds.
2. The site-risk CVaR values are small in absolute terms (one month of M2.8+ seismicity rarely
   threatens a given site); only the ratios are meaningful.
3. The moment-rate reference is the catalog's own release, not an independent geodetic deformation
   model. An external budget would strengthen the constraint and is the natural anchor for M13.
4. SIGReg was evaluated at one weight (gradient-norm matched at initialisation). A weight sweep might
   find a setting that helps, but the arm-C result makes it unlikely to be the binding factor.
5. Time-varying `M_c(t)` is still not implemented; threshold sensitivity was tested instead.
