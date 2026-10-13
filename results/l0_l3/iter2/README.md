# L0–L3, second iteration — results

Addresses the ten open defects listed for the first build. Narrative and revised hypotheses:
[`docs/l0_l3/EQ-JEPA_sequential_definition.md` §10](../../../docs/l0_l3/EQ-JEPA_sequential_definition.md)
and [`docs/l0_l3/L0_L3_verification_report_v2.md`](../../../docs/l0_l3/L0_L3_verification_report_v2.md).

**Two numbers from the first build are superseded**: the reported 3.5x rate *under*-forecast was an
artifact of `train.py`'s `write_forecast` (which averages rates over a shuffled-`DataLoader` batch
mixing the 1/7/30-day horizons), and the two-parameter "Hawkes-like" rate correction was fitting a
constant — the declustering ratio between the training catalog (19,152 raw events) and the scoring
catalog (5,491 independent). Do not reuse the §8 numbers.

## New code

| path | what it is |
|---|---|
| `src/cvar_psha/eq_jepa/model_v2.py` | normalized + Fourier/Time2Vec event embedding, log-compressed long-history channels, and a differentiable Epps-Pulley SIGReg isotropy penalty. Additive: `model.py` is untouched |
| `src/cvar_psha/eq_jepa/model_pp.py` | continuous marked point-process decoder `λ(t,x,y,z,m)` with exactly normalized truncated-Gaussian mixture in (lon, lat, depth) and a truncated-GR magnitude law; exact grid integration and catalog simulation |

## Scripts (`scripts/`, run in order)

`step1_forecasts.py` → `step1_score.py` → `step2_collapse.py` → `step3_build_cache.py` →
`step3_train_ablate.py` → `step3_score.py` → `step4_energy.py` → `step5_build_targets.py` →
`step5_train_pp.py` → `step5_score_pp.py` → `step6_depth.py` → `step7_horizons.py` →
`step8_dispersion.py` → `step9_crossregion.py` → `step10_deck.py`.
`csep_scoring.py` holds the shared scoring helpers.

Two environment notes that will bite a re-run:

- pandas 3.0 gives `to_datetime` **microsecond** resolution, so converting to epoch units via
  `astype('int64')` with a nanosecond constant is wrong by 1000x — divide a `pd.Timedelta`.
- pycsep 0.8.0's `poisson_evaluations._simulate_catalog` raises `IndexError` on strongly peaked rate
  maps (its cumulative sampling weights can end a few ulp below 1.0). `csep_scoring.py` monkeypatches
  it with a version that clips the insertion index.

Scoring goes through `csep.core.poisson_evaluations` and `csep.core.catalog_evaluations` directly —
the same functions floatCSEP's `tests.yml` wraps — rather than through floatCSEP's run directories.

## Data (`data/`)

Per-window CSEP test results, T-tests and summaries for every model variant (`step1_*`, `step3_*`,
`step5_*`); latent collapse diagnostics and the frozen-encoder control (`step2_*`); moment-rate
budget, E-test and b-values (`step4_*`); the new depth test, the `R_hyp` propagation and the L2 fit
(`step6_*`); horizon coherence, additivity, PIT and temporal skill (`step7_*`); per-cell dispersion
and the CVaR sensitivity tables (`step8_*`); and the four-region predictability table (`step9_*`).

## Figures (`figures/`)

One per step, matching the deck
[`docs/l0_l3/EQ-JEPA_next-iteration_findings.pptx`](../../../docs/l0_l3/EQ-JEPA_next-iteration_findings.pptx).

## Checked in vs regenerable

`step5_ckpt_pointprocess.pt` is the trained point-process L0 and is checked in. Left out as
regenerable from the scripts above: the four ablation checkpoints (`step3_ckpt_*.pt`, ~8 MB each),
the window-sample cache (`samples_cache.pt`), the per-window forecast arrays
(`forecasts_step1.npz`, `step3_arms.npz`, `step5_pp_forecast.npz`) and the 500-catalog simulation
set (`step5_pp_simulations.npz`). Ask if you want them packaged as a release asset.
