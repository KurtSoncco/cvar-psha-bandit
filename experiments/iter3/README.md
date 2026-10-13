# Iteration 3 — H9: does L0 beat ETAS on the undeclustered, sub-week target?

L0's earlier target (declustered catalog, 30-day windows) had no measurable
signal to learn: variance/mean 0.98, lag-1 autocorrelation -0.08, and the
out-of-sample gain available to *any* forecaster was +0.0001 nats/event.
H9 moves the target to the undeclustered California M2.8+ catalog at 1-, 3-
and 7-day horizons, where temporal ETAS earns +0.663 / +0.135 / +0.462
nats/event, so a failure there is informative.

Run order (needs `torch`, `pandas>=2.0`, `scipy`; `src/` on the path is
handled by the scripts):

    python iter3_build_cache.py          # ~1 min -> iter3_cache.pt
    python -u iter3_train_h9_v2.py --epochs 60 --batch 64 \
        --objective mle --stratify 1 --tag gpu_mle_strat

`iter3_build_cache.py` must print 5447 / 1815 / 778 windows with mean counts
2.30 / 6.92 / 16.14 and `truncated = 0`; different numbers mean the catalog
is not the one these results were measured on.

Training arms:

  --objective mle        --stratify 1   the fix: batches stratified by event-count
                                        decile so bursty and quiet windows appear
                                        in every batch. Leaves the likelihood
                                        alone, so it is still the MLE.
  --objective normalized --stratify 1   divides each window's log-likelihood by
                                        (1 + n_events). A different estimator,
                                        not the MLE - a diagnostic of how much
                                        the aftershock bursts were dominating.
  --objective mle        --stratify 0   control: isolates what stratification bought.

Why this was needed: single windows hold up to 1,116 events against a 90th
percentile of 9, so unstratified batches made the per-window marked-Poisson
likelihood swing between -12 and -24 nats across epochs.

Output `iter3_h9_summary_<tag>.csv` carries one row per horizon. H9 is
supported only if `gain_per_event` at `horizon_days = 1` rises well above the
preliminary +0.091 and `l0_minus_etas` closes from -0.572 toward zero.
