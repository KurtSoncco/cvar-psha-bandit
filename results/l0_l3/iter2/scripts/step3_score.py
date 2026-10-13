"""Step 3c: score the four ablation arms through the CSEP consistency tests, against
the same historical-rate reference used in step 1."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # PYTHONSAFEPATH is set here
import numpy as np, pandas as pd
from csep_scoring import score_variants, pass_rates

z3 = np.load("step3_arms.npz", allow_pickle=False)
z1 = np.load("forecasts_step1.npz", allow_pickle=False)
times = pd.to_datetime([t for t in z3["eval_times"]], utc=True)
starts = times
ends = pd.to_datetime([t for t in z1["ends"]], utc=True)
assert (pd.to_datetime([t for t in z1["starts"]], utc=True) == starts).all(), "window mismatch"
cal = np.where(starts < pd.Timestamp("2019-01-01", tz="UTC"))[0]
tst = np.where(starts >= pd.Timestamp("2019-01-01", tz="UTC"))[0]
obs = z3["obs"]

arms = [k[len("rates_"):] for k in z3.files if k.startswith("rates_")]
variants = {}
for a in arms:
    r = z3[f"rates_{a}"]
    variants[a] = r * (obs[cal].sum() / r[cal].sum())
variants["HistoricalRateGR"] = z1["rates_gr"] * (obs[cal].sum() / z1["rates_gr"][cal].sum())
variants["L0_original_checkpoint"] = z1["rates_l0"] * (obs[cal].sum() / z1["rates_l0"][cal].sum())

pairs = [(a, "HistoricalRateGR") for a in arms]
sc, tt = score_variants(variants, starts, ends, tst, pairs=pairs)
sc.to_csv("step3_window_scores.csv", index=False)
tt.to_csv("step3_ttests.csv", index=False)
pr = pass_rates(sc)
pr.to_csv("step3_pass_rates.csv")
ttv = tt.groupby(["model", "reference"]).agg(
    better=("verdict", lambda s: (s == "better").sum()),
    tied=("verdict", lambda s: (s == "tied").sum()),
    worse=("verdict", lambda s: (s == "worse").sum()),
    mean_information_gain=("information_gain", "mean")).round(4)
ttv.to_csv("step3_ttest_summary.csv")
print(pr.round(1).to_string()); print(); print(ttv.to_string())
