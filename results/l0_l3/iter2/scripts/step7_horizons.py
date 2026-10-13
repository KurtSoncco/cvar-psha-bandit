"""Step 7: horizon coherence, temporal additivity, and whether a mean intensity is enough.

Three questions, all answered with the trained point-process L0 (no retrain):

1. Horizon coherence. Decode the same latent at 1, 7, 30, 90 and 365 days and look at
   total(z, delta)/delta. Exact temporal additivity (the time half of invariant I2)
   would make that ratio constant; the learned horizon correction deliberately bends it,
   so the question is by how much.
2. Additivity across issue times. Sum three consecutive one-month forecasts (each
   decoded from its own latent) and compare against a single three-month decode from
   the first issue time. This is the coherence a hierarchical multi-horizon model is
   supposed to satisfy by construction.
3. Is a deterministic intensity enough? The decoder emits a mean, so the predictive
   count law is Poisson(total). Randomized PIT values of the observed counts test that
   directly: a mean-only decoder that is right on average but too narrow shows up as a
   U-shaped PIT, and the variance ratio says by how much. This is the quantitative
   version of the objection that one-step latent regression identifies a conditional
   mean rather than a transition kernel.

Outputs: step7_horizon_coherence.csv, step7_additivity.csv, step7_pit.csv,
step7_longhorizon.csv
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "repo/src")
from datetime import datetime, timedelta, timezone
import numpy as np, pandas as pd, torch
from scipy import stats
from cvar_psha.eq_jepa.model import EQJEPAConfig
from cvar_psha.eq_jepa.model_v2 import build_model
from cvar_psha.eq_jepa.model_pp import PointProcessHead, PPConfig
from step3_build_cache import FastWindows, GRID, EPOCH, CAT, days
from cvar_psha.eq_jepa.data import RegionGrid
from csep_scoring import load_independent_catalog

ck = torch.load("step5_ckpt_pointprocess.pt", map_location="cpu", weights_only=False)
cfg = ck["cfg"]
model = build_model(cfg, v2_embedding=True); model.load_state_dict(ck["encoder"]); model.eval()
head = PointProcessHead(PPConfig(dim=cfg.dim)); head.load_state_dict(ck["head"]); head.eval()

cache = torch.load("samples_cache.pt", weights_only=False)
EV = {k: v for k, v in cache["eval"].items() if torch.is_tensor(v)}
times = pd.to_datetime(cache["eval_times"], utc=True)
tst = np.where(times >= pd.Timestamp("2019-01-01", tz="UTC"))[0]
with torch.no_grad():
    z = model.encode(EV)

# ---------- 1. horizon coherence ----------
HOR = [1.0, 7.0, 30.0, 90.0, 365.0]
rows = []
with torch.no_grad():
    for h in HOR:
        par = head(z, torch.full((z.shape[0],), h))
        tot = par["total"].numpy()
        for w in tst:
            rows.append(dict(window_start=times[w].strftime("%Y-%m-%d"), horizon_days=h,
                             total=float(tot[w]), daily_rate=float(tot[w] / h)))
hc = pd.DataFrame(rows); hc.to_csv("step7_horizon_coherence.csv", index=False)
piv = hc.pivot(index="window_start", columns="horizon_days", values="daily_rate")
print("daily rate implied by each horizon (median over 71 windows):")
print(piv.median().round(4).to_string(), flush=True)
print(f"max/min across horizons, median over windows: "
      f"{float((piv.max(1) / piv.min(1)).median()):.3f}", flush=True)

# ---------- 2. additivity across issue times ----------
add = []
with torch.no_grad():
    par30 = head(z, torch.full((z.shape[0],), 30.0))
    par90 = head(z, torch.full((z.shape[0],), 90.0))
t30, t90 = par30["total"].numpy(), par90["total"].numpy()
for i, w in enumerate(tst[:-2]):
    three = float(t30[w] + t30[w + 1] + t30[w + 2])
    one = float(t90[w])
    add.append(dict(window_start=times[w].strftime("%Y-%m-%d"), sum_of_three_monthly=three,
                    single_three_month=one, residual=three - one,
                    relative_residual=(three - one) / one))
ad = pd.DataFrame(add); ad.to_csv("step7_additivity.csv", index=False)
print(f"additivity: median relative residual {ad.relative_residual.median():+.3f}, "
      f"IQR [{ad.relative_residual.quantile(.25):+.3f}, {ad.relative_residual.quantile(.75):+.3f}]",
      flush=True)

# ---------- 3. predictive distribution ----------
obs = cache["eval"]["counts_ind"].numpy().sum((1, 2))
c0 = float(obs[:len(times) - len(tst)].sum() / t30[:len(times) - len(tst)].sum())
lam = t30 * c0
rng = np.random.default_rng(3)
pit = []
for w in tst:
    k, m = int(obs[w]), float(lam[w])
    lo = stats.poisson.cdf(k - 1, m); hi = stats.poisson.cdf(k, m)
    pit.append(dict(window_start=times[w].strftime("%Y-%m-%d"), observed=k, forecast=m,
                    pit=float(lo + rng.random() * (hi - lo))))
pt = pd.DataFrame(pit); pt.to_csv("step7_pit.csv", index=False)
ks = stats.kstest(pt.pit.to_numpy(), "uniform")
var_ratio = float(pt.observed.var() / pt.forecast.mean())
print(f"PIT uniformity KS p = {ks.pvalue:.4f}; observed variance / mean forecast = "
      f"{var_ratio:.2f} (Poisson predicts 1.0)", flush=True)

# ---------- longer-horizon decode vs observed ----------
df = load_independent_catalog(); tt = days(df.Time)
lh = []
with torch.no_grad():
    for h in (30.0, 90.0, 365.0):
        par = head(z, torch.full((z.shape[0],), h)); tot = par["total"].numpy() * c0
        for w in tst:
            t0 = (times[w].to_pydatetime() - EPOCH).total_seconds() / 86400.0
            a, b = np.searchsorted(tt, [t0, t0 + h])
            if times[w] + pd.Timedelta(days=h) > pd.Timestamp("2024-12-01", tz="UTC"):
                continue
            lh.append(dict(window_start=times[w].strftime("%Y-%m-%d"), horizon_days=h,
                           forecast=float(tot[w]), observed=int(b - a)))
lhd = pd.DataFrame(lh); lhd.to_csv("step7_longhorizon.csv", index=False)
print(lhd.groupby("horizon_days").apply(
    lambda g: pd.Series({"n_windows": len(g), "mean_forecast": g.forecast.mean(),
                         "mean_observed": g.observed.mean(),
                         "ratio": g.forecast.mean() / g.observed.mean(),
                         "corr": np.corrcoef(g.forecast, g.observed)[0, 1]}),
    include_groups=False).round(3).to_string(), flush=True)
