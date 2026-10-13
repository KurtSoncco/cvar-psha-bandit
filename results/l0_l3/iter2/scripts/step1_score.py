"""Step 1b: fit the Hawkes-like rate correction symmetrically to all three models on a
pre-2019 calibration period, then score every model over the 71 prospective test windows
with pycsep's Poisson consistency tests and paired T-tests.

Deliverables: step1_window_scores.csv, step1_ttests.csv, step1_window_totals.csv,
step1_corrections.json
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from csep.core.regions import CartesianGrid2D
from csep.core.forecasts import GriddedForecast
from csep.core.catalogs import CSEPCatalog
from csep.core import poisson_evaluations as pe

ALPHA = 0.05
NSIM = 1000
MAGS = np.array([2.8, 3.5, 4.5, 5.5])
N_LAT, N_LON, DH, MIN_LAT, MIN_LON = 20, 22, 0.5, 32.0, -125.0
CAT = "repo/results/l0_l3/data/california_catalog_declustered_vetted.csv"

z = np.load("forecasts_step1.npz", allow_pickle=False)
starts = pd.to_datetime([s for s in z["starts"]], utc=True)
ends = pd.to_datetime([s for s in z["ends"]], utc=True)
feat, taus = z["feat"], z["taus"]
RAW = {"EQJEPA_L0": z["rates_l0"], "HistoricalRateGR": z["rates_gr"], "ETASLite": z["rates_etas"]}
obs = z["obs"]
obs_tot = obs.sum((1, 2))
cal = np.where(starts < pd.Timestamp("2019-01-01", tz="UTC"))[0]
tst = np.where(starts >= pd.Timestamp("2019-01-01", tz="UTC"))[0]

def fit_correction(lam_tot, n_obs, idx):
    """Poisson MLE of  mu_w = c0 * Lambda_w * (1 + amp * feat_w(tau)).

    c0 (a constant multiplicative scale, which absorbs any train/score target mismatch)
    has a closed-form Poisson MLE given the excitation term, so it is profiled out and
    only amp is optimized numerically, for each tau on the precomputed grid.
    Applied identically to every model, so no model gets a correction the others don't.
    """
    best = None
    for ti, tau in enumerate(taus):
        f = feat[idx, ti]; lam = lam_tot[idx]; n = n_obs[idx]
        def nll(log_amp):
            g = lam * (1.0 + np.exp(log_amp) * f)
            c0 = n.sum() / max(g.sum(), 1e-12)
            mu = np.clip(c0 * g, 1e-12, None)
            return -(n * np.log(mu) - mu).sum()
        r = minimize_scalar(nll, bounds=(-12, 6), method="bounded")
        if best is None or r.fun < best[0]:
            amp = float(np.exp(r.x))
            g = lam * (1.0 + amp * f)
            best = (float(r.fun), amp, float(tau), ti, float(n.sum() / g.sum()))
    nll, amp, tau, ti, c0 = best
    # null model: constant scale only (amp = 0), for a like-for-like likelihood comparison
    g0 = lam_tot[idx]; c0_only = n_obs[idx].sum() / g0.sum()
    mu0 = np.clip(c0_only * g0, 1e-12, None)
    nll0 = -(n_obs[idx] * np.log(mu0) - mu0).sum()
    return dict(nll=nll, nll_scale_only=float(nll0), amp=amp, tau=tau, tau_index=ti,
                scale_c0=c0, scale_only_c0=float(c0_only))

corr = {}
for name, r in RAW.items():
    lam_tot = r.sum((1, 2))
    corr[name] = {"calibration_2010_2018": fit_correction(lam_tot, obs_tot, cal),
                  "in_sample_2019_2024": fit_correction(lam_tot, obs_tot, tst)}
    c = corr[name]["calibration_2010_2018"]
    print(f"{name}: c0={c['scale_c0']:.4f} amp={c['amp']:.4f} tau={c['tau']:.2f} d  "
          f"(nll {c['nll']:.1f} vs scale-only {c['nll_scale_only']:.1f}; "
          f"calibrated on {len(cal)} pre-2019 windows)", flush=True)

FC = dict(RAW)
for name, r in RAW.items():
    c = corr[name]["calibration_2010_2018"]
    mult = c["scale_c0"] * (1.0 + c["amp"] * feat[:, c["tau_index"]])
    FC[name + "_corrected"] = r * mult[:, None, None]
json.dump(corr, open("step1_corrections.json", "w"), indent=2)

origins = np.array([[MIN_LON + (c % N_LON) * DH, MIN_LAT + (c // N_LON) * DH]
                    for c in range(N_LAT * N_LON)])
region = CartesianGrid2D.from_origins(origins, dh=DH, magnitudes=MAGS)

df = pd.read_csv(CAT)
df["Time"] = pd.to_datetime(df["Time"], format="mixed", utc=True)
df = df[df["is_mainshock"].astype(bool)]
df = df[(df.Latitude >= MIN_LAT) & (df.Latitude < MIN_LAT + N_LAT * DH) &
        (df.Longitude >= MIN_LON) & (df.Longitude < MIN_LON + N_LON * DH) & (df.Magnitude >= MAGS[0])]

def catalog_for(w):
    sub = df[(df.Time >= starts[w]) & (df.Time < ends[w])]
    arr = np.zeros(len(sub), dtype=CSEPCatalog.dtype)
    arr["longitude"] = sub.Longitude.to_numpy(); arr["latitude"] = sub.Latitude.to_numpy()
    arr["magnitude"] = sub.Magnitude.to_numpy(); arr["depth"] = sub["Depth/km"].to_numpy()
    arr["origin_time"] = (sub.Time.astype("int64") // 10**6).to_numpy()
    arr["id"] = [str(x).encode() for x in sub.EventID]
    return CSEPCatalog(data=arr, region=region)

def forecast_for(name, w):
    return GriddedForecast(data=FC[name][w], region=region, magnitudes=MAGS,
                           start_time=starts[w].to_pydatetime(), end_time=ends[w].to_pydatetime())

rows, trows, totals = [], [], []
PAIRS = [("EQJEPA_L0", "HistoricalRateGR"), ("EQJEPA_L0", "ETASLite"),
         ("EQJEPA_L0_corrected", "HistoricalRateGR_corrected"),
         ("EQJEPA_L0_corrected", "ETASLite_corrected")]
for k, w in enumerate(tst):
    cat = catalog_for(w)
    rec = {"window_start": starts[w].strftime("%Y-%m-%d"), "obs_total": int(obs_tot[w])}
    for name in FC:
        f = forecast_for(name, w)
        rec[name] = float(f.event_count)
        nt = pe.number_test(f, cat)
        d1, d2 = nt.quantile
        rows.append(dict(window_start=rec["window_start"], model=name, test="N",
                         observed=float(nt.observed_statistic), forecast=float(f.event_count),
                         q1=float(d1), q2=float(d2), passes=bool(d1 >= ALPHA/2 and d2 >= ALPHA/2)))
        if cat.event_count > 0:
            for tname, fn in (("S", pe.spatial_test), ("M", pe.magnitude_test),
                              ("CL", pe.conditional_likelihood_test)):
                r = fn(f, cat, num_simulations=NSIM, seed=1000 + w)
                rows.append(dict(window_start=rec["window_start"], model=name, test=tname,
                                 observed=float(r.observed_statistic), forecast=float(f.event_count),
                                 q1=float(r.quantile), q2=np.nan, passes=bool(r.quantile >= ALPHA)))
    for a, b in PAIRS:
        t = pe.paired_t_test(forecast_for(a, w), forecast_for(b, w), cat)
        lo, hi = t.test_distribution
        trows.append(dict(window_start=rec["window_start"], model=a, reference=b,
                          information_gain=float(t.observed_statistic), ci_low=float(lo), ci_high=float(hi),
                          verdict="better" if lo > 0 else ("worse" if hi < 0 else "tied")))
    totals.append(rec)
    if k % 10 == 0: print(f"  scored {k+1}/{len(tst)} windows", flush=True)

pd.DataFrame(rows).to_csv("step1_window_scores.csv", index=False)
pd.DataFrame(trows).to_csv("step1_ttests.csv", index=False)
pd.DataFrame(totals).to_csv("step1_window_totals.csv", index=False)
print("wrote step1_window_scores.csv, step1_ttests.csv, step1_window_totals.csv", flush=True)
