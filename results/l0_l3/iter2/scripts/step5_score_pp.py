"""Step 5c: score the point-process L0 two ways.

(a) Gridded track: integrate the intensity exactly onto the 20x22x4 grid and run the
    same N/S/M/CL battery and paired T-test used in steps 1 and 3, so the redesign is
    comparable with its own gridded predecessor and the baselines.
(b) Catalog track: feed the simulated catalogs to pycsep's catalog_evaluations
    (number, spatial, magnitude, pseudolikelihood) and the calibration test across
    windows -- the test path a depth-aware, ungridded L0 is supposed to live on.
Also recomputes the moment-rate budget under the learned truncated GR.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from csep.core.catalogs import CSEPCatalog
from csep.core.forecasts import CatalogForecast
from csep.core import catalog_evaluations as ce
from csep_scoring import (score_variants, pass_rates, build_region, load_independent_catalog,
                          catalog_window, MAGS)

SEC_PER_YEAR = 365.25 * 86400
z5 = np.load("step5_pp_forecast.npz", allow_pickle=False)
z3 = np.load("step3_arms.npz", allow_pickle=False)
z1 = np.load("forecasts_step1.npz", allow_pickle=False)
sim = np.load("step5_pp_simulations.npz", allow_pickle=False)
times = pd.to_datetime([t for t in z5["eval_times"]], utc=True)
ends = pd.to_datetime([t for t in z1["ends"]], utc=True)
cal = np.where(times < pd.Timestamp("2019-01-01", tz="UTC"))[0]
tst = np.where(times >= pd.Timestamp("2019-01-01", tz="UTC"))[0]
obs = z3["obs"]

# ---------- (a) gridded track ----------
grid = z5["grid"]
c0_pp = float(obs[cal].sum() / grid[cal].sum())
variants = {"EQJEPA_L0_pointprocess": grid * c0_pp,
            "C_fixed_embedding": z3["rates_C_fixed_embedding"] *
                                 (obs[cal].sum() / z3["rates_C_fixed_embedding"][cal].sum()),
            "HistoricalRateGR": z1["rates_gr"] * (obs[cal].sum() / z1["rates_gr"][cal].sum())}
print(f"point-process calibration scale c0 = {c0_pp:.3f}", flush=True)
sc, tt = score_variants(variants, times, ends, tst,
                        pairs=[("EQJEPA_L0_pointprocess", "HistoricalRateGR"),
                               ("EQJEPA_L0_pointprocess", "C_fixed_embedding")])
sc.to_csv("step5_gridded_scores.csv", index=False); tt.to_csv("step5_gridded_ttests.csv", index=False)
pr = pass_rates(sc); pr.to_csv("step5_gridded_pass_rates.csv")
print(pr.round(1).to_string(), flush=True)
print(tt.groupby(["model", "reference"]).verdict.value_counts().unstack(fill_value=0).to_string(),
      flush=True)

# ---------- (b) catalog track ----------
region = build_region(); df = load_independent_catalog()
n_cat = int(sim["n_cat"][0])
rows, results_by_test = [], {k: [] for k in ("number", "spatial", "magnitude", "pseudolikelihood")}
for k, w in enumerate(tst):
    lens, ev = sim[f"len_{w}"], sim[f"ev_{w}"]
    mid = times[w] + (ends[w] - times[w]) / 2
    off, cats = 0, []
    for i, n in enumerate(lens):
        arr = np.zeros(int(n), dtype=CSEPCatalog.dtype)
        if n:
            e = ev[off:off + int(n)]
            arr["longitude"] = e[:, 0]; arr["latitude"] = e[:, 1]
            arr["depth"] = e[:, 2]; arr["magnitude"] = e[:, 3]
            arr["origin_time"] = ((mid - pd.Timestamp("1970-01-01", tz="UTC"))
                                  // pd.Timedelta(milliseconds=1))
            arr["id"] = [f"s{i}_{j}".encode() for j in range(int(n))]
        off += int(n)
        cats.append(CSEPCatalog(data=arr, region=region, catalog_id=i))
    fc = CatalogForecast(catalogs=cats, region=region, n_cat=n_cat,
                         start_time=times[w].to_pydatetime(), end_time=ends[w].to_pydatetime(),
                         name="EQJEPA_L0_pointprocess")
    cat = catalog_window(df, region, times[w], ends[w])
    for nm, fn in (("number", ce.number_test), ("spatial", ce.spatial_test),
                   ("magnitude", ce.magnitude_test), ("pseudolikelihood", ce.pseudolikelihood_test)):
        try:
            r = fn(fc, cat)
            q = r.quantile if np.isscalar(r.quantile) else float(np.atleast_1d(r.quantile)[0])
            rows.append(dict(window_start=times[w].strftime("%Y-%m-%d"), test=nm,
                             observed_statistic=float(np.atleast_1d(r.observed_statistic)[0]),
                             quantile=float(q), passes=bool(0.025 <= float(q) <= 0.975)))
            results_by_test[nm].append(r)
        except Exception as exc:
            rows.append(dict(window_start=times[w].strftime("%Y-%m-%d"), test=nm,
                             observed_statistic=np.nan, quantile=np.nan, passes=False,
                             error=str(exc)[:120]))
    if k % 10 == 0: print(f"  catalog-track window {k+1}/{len(tst)}", flush=True)
cb = pd.DataFrame(rows); cb.to_csv("step5_catalog_scores.csv", index=False)
summ = cb.groupby("test").agg(pass_pct=("passes", lambda s: 100 * s.mean()),
                              median_quantile=("quantile", "median")).round(3)
calib = {}
for nm, res in results_by_test.items():
    if len(res) > 5:
        try:
            cr = ce.calibration_test(res)
            calib[nm] = dict(statistic=float(np.atleast_1d(cr.observed_statistic)[0]),
                             quantile=float(np.atleast_1d(cr.quantile)[0]))
        except Exception as exc:
            calib[nm] = dict(error=str(exc)[:160])
json.dump(calib, open("step5_calibration_test.json", "w"), indent=2)
summ.to_csv("step5_catalog_summary.csv")
print(summ.to_string(), flush=True); print(json.dumps(calib, indent=2), flush=True)

# ---------- moment-rate budget under the learned truncated GR ----------
dur = float((ends[tst[-1]] - times[tst[0]]).total_seconds() / SEC_PER_YEAR)
pp_m0 = float((z5["total"][tst] * z5["e_m0_per_event"][tst]).sum() / dur)
mr = pd.read_csv("step4_moment_rate.csv")
obs_dec = float(mr.loc[mr.source.str.startswith("observed (declustered"), "moment_rate_Nm_per_yr"].iloc[0])
pd.DataFrame([dict(source="EQJEPA_L0_pointprocess (learned truncated GR)",
                   moment_rate_Nm_per_yr=pp_m0, ratio_to_observed=pp_m0 / obs_dec,
                   mmax_learned=float(z5["mmax"][0]),
                   mean_b_learned=float(z5["beta"][tst].mean() / np.log(10)))]
             ).to_csv("step5_moment_rate.csv", index=False)
print(f"point-process implied moment rate = {pp_m0:.3e} N m/yr "
      f"({pp_m0/obs_dec:.2f}x observed); Mmax={float(z5['mmax'][0]):.2f}, "
      f"b={float(z5['beta'][tst].mean()/np.log(10)):.3f}", flush=True)
