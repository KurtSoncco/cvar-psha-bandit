"""Iteration 3, step 2a: window samples for the undeclustered 1/3/7-day target.

Same encoder inputs as before (30-day event-token history, 120-month summary history,
both from the undeclustered catalog) but the TARGET is now every event in the forecast
window, not just the independent ones, and windows are 1, 3 or 7 days instead of monthly.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime, timezone
import numpy as np, pandas as pd, torch
sys.path.insert(0, "repo/src")
from step3_build_cache import FastWindows, GRID, EPOCH
from cvar_psha.eq_jepa.data import RegionGrid

HORIZONS = [1.0, 3.0, 7.0]
T_START, T_SPLIT, T_END = "2010-01-01", "2019-01-01", "2024-12-01"
MAXEV_TARGET = 1400
MAG_MIN = 2.8

def days(ts):
    return ((ts - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(days=1)).to_numpy()

def main():
    cat = pd.read_csv("california_m2_catalog.csv")
    cat["time"] = pd.to_datetime(cat["time"], format="mixed", utc=True)
    cat = cat[(cat.mag >= MAG_MIN) & (cat.lat >= 32) & (cat.lat <= 42) &
              (cat.lon >= -125) & (cat.lon <= -114)].sort_values("time").reset_index(drop=True)
    print(f"{len(cat)} undeclustered events M>={MAG_MIN}", flush=True)
    grid = RegionGrid(**GRID)
    df = pd.DataFrame({"Time": cat.time, "Latitude": cat.lat, "Longitude": cat.lon,
                       "Depth/km": cat.depth.fillna(8.0).clip(-5 + 1e-3, 60 - 1e-3),
                       "Magnitude": cat.mag})
    fw = FastWindows(df, grid)
    t_ev = fw.t

    rows = {k: [] for k in ("events", "event_mask", "long_history", "future_events",
                            "future_mask", "future_long_history", "counts", "horizon_days",
                            "target_events", "target_mask")}
    meta = []
    for h in HORIZONS:
        edges = np.arange(days(pd.Series([pd.Timestamp(T_START, tz="UTC")]))[0],
                          days(pd.Series([pd.Timestamp(T_END, tz="UTC")]))[0], h)
        for t0 in edges[:-1]:
            t1 = t0 + h
            pa, pb = fw.slice(t0 - 30.0, t0)
            fa, fb = fw.slice(t0, t1)
            tok, mask = fw.tokens(t0, pa, pb, backward=True)
            ftok, fmask = fw.tokens(t0, fa, fb, backward=False)
            long = fw.long_history(t0)
            flong = np.zeros_like(long)
            flong[-1] = [fb - fa, fw.cmom[fb] - fw.cmom[fa]]
            n = min(fb - fa, MAXEV_TARGET)
            tev = np.zeros((MAXEV_TARGET, 4), np.float32); tmask = np.zeros(MAXEV_TARGET, np.float32)
            if n:
                tev[:n] = np.stack([fw.lo[fa:fa+n], fw.la[fa:fa+n],
                                    fw.dz[fa:fa+n], fw.m[fa:fa+n]], 1)
                tmask[:n] = 1.0
            rows["events"].append(tok); rows["event_mask"].append(mask)
            rows["long_history"].append(long); rows["future_events"].append(ftok)
            rows["future_mask"].append(fmask); rows["future_long_history"].append(flong)
            rows["counts"].append(fw.counts(fa, fb)); rows["horizon_days"].append(np.float32(h))
            rows["target_events"].append(tev); rows["target_mask"].append(tmask)
            meta.append(dict(horizon_days=h, t0=float(t0),
                             window_start=str(pd.Timestamp("1970-01-01", tz="UTC")
                                              + pd.Timedelta(days=float(t0)))[:10],
                             n_events=int(fb - fa), truncated=bool(fb - fa > MAXEV_TARGET)))
    out = {k: torch.from_numpy(np.stack(v)) for k, v in rows.items()}
    md = pd.DataFrame(meta)
    md["is_train"] = md.window_start < T_SPLIT
    torch.save(dict(data=out, meta=md.to_dict("list")), "iter3_cache.pt")
    print(md.groupby("horizon_days").agg(n_windows=("n_events", "size"),
                                         n_train=("is_train", "sum"),
                                         mean_count=("n_events", "mean"),
                                         max_count=("n_events", "max"),
                                         truncated=("truncated", "sum")).round(2).to_string(),
          flush=True)

if __name__ == "__main__":
    main()
