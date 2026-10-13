"""Step 3a: precompute every CatalogWindowDataset sample once, vectorized.

Reproduces data.py's sample semantics exactly (verified against the original class) but
with searchsorted instead of per-event list comprehensions, so the 2x2 ablation can
reuse one cache instead of rebuilding tensors for every arm.

Two count targets are stored per sample:
  counts_raw : future-window counts from the aftershock-inclusive catalog (what the
               existing checkpoint was trained on)
  counts_ind : future-window counts from the declustered independent catalog (what
               floatCSEP actually scores)
"""
from __future__ import annotations
import os, sys
from datetime import datetime, timedelta, timezone
import numpy as np, pandas as pd, torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in ("repo/src", os.path.join(_HERE, "..", "..", "src"),
           os.path.join(_HERE, "repo", "src"), os.path.join(_HERE, "src")):
    if os.path.isdir(os.path.join(_c, "cvar_psha")):
        sys.path.insert(0, os.path.abspath(_c))
        break
else:
    raise SystemExit("cannot locate the cvar_psha package: looked for src/cvar_psha "
                     "beside this script, two levels up, and under ./repo/src")
from cvar_psha.eq_jepa.catalog import EarthquakeEvent
from cvar_psha.eq_jepa.data import CatalogWindowDataset, RegionGrid

CAT = "repo/results/l0_l3/data/california_catalog_declustered_vetted.csv"
GRID = dict(min_lat=32.0, max_lat=42.0, min_lon=-125.0, max_lon=-114.0,
            n_lat=20, n_lon=22, magnitude_edges=(2.8, 3.5, 4.5, 5.5, 8.5))
HIST_DAYS, LONG_MONTHS, MAX_EVENTS = 30, 120, 256
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

def days(ts):
    """pandas UTC timestamps -> float days since epoch.

    Resolution-agnostic on purpose: pandas 3.0 gives to_datetime microsecond dtype, so
    dividing a raw astype('int64') by a nanosecond constant is wrong by 1000x.
    """
    return ((ts - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(days=1)).to_numpy()

class FastWindows:
    def __init__(self, df, grid):
        self.grid = grid
        self.t = days(df.Time); self.la = df.Latitude.to_numpy(); self.lo = df.Longitude.to_numpy()
        self.dz = df["Depth/km"].to_numpy(); self.m = df.Magnitude.to_numpy()
        self.mom = 10.0 ** (1.5 * self.m) / 1e18
        self.cmom = np.concatenate([[0.0], np.cumsum(self.mom)])
        inside = ((self.la >= grid.min_lat) & (self.la <= grid.max_lat) &
                  (self.lo >= grid.min_lon) & (self.lo <= grid.max_lon))
        i = np.clip(((self.la - grid.min_lat) / (grid.max_lat - grid.min_lat) * grid.n_lat).astype(int),
                    0, grid.n_lat - 1)
        j = np.clip(((self.lo - grid.min_lon) / (grid.max_lon - grid.min_lon) * grid.n_lon).astype(int),
                    0, grid.n_lon - 1)
        self.cell = np.where(inside, i * grid.n_lon + j, -1)
        mi = np.searchsorted(np.asarray(grid.magnitude_edges), self.m, side="right") - 1
        self.magi = np.where((mi >= 0) & (mi < grid.n_mag), mi, -1)

    def slice(self, t0, t1):
        a, b = np.searchsorted(self.t, [t0, t1])
        return a, b

    def tokens(self, t_issue, a, b, backward=True):
        """Event tokens for events [a,b), last MAX_EVENTS kept (matches data.py)."""
        a = max(a, b - MAX_EVENTS)
        n = b - a
        tok = np.zeros((MAX_EVENTS, 5), np.float32); mask = np.zeros(MAX_EVENTS, bool)
        if n > 0:
            dt = (t_issue - self.t[a:b]) if backward else (self.t[a:b] - t_issue)
            tok[:n] = np.stack([dt, self.la[a:b], self.lo[a:b], self.dz[a:b], self.m[a:b]], 1)
            mask[:n] = True
        return tok, mask

    def long_history(self, t_issue):
        ends = t_issue - 30.0 * (LONG_MONTHS - np.arange(LONG_MONTHS) - 1)
        starts = ends - 30.0
        ia = np.searchsorted(self.t, starts); ib = np.searchsorted(self.t, ends)
        return np.stack([(ib - ia).astype(np.float32), (self.cmom[ib] - self.cmom[ia]).astype(np.float32)], 1)

    def counts(self, a, b):
        c = np.zeros((self.grid.n_cells, self.grid.n_mag), np.float32)
        cell, magi = self.cell[a:b], self.magi[a:b]
        ok = (cell >= 0) & (magi >= 0)
        if ok.any():
            np.add.at(c, (cell[ok], magi[ok]), 1.0)
        return c

def main():
    df = pd.read_csv(CAT); df["Time"] = pd.to_datetime(df["Time"], format="mixed", utc=True)
    df = df.sort_values("Time").reset_index(drop=True)
    grid = RegionGrid(**GRID)
    raw = FastWindows(df, grid)
    ind = FastWindows(df[df.is_mainshock.astype(bool)].reset_index(drop=True), grid)

    def build(t_issue_dt, horizon_days):
        t0 = (t_issue_dt - EPOCH).total_seconds() / 86400.0
        t1 = t0 + float(horizon_days)
        pa, pb = raw.slice(t0 - HIST_DAYS, t0)
        fa, fb = raw.slice(t0, t1)
        ia, ib = ind.slice(t0, t1)
        tok, mask = raw.tokens(t0, pa, pb, backward=True)
        ftok, fmask = raw.tokens(t0, fa, fb, backward=False)
        long = raw.long_history(t0)
        flong = np.zeros_like(long)
        flong[-1] = [fb - fa, raw.cmom[fb] - raw.cmom[fa]]
        return dict(events=tok, event_mask=mask, long_history=long,
                    future_events=ftok, future_mask=fmask, future_long_history=flong,
                    counts_raw=raw.counts(fa, fb), counts_ind=ind.counts(ia, ib),
                    horizon_days=np.float32(horizon_days))

    # fidelity check against the original class on a few (issue time, horizon) pairs
    events = [EarthquakeEvent(t.to_pydatetime(), la, lo, dz, m, e) for t, la, lo, dz, m, e in
              zip(df.Time, df.Latitude, df.Longitude, df["Depth/km"], df.Magnitude, df.EventID)]
    for t_chk, h_chk in [(datetime(2013, 5, 1, tzinfo=timezone.utc), 30),
                         (datetime(2019, 7, 1, tzinfo=timezone.utc), 7),
                         (datetime(2016, 2, 1, tzinfo=timezone.utc), 1)]:
        ref = CatalogWindowDataset(events, grid, [t_chk], history_days=HIST_DAYS,
                                   long_months=LONG_MONTHS, horizon_days=h_chk,
                                   max_events=MAX_EVENTS)[0]
        got = build(t_chk, h_chk)
        for k in ("events", "event_mask", "long_history", "future_events", "future_mask",
                  "future_long_history"):
            r = ref[k].numpy()
            assert np.allclose(r, got[k], atol=2e-3), f"{k} mismatch at {t_chk} h={h_chk}"
        assert np.allclose(ref["counts"].numpy(), got["counts_raw"]), f"counts mismatch {t_chk}"
    print("fidelity check vs CatalogWindowDataset passed", flush=True)

    def month_starts(y0, m0, y1, m1):
        out, y, m = [], y0, m0
        while (y, m) <= (y1, m1):
            out.append(datetime(y, m, 1, tzinfo=timezone.utc)); y, m = (y+1, 1) if m == 12 else (y, m+1)
        return out
    train_times = month_starts(2010, 1, 2018, 12)
    all_times = month_starts(2010, 1, 2024, 11)
    win_len = [ (datetime(t.year + (t.month == 12), t.month % 12 + 1, 1, tzinfo=timezone.utc) - t).days
                for t in all_times ]

    train = [build(t, h) for h in (1, 7, 30) for t in train_times]
    evalw = [build(t, h) for t, h in zip(all_times, win_len)]
    def stack(rows):
        out = {}
        for k in rows[0]:
            arr = np.stack([r[k] for r in rows])
            out[k] = torch.from_numpy(arr) if arr.dtype != bool else torch.from_numpy(arr)
        return out
    torch.save(dict(train=stack(train), eval=stack(evalw),
                    train_times=[t.isoformat() for t in train_times],
                    eval_times=[t.isoformat() for t in all_times],
                    eval_horizons=win_len), "samples_cache.pt")
    print(f"cached {len(train)} training samples, {len(evalw)} evaluation windows", flush=True)

if __name__ == "__main__":
    main()
