"""Step 5a: per-window independent-event coordinates for the point-process likelihood.

Same window order as samples_cache.pt (train = [h in (1,7,30)] x [108 monthly issue
times]; eval = 179 monthly windows at their own lengths), so the two files index
together. Stores lon/lat/depth/magnitude of the declustered independent events inside
the modelling domain, padded with a mask.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime, timezone
import numpy as np, pandas as pd, torch
from step3_build_cache import days, CAT, GRID, EPOCH

MAXEV = 512
LON0, LON1, LAT0, LAT1 = -125.0, -114.0, 32.0, 42.0
DEP0, DEP1 = -5.0, 60.0

def main():
    df = pd.read_csv(CAT); df["Time"] = pd.to_datetime(df["Time"], format="mixed", utc=True)
    df = df[df.is_mainshock.astype(bool)].sort_values("Time").reset_index(drop=True)
    df = df[(df.Longitude >= LON0) & (df.Longitude < LON1) & (df.Latitude >= LAT0) &
            (df.Latitude < LAT1) & (df.Magnitude >= 2.8)].reset_index(drop=True)
    dep = df["Depth/km"].to_numpy().clip(DEP0 + 1e-3, DEP1 - 1e-3)
    t = days(df.Time); lon = df.Longitude.to_numpy(); lat = df.Latitude.to_numpy()
    mag = df.Magnitude.to_numpy()
    print(f"{len(df)} independent events inside the domain", flush=True)

    def window(t_issue: datetime, horizon: float):
        t0 = (t_issue - EPOCH).total_seconds() / 86400.0
        a, b = np.searchsorted(t, [t0, t0 + horizon])
        n = min(b - a, MAXEV)
        arr = np.zeros((MAXEV, 4), np.float32); msk = np.zeros(MAXEV, np.float32)
        if n:
            arr[:n] = np.stack([lon[a:a+n], lat[a:a+n], dep[a:a+n], mag[a:a+n]], 1)
            msk[:n] = 1.0
        return arr, msk, b - a

    def month_starts(y0, m0, y1, m1):
        out, y, m = [], y0, m0
        while (y, m) <= (y1, m1):
            out.append(datetime(y, m, 1, tzinfo=timezone.utc)); y, m = (y+1, 1) if m == 12 else (y, m+1)
        return out
    train_times = month_starts(2010, 1, 2018, 12)
    all_times = month_starts(2010, 1, 2024, 11)
    win_len = [(datetime(x.year + (x.month == 12), x.month % 12 + 1, 1, tzinfo=timezone.utc) - x).days
               for x in all_times]

    def pack(pairs):
        A, M, N = [], [], []
        for tt, hh in pairs:
            a, m, n = window(tt, hh); A.append(a); M.append(m); N.append(n)
        return np.stack(A), np.stack(M), np.array(N)

    tr = pack([(tt, h) for h in (1, 7, 30) for tt in train_times])
    ev = pack(list(zip(all_times, win_len)))
    cache = torch.load("samples_cache.pt", weights_only=False)
    assert tr[0].shape[0] == cache["train"]["events"].shape[0]
    assert ev[0].shape[0] == cache["eval"]["events"].shape[0]
    # the per-window independent counts must match the cached binned targets
    assert np.allclose(ev[2], cache["eval"]["counts_ind"].numpy().sum((1, 2))), "count mismatch"
    np.savez_compressed("pp_targets.npz", train_events=tr[0], train_mask=tr[1], train_n=tr[2],
                        eval_events=ev[0], eval_mask=ev[1], eval_n=ev[2],
                        eval_horizons=np.array(win_len),
                        eval_times=np.array([x.isoformat() for x in all_times]))
    print(f"train windows {tr[0].shape}, eval windows {ev[0].shape}, "
          f"max events in a window {max(ev[2].max(), tr[2].max())}", flush=True)

if __name__ == "__main__":
    main()
