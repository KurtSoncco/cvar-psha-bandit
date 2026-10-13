"""Step 1a: regenerate per-window gridded forecasts for L0 and the two baselines.

Writes forecasts_step1.npz with, for every monthly window in 2010-01..2024-11:
  rates_l0, rates_gr, rates_etas : (n_win, n_cells, n_mag) expected counts
  obs_counts                     : (n_win, n_cells, n_mag) observed independent events
  hawkes_feat                    : (n_win,) sum of exp(-dt/tau) over the raw catalog,
                                   stored for a grid of tau values
  window_start/window_end/horizon_days
"""
from __future__ import annotations
import sys, json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "repo/src")
from cvar_psha.eq_jepa.catalog import EarthquakeEvent
from cvar_psha.eq_jepa.data import CatalogWindowDataset, RegionGrid
from cvar_psha.eq_jepa.model import EQJEPA, EQJEPAConfig
from cvar_psha.eq_jepa.baselines import HistoricalRateBaseline, ETASLiteBaseline

CAT = "repo/results/l0_l3/data/california_catalog_declustered_vetted.csv"
CKPT = "repo/results/l0_l3/model/eq_jepa_model.pt"
GRID = dict(min_lat=32.0, max_lat=42.0, min_lon=-125.0, max_lon=-114.0,
            n_lat=20, n_lon=22, magnitude_edges=(2.8, 3.5, 4.5, 5.5, 8.5))

def load_catalog():
    df = pd.read_csv(CAT)
    df["Time"] = pd.to_datetime(df["Time"], format="mixed", utc=True)
    df = df.sort_values("Time").reset_index(drop=True)
    def to_events(sub):
        return [EarthquakeEvent(t.to_pydatetime(), la, lo, dz, m, eid)
                for t, la, lo, dz, m, eid in zip(sub["Time"], sub["Latitude"], sub["Longitude"],
                                                 sub["Depth/km"], sub["Magnitude"], sub["EventID"])]
    raw = to_events(df)                                  # aftershock-inclusive (model input)
    ind = to_events(df[df["is_mainshock"].astype(bool)])  # independent events (scoring target)
    return df, raw, ind

def month_starts(y0, m0, y1, m1):
    out, y, m = [], y0, m0
    while (y, m) <= (y1, m1):
        out.append(datetime(y, m, 1, tzinfo=timezone.utc))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out

def main():
    df, raw, ind = load_catalog()
    print(f"catalog: {len(raw)} events, {len(ind)} independent, Mmax={df.Magnitude.max()}", flush=True)
    grid = RegionGrid(**GRID)
    starts = month_starts(2010, 1, 2024, 11)
    ends = [s + timedelta(days=(datetime(s.year + (s.month == 12), s.month % 12 + 1, 1,
                                         tzinfo=timezone.utc) - s).days) for s in starts]
    horizons = np.array([(e - s).days for s, e in zip(starts, ends)], float)
    n_win, n_c, n_m = len(starts), grid.n_cells, grid.n_mag
    print(f"{n_win} windows, {n_c} cells, {n_m} mag bins", flush=True)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ck["config"] if isinstance(ck.get("config"), EQJEPAConfig) else EQJEPAConfig(n_c, n_m)
    model = EQJEPA(cfg); model.load_state_dict(ck["state_dict"]); model.eval()

    gr, etas = HistoricalRateBaseline(grid), ETASLiteBaseline(grid)
    rates_l0 = np.zeros((n_win, n_c, n_m)); rates_gr = np.zeros_like(rates_l0)
    rates_etas = np.zeros_like(rates_l0); obs = np.zeros_like(rates_l0)
    zlat = np.zeros((n_win, cfg.dim), np.float32)

    ind_times = np.array([e.time for e in ind]); raw_times = np.array([e.time for e in raw])
    for w, (t, h) in enumerate(zip(starts, horizons)):
        ds = CatalogWindowDataset(raw, grid, [t], history_days=30, long_months=120,
                                  horizon_days=int(h), max_events=256)
        sample = {k: v[None] if v.ndim >= 1 else v.reshape(1) for k, v in ds[0].items()}
        sample["horizon_days"] = torch.tensor([float(h)])
        with torch.no_grad():
            z = model.encode(sample)
            r, _ = model.forecast_rates(sample, horizon_days=sample["horizon_days"])
        rates_l0[w] = r[0].numpy(); zlat[w] = z[0].numpy()

        hist_ind = [e for e in ind if e.time < t]                       # all independent history
        hist_raw = [e for e in raw if t - timedelta(days=30) <= e.time < t]
        rates_gr[w] = gr.fit_predict(hist_ind, h)
        rates_etas[w] = etas.fit_predict(hist_ind, h, issue_time=t)

        te = ends[w]
        for e in (e for e in ind if t <= e.time < te):
            c, m = grid.cell_index(e), grid.mag_index(e.magnitude)
            if c is not None and m is not None: obs[w, c, m] += 1
        if w % 20 == 0: print(f"  window {w}/{n_win} {t:%Y-%m} obs={obs[w].sum():.0f} "
                              f"l0={rates_l0[w].sum():.2f} gr={rates_gr[w].sum():.2f} "
                              f"etas={rates_etas[w].sum():.2f}", flush=True)

    # Hawkes-like excitation feature over a grid of decay constants (raw catalog, 60-day lookback)
    taus = np.concatenate([np.linspace(0.5, 20, 40), np.linspace(21, 60, 14)])
    feat = np.zeros((n_win, len(taus)))
    for w, t in enumerate(starts):
        dt = np.array([(t - e.time).total_seconds() / 86400 for e in raw
                       if t - timedelta(days=60) <= e.time < t])
        if dt.size: feat[w] = np.exp(-dt[:, None] / taus[None, :]).sum(0)

    np.savez_compressed("forecasts_step1.npz", rates_l0=rates_l0, rates_gr=rates_gr,
        rates_etas=rates_etas, obs=obs, zlat=zlat, horizons=horizons, taus=taus, feat=feat,
        starts=np.array([s.isoformat() for s in starts]), ends=np.array([e.isoformat() for e in ends]))
    print("saved forecasts_step1.npz", flush=True)

if __name__ == "__main__":
    main()
