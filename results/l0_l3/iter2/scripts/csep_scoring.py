"""Shared pycsep scoring helpers for the EQ-JEPA L0 grid.

Scores gridded rate arrays with the same functions floatCSEP's tests.yml wraps
(csep.core.poisson_evaluations), without needing floatCSEP's run-directory scaffolding.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from csep.core.regions import CartesianGrid2D
from csep.core.forecasts import GriddedForecast
from csep.core.catalogs import CSEPCatalog
from csep.core import poisson_evaluations as pe

def _simulate_catalog_clipped(num_events, sampling_weights, sim_fore, random_numbers=None):
    """pycsep 0.8.0's _simulate_catalog, with the insertion index clipped in range.

    Upstream builds sampling_weights = cumsum(rates)/sum(rates) and does
    searchsorted(..., side='right'); when the final cumulative value lands a few ulp
    below 1.0 (which happens for strongly peaked rate maps) a draw in that gap returns
    index == n_cells and raises IndexError. Clipping assigns that float-error-sized
    probability mass to the last bin, where it belongs.
    """
    if random_numbers is None:
        random_numbers = np.random.rand(num_events)
    sim_fore.fill(0)
    pnts = np.searchsorted(sampling_weights, random_numbers, side="right")
    np.add.at(sim_fore, np.clip(pnts, 0, len(sampling_weights) - 1), 1)
    assert sim_fore.sum() == num_events, "simulated the wrong number of events!"
    return sim_fore

pe._simulate_catalog = _simulate_catalog_clipped

ALPHA, NSIM = 0.05, 1000
MAGS = np.array([2.8, 3.5, 4.5, 5.5])
N_LAT, N_LON, DH, MIN_LAT, MIN_LON = 20, 22, 0.5, 32.0, -125.0
CAT = "repo/results/l0_l3/data/california_catalog_declustered_vetted.csv"

def build_region():
    origins = np.array([[MIN_LON + (c % N_LON) * DH, MIN_LAT + (c // N_LON) * DH]
                        for c in range(N_LAT * N_LON)])
    return CartesianGrid2D.from_origins(origins, dh=DH, magnitudes=MAGS)

def load_independent_catalog():
    df = pd.read_csv(CAT)
    df["Time"] = pd.to_datetime(df["Time"], format="mixed", utc=True)
    df = df[df["is_mainshock"].astype(bool)]
    return df[(df.Latitude >= MIN_LAT) & (df.Latitude < MIN_LAT + N_LAT * DH) &
              (df.Longitude >= MIN_LON) & (df.Longitude < MIN_LON + N_LON * DH) &
              (df.Magnitude >= MAGS[0])]

def catalog_window(df, region, t0, t1):
    sub = df[(df.Time >= t0) & (df.Time < t1)]
    arr = np.zeros(len(sub), dtype=CSEPCatalog.dtype)
    arr["longitude"] = sub.Longitude.to_numpy(); arr["latitude"] = sub.Latitude.to_numpy()
    arr["magnitude"] = sub.Magnitude.to_numpy(); arr["depth"] = sub["Depth/km"].to_numpy()
    # epoch MILLISECONDS; pandas 3.0 datetimes are microsecond-resolution, so divide a
    # Timedelta rather than an astype('int64') view
    arr["origin_time"] = ((sub.Time - pd.Timestamp("1970-01-01", tz="UTC"))
                          // pd.Timedelta(milliseconds=1)).to_numpy()
    arr["id"] = [str(x).encode() for x in sub.EventID]
    return CSEPCatalog(data=arr, region=region)

def score_variants(variants: dict[str, np.ndarray], starts, ends, idx, nsim: int = NSIM,
                   pairs: list[tuple[str, str]] | None = None, verbose: bool = True):
    """variants: name -> (n_windows, n_cells, n_mag) expected counts, already rescaled."""
    region, df = build_region(), load_independent_catalog()
    rows, trows = [], []
    for k, w in enumerate(idx):
        cat = catalog_window(df, region, starts[w], ends[w])
        fc = {nm: GriddedForecast(data=r[w], region=region, magnitudes=MAGS,
                                  start_time=starts[w].to_pydatetime(),
                                  end_time=ends[w].to_pydatetime())
              for nm, r in variants.items()}
        ws = starts[w].strftime("%Y-%m-%d")
        for nm, f in fc.items():
            nt = pe.number_test(f, cat); d1, d2 = nt.quantile
            rows.append(dict(window_start=ws, model=nm, test="N", q1=float(d1), q2=float(d2),
                             forecast=float(f.event_count), observed=float(cat.event_count),
                             passes=bool(d1 >= ALPHA / 2 and d2 >= ALPHA / 2)))
            if cat.event_count:
                for tn, fn in (("S", pe.spatial_test), ("M", pe.magnitude_test),
                               ("CL", pe.conditional_likelihood_test)):
                    r = fn(f, cat, num_simulations=nsim, seed=1000 + w)
                    rows.append(dict(window_start=ws, model=nm, test=tn, q1=float(r.quantile),
                                     q2=np.nan, forecast=float(f.event_count),
                                     observed=float(cat.event_count),
                                     passes=bool(r.quantile >= ALPHA)))
        for a, b in (pairs or []):
            t = pe.paired_t_test(fc[a], fc[b], cat); lo, hi = t.test_distribution
            trows.append(dict(window_start=ws, model=a, reference=b,
                              information_gain=float(t.observed_statistic),
                              ci_low=float(lo), ci_high=float(hi),
                              verdict="better" if lo > 0 else ("worse" if hi < 0 else "tied")))
        if verbose and k % 20 == 0: print(f"  scored {k+1}/{len(idx)}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(trows)

def pass_rates(scores: pd.DataFrame) -> pd.DataFrame:
    return scores.pivot_table(index="model", columns="test", values="passes",
                              aggfunc="mean")[["N", "S", "M", "CL"]] * 100
