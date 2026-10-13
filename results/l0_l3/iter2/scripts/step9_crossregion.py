"""Step 9: is the absence of monthly predictability specific to this target?

Step 7 showed that monthly counts of declustered M2.8+ California events behave like a
homogeneous Poisson process, which bounds what any L0 can achieve there. That makes the
useful cross-region question not "does L0 win elsewhere" but "where does predictable
signal exist at all". So for each (region, magnitude threshold, aggregation window,
declustered or raw) this measures:

  * dispersion (variance/mean) and the autocorrelation of the count series;
  * whether recent activity predicts the next window -- the correlation between an
    exponential-decay excitation feature over the preceding 60 days and the next
    window's count, which is the signal an ETAS or a JEPA encoder could exploit;
  * the Poisson log-likelihood gain of that excitation predictor over a constant rate,
    evaluated out of sample on the second half of each series.

Catalogs come from USGS ComCat; declustering uses the vetted seismostats
Gardner-Knopoff implementation, the same one that fixed this project's catalog bugs.

Outputs: step9_predictability.csv, step9_catalog_summary.csv
"""
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen
import numpy as np, pandas as pd
from scipy import optimize, special

REGIONS = {
    "California":  dict(minlat=32.0, maxlat=42.0, minlon=-125.0, maxlon=-114.0, mags=(2.8, 3.5)),
    "Japan":       dict(minlat=30.0, maxlat=46.0, minlon=129.0, maxlon=146.0, mags=(4.0, 4.5)),
    "Chile":       dict(minlat=-40.0, maxlat=-17.0, minlon=-75.0, maxlon=-67.0, mags=(4.0, 4.5)),
    "Italy_Greece": dict(minlat=34.0, maxlat=47.0, minlon=6.0, maxlon=29.0, mags=(3.5, 4.0)),
}
WINDOWS_DAYS = [7, 30, 90]
TAU_FEATURE, LOOKBACK = 10.0, 60.0
START, END = "2000-01-01", "2025-01-01"

def fetch_usgs(box, minmag):
    """Year-by-year FDSN text queries to stay under the per-request event cap."""
    frames = []
    for y in range(2000, 2025):
        q = {"format": "text", "starttime": f"{y}-01-01", "endtime": f"{y+1}-01-01",
             "minlatitude": box["minlat"], "maxlatitude": box["maxlat"],
             "minlongitude": box["minlon"], "maxlongitude": box["maxlon"],
             "minmagnitude": minmag, "orderby": "time-asc", "limit": 20000}
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + urlencode(q)
        for attempt in range(3):
            try:
                with urlopen(url, timeout=120) as r:
                    txt = r.read().decode("utf-8")
                break
            except Exception as exc:
                if attempt == 2: raise
                time.sleep(5)
        lines = [l for l in txt.strip().split("\n") if l and not l.startswith("#")]
        if not lines: continue
        rows = [l.split("|") for l in lines]
        frames.append(pd.DataFrame(rows, columns=["id", "time", "lat", "lon", "depth", "author",
                                                  "catalog", "contributor", "contributorID",
                                                  "magtype", "mag", "magauthor", "place",
                                                  "type"][:len(rows[0])]))
        time.sleep(0.5)
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
    for c in ("lat", "lon", "depth", "mag"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["lat", "lon", "mag"]).sort_values("time").reset_index(drop=True)

def decluster(df):
    from seismostats.analysis.declustering import GardnerKnopoffType1, GardnerKnopoffWindow
    gk = GardnerKnopoffType1(GardnerKnopoffWindow())
    cat = pd.DataFrame({"time": df.time, "latitude": df.lat, "longitude": df.lon,
                        "magnitude": df.mag, "depth": df.depth.fillna(10.0)})
    out = gk(cat)
    return np.asarray(out).astype(bool) if not isinstance(out, pd.Series) else out.to_numpy().astype(bool)

def days_since_epoch(ts):
    return ((ts - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(days=1)).to_numpy()

def diagnostics(t_days, win):
    """Counts per window plus the excitation feature evaluated at each window start."""
    t0, t1 = t_days.min(), t_days.max()
    edges = np.arange(t0, t1, win)
    counts = np.histogram(t_days, bins=edges)[0]
    feat = np.empty(len(counts))
    for i, s in enumerate(edges[:-1]):
        sel = t_days[(t_days >= s - LOOKBACK) & (t_days < s)]
        feat[i] = np.exp(-(s - sel) / TAU_FEATURE).sum() if sel.size else 0.0
    return counts.astype(float), feat

def poisson_gain(counts, feat):
    """Out-of-sample Poisson log-lik gain per window of  mu = c0*(1+a*feat)  over a constant."""
    n = len(counts); half = n // 2
    tr_c, tr_f, te_c, te_f = counts[:half], feat[:half], counts[half:], feat[half:]
    if tr_c.sum() < 10 or te_c.sum() < 10: return np.nan, np.nan
    def nll(log_a, c, f):
        g = 1.0 + np.exp(log_a) * f
        c0 = c.sum() / g.sum()
        mu = np.clip(c0 * g, 1e-12, None)
        return -float((c * np.log(mu) - mu - special.gammaln(c + 1)).sum())
    r = optimize.minimize_scalar(nll, bounds=(-12, 6), method="bounded", args=(tr_c, tr_f))
    a = float(np.exp(r.x))
    g_te = 1.0 + a * te_f
    c0 = tr_c.sum() / (1.0 + a * tr_f).sum()
    mu = np.clip(c0 * g_te, 1e-12, None)
    ll_feat = float((te_c * np.log(mu) - mu - special.gammaln(te_c + 1)).sum())
    mu0 = np.full_like(te_c, tr_c.mean())
    ll_const = float((te_c * np.log(mu0) - mu0 - special.gammaln(te_c + 1)).sum())
    return (ll_feat - ll_const) / len(te_c), a

def main():
    rows, summary = [], []
    for region, box in REGIONS.items():
        if region == "California":
            cal = pd.read_csv("repo/results/l0_l3/data/california_catalog_declustered_vetted.csv")
            cal["time"] = pd.to_datetime(cal["Time"], format="mixed", utc=True)
            df = pd.DataFrame({"time": cal.time, "lat": cal.Latitude, "lon": cal.Longitude,
                               "depth": cal["Depth/km"], "mag": cal.Magnitude})
            is_main = cal.is_mainshock.astype(bool).to_numpy()
        else:
            df = fetch_usgs(box, min(box["mags"]))
            print(f"{region}: fetched {len(df)} events M>={min(box['mags'])}", flush=True)
            is_main = decluster(df)
        summary.append(dict(region=region, n_events=len(df), min_mag=float(df.mag.min()),
                            max_mag=float(df.mag.max()), n_independent=int(is_main.sum()),
                            independent_fraction=float(is_main.mean())))
        print(f"{region}: {len(df)} events, {is_main.sum()} independent "
              f"({100*is_main.mean():.1f}%)", flush=True)
        for mag in box["mags"]:
            for kind, sel in (("raw", df.mag >= mag), ("declustered", (df.mag >= mag) & is_main)):
                t = days_since_epoch(df.time[sel.to_numpy()])
                if len(t) < 200: continue
                for win in WINDOWS_DAYS:
                    counts, feat = diagnostics(t, win)
                    if len(counts) < 24: continue
                    x = counts - counts.mean()
                    acf = [float(np.corrcoef(x[:-l], x[l:])[0, 1]) for l in (1, 2, 3)]
                    gain, a = poisson_gain(counts, feat)
                    rows.append(dict(region=region, min_magnitude=mag, catalog=kind,
                                     window_days=win, n_windows=len(counts),
                                     n_events=int(counts.sum()), mean_count=float(counts.mean()),
                                     variance_over_mean=float(counts.var() / max(counts.mean(), 1e-9)),
                                     acf_lag1=acf[0], acf_lag2=acf[1], acf_lag3=acf[2],
                                     corr_feature_next_count=float(np.corrcoef(feat, counts)[0, 1]),
                                     poisson_loglik_gain_per_window=gain, fitted_amplitude=a))
    pd.DataFrame(rows).to_csv("step9_predictability.csv", index=False)
    pd.DataFrame(summary).to_csv("step9_catalog_summary.csv", index=False)
    out = pd.DataFrame(rows)
    print(out[["region", "min_magnitude", "catalog", "window_days", "n_events",
               "variance_over_mean", "acf_lag1", "corr_feature_next_count",
               "poisson_loglik_gain_per_window"]].round(3).to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
