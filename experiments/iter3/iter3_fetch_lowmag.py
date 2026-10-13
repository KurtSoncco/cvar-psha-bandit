"""Iteration 3, step 1a: a California catalog down to M2.0 for estimating Mc(t).

The project's working catalogs are cut at M2.5 (query cutoff) and M2.8 (adopted
completeness), so neither can test whether 2.8 was complete in every year. This fetches
down to M2.0 in monthly slices, subdividing any slice that hits the per-request event
cap, so a maximum-curvature Mc(t) estimate has magnitudes below the adopted cutoff to
work with.
"""
from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from urllib.request import urlopen
import pandas as pd

BOX = dict(minlat=32.0, maxlat=42.0, minlon=-125.0, maxlon=-114.0)
MINMAG, LIMIT = 2.0, 20000
COLS = ["id", "time", "lat", "lon", "depth", "author", "catalog", "contributor",
        "contributorID", "magtype", "mag", "magauthor", "place"]

def query(t0: datetime, t1: datetime, depth: int = 0):
    q = {"format": "text", "starttime": t0.isoformat(), "endtime": t1.isoformat(),
         "minlatitude": BOX["minlat"], "maxlatitude": BOX["maxlat"],
         "minlongitude": BOX["minlon"], "maxlongitude": BOX["maxlon"],
         "minmagnitude": MINMAG, "orderby": "time-asc", "limit": LIMIT}
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + urlencode(q)
    for attempt in range(4):
        try:
            with urlopen(url, timeout=180) as r:
                txt = r.read().decode("utf-8")
            break
        except Exception:
            if attempt == 3: raise
            time.sleep(4 * (attempt + 1))
    lines = [l for l in txt.strip().split("\n") if l and not l.startswith("#")]
    if len(lines) >= LIMIT and depth < 4:       # hit the cap: split the interval
        mid = t0 + (t1 - t0) / 2
        return query(t0, mid, depth + 1) + query(mid, t1, depth + 1)
    return [l.split("|") for l in lines]

def main():
    rows, t = [], datetime(2000, 1, 1, tzinfo=timezone.utc)
    end = datetime(2025, 1, 1, tzinfo=timezone.utc)
    while t < end:
        nxt = datetime(t.year + (t.month == 12), t.month % 12 + 1, 1, tzinfo=timezone.utc)
        rows += query(t, nxt)
        if t.month == 1:
            print(f"  through {t:%Y-%m}: {len(rows)} events", flush=True)
        t = nxt
        time.sleep(0.3)
    df = pd.DataFrame(rows, columns=COLS[:len(rows[0])])
    df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
    for c in ("lat", "lon", "depth", "mag"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "mag"]).drop_duplicates("id")
    df = df.sort_values("time").reset_index(drop=True)
    df[["id", "time", "lat", "lon", "depth", "mag", "magtype"]].to_csv(
        "california_m2_catalog.csv", index=False)
    print(f"saved {len(df)} events, M{df.mag.min():.1f}-{df.mag.max():.1f}", flush=True)

if __name__ == "__main__":
    main()
