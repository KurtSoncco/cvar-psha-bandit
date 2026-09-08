"""Transparent CSEP-ready baselines; ETASLite is a benchmark, not an ETAS replacement."""
from __future__ import annotations
import numpy as np
from cvar_psha.eq_jepa.catalog import EarthquakeEvent
from cvar_psha.eq_jepa.data import RegionGrid

class HistoricalRateBaseline:
    def __init__(self, grid: RegionGrid, smoothing: float = 0.1): self.grid, self.smoothing = grid, smoothing
    def fit_predict(self, history: list[EarthquakeEvent], horizon_days: float) -> np.ndarray:
        rates = np.full((self.grid.n_cells, self.grid.n_mag), self.smoothing, float)
        if not history: return rates
        duration = max((history[-1].time-history[0].time).total_seconds()/86400, 1.)
        for e in history:
            c, m = self.grid.cell_index(e), self.grid.mag_index(e.magnitude)
            if c is not None and m is not None: rates[c,m] += 1
        return rates/duration*horizon_days

class ETASLiteBaseline(HistoricalRateBaseline):
    """Simple isotropic triggering kernel; use a validated ETAS implementation for publication claims."""
    def __init__(self, grid, k=0.03, alpha=1.0, c_days=0.05, p=1.1, d_km=10.0, q=1.5):
        super().__init__(grid); self.k, self.alpha, self.c_days, self.p, self.d_km, self.q = k, alpha, c_days, p, d_km, q
    def fit_predict(self, history, horizon_days, issue_time=None):
        out = super().fit_predict(history, horizon_days)
        if not history: return out
        now = issue_time or history[-1].time
        for e in history:
            age = max((now-e.time).total_seconds()/86400, 0.0); temporal = (age+self.c_days)**(-self.p)
            productivity = self.k*10**(self.alpha*(e.magnitude-3.0))*temporal*horizon_days
            for i in range(self.grid.n_lat):
                for j in range(self.grid.n_lon):
                    lat=self.grid.min_lat+(i+.5)*(self.grid.max_lat-self.grid.min_lat)/self.grid.n_lat; lon=self.grid.min_lon+(j+.5)*(self.grid.max_lon-self.grid.min_lon)/self.grid.n_lon
                    dist=111*np.hypot(lat-e.latitude, (lon-e.longitude)*np.cos(np.deg2rad(lat))); spatial=(dist+self.d_km)**(-self.q)
                    m=self.grid.mag_index(e.magnitude)
                    if m is not None: out[i*self.grid.n_lon+j,m] += productivity*spatial
        return out
