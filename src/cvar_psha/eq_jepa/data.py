"""Leakage-safe multi-resolution catalogue tensors for EQ-JEPA."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import numpy as np
import torch
from torch.utils.data import Dataset
from cvar_psha.eq_jepa.catalog import EarthquakeEvent


@dataclass(frozen=True)
class RegionGrid:
    min_lat: float; max_lat: float; min_lon: float; max_lon: float
    n_lat: int = 8; n_lon: int = 8; magnitude_edges: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 10.0)
    @property
    def n_cells(self) -> int: return self.n_lat * self.n_lon
    @property
    def n_mag(self) -> int: return len(self.magnitude_edges) - 1
    def cell_index(self, event: EarthquakeEvent) -> int | None:
        if not (self.min_lat <= event.latitude <= self.max_lat and self.min_lon <= event.longitude <= self.max_lon): return None
        i = min(self.n_lat - 1, int((event.latitude-self.min_lat)/(self.max_lat-self.min_lat)*self.n_lat))
        j = min(self.n_lon - 1, int((event.longitude-self.min_lon)/(self.max_lon-self.min_lon)*self.n_lon))
        return i * self.n_lon + j
    def mag_index(self, magnitude: float) -> int | None:
        idx = int(np.searchsorted(self.magnitude_edges, magnitude, side="right") - 1)
        return idx if 0 <= idx < self.n_mag else None


class CatalogWindowDataset(Dataset):
    """Each sample uses events strictly before ``issue_time``; target is future count tensor."""
    def __init__(self, events: list[EarthquakeEvent], grid: RegionGrid, issue_times: list[datetime],
                 history_days: int = 30, long_months: int = 120, horizon_days: int = 1, max_events: int = 256):
        self.events = sorted(events, key=lambda e: e.time)
        self.grid, self.issue_times = grid, issue_times
        self.history_days, self.long_months, self.horizon_days, self.max_events = history_days, long_months, horizon_days, max_events

    def __len__(self) -> int: return len(self.issue_times)
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        t = self.issue_times[index]
        if t.tzinfo is None: t = t.replace(tzinfo=timezone.utc)
        past = [e for e in self.events if t-timedelta(days=self.history_days) <= e.time < t][-self.max_events:]
        future = [e for e in self.events if t <= e.time < t+timedelta(days=self.horizon_days)]
        tokens = np.zeros((self.max_events, 5), np.float32); mask = np.zeros(self.max_events, bool)
        for k, e in enumerate(past):
            tokens[k] = [(t-e.time).total_seconds()/86400, e.latitude, e.longitude, e.depth_km, e.magnitude]; mask[k] = True
        # oldest-to-newest monthly rate/moment summaries; no future information
        long = np.zeros((self.long_months, 2), np.float32)
        for m in range(self.long_months):
            end = t - timedelta(days=30*(self.long_months-m-1)); start = end-timedelta(days=30)
            es = [e for e in self.events if start <= e.time < end]
            long[m] = [len(es), sum(10 ** (1.5*e.magnitude) for e in es) / 1e18]
        counts = np.zeros((self.grid.n_cells, self.grid.n_mag), np.float32)
        for e in future:
            c, mag = self.grid.cell_index(e), self.grid.mag_index(e.magnitude)
            if c is not None and mag is not None: counts[c, mag] += 1
        future_tokens = np.zeros((self.max_events, 5), np.float32); future_mask = np.zeros(self.max_events, bool)
        for k, e in enumerate(future[-self.max_events:]):
            future_tokens[k] = [(e.time-t).total_seconds()/86400, e.latitude, e.longitude, e.depth_km, e.magnitude]; future_mask[k] = True
        # The target branch sees only the held-out future window. Its last
        # summary token makes a non-event window distinguishable from padding.
        future_long = np.zeros_like(long)
        future_long[-1] = [len(future), sum(10 ** (1.5*e.magnitude) for e in future) / 1e18]
        return {"events": torch.from_numpy(tokens), "event_mask": torch.from_numpy(mask), "long_history": torch.from_numpy(long),
                "future_events": torch.from_numpy(future_tokens), "future_mask": torch.from_numpy(future_mask),
                "future_long_history": torch.from_numpy(future_long), "counts": torch.from_numpy(counts)}
