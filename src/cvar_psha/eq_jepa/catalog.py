"""Catalog adapters. Network access is explicit; no client polls continuously."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


@dataclass(frozen=True)
class EarthquakeEvent:
    time: datetime
    latitude: float
    longitude: float
    depth_km: float
    magnitude: float
    event_id: str = ""


@dataclass(frozen=True)
class CatalogQuery:
    start_time: datetime
    end_time: datetime
    min_latitude: float
    max_latitude: float
    min_longitude: float
    max_longitude: float
    min_magnitude: float = 2.5
    max_depth_km: float = 700.0
    limit: int = 20_000

    def fdsn_params(self) -> dict[str, str | int | float]:
        return {
            "format": "text", "starttime": self.start_time.isoformat(), "endtime": self.end_time.isoformat(),
            "minlatitude": self.min_latitude, "maxlatitude": self.max_latitude,
            "minlongitude": self.min_longitude, "maxlongitude": self.max_longitude,
            "minmagnitude": self.min_magnitude, "maxdepth": self.max_depth_km,
            "orderby": "time-asc", "limit": self.limit,
        }


class FDSNEventClient:
    """Minimal standard FDSN event client; returns only canonical event fields."""
    def __init__(self, endpoint: str, timeout_s: int = 60):
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s

    def fetch(self, query: CatalogQuery) -> list[EarthquakeEvent]:
        url = f"{self.endpoint}/query?{urlencode(query.fdsn_params())}"
        with urlopen(url, timeout=self.timeout_s) as response:  # nosec B310: caller selects known catalog endpoint
            text = response.read().decode("utf-8")
        return _parse_fdsn_text(text)


class KikNetCatalogClient:
    """KiK-net/K-NET catalogue adapter for an exported CSV or approved URL.

    NIED's waveform archive has access/usage rules and no stable public FDSN event
    endpoint. This adapter deliberately does not scrape it: provide an exported
    event CSV (or a project-approved CSV URL) with the documented columns.
    """
    required = {"time", "latitude", "longitude", "depth_km", "magnitude"}

    def fetch_csv(self, source: str | Path) -> list[EarthquakeEvent]:
        source = str(source)
        if source.startswith(("https://", "http://")):
            with urlopen(source, timeout=60) as response:  # nosec B310: explicit user URL
                raw = response.read().decode("utf-8")
        else:
            raw = Path(source).read_text(encoding="utf-8")
        rows = csv.DictReader(io.StringIO(raw))
        if not rows.fieldnames or not self.required.issubset(set(rows.fieldnames)):
            raise ValueError(f"KiK-net CSV needs columns {sorted(self.required)}")
        return [
            EarthquakeEvent(_parse_time(r["time"]), float(r["latitude"]), float(r["longitude"]),
                            float(r["depth_km"]), float(r["magnitude"]), r.get("event_id", ""))
            for r in rows
        ]


def get_catalog_client(name: str) -> FDSNEventClient | KikNetCatalogClient:
    """Return the named supported source adapter.

    ``iris`` uses EarthScope/IRIS FDSN; ``epos`` uses the EMSC FDSN endpoint
    integrated with EPOS. ``kiknet`` expects a user-supplied exported CSV.
    """
    key = name.lower()
    if key == "iris":
        return FDSNEventClient("https://service.iris.edu/fdsnws/event/1")
    if key == "epos":
        return FDSNEventClient("https://www.seismicportal.eu/fdsnws/event/1")
    if key in {"kiknet", "k-net", "knet"}:
        return KikNetCatalogClient()
    raise ValueError("source must be iris, epos, or kiknet")


def _parse_time(value: str) -> datetime:
    value = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _parse_fdsn_text(text: str) -> list[EarthquakeEvent]:
    events = []
    for row in csv.reader(io.StringIO(text), delimiter="|"):
        if not row or row[0] == "EventID" or len(row) < 11:
            continue
        try:
            events.append(EarthquakeEvent(_parse_time(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[10]), row[0]))
        except (ValueError, IndexError):
            continue
    return events
