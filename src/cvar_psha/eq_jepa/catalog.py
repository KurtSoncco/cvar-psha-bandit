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


class CSVCatalogClient:
    """Shared CSV-loading logic for catalog sources with no live FDSN feed.

    Subclasses only differ in the error message / doc string, since the
    required standardized column set (``time, latitude, longitude, depth_km,
    magnitude`` plus an optional ``event_id``) is the same either way.
    """
    required = {"time", "latitude", "longitude", "depth_km", "magnitude"}
    label = "CSV"

    def fetch_csv(self, source: str | Path) -> list[EarthquakeEvent]:
        source = str(source)
        if source.startswith(("https://", "http://")):
            with urlopen(source, timeout=60) as response:  # nosec B310: explicit user URL
                raw = response.read().decode("utf-8")
        else:
            raw = Path(source).read_text(encoding="utf-8")
        rows = csv.DictReader(io.StringIO(raw))
        if not rows.fieldnames or not self.required.issubset(set(rows.fieldnames)):
            raise ValueError(f"{self.label} CSV needs columns {sorted(self.required)}")
        return [
            EarthquakeEvent(_parse_time(r["time"]), float(r["latitude"]), float(r["longitude"]),
                            float(r["depth_km"]), float(r["magnitude"]), r.get("event_id", ""))
            for r in rows
        ]


class KikNetCatalogClient(CSVCatalogClient):
    """KiK-net/K-NET catalogue adapter for an exported CSV or approved URL.

    NIED's waveform archive has access/usage rules and no stable public FDSN event
    endpoint. This adapter deliberately does not scrape it: provide an exported
    event CSV (or a project-approved CSV URL) with the documented columns.
    """
    label = "KiK-net"


class LocalCatalogClient(CSVCatalogClient):
    """Adapter for a previously-fetched, standardized-column catalog CSV.

    Use this to retrain/rerun deterministically against a frozen snapshot of a
    live source (e.g. the USGS pull cached for this project) without re-issuing
    network queries on every run. The CSV must already be in the standardized
    ``time, latitude, longitude, depth_km, magnitude[, event_id]`` schema --
    see ``write_standard_csv`` for producing one from a list of events.
    """
    label = "local catalog"


def write_standard_csv(events: list[EarthquakeEvent], path: str | Path) -> None:
    """Write events to the standardized CSV schema ``LocalCatalogClient`` reads back."""
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "latitude", "longitude", "depth_km", "magnitude", "event_id"])
        for e in events:
            w.writerow([e.time.isoformat(), e.latitude, e.longitude, e.depth_km, e.magnitude, e.event_id])


def get_catalog_client(name: str) -> FDSNEventClient | KikNetCatalogClient | LocalCatalogClient:
    """Return the named supported source adapter.

    ``usgs`` uses the USGS ComCat FDSN event service (global, includes the ANSS
    composite catalog used for California); ``epos`` uses the EMSC FDSN endpoint
    integrated with EPOS (Europe-Mediterranean). ``kiknet`` expects a
    user-supplied exported CSV. ``local`` expects a previously-cached,
    standardized-schema CSV (see ``write_standard_csv``/``LocalCatalogClient``).

    NOTE: EarthScope/IRIS retired ``service.iris.edu/fdsnws/event`` (returns
    HTTP 410 Gone as of 2023); the former ``iris`` adapter has been removed.
    Use ``usgs`` for a global/California catalog instead.
    """
    key = name.lower()
    if key == "usgs":
        return FDSNEventClient("https://earthquake.usgs.gov/fdsnws/event/1")
    if key == "epos":
        return FDSNEventClient("https://www.seismicportal.eu/fdsnws/event/1")
    if key in {"kiknet", "k-net", "knet"}:
        return KikNetCatalogClient()
    if key in {"local", "cached"}:
        return LocalCatalogClient()
    raise ValueError("source must be usgs, epos, kiknet, or local")


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
