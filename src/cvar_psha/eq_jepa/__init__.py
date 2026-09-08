"""EQ-JEPA: hierarchical, prospective earthquake-rate forecasting."""

from cvar_psha.eq_jepa.catalog import CatalogQuery, EarthquakeEvent, get_catalog_client
from cvar_psha.eq_jepa.model import EQJEPA, EQJEPAConfig

__all__ = ["CatalogQuery", "EarthquakeEvent", "EQJEPA", "EQJEPAConfig", "get_catalog_client"]
