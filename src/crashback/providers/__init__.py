"""Provider-neutral market-data interfaces and concrete adapters.

Downstream stages import the canonical schemas and the ``MarketDataProvider`` interface
from here — never a vendor's column names. Vendor-specific knowledge is confined to
``normalize`` (pure mapping functions) and the concrete adapters. (STU-45)
"""
from __future__ import annotations

from crashback.providers import normalize, schemas
from crashback.providers.base import MarketDataProvider
from crashback.providers.schemas import (
    SCHEMAS,
    SchemaError,
    empty_frame,
    validate_schema,
)
from crashback.providers.synthetic import SyntheticProvider

__all__ = [
    "MarketDataProvider",
    "SyntheticProvider",
    "SCHEMAS",
    "SchemaError",
    "empty_frame",
    "validate_schema",
    "normalize",
    "schemas",
]

# NOTE: WRDSProvider is intentionally NOT imported here so that importing the interface
# never requires the `wrds` driver. Import it explicitly:
#     from crashback.providers.wrds_provider import WRDSProvider
