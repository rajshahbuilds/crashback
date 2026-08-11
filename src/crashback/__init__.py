"""crashback — stock-crash recovery prediction research pipeline.

Subpackages follow the pipeline stages (CLAUDE.md sec 8):
  providers    provider-neutral market-data interfaces + WRDS adapter
  ingestion    raw extraction / normalization into canonical schemas
  securities   survivorship-safe security master
  events       crash-event detection
  labels       recovery labels and continuous outcomes
  features     crash-day, pre-crash, recent-crash, market/sector features
  fundamentals point-in-time fundamentals and as-of joins
  datasets     assembly of the canonical event-level modeling dataset
  models       baseline / logistic / gradient-boosted models
  evaluation   calibration, discrimination, robustness
  storage      Parquet + Supabase/Postgres metadata helpers
"""
from __future__ import annotations

__version__ = "0.1.0"

from crashback.config import Config, load_config
from crashback.logging_utils import configure_logging, get_logger

__all__ = ["Config", "__version__", "configure_logging", "get_logger", "load_config"]
