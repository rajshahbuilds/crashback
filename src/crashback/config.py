"""Typed configuration loading and validation.

The whole pipeline is reproducible from source + config, so every methodological
assumption (crash threshold, horizons, universe filters, splits, missing-data policy)
is declared here as a validated Pydantic model rather than hard-coded downstream.

Usage:
    from crashback.config import load_config
    cfg = load_config()                       # loads configs/default.yaml
    cfg = load_config("configs/experiments/foo.yaml")
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Repository root = two levels up from this file (src/crashback/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CrashConfig(_Base):
    threshold: float = Field(..., description="Daily return <= threshold qualifies as a crash.")

    @field_validator("threshold")
    @classmethod
    def _must_be_negative(cls, v: float) -> float:
        if v >= 0:
            raise ValueError(f"crash.threshold must be negative, got {v}")
        return v


class PrimaryTarget(_Base):
    recovery_threshold: float = Field(..., gt=0)
    horizon_trading_days: int = Field(..., gt=0)
    basis: str = "close"

    @field_validator("basis")
    @classmethod
    def _valid_basis(cls, v: str) -> str:
        if v not in {"close", "intraday_high"}:
            raise ValueError(f"labels.primary.basis must be close|intraday_high, got {v!r}")
        return v


class LabelsConfig(_Base):
    recovery_thresholds: list[float]
    horizons_trading_days: list[int]
    primary: PrimaryTarget

    @field_validator("recovery_thresholds")
    @classmethod
    def _positive_thresholds(cls, v: list[float]) -> list[float]:
        if not v or any(t <= 0 for t in v):
            raise ValueError("labels.recovery_thresholds must be non-empty and all > 0")
        return v

    @field_validator("horizons_trading_days")
    @classmethod
    def _positive_horizons(cls, v: list[int]) -> list[int]:
        if not v or any(h <= 0 for h in v):
            raise ValueError("labels.horizons_trading_days must be non-empty and all > 0")
        return v

    @model_validator(mode="after")
    def _primary_in_grid(self) -> LabelsConfig:
        if self.primary.horizon_trading_days not in self.horizons_trading_days:
            raise ValueError("labels.primary.horizon_trading_days must be in horizons_trading_days")
        if self.primary.recovery_threshold not in self.recovery_thresholds:
            raise ValueError("labels.primary.recovery_threshold must be in recovery_thresholds")
        return self


class UniverseConfig(_Base):
    # CRSP CIZ common-stock filter (see configs/default.yaml). REITs are kept by not
    # filtering on issuer type; foreign-incorporated common is dropped via us_incorporated_only.
    share_types: list[str]
    security_types: list[str]
    security_subtypes: list[str]
    exchanges: list[str]
    us_incorporated_only: bool = True
    min_price: float | None = None
    min_trailing_dollar_volume: float | None = None


class DataSourceConfig(_Base):
    provider: str = "wrds"
    crsp_format: str = "ciz"

    @field_validator("crsp_format")
    @classmethod
    def _valid_format(cls, v: str) -> str:
        if v not in {"ciz", "siz"}:
            raise ValueError(f"data_source.crsp_format must be ciz|siz, got {v!r}")
        return v


class SplitsConfig(_Base):
    train: tuple[date, date]
    validation: tuple[date, date]
    test: tuple[date, date]
    embargo_trading_days: int = Field(0, ge=0)

    @model_validator(mode="after")
    def _ordered_non_overlapping(self) -> SplitsConfig:
        for name, (lo, hi) in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            if lo > hi:
                raise ValueError(f"splits.{name}: start {lo} is after end {hi}")
        if not (self.train[1] < self.validation[0] < self.validation[1] < self.test[0]):
            raise ValueError(
                "splits must be chronological and non-overlapping: train < validation < test"
            )
        return self


class MissingDataConfig(_Base):
    policy: str = "flag"

    @field_validator("policy")
    @classmethod
    def _valid_policy(cls, v: str) -> str:
        if v not in {"flag", "drop", "impute"}:
            raise ValueError(f"missing_data.policy must be flag|drop|impute, got {v!r}")
        return v


class PathsConfig(_Base):
    data_raw: str = "data/raw"
    data_normalized: str = "data/normalized"
    data_processed: str = "data/processed"
    data_events: str = "data/events"
    reports: str = "reports"

    def resolve(self, name: str, root: Path = REPO_ROOT) -> Path:
        """Absolute path for a named directory, rooted at the repo by default."""
        return (root / getattr(self, name)).resolve()


class LoggingConfig(_Base):
    level: str = "INFO"


class Config(_Base):
    crash: CrashConfig
    labels: LabelsConfig
    universe: UniverseConfig
    data_source: DataSourceConfig
    splits: SplitsConfig
    missing_data: MissingDataConfig
    paths: PathsConfig
    logging: LoggingConfig


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate a YAML config into a typed Config (raises on invalid input)."""
    path = Path(path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    with path.open("r") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raise ValueError(f"config file is empty: {path}")
    return Config.model_validate(raw)
