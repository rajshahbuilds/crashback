"""Config loads, validates, and rejects invalid methodological assumptions."""
from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from crashback import load_config
from crashback.config import DEFAULT_CONFIG_PATH, Config


def test_default_config_loads_and_validates():
    cfg = load_config()
    assert isinstance(cfg, Config)
    # Crash threshold is negative (a crash is a decline).
    assert cfg.crash.threshold < 0
    # Primary target matches the CLAUDE.md contract: +10% close-based within 20 trading days.
    assert cfg.labels.primary.recovery_threshold == 0.10
    assert cfg.labels.primary.horizon_trading_days == 20
    assert cfg.labels.primary.basis == "close"
    # Canonical source decision from STU-43.
    assert cfg.data_source.crsp_format == "ciz"
    # Splits are chronological and non-overlapping.
    assert cfg.splits.train[1] < cfg.splits.validation[0]
    assert cfg.splits.validation[1] < cfg.splits.test[0]


def test_paths_resolve_under_repo_root():
    cfg = load_config()
    raw = cfg.paths.resolve("data_raw")
    assert raw.name == "raw"
    assert raw.is_absolute()


def _default_config_dict() -> dict:
    with open(DEFAULT_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def test_positive_crash_threshold_is_rejected(tmp_path):
    cfg = _default_config_dict()
    cfg["crash"]["threshold"] = 0.10  # invalid: a crash must be negative
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValidationError):
        load_config(bad)


def test_primary_target_must_be_in_grid(tmp_path):
    cfg = _default_config_dict()
    cfg["labels"]["primary"]["horizon_trading_days"] = 999  # not in horizons grid
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValidationError):
        load_config(bad)
