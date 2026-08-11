"""Versioned Parquet artifacts with a JSON provenance sidecar.

Writes ``<name>_<version>.parquet`` plus ``<name>_<version>.meta.json`` capturing row
count, sha256, git commit, generated-at, and arbitrary build metadata (e.g. the filter
config). The sidecar mirrors what the Supabase `artifacts` / `dataset_versions` tables
(STU-46) will record, so a Parquet file on disk is self-describing and reconstructible.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import polars as pl


def current_git_commit(default: str | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return default


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_versioned_parquet(
    df: pl.DataFrame,
    out_dir: str | Path,
    name: str,
    version: str,
    *,
    meta: dict | None = None,
    git_commit: str | None = None,
) -> Path:
    """Write a versioned Parquet file + sidecar; return the Parquet path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{name}_{version}.parquet"
    df.write_parquet(parquet_path)

    sidecar = {
        "name": name,
        "version": version,
        "parquet": parquet_path.name,
        "row_count": df.height,
        "columns": df.columns,
        "sha256": _sha256(parquet_path),
        "size_bytes": parquet_path.stat().st_size,
        "git_commit": git_commit if git_commit is not None else current_git_commit(),
        "generated_at": datetime.now(UTC).isoformat(),
        "meta": meta or {},
    }
    meta_path = out_dir / f"{name}_{version}.meta.json"
    meta_path.write_text(json.dumps(sidecar, indent=2))
    return parquet_path
