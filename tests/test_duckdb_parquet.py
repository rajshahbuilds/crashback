"""DuckDB can query a Parquet file written by Polars/PyArrow.

This is the core analytics path for the project (bulk bars/feature tables live in
Parquet, queried by DuckDB), so we smoke-test it end to end.
"""
from __future__ import annotations

import duckdb
import polars as pl


def test_duckdb_queries_parquet(tmp_path):
    df = pl.DataFrame(
        {
            "security_id": [1, 1, 2, 2],
            "daily_return": [-0.12, 0.03, -0.11, -0.25],
        }
    )
    path = tmp_path / "sample.parquet"
    df.write_parquet(path)

    con = duckdb.connect()
    n_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
    n_crashes = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path}') WHERE daily_return <= -0.10"
    ).fetchone()[0]
    con.close()

    assert n_rows == 4
    assert n_crashes == 3  # -0.12, -0.11, -0.25
