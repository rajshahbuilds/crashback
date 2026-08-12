# crashback — Stock Crash Recovery Prediction

Research pipeline studying **short-term stock-price recovery after large crashes**:

> After a major decline, can we predict which individual companies recover, by how much,
> and how quickly — using only information available at the time of the crash?

V1 is an empirical, calibrated-probability research project (not a trading bot). See
[`CLAUDE.md`](./CLAUDE.md) for the full prediction contract, methodology, and non-goals.

## Local setup

Requires Python **>= 3.12, < 3.14**. WRDS access is validated (see
[`reports/STU-43_wrds_validation.md`](./reports/STU-43_wrds_validation.md)).

> **Not Python 3.14:** `wrds` pins `pandas<2.3`, and pandas <2.3 segfaults on 3.14
> (`pd.to_datetime`). Use 3.12 or 3.13. Create the venv with `python3.12 -m venv .venv`.
>
> **macOS — LightGBM:** needs the OpenMP runtime — `brew install libomp` once
> (otherwise `import lightgbm` fails with a missing `libomp.dylib`).
>
> **macOS + editable install:** if `import crashback` fails after `pip install -e .` (the
> editable `.pth` can get marked hidden, which `site.py` skips), run
> `chflags nohidden .venv/lib/python3.*/site-packages/_editable_impl_crashback.pth`. The test
> suite is unaffected — it adds `src/` to the path via `pyproject.toml`'s `pythonpath`.

```bash
python3 -m venv .venv
source .venv/bin/activate

# install the package (editable) + dev tooling
pip install -e ".[dev]"
# optional: notebooks
pip install -e ".[dev,notebooks]"

pytest                      # run the test suite
```

### WRDS credentials

The WRDS client reads your password from `~/.pgpass`. Set it up interactively once:

```bash
python -c "import wrds; wrds.Connection(wrds_username='YOUR_WRDS_USER')"
# enter password, answer 'y' to create ~/.pgpass
```

Then validate access / refresh the coverage report:

```bash
python scripts/validate_wrds.py --username YOUR_WRDS_USER
python scripts/export_wrds_sample.py --username YOUR_WRDS_USER   # small Parquet sample
```

## Configuration

All methodological assumptions live in [`configs/default.yaml`](./configs/default.yaml)
(crash threshold, recovery targets/horizons, universe filters, chronological splits,
missing-data policy). Load and validate it with a typed model:

```python
from crashback import load_config
cfg = load_config()                              # -> validated crashback.config.Config
cfg = load_config("configs/experiments/foo.yaml")
```

Per-experiment overrides go under `configs/experiments/`.

## Logging

```python
from crashback import get_logger
log = get_logger(__name__)      # consistent timestamped format; configured once per entry point
```

## Directory conventions

```
configs/          default.yaml + experiments/ (all tunable assumptions)
data/
  raw/            provider extracts (gitignored; incl. WRDS sample)
  normalized/     canonical schemas (prices, fundamentals) — gitignored
  processed/      derived feature tables / model matrices — gitignored
  events/         crash-event tables — gitignored
src/crashback/    installable package, one subpackage per pipeline stage:
                  providers, ingestion, securities, events, labels, features,
                  fundamentals, datasets, models, evaluation, storage
scripts/          runnable utilities (WRDS validation, sample export)
notebooks/        thin notebooks that call tested src/ functions
reports/          human-readable findings (e.g. STU-43 validation)
tests/            pytest suite (dangerous paths: crashes, labels, joins, splits)
```

Bulk data lives in **Parquet** (immutable/versioned); high-volume transforms use
**Polars/DuckDB**. Durable relational project state lives in **Supabase/Postgres**.

## Metadata store (Supabase) — what belongs where

**Rule:** Postgres holds *provenance and pointers*; Parquet holds the *bulk data*. Raw daily
bars and wide feature/training matrices never go into Postgres.

| Lives in **Supabase/Postgres** (small, relational) | Lives in **Parquet** (bulk, on disk) |
|---|---|
| dataset / feature / model **versions**, pipeline **runs** | daily price bars, normalized fundamentals |
| evaluation **metrics**, per-event **predictions** | crash-event tables, wide feature tables |
| **artifacts** registry (path + sha256 + row_count) | model matrices / serialized models (files) |
| light **securities_ref** (one row per security) | canonical security master (full history) |

The schema is defined as versioned migrations in [`supabase/migrations/`](./supabase/migrations/)
and applied to the project (ref `xhnzdcmswdeasindpwga`). Any dataset or model run is
reconstructible from its row + the referenced `artifacts` row + `git_commit` + `config`.

Tables: `artifacts`, `dataset_versions`, `feature_versions`, `pipeline_runs`, `model_runs`,
`metrics`, `predictions`, `securities_ref`. RLS is enabled on all with no policies — the
pipeline writes via the **service role** or a direct Postgres connection (both bypass RLS);
the public REST API is denied. Provide credentials via env vars (never committed), e.g.
`SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` or a `DATABASE_URL` connection string.

## Provider interface

Downstream code reads market data through a provider-neutral interface returning canonical
Polars schemas — never vendor column names (see
[`reports/STU-45_provider_interfaces.md`](./reports/STU-45_provider_interfaces.md)):

```python
from datetime import date
from crashback.providers.wrds_provider import WRDSProvider   # live CRSP CIZ + Compustat
from crashback.providers import SyntheticProvider            # in-memory, for tests/offline

p = WRDSProvider(username="YOUR_WRDS_USER")
prices = p.get_daily_prices([14593], date(2008, 9, 1), date(2008, 9, 30))  # canonical daily_price
```

## Status

**M1 — Data Source & Research Foundation:** STU-43 (WRDS validation), STU-44 (repo
skeleton), STU-45 (provider interfaces), STU-46 (Supabase metadata) — done.
**M2 — Security Master & Historical Prices:** STU-47 (survivorship-safe security master)
and STU-48 (85.3M-row daily price history, 1925–2025) done — see
[`reports/STU-47_security_master.md`](./reports/STU-47_security_master.md) and
[`reports/STU-48_daily_prices.md`](./reports/STU-48_daily_prices.md).
**M3 — Crash Events & Recovery Labels:** STU-49 (crash-event detection — 1.15M events,
268k in-universe & liquid) and STU-50 (recovery labels — base rate P(+10% / 20d) ≈ 56% on
the clean pool) done — see [`reports/STU-49_crash_events.md`](./reports/STU-49_crash_events.md)
and [`reports/STU-50_recovery_labels.md`](./reports/STU-50_recovery_labels.md).
**M4 — Feature Engineering:** STU-51 (crash-day & pre-crash price features, point-in-time)
done — see [`reports/STU-51_price_features.md`](./reports/STU-51_price_features.md).
Next: STU-52 (recent-crash history features).
