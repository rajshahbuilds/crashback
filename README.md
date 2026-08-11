# crashback — Stock Crash Recovery Prediction

Research pipeline studying **short-term stock-price recovery after large crashes**:

> After a major decline, can we predict which individual companies recover, by how much,
> and how quickly — using only information available at the time of the crash?

V1 is an empirical, calibrated-probability research project (not a trading bot). See
[`CLAUDE.md`](./CLAUDE.md) for the full prediction contract, methodology, and non-goals.

## Local setup

Requires Python **>= 3.12** (developed on 3.14). WRDS access is validated (see
[`reports/STU-43_wrds_validation.md`](./reports/STU-43_wrds_validation.md)).

> **macOS:** LightGBM needs the OpenMP runtime. Install it once with `brew install libomp`
> (otherwise `import lightgbm` fails with a missing `libomp.dylib`).

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
**Polars/DuckDB**. Durable relational project state (dataset/model/run metadata) will
live in **Supabase/Postgres** (STU-46).

## Status

Milestone **M1 — Data Source & Research Foundation**. Done: STU-43 (WRDS validation).
In progress: STU-44 (this repo skeleton). Next: STU-45 (provider-neutral interfaces).
