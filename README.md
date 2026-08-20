# crashback — Stock Crash Recovery, Empirically

A survivorship-safe empirical study of what happens to a US stock **after a large single-day
crash** (a daily total return of $-10\%$ or worse):

> After a major decline, do stocks recover — how often, by how much, and how fast — and which
> characteristics separate the ones that rebound from the ones that keep falling?

This is a **descriptive research project**, not a trading system. The deliverable is the paper
in [`paper/crashback.tex`](./paper/crashback.tex), built entirely from the pipeline in this repo.

## What it finds (in one breath)

Buying the dip is a losing bet for the *typical* crashed stock — at one year only ~47% earn
money (median $-5\%$); the positive mean is a fat right tail. Recovery is strongly
**regime-dependent** (the base rate swings from $+21\%$ mean in the 2010s to $-22\%$ median
for 2022+ crashes), and it has **two clocks that disagree**: the one-week bounce favours the
most *oversold* names (small, repeatedly-crashed, crashing into a panic), while durable
one-year recovery favours *cheap, profitable, systemically-crashed* names. The one thing
consistent across horizons is quality — operating losses are the worst bucket at every horizon.

See the paper for the full analysis and figures.

## Repository layout

```
paper/            crashback.tex + figures/ (the writeup and every figure it renders)
configs/          default.yaml + experiments/ — all tunable assumptions
data/             gitignored artifacts (raw / normalized / processed / events); dirs kept via .gitkeep
src/crashback/    installable package, one subpackage per pipeline stage:
                  providers, ingestion, securities, events, labels,
                  features, fundamentals, datasets, analysis, evaluation, storage
scripts/          runnable pipeline steps (build_*), WRDS utilities, and the fig_* figure generators
tests/            pytest suite (dangerous paths: crashes, labels, as-of joins, rolling windows)
```

Bulk data lives in **Parquet** (immutable/versioned); high-volume transforms use
**Polars / DuckDB**.

## Setup

Requires Python **>= 3.12, < 3.14** (the `wrds` client pins `pandas<2.3`, which segfaults on
3.14). Create the venv with `python3.12 -m venv .venv`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # package + pytest/ruff
pip install -e ".[dev,notebooks]"  # optional: notebooks
pytest                            # run the suite
```

> **macOS editable install:** if `import crashback` fails after `pip install -e .` (the editable
> `.pth` can get marked hidden, which `site.py` skips), run
> `chflags nohidden .venv/lib/python3.*/site-packages/_editable_impl_crashback.pth`. The test
> suite is unaffected — `pyproject.toml`'s `pythonpath` adds `src/`.

## Data source (WRDS)

All market data come from **WRDS** — CRSP CIZ tables for securities, prices, and delistings, and
Compustat for point-in-time fundamentals. To avoid survivorship bias the universe is built from
the full historical cross-section of US-listed common stock (including delisted, acquired,
renamed, and bankrupt names), restricted to common shares on NYSE / NYSE American / Nasdaq.
Fundamentals are dated by *public availability* (`rdq`), never by fiscal period end, so nothing is
knowable before it was released.

The WRDS client reads your password from `~/.pgpass`:

```bash
python -c "import wrds; wrds.Connection(wrds_username='YOUR_WRDS_USER')"  # creates ~/.pgpass
python scripts/validate_wrds.py --username YOUR_WRDS_USER                 # validate access
```

Downstream code reads market data only through a provider-neutral interface returning canonical
Polars schemas — never vendor column names:

```python
from datetime import date
from crashback.providers.wrds_provider import WRDSProvider   # live CRSP + Compustat
from crashback.providers import SyntheticProvider            # in-memory, for tests/offline

p = WRDSProvider(username="YOUR_WRDS_USER")
prices = p.get_daily_prices([14593], date(2008, 9, 1), date(2008, 9, 30))
```

## Reproducing the paper

The pipeline runs source → dataset → figures:

```bash
# 1) build the dataset (each writes a versioned Parquet under data/)
python scripts/build_security_master.py
python scripts/ingest_daily_prices.py
python scripts/build_crash_events.py
python scripts/build_recovery_labels.py
python scripts/build_price_features.py
python scripts/build_recent_crash_features.py
python scripts/build_market_sector_features.py
python scripts/ingest_fundamentals.py
python scripts/build_fundamental_features.py
python scripts/build_events_dataset.py      # -> data/processed/events_v1.parquet

# 2) render the figures (into paper/figures/)
python scripts/fig_m0_return_hist.py        # base-rate histograms
python scripts/fig_recovery_by_marketcap.py # and the other fig_recovery_by_* / fig_features_* / fig_m0_2022
...

# 3) build the paper
cd paper && pdflatex crashback.tex
```

All methodological assumptions (crash threshold, recovery targets/horizons, universe and
liquidity filters, missing-data policy) live in
[`configs/default.yaml`](./configs/default.yaml) and are loaded as a typed, validated model:

```python
from crashback import load_config
cfg = load_config()
```

## Principles

Point-in-time correctness is treated as a **correctness bug, not a modeling nuance**: every
feature must be knowable at the crash-day close, corporate actions can never manufacture a false
crash (returns are split/dividend-adjusted), and delisting/bankruptcy terminal returns are kept
so a stock that goes to zero contributes its real loss rather than disappearing. The dangerous
transforms — crash detection, recovery labels, as-of fundamental joins, rolling windows — carry
dedicated tests.
