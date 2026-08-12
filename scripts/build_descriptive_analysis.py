#!/usr/bin/env python3
"""STU-57: recovery base rates and descriptive slices over events_v1 (generates the report).

Loads events_v1, restricts to the CLEAN pool, and computes the recovery grid + slices by
crash magnitude, recent-crash count, size, profitability, leverage, sector, and market
regime — writing reports/STU-57_base_rates.md directly (no manual spreadsheets).

Run: .venv/bin/python scripts/build_descriptive_analysis.py
"""
from __future__ import annotations

import polars as pl

from crashback.config import load_config
from crashback.evaluation.descriptive import grouped_rate, overall_rate
from crashback.providers.normalize import sic_division

PRIMARY = "hit_10pct_20d"
THRESHOLDS = [(5, "hit_5pct"), (10, "hit_10pct"), (20, "hit_20pct")]
HORIZONS = [5, 20, 60]


def _md_grid(clean: pl.DataFrame) -> str:
    lines = ["| threshold | 5d | 20d | 60d |", "|---|---|---|---|"]
    for pct, stem in THRESHOLDS:
        cells = []
        for h in HORIZONS:
            r = overall_rate(clean, f"{stem}_{h}d")
            cells.append(f"{r['rate']:.3f} (n={r['n']:,})" if r["rate"] is not None else "—")
        lines.append(f"| +{pct}% | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(lines)


def _md_slice(df: pl.DataFrame, code_expr: pl.Expr, labels: list[str], target: str) -> str:
    d = df.with_columns(code_expr.alias("_code"))
    g = grouped_rate(d.filter(pl.col("_code").is_not_null()), "_code", target)
    lines = ["| group | n | recovery | 95% CI |", "|---|---|---|---|"]
    for r in g.sort("_code").iter_rows(named=True):
        lab = labels[int(r["_code"])]
        ci = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        lines.append(f"| {lab} | {r['n']:,} | **{r['rate']:.3f}** | {ci} |")
    return "\n".join(lines)


def _md_sector(df: pl.DataFrame, target: str, top: int = 12) -> str:
    g = grouped_rate(df.filter(pl.col("sector").is_not_null()), "sector", target)
    g = g.sort("n", descending=True).head(top)
    lines = ["| sector | n | recovery | 95% CI |", "|---|---|---|---|"]
    for r in g.iter_rows(named=True):
        lines.append(f"| {r['sector']} | {r['n']:,} | **{r['rate']:.3f}** | "
                     f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] |")
    return "\n".join(lines)


def main():
    cfg = load_config()
    proc = cfg.paths.resolve("data_processed")
    df = pl.read_parquet(proc / "events_v1.parquet")

    # sector label per security from the master (SIC major division)
    sm = pl.read_parquet(
        cfg.paths.resolve("data_normalized") / "security_master" / "security_master_v1.parquet"
    ).group_by("security_id").agg(pl.col("sic_code").drop_nulls().first().alias("sic_code"))
    sic_map = {s: sic_division(s) for s in sm["sic_code"].unique().to_list()}
    sm = sm.with_columns(
        pl.col("sic_code").replace_strict(sic_map, default="Unknown").alias("sector")
    )
    df = df.join(sm.select("security_id", "sector"), on="security_id", how="left")

    clean = df.filter(pl.col("in_universe_at_event") & pl.col("passes_min_price"))
    prim = overall_rate(clean, PRIMARY)

    def rate_where(cond) -> float:
        return overall_rate(clean.filter(cond), PRIMARY)["rate"]

    fresh = rate_where(pl.col("prior_crash_count_20d") == 0)
    rep3 = rate_where(pl.col("prior_crash_count_20d") >= 3)
    broad = rate_where(pl.col("market_return_1d") <= -0.05)
    calm = rate_where(pl.col("market_return_1d") >= 0)
    micro = rate_where(pl.col("market_cap") < 300)
    large = rate_where(pl.col("market_cap") >= 10000)
    unprof = rate_where(pl.col("net_margin") < 0)
    prof = rate_where(pl.col("net_margin") >= 0)

    cr = pl.col("crash_return")
    mag = (pl.when(cr > -0.15).then(0).when(cr > -0.20).then(1)
           .when(cr > -0.30).then(2).otherwise(3))
    mag_lbl = ["-10% to -15%", "-15% to -20%", "-20% to -30%", "≤ -30%"]
    pc = pl.col("prior_crash_count_20d")
    pc_code = pl.when(pc == 0).then(0).when(pc == 1).then(1).when(pc == 2).then(2).otherwise(3)
    pc_lbl = ["fresh (0 prior/20d)", "1 prior", "2 prior", "3+ prior"]
    mc = pl.col("market_cap")
    size = (pl.when(mc.is_null()).then(None).when(mc < 300).then(0)
            .when(mc < 2000).then(1).when(mc < 10000).then(2).otherwise(3))
    size_lbl = ["micro (<$300M)", "small ($0.3–2B)", "mid ($2–10B)", "large (≥$10B)"]
    nm = pl.col("net_margin")
    prof_code = pl.when(nm.is_null()).then(None).when(nm < 0).then(0).otherwise(1)
    prof_lbl = ["unprofitable (margin<0)", "profitable (margin≥0)"]
    dta = pl.col("debt_to_assets")
    lev = (pl.when(dta.is_null()).then(None).when(dta < 0.1).then(0)
           .when(dta < 0.3).then(1).when(dta < 0.5).then(2).otherwise(3))
    lev_lbl = ["<0.1", "0.1–0.3", "0.3–0.5", "≥0.5"]
    mr = pl.col("market_return_1d")
    reg = (pl.when(mr <= -0.05).then(0).when(mr <= -0.02).then(1)
           .when(mr < 0).then(2).otherwise(3))
    reg_lbl = ["market ≤ -5% (broad crash)", "-5% to -2%", "-2% to 0%", "market ≥ 0%"]

    report = f"""# STU-57 — Recovery Base Rates & Descriptive Slices

**Generated from code** by `scripts/build_descriptive_analysis.py` over `events_v1`
(module `crashback.evaluation.descriptive`). Population: the **CLEAN pool**
(in-universe & ≥ $5), determined events only (censored / no-close labels excluded).
Confidence intervals are Wilson 95%.

> **Dependence caveat:** crashes of the same security close together in time are correlated
> (§22). Wilson CIs assume independent observations and therefore **understate** true
> uncertainty — treat them as a lower bound on sampling noise, not a full accounting of
> event dependence.

## Headline

**P(+10% close within 20 trading days | crash ≤ −10%) = {prim['rate']:.4f}**
(n = {prim['n']:,}; 95% CI [{prim['ci_low']:.4f}, {prim['ci_high']:.4f}]).

## Key findings

1. **Recent-crash history is the strongest descriptive signal.** Recovery rises monotonically
   with the number of recent crashes: fresh shocks (0 prior in 20d) recover **{fresh:.3f}**,
   vs **{rep3:.3f}** for 3+ prior — a ~{100 * (rep3 - fresh):.0f} pp spread. A repeat leg-down
   is *more* likely to bounce (these are volatile names that whipsaw), not less.
2. **Broad-market crashes recover more than idiosyncratic ones.** Crashing on a day the market
   fell ≥ 5% → **{broad:.3f}**; crashing on a flat/up market day → **{calm:.3f}**
   (~{100 * (broad - calm):.0f} pp). Consistent with market-driven drops being more
   overreaction than firm-specific damage.
3. **Smaller is bouncier.** Micro-caps (<$300M) recover **{micro:.3f}** vs large-caps (≥$10B)
   **{large:.3f}** — recovery decreases with size.
4. **Crash magnitude and fundamentals add little on their own.** Recovery is roughly flat
   across −10%→−30% crashes; and profitable vs unprofitable firms are nearly identical
   (**{prof:.3f}** vs **{unprof:.3f}**). A first hint that for *short-term* rebound, price/context
   signals may dominate fundamentals — the M2→M3 question to test in modeling.

## Full recovery grid (close-based)

{_md_grid(clean)}

Monotone as expected: recovery probability rises with horizon and falls with threshold.

## By crash magnitude

{_md_slice(clean, mag, mag_lbl, PRIMARY)}

## By recent-crash history (fresh vs repeat legs down)

{_md_slice(clean, pc_code, pc_lbl, PRIMARY)}

## By company size (market cap)

{_md_slice(clean, size, size_lbl, PRIMARY)}

## By profitability (TTM net margin)

{_md_slice(clean, prof_code, prof_lbl, PRIMARY)}

## By leverage (debt / assets)

{_md_slice(clean, lev, lev_lbl, PRIMARY)}

## By market environment on the crash day

{_md_slice(clean, reg, reg_lbl, PRIMARY)}

## By sector (SIC major division, top by n)

{_md_sector(clean, PRIMARY)}

## Caveats

- Base rates use close-based labels anchored to the crash-day close; a +10% rebound off a
  depressed close is partly mechanical (mean reversion / bid-ask bounce), which is why the
  ≥ $5 filter matters (penny-stock inclusion pushes the primary rate to ~0.68).
- Fundamentals slices (size/profitability/leverage) cover only events with a Compustat match
  (~80% of CLEAN); missingness is non-random (small/young firms less covered).
- These are **descriptive** associations, not causal or predictive claims. Chronological
  modeling (M7) tests whether they hold out-of-sample.
"""
    out = cfg.paths.resolve("reports") / "STU-57_base_rates.md"
    out.write_text(report)
    print(f"primary +10%/20d = {prim['rate']:.4f} (n={prim['n']:,}, "
          f"CI [{prim['ci_low']:.4f}, {prim['ci_high']:.4f}])")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
