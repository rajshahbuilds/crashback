#!/usr/bin/env python3
"""STU-43: extract a small WRDS sample and export it to Parquet.

Covers the final STU-43 acceptance criterion — a small sanity sample spanning:
  * one active stock       : AAPL
  * one delisted stock     : Lehman Brothers Holdings (ticker LEH, delisted 2008-09)
  * one known crash event  : detected as any day with CRSP daily return <= -10%
                             in the sampled windows (Lehman's Sep-2008 collapse and
                             AAPL's 2008 selloff both surface here)

Sources use the CRSP CIZ format (crsp.dsf_v2 / crsp.stocknames_v2) chosen in the
STU-43 validation, plus crsp.dsedelist and Compustat via the CCM link. Output goes
to data/raw/wrds_sample/ (gitignored — this is prototype data, not the research set).

Run: .venv/bin/python scripts/export_wrds_sample.py --username r43shah
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "wrds_sample"
CRASH_THRESHOLD = -0.10

# CIZ daily columns we keep for the canonical price schema (STU-48 will normalize these)
PRICE_COLS = (
    "permno, permco, ticker, primaryexch, siccd, dlycaldt, "
    "dlyopen, dlyhigh, dlylow, dlyclose, dlyprc, dlyret, dlyretx, "
    "dlyvol, dlyprcvol, dlycap, shrout, dlycumfacpr, dlycumfacshr"
)


def resolve_permno(db, ticker: str, name_prefix: str) -> tuple[int, pd.DataFrame]:
    """Find a permno by ticker + issuer name, preferring US common shares.

    Filters the issuer name in pandas rather than SQL LIKE, because a literal '%'
    in the SQL text is misparsed as a bind-parameter marker by SQLAlchemy.
    """
    cand = db.raw_sql(
        "SELECT permno, ticker, issuernm, primaryexch, sharetype, securitytype, "
        "securitybegdt, securityenddt "
        f"FROM crsp.stocknames_v2 WHERE ticker = '{ticker}' ORDER BY securitybegdt"
    )
    cand = cand[cand["issuernm"].astype(str).str.upper().str.startswith(name_prefix.upper())]
    if cand.empty:
        raise RuntimeError(f"No permno found for ticker={ticker} name~{name_prefix}")
    # prefer common stock (sharetype NS = New/common), else first match
    common = cand[cand["sharetype"].astype(str).str.upper().eq("NS")]
    permno = int((common if not common.empty else cand).iloc[0]["permno"])
    return permno, cand


def get_prices(db, permno: int, start: str, end: str) -> pd.DataFrame:
    df = db.raw_sql(
        f"SELECT {PRICE_COLS} FROM crsp.dsf_v2 "
        f"WHERE permno = {permno} AND dlycaldt BETWEEN '{start}' AND '{end}' "
        f"ORDER BY dlycaldt"
    )
    return df


def get_delist(db, permno: int) -> pd.DataFrame:
    return db.raw_sql(
        "SELECT permno, dlstdt, dlstcd, dlret, dlretx, dlprc, nwperm, cusip "
        f"FROM crsp.dsedelist WHERE permno = {permno}"
    )


def get_fundamentals(db, permno: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Link permno -> gvkey via CCM, then pull recent quarterly fundamentals."""
    link = db.raw_sql(
        "SELECT gvkey, lpermno, linktype, linkprim, linkdt, linkenddt "
        f"FROM crsp.ccmxpf_lnkhist WHERE lpermno = {permno} "
        "AND linktype IN ('LU','LC') AND linkprim IN ('P','C')"
    )
    if link.empty:
        return link, pd.DataFrame()
    gvkey = str(link.iloc[0]["gvkey"])
    fq = db.raw_sql(
        "SELECT gvkey, datadate, rdq, fyearq, fqtr, saleq, revtq, niq, atq, "
        "ltq, cheq, dlttq, cshoq "
        f"FROM comp.fundq WHERE gvkey = '{gvkey}' AND indfmt='INDL' AND datafmt='STD' "
        "AND consol='C' AND popsrc='D' "
        "ORDER BY datadate DESC LIMIT 4"
    )
    return link, fq


def write_parquet(df: pd.DataFrame, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    df.to_parquet(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", default=os.environ.get("WRDS_USERNAME"))
    args = ap.parse_args()

    import wrds

    db = wrds.Connection(wrds_username=args.username)
    print("connected.\n")

    # --- resolve identifiers ---
    aapl_permno, aapl_cand = resolve_permno(db, "AAPL", "APPLE")
    leh_permno, leh_cand = resolve_permno(db, "LEH", "LEHMAN")
    print(f"AAPL permno = {aapl_permno}")
    print(f"LEH  permno = {leh_permno}\n")

    # --- prices (CIZ) ---
    aapl_px = get_prices(db, aapl_permno, "2008-09-01", "2009-03-31")
    leh_px = get_prices(db, leh_permno, "2008-01-01", "2008-09-30")

    # --- delisting record (Lehman) ---
    leh_delist = get_delist(db, leh_permno)

    # --- fundamentals (AAPL via CCM) ---
    aapl_link, aapl_fq = get_fundamentals(db, aapl_permno)

    # --- crash-event detection on the sampled prices ---
    both = pd.concat([aapl_px, leh_px], ignore_index=True)
    crashes = both[both["dlyret"] <= CRASH_THRESHOLD].copy().sort_values("dlyret")

    db.close()

    # --- export ---
    written = {
        "prices_active_aapl.parquet": aapl_px,
        "prices_delisted_leh.parquet": leh_px,
        "delist_leh.parquet": leh_delist,
        "fundamentals_aapl_fundq.parquet": aapl_fq,
        "crash_events_sample.parquet": crashes,
    }
    print("=== written ===")
    for name, df in written.items():
        p = write_parquet(df, name)
        print(f"  {name:34s} rows={len(df):>4}  -> {p}")

    # --- sanity checks (read back + summarize) ---
    print("\n=== sanity checks ===")
    for name in ("prices_active_aapl.parquet", "prices_delisted_leh.parquet"):
        rt = pd.read_parquet(OUT_DIR / name)
        assert (rt["permno"].nunique() == 1), f"{name}: expected one permno"
        d0, d1 = rt["dlycaldt"].min(), rt["dlycaldt"].max()
        print(f"  {name}: {len(rt)} rows, {d0}..{d1}, "
              f"close {rt['dlyclose'].min():.2f}..{rt['dlyclose'].max():.2f}, "
              f"min_ret={rt['dlyret'].min():.4f}")

    print(f"\n  crash days (ret <= {CRASH_THRESHOLD:.0%}): {len(crashes)}")
    for _, r in crashes.head(8).iterrows():
        print(f"    {r['ticker']:5s} {str(r['dlycaldt'])[:10]}  "
              f"ret={r['dlyret']:.4f}  close={r['dlyclose']:.2f}")

    if not leh_delist.empty:
        d = leh_delist.iloc[0]
        print(f"\n  Lehman delist: dlstdt={str(d['dlstdt'])[:10]} "
              f"dlstcd={d['dlstcd']} dlret={d['dlret']} dlprc={d['dlprc']}")

    if not aapl_fq.empty:
        f = aapl_fq.iloc[0]
        print(f"\n  AAPL latest fundq: gvkey={aapl_link.iloc[0]['gvkey']} "
              f"datadate={str(f['datadate'])[:10]} rdq={str(f['rdq'])[:10]} "
              f"saleq={f['saleq']} niq={f['niq']} atq={f['atq']}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
