"""WRDS adapter: CRSP CIZ + Compustat -> canonical schemas.

Thin layer — it runs SQL against the validated tables (STU-43) and delegates all column
mapping to ``crashback.providers.normalize``. Vendor column names appear only here and in
``normalize``. Prices/security master come from the CIZ (``*_v2``) tables; fundamentals
from Compustat ``fundq``/``funda``; the permno<->gvkey link from ``crsp.ccmxpf_lnkhist``.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import polars as pl

from crashback.logging_utils import get_logger
from crashback.providers import normalize as norm
from crashback.providers.base import MarketDataProvider
from crashback.providers.universe import UniverseFilter

log = get_logger(__name__)

_LINK_FILTER = "linktype IN ('LU','LC') AND linkprim IN ('P','C')"
_FUNDQ_FILTER = "indfmt='INDL' AND datafmt='STD' AND consol='C' AND popsrc='D'"


def _in_clause(column: str, ids: Sequence[int] | None) -> str:
    """`` AND col IN (1,2,3)`` for integer id columns (e.g. permno), or '' for None/empty."""
    if not ids:
        return ""
    joined = ",".join(str(int(i)) for i in ids)
    return f" AND {column} IN ({joined})"


def _in_clause_gvkey(column: str, ids: Sequence[int | str] | None) -> str:
    """`` AND col IN ('001690',...)`` for Compustat gvkey (a 6-digit zero-padded string)."""
    if not ids:
        return ""
    joined = ",".join(f"'{int(i):06d}'" for i in ids)
    return f" AND {column} IN ({joined})"


def _universe_where(u: UniverseFilter | None) -> str:
    """SQL WHERE fragment mapping a UniverseFilter to CIZ stocknames_v2 columns."""
    if u is None:
        return ""

    def q(vals):
        return ",".join(f"'{v}'" for v in vals)

    parts = []
    if u.share_types:
        parts.append(f"sharetype IN ({q(u.share_types)})")
    if u.security_types:
        parts.append(f"securitytype IN ({q(u.security_types)})")
    if u.security_subtypes:
        parts.append(f"securitysubtype IN ({q(u.security_subtypes)})")
    if u.exchanges:
        parts.append(f"primaryexch IN ({q(u.exchanges)})")
    if u.us_incorporated_only:
        parts.append("usincflg = 'Y'")
    return "".join(f" AND {p}" for p in parts)


class WRDSProvider(MarketDataProvider):
    """MarketDataProvider backed by WRDS (CRSP CIZ + Compustat NA)."""

    def __init__(self, username: str | None = None, connection=None):
        self._username = username
        self._conn = connection  # inject a live wrds.Connection, or lazily create one

    @property
    def conn(self):
        if self._conn is None:
            import wrds

            log.info("opening WRDS connection (user=%s)", self._username or "from-pgpass")
            self._conn = wrds.Connection(wrds_username=self._username)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # --- helpers -----------------------------------------------------------------

    def _sql(self, query: str, date_cols: Sequence[str] = ()) -> pl.DataFrame:
        """Run SQL, coerce date-like columns, return a Polars DataFrame."""
        import pandas as pd

        pdf = self.conn.raw_sql(query)
        for c in date_cols:
            if c in pdf.columns:
                pdf[c] = pd.to_datetime(pdf[c], errors="coerce")
        return pl.from_pandas(pdf)

    # --- interface ---------------------------------------------------------------

    def get_daily_prices(
        self, security_ids: Sequence[int] | None, start: date, end: date
    ) -> pl.DataFrame:
        q = (
            "SELECT permno, dlycaldt, dlyopen, dlyhigh, dlylow, dlyclose, dlyret, dlyretx, "
            "dlyvol, dlycumfacpr, dlycumfacshr, shrout "
            "FROM crsp.dsf_v2 "
            f"WHERE dlycaldt BETWEEN '{start.isoformat()}' AND '{end.isoformat()}'"
            f"{_in_clause('permno', security_ids)} ORDER BY permno, dlycaldt"
        )
        return norm.normalize_ciz_prices(self._sql(q, date_cols=["dlycaldt"]))

    def get_security_master(
        self,
        security_ids: Sequence[int] | None = None,
        universe: UniverseFilter | None = None,
    ) -> pl.DataFrame:
        names = self._sql(
            "SELECT permno, permco, namedt, nameenddt, ticker, primaryexch, securitytype, "
            "siccd, securitybegdt, securityenddt FROM crsp.stocknames_v2 "
            f"WHERE 1=1{_in_clause('permno', security_ids)}{_universe_where(universe)}",
            date_cols=["namedt", "nameenddt", "securitybegdt", "securityenddt"],
        )
        delist = self._sql(
            "SELECT permno, dlstdt, dlstcd, dlret FROM crsp.dsedelist "
            f"WHERE 1=1{_in_clause('permno', security_ids)}",
            date_cols=["dlstdt"],
        )
        return norm.normalize_ciz_security_master(names, delist)

    def get_corporate_actions(
        self, security_ids: Sequence[int] | None = None
    ) -> pl.DataFrame:
        df = self._sql(
            "SELECT permno, dlstdt, dlstcd, dlret FROM crsp.dsedelist "
            f"WHERE 1=1{_in_clause('permno', security_ids)}",
            date_cols=["dlstdt"],
        )
        return norm.normalize_ciz_delistings(df)

    def get_sector_metadata(
        self, security_ids: Sequence[int] | None = None
    ) -> pl.DataFrame:
        names = self._sql(
            "SELECT permno, siccd FROM crsp.stocknames_v2 "
            f"WHERE 1=1{_in_clause('permno', security_ids)}"
        )
        return norm.normalize_ciz_sector(names)

    def get_fundamentals(
        self,
        *,
        company_ids: Sequence[int] | None = None,
        security_ids: Sequence[int] | None = None,
        freq: str = "Q",
    ) -> pl.DataFrame:
        # Resolve permnos -> gvkeys when only security_ids are given. gvkey is a
        # zero-padded string in Compustat/CCM, so it needs quoted, padded literals.
        links = self._sql(
            f"SELECT gvkey, lpermno FROM crsp.ccmxpf_lnkhist WHERE {_LINK_FILTER}"
            f"{_in_clause('lpermno', security_ids)}"
            f"{_in_clause_gvkey('gvkey', company_ids)}"
        )
        gvkeys = company_ids
        if gvkeys is None and security_ids is not None:
            gvkeys = links["gvkey"].unique().to_list()

        if freq == "Q":
            cols = ("gvkey, datadate, rdq, fyearq, fqtr, revtq, niq, atq, ltq, cheq, "
                    "dlttq, dlcq, cshoq")
            table, dcols = "comp.fundq", ["datadate", "rdq"]
            where = _FUNDQ_FILTER
        elif freq == "A":
            cols = "gvkey, datadate, pdate, fyear, revt, ni, at, lt, che, dltt, dlc, csho"
            table, dcols = "comp.funda", ["datadate", "pdate"]
            where = "indfmt='INDL' AND datafmt='STD' AND consol='C' AND popsrc='D'"
        else:
            raise ValueError(f"freq must be 'Q' or 'A', got {freq!r}")

        df = self._sql(
            f"SELECT {cols} FROM {table} WHERE {where}{_in_clause_gvkey('gvkey', gvkeys)} "
            "ORDER BY gvkey, datadate",
            date_cols=dcols,
        )
        return norm.normalize_compustat_fundamentals(df, freq=freq, links=links)
