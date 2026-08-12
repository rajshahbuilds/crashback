"""SEC EDGAR document provider — the first concrete ``DocumentProvider`` (STU-65).

EDGAR is free, reachable, and **point-in-time-safe by construction**: every filing carries an
``acceptanceDateTime`` (the exact moment it became public), which we map to the canonical
``available_at``. 8-Ks in particular capture most crash-cause events (earnings via item 2.02,
guidance cuts, material events). Historical *general-news* archives are a paid source and are out
of scope here; the provider-neutral interface (``crashback.providers.documents``) lets a news
provider slot in later behind the same point-in-time guarantee.

Network is isolated from parsing: ``parse_submissions`` is pure (unit-tested on fixtures) and
``_fetch`` does the HTTP. EDGAR requires a descriptive ``User-Agent`` and rate-limits to ~10
req/s.

**Timestamp convention:** ``acceptanceDateTime`` is Eastern Time; we treat it as naive ET and
compare to the crash-day 16:00 ET prediction cutoff. A filing accepted after the crash close (or
on a later day) is excluded by the base-class point-in-time filter.

**Identifier caveat:** ticker→CIK via SEC ``company_tickers.json`` is the *current* mapping, so a
historically reused ticker could resolve to the wrong issuer. For production backfill, resolve
``gvkey``→``cik`` via Compustat (``comp.company.cik``); ticker resolution is fine for a curated
recent sample.
"""
from __future__ import annotations

import html
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta

import polars as pl

from crashback.providers.documents import (
    DOCUMENT_SCHEMA,
    DocumentProvider,
    DocumentQuery,
    empty_documents,
)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _accession_index_url(cik: str, accession: str) -> str:
    nodash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nodash}/{accession}-index.htm"


def _parse_acceptance(s: str | None) -> datetime | None:
    """Parse EDGAR acceptanceDateTime ('2022-02-02T16:30:00.000Z') to a naive ET datetime."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "")).replace(microsecond=0)
    except ValueError:
        return None


def _html_to_text(raw: str) -> str:
    """Strip HTML/XBRL markup to readable plain text (best-effort, stdlib-only)."""
    raw = re.sub(r"(?is)<(script|style|head).*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?i)</(p|div|tr|h[1-6]|li)>", "\n", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(raw)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _parse_date(s: str | None):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


class EDGARProvider(DocumentProvider):
    """Retrieve SEC filings as contemporaneous documents for a crash prediction timestamp."""

    source = "sec_edgar"

    def __init__(self, user_agent: str, *, retrieved_at: datetime | None = None,
                 min_interval: float = 0.15):
        if not user_agent or "@" not in user_agent:
            raise ValueError("EDGAR requires a descriptive User-Agent incl. a contact email")
        self.user_agent = user_agent
        self._retrieved_at = retrieved_at
        self._min_interval = min_interval           # polite spacing (EDGAR ~10 req/s)
        self._last_fetch = 0.0
        self._ticker_map: dict[str, int] | None = None

    # --- network (kept separate from parsing so tests stay hermetic) ---------------
    def _get(self, url: str) -> bytes:
        wait = self._min_interval - (time.monotonic() - self._last_fetch)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed SEC host
            self._last_fetch = time.monotonic()
            return resp.read()

    def _get_json(self, url: str) -> dict:
        return json.loads(self._get(url).decode("utf-8"))

    def fetch_document_text(self, index_url: str, *, max_chars: int = 40000) -> str:
        """Fetch a filing's main text: prefer the EX-99.x exhibit (earnings release), else the
        primary document. HTML is stripped to plain text and truncated to ``max_chars``.

        Best-effort and heuristic — enough to feed the extractor, not a full XBRL parser. Returns
        ``""`` on any fetch/parse failure so one bad filing never aborts a batch.
        """
        try:
            base = index_url.rsplit("/", 1)[0]
            # deterministic directory listing (more robust than scraping the index HTML)
            listing = self._get_json(f"{base}/index.json")
            names = [it.get("name", "") for it in listing.get("directory", {}).get("item", [])]
            docs = [n for n in names
                    if re.search(r"\.(htm|html|txt)$", n, re.I) and "index" not in n.lower()]
            # prefer an EX-99 exhibit (press/earnings release), then the largest remaining doc
            exhibit = next((d for d in docs if re.search(r"ex[-_]?99", d, re.I)), None)
            pick = exhibit or (docs[0] if docs else None)
            if pick is None:
                return ""
            raw = self._get(f"{base}/{pick}").decode("utf-8", "ignore")
            return _html_to_text(raw)[:max_chars]
        except (urllib.error.URLError, ValueError, KeyError):
            return ""

    def resolve_cik(self, ticker: str) -> str | None:
        """Current ticker→CIK via SEC company_tickers.json (see identifier caveat)."""
        if self._ticker_map is None:
            data = self._get_json(TICKERS_URL)
            self._ticker_map = {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}
        cik = self._ticker_map.get(ticker.upper())
        return str(cik) if cik is not None else None

    # --- pure normalization (unit-tested on a fixture) -----------------------------
    def parse_recent_dict(
        self, recent: dict, *, cik: str, security_id: int | None = None,
        company_id: int | None = None, retrieved_at: datetime,
    ) -> pl.DataFrame:
        """Normalize an EDGAR filing dict (the ``recent`` block or an older shard, same shape)."""
        forms = recent.get("form", [])
        if not forms:
            return empty_documents()
        acc = recent.get("accessionNumber", [])
        adt = recent.get("acceptanceDateTime", [])
        rdate = recent.get("reportDate", [])
        pdesc = recent.get("primaryDocDescription", [])
        rows = []
        for i, form in enumerate(forms):
            a = acc[i] if i < len(acc) else None
            rows.append({
                "doc_id": a,
                "security_id": security_id,
                "company_id": company_id,
                "source": self.source,
                "source_type": form,
                "title": (pdesc[i] if i < len(pdesc) and pdesc[i] else form),
                "available_at": _parse_acceptance(adt[i] if i < len(adt) else None),
                "period_of_report": _parse_date(rdate[i] if i < len(rdate) else None),
                "url": _accession_index_url(cik, a) if a else None,
                "snippet": None,
                "retrieved_at": retrieved_at,
            })
        return pl.DataFrame(rows, schema=DOCUMENT_SCHEMA)

    def parse_submissions(
        self, submissions: dict, *, security_id: int | None = None,
        company_id: int | None = None, retrieved_at: datetime,
    ) -> pl.DataFrame:
        """Normalize a submissions JSON's ``recent`` block into the canonical schema."""
        return self.parse_recent_dict(
            submissions.get("filings", {}).get("recent", {}),
            cik=str(submissions.get("cik", "")) or "0",
            security_id=security_id, company_id=company_id, retrieved_at=retrieved_at)

    def retrieve_raw(
        self, cik: str, *, start: datetime, end: datetime,
        security_id: int | None = None, company_id: int | None = None,
    ) -> pl.DataFrame:
        """Unfiltered documents for a CIK spanning [start, end], paginating older shards.

        EDGAR's submissions feed holds only the most recent ~1000 filings; heavy filers don't
        reach back years. We additionally fetch the older ``filings.files`` shards whose
        [filingFrom, filingTo] range overlaps the requested window, so a 2022 crash is reachable
        for active filers. No point-in-time filter here — the base ``get_documents`` applies it.
        """
        retrieved_at = self._retrieved_at or datetime.now().replace(microsecond=0)
        subs = self._get_json(SUBMISSIONS_URL.format(cik10=str(int(cik)).zfill(10)))
        frames = [self.parse_submissions(subs, security_id=security_id, company_id=company_id,
                                         retrieved_at=retrieved_at)]
        cik_str = str(subs.get("cik", "")) or str(int(cik))
        for shard in subs.get("filings", {}).get("files", []):
            f_from, f_to = _parse_date(shard.get("filingFrom")), _parse_date(shard.get("filingTo"))
            if f_from and f_to and not (f_to < start.date() or f_from > end.date()):
                sj = self._get_json(f"https://data.sec.gov/submissions/{shard['name']}")
                frames.append(self.parse_recent_dict(
                    sj, cik=cik_str, security_id=security_id, company_id=company_id,
                    retrieved_at=retrieved_at))
        return pl.concat(frames) if frames else empty_documents()

    def _fetch(self, query: DocumentQuery) -> pl.DataFrame:
        cik = query.cik or (self.resolve_cik(query.ticker) if query.ticker else None)
        if cik is None:
            return empty_documents()
        start = query.as_of - timedelta(days=query.lookback_days or 0)
        return self.retrieve_raw(cik, start=start, end=query.as_of,
                                 security_id=query.security_id, company_id=query.company_id)
