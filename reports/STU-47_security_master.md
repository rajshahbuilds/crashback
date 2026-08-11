# STU-47 — Survivorship-Safe Security Master

**Status:** ✅ complete. The canonical historical universe of US common stocks (active +
delisted) is built from CRSP CIZ and written to versioned Parquet.

**Build:** `.venv/bin/python scripts/build_security_master.py --username r43shah --version v1`
**Artifact:** `data/normalized/security_master/security_master_v1.parquet` (+ `.meta.json`
sidecar; gitignored — bulk artifact, provenance in the sidecar).

## Universe definition (reproducible from config)

Equivalent to the classic CRSP `shrcd IN (10,11)` on `exchcd IN (1,2,3)`, expressed in CIZ
terms in `configs/default.yaml`:

```
sharetype='NS' AND securitytype='EQTY' AND securitysubtype='COM'
AND usincflg='Y' AND primaryexch IN ('N','A','Q')
```

- **Excludes** ETFs/funds (`FUND`), ADRs (`AD`), warrants/rights (`DERV`), units, and
  non-major exchanges (Arca/other).
- **Excludes** foreign-incorporated common (`usincflg='N'`, ≈ legacy shrcd 12) — e.g.
  Accenture, Medtronic, Chubb, Schlumberger.
- **Includes** REITs (no issuer-type filter, so `CORP`/`ACOR`/`REIT` are all kept).

## Result

| metric | value |
|---|---|
| securities (distinct permno) | **27,363** |
| — active | 4,171 |
| — delisted | 23,192 |
| companies (distinct permco) | 26,727 |
| ticker-period rows | 61,274 |
| coverage | 1925-12-31 → 2025-12-31 |
| exchanges | NYSE, NYSE American, Nasdaq |

The active count (~4.2k) matches the real count of US common stocks on major exchanges;
the 23k delisted securities are what makes the universe **survivorship-safe**.

## Grain & identifiers

One row per **(security_id, ticker period)**. `security_id` = CRSP **permno** is the stable
identity, so a ticker/name change adds rows to the SAME security — it never creates a new
security or duplicate company. `company_id` = permco.

## Delisting correctness

`delisting_*` reflect a genuine delisting event only (CRSP `dsedelist` code **≥ 200**). Code
**100** ("still trading") and securities with no delisting row are **active** (null delisting
fields). This fixed an initial bug where `securityenddt` (the name-record end = data cutoff
for active names) was mistaken for a delisting date, which had marked every security delisted.

## Sample checks (acceptance criterion)

- **Active:** e.g. `JJSF` (J&J Snack Foods) — null delisting.
- **Renamed / ticker-changed:** security `38149` carries `APAT→GATI→UGAM→ALLY→AGI→BYI`
  (Bally lineage) under one permno across NYSE/Nasdaq — a single security, not six.
- **Acquired:** security `10001` — delisting code `233` (merger), 2017-08-03.
- **Delisted/dropped:** securities `10000`/`10005` — codes `560`/`575` (dropped: insufficient
  capital / bankruptcy).

## Tests

Hermetic (`tests/test_security_master.py`, `tests/test_normalization.py`): universe-filter →
CIZ SQL mapping, active-vs-delisted gating (code 100 stays active), builder active/delisted
counts on the synthetic provider, and the versioned-Parquet writer round-trip + sidecar. 22
tests passing.
