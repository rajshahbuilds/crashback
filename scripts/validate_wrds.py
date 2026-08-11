#!/usr/bin/env python3
"""STU-43: Validate WRDS access and historical data coverage.

Discovery-driven validation of the 7 items in CLAUDE.md sec 5:
  1. programmatic Python access
  2. CRSP entitlement
  3. Compustat North America entitlement
  4. delisted-security coverage
  5. exact historical date coverage
  6. whether Compustat Point-in-Time is available
  7. the precise tables and columns needed

The script assumes the password lives in ~/.pgpass (or the wrds interactive
setup has been run). Username comes from --username or $WRDS_USERNAME.

It probes with cheap server-side aggregates (min/max/count) and LIMIT queries;
it does NOT pull bulk data. Every check is isolated so one failure does not
abort the rest. Results are written as JSON to reports/ and summarized to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


def _jsonable(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)


class Validator:
    def __init__(self, db):
        self.db = db
        self.results: dict = {"checks": {}}

    def record(self, key, **payload):
        self.results["checks"][key] = payload

    def safe(self, key, fn):
        """Run a check, capturing exceptions as a failed result."""
        try:
            payload = fn()
            payload.setdefault("ok", True)
            self.record(key, **payload)
            status = "OK " if payload.get("ok") else "FAIL"
            print(f"  [{status}] {key}: {payload.get('summary', '')}")
        except Exception as e:  # noqa: BLE001 - we want every failure captured
            self.record(key, ok=False, error=str(e), trace=traceback.format_exc())
            print(f"  [FAIL] {key}: {e}")

    # --- individual checks -------------------------------------------------

    def check_libraries(self):
        def fn():
            libs = sorted(self.db.list_libraries())
            crsp = [lib for lib in libs if "crsp" in lib.lower()]
            comp = [lib for lib in libs if "comp" in lib.lower()]
            pit = [
                lib for lib in libs
                if any(k in lib.lower() for k in ("snapshot", "_pit", "point", "unrestated"))
            ]
            return {
                "n_libraries": len(libs),
                "crsp_libraries": crsp,
                "compustat_libraries": comp,
                "possible_pit_libraries": pit,
                "all_libraries_sample": libs[:60],
                "summary": f"{len(libs)} libraries; crsp={crsp}; comp={comp}",
            }
        self.safe("entitlements.libraries", fn)

    def check_tables(self, library):
        def fn():
            tables = sorted(self.db.list_tables(library=library))
            return {
                "library": library,
                "n_tables": len(tables),
                "tables": tables,
                "summary": f"{len(tables)} tables in {library}",
            }
        self.safe(f"tables.{library}", fn)

    def probe_table(self, key, library, table, date_col=None, id_col=None):
        """Row count, optional date range, optional distinct id count, columns."""
        def fn():
            fq = f"{library}.{table}"
            cols = self.db.get_row_count  # noqa: F841 (placeholder to avoid lint)
            # column names
            desc = self.db.describe_table(library=library, table=table)
            columns = list(desc["name"]) if hasattr(desc, "__getitem__") else None
            out = {"table": fq, "n_columns": len(columns) if columns else None,
                   "columns_sample": columns[:40] if columns else None}
            if date_col:
                r = self.db.raw_sql(
                    f"SELECT MIN({date_col}) AS min_d, MAX({date_col}) AS max_d, "
                    f"COUNT(*) AS n FROM {fq}"
                )
                out["date_col"] = date_col
                out["min_date"] = r.iloc[0]["min_d"]
                out["max_date"] = r.iloc[0]["max_d"]
                out["n_rows"] = int(r.iloc[0]["n"])
            if id_col:
                r2 = self.db.raw_sql(f"SELECT COUNT(DISTINCT {id_col}) AS n_ids FROM {fq}")
                out["id_col"] = id_col
                out["n_distinct_ids"] = int(r2.iloc[0]["n_ids"])
            bits = []
            if date_col:
                bits.append(f"{out['min_date']}..{out['max_date']} ({out.get('n_rows')} rows)")
            if id_col:
                bits.append(f"{out['n_distinct_ids']} distinct {id_col}")
            out["summary"] = "; ".join(bits) or f"{out['n_columns']} cols"
            return out
        self.safe(key, fn)

    def check_delistings(self, library, table, code_col):
        def fn():
            fq = f"{library}.{table}"
            r = self.db.raw_sql(
                f"SELECT {code_col} AS code, COUNT(*) AS n FROM {fq} "
                f"GROUP BY {code_col} ORDER BY n DESC LIMIT 25"
            )
            total = self.db.raw_sql(f"SELECT COUNT(*) AS n FROM {fq}").iloc[0]["n"]
            return {
                "table": fq,
                "total_delist_records": int(total),
                "top_delist_codes": r.to_dict("records"),
                "summary": f"{int(total)} delist records across {len(r)} codes shown",
            }
        self.safe(f"delistings.{library}.{table}", fn)


def run(username: str | None):
    from importlib.metadata import version as _pkg_version

    import wrds

    wrds_ver = _pkg_version("wrds")
    print(f"wrds version: {wrds_ver}")
    print(f"connecting (username={username or 'from-pgpass/prompt'}) ...")
    conn_kwargs = {}
    if username:
        conn_kwargs["wrds_username"] = username
    db = wrds.Connection(**conn_kwargs)
    print("connected.\n")

    v = Validator(db)
    v.results["meta"] = {
        "wrds_version": wrds_ver,
        "username": username,
    }

    print("== entitlements ==")
    v.check_libraries()
    libs = v.results["checks"].get("entitlements.libraries", {})
    crsp_libs = libs.get("crsp_libraries", [])
    comp_libs = libs.get("compustat_libraries", [])

    # Pick primary schemas (classic names first, else first match)
    crsp_lib = "crsp" if "crsp" in crsp_libs else (crsp_libs[0] if crsp_libs else None)
    comp_lib = "comp" if "comp" in comp_libs else (comp_libs[0] if comp_libs else None)

    print("\n== list tables ==")
    if crsp_lib:
        v.check_tables(crsp_lib)
    if comp_lib:
        v.check_tables(comp_lib)

    print("\n== CRSP coverage ==")
    if crsp_lib:
        # Daily stock file — the canonical price source
        v.probe_table("crsp.dsf", crsp_lib, "dsf", date_col="date", id_col="permno")
        # Security master / ticker history
        v.probe_table("crsp.stocknames", crsp_lib, "stocknames", id_col="permno")
        v.probe_table("crsp.dsenames", crsp_lib, "dsenames", id_col="permno")
        # Delistings
        v.probe_table("crsp.dsedelist", crsp_lib, "dsedelist", date_col="dlstdt")
        v.check_delistings(crsp_lib, "dsedelist", "dlstcd")

    print("\n== Compustat coverage ==")
    if comp_lib:
        v.probe_table("comp.company", comp_lib, "company", id_col="gvkey")
        # Annual & quarterly fundamentals; fundq.rdq = report date (PIT proxy)
        v.probe_table("comp.funda", comp_lib, "funda", date_col="datadate", id_col="gvkey")
        v.probe_table("comp.fundq", comp_lib, "fundq", date_col="datadate", id_col="gvkey")

    print("\n== Compustat point-in-time probe ==")
    def pit_probe():
        out = {"possible_pit_libraries": libs.get("possible_pit_libraries", [])}
        # rdq (earnings report date) presence in fundq = filing-availability signal
        if comp_lib:
            r = db.raw_sql(
                f"SELECT COUNT(*) AS n, COUNT(rdq) AS n_rdq, MIN(rdq) AS min_rdq, "
                f"MAX(rdq) AS max_rdq FROM {comp_lib}.fundq"
            )
            out["fundq_rows"] = int(r.iloc[0]["n"])
            out["fundq_rows_with_rdq"] = int(r.iloc[0]["n_rdq"])
            out["fundq_rdq_min"] = r.iloc[0]["min_rdq"]
            out["fundq_rdq_max"] = r.iloc[0]["max_rdq"]
        out["summary"] = (
            f"pit_libs={out['possible_pit_libraries']}; "
            f"fundq rdq coverage {out.get('fundq_rows_with_rdq')}/{out.get('fundq_rows')}"
        )
        return out
    v.safe("compustat.point_in_time", pit_probe)

    # Compustat<->CRSP link (CCM) — needed to join fundamentals to prices
    print("\n== CRSP/Compustat link (CCM) ==")
    def ccm_probe():
        candidates = [
            ("crsp", "ccmxpf_linktable"),
            ("crsp", "ccmxpf_lnkhist"),
            ("crsp", "ccm_lookup"),
        ]
        found = []
        for lib, tbl in candidates:
            try:
                n = db.raw_sql(f"SELECT COUNT(*) AS n FROM {lib}.{tbl}").iloc[0]["n"]
                found.append({"table": f"{lib}.{tbl}", "n_rows": int(n)})
            except Exception:  # noqa: BLE001
                continue
        return {"link_tables_found": found,
                "summary": f"{len(found)} CCM link table(s) found"}
    v.safe("link.ccm", ccm_probe)

    db.close()
    return v.results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", default=os.environ.get("WRDS_USERNAME"))
    args = ap.parse_args()

    started = datetime.now()
    try:
        results = run(args.username)
        results["meta"]["started"] = started.isoformat()
        results["meta"]["finished"] = datetime.now().isoformat()
    except Exception as e:  # noqa: BLE001
        print(f"\nFATAL: could not complete validation: {e}", file=sys.stderr)
        traceback.print_exc()
        results = {"fatal_error": str(e), "trace": traceback.format_exc(),
                   "meta": {"started": started.isoformat()}}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "wrds_validation.json"
    out_path.write_text(json.dumps(results, indent=2, default=_jsonable))
    print(f"\nWrote {out_path}")

    # Exit non-zero if core entitlements failed
    checks = results.get("checks", {})
    core_ok = (
        checks.get("crsp.dsf", {}).get("ok")
        and checks.get("comp.funda", {}).get("ok")
    )
    sys.exit(0 if core_ok else 1)


if __name__ == "__main__":
    main()
