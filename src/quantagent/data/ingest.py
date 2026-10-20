"""The only code that writes to the QuantAgent database.

Builds a fresh database next to the target, validates every input row (rejects go to
reports/rejected_rows.csv with a reason), builds the trading calendar and metrics cache,
bumps data_version, then atomically replaces the target file.
"""
import csv
import hashlib
import json
import logging
import os
import uuid
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, insert, select

from ..analytics import CALC_VERSION
from ..analytics.returns import daily_rf
from ..analytics.windows import PERIOD_YEARS, clip, metric_bundle, window_start, window_sufficiency
from ..config import Settings
from . import models as m
from .calendar import build_calendar
from .engine import make_readonly_engine
from .validation import (
    base_name,
    option_type_from_name,
    run_checks,
    status_of,
    validate_schemes,
    validate_series,
)

log = logging.getLogger("quantagent.ingest")

SCHEMES, NAV, BENCH, TER = "schemes.csv", "nav_history.csv", "benchmark_indices.csv", "expense_ratios.csv"
RISK_FREE, HOLIDAYS, SOURCES = "risk_free_rates.csv", "nse_holidays.csv", "sources.json"
HOLIDAY_SHARE = 0.5  # a weekday is a holiday when fewer than half the live schemes publish a NAV


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _opt_date(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    d = pd.to_datetime(v, errors="coerce")
    return None if pd.isna(d) else d.date()


def _opt_dt(v):
    if not v:
        return None
    d = pd.to_datetime(v, errors="coerce", utc=True)
    return None if pd.isna(d) else d.tz_localize(None).to_pydatetime()


def _previous_version(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    try:
        eng = make_readonly_engine(f"sqlite:///{db_path.as_posix()}")
        with eng.connect() as c:
            v = c.execute(select(m.meta.c.value).where(m.meta.c.key == "data_version")).scalar()
        eng.dispose()
        return int(v or 0)
    except Exception:  # unreadable old file: start a new version line
        return 0


def derive_holidays(nav: pd.DataFrame) -> list[date]:
    """Weekdays on which fewer than HOLIDAY_SHARE of the schemes alive that day published a NAV."""
    if nav.empty:
        return []
    span = nav.groupby("key")["date"].agg(["min", "max"])
    days = pd.bdate_range(min(nav["date"]), max(nav["date"])).date
    started = np.searchsorted(np.sort(span["min"].to_numpy()), days, side="right")
    ended = np.searchsorted(np.sort(span["max"].to_numpy()), days, side="left")
    alive = started - ended
    published = pd.Series(nav["date"]).value_counts().reindex(days, fill_value=0).to_numpy()
    return [d for d, a, n in zip(days, alive, published, strict=True) if a and n < HOLIDAY_SHARE * a]


def ingest(raw_dir: Path, db_path: Path, reference_dir: Path, rejected_path: Path, settings: Settings,
           data_nature: str | None = None, today: date | None = None) -> dict:
    today = today or date.today()
    started = datetime.now(UTC).replace(tzinfo=None)
    run_id = str(uuid.uuid4())
    version = _previous_version(db_path) + 1
    rejected: list[dict] = []
    assumptions: list[str] = []
    provenance = json.loads((raw_dir / SOURCES).read_text(encoding="utf-8")) if (raw_dir / SOURCES).exists() else {}
    data_nature = data_nature or provenance.get("data_nature") or "synthetic"
    file_info = provenance.get("files", {})

    # ---------------- schemes
    master_raw = pd.read_csv(raw_dir / SCHEMES, dtype={"amfi_code": "string"})
    master, rej = validate_schemes(master_raw, SCHEMES)
    rejected += rej
    bmap_path = reference_dir / "benchmark_map.csv"
    bmap = pd.read_csv(bmap_path) if bmap_path.exists() else pd.DataFrame(
        columns=["benchmark_name", "index_name", "index_type"])
    name_to_index = dict(zip(bmap["benchmark_name"], bmap["index_name"], strict=True))

    # ---------------- series
    nav, rej = validate_series(pd.read_csv(raw_dir / NAV, dtype={"amfi_code": "string"}), "amfi_code", "date",
                               "nav", set(master["scheme_code"]), today, NAV)
    rejected += rej
    if (raw_dir / BENCH).exists():
        bench_raw = pd.read_csv(raw_dir / BENCH)
        bench, rej = validate_series(bench_raw, "index_name", "date", "close_value", None, today, BENCH)
        rejected += rej
    else:
        bench_raw, bench = pd.DataFrame(), pd.DataFrame(columns=["key", "date", "value"])
        assumptions.append("No benchmark series loaded; benchmark-relative statistics are missing.")
    index_types = dict(zip(bmap["index_name"], bmap["index_type"], strict=True))
    if "index_type" in bench_raw:
        index_types |= dict(zip(bench_raw["index_name"], bench_raw["index_type"], strict=False))
    bench["index_type"] = [index_types.get(n, "TRI" if "TRI" in n.upper() else "PRI") for n in bench["key"]]
    if (bench["index_type"] == "PRI").any():
        assumptions.append("Benchmarks without a TRI tag are stored as PRI (price index; excludes dividends).")

    rf = pd.DataFrame(columns=["key", "date", "value"])
    if (raw_dir / RISK_FREE).exists():
        rf_raw = pd.read_csv(raw_dir / RISK_FREE)
        rf_raw["key"] = rf_raw.get("source", "file")
        rf, rej = validate_series(rf_raw, "key", "date", "annual_rate", None, today, RISK_FREE)
        rejected += rej
    else:
        assumptions.append(f"No risk-free series; the configured default {settings.risk_free_annual:.2%} "
                           f"is used ({settings.risk_free_source}).")

    if (raw_dir / HOLIDAYS).exists():
        holidays = list(pd.to_datetime(pd.read_csv(raw_dir / HOLIDAYS)["date"]).dt.date)
        calendar_source = f"Mon-Fri minus {len(holidays)} NSE holidays from {HOLIDAYS}"
    else:
        holidays = derive_holidays(nav)
        calendar_source = (f"Mon-Fri minus {len(holidays)} weekdays on which fewer than "
                           f"{HOLIDAY_SHARE:.0%} of live schemes published a NAV (derived; no holiday list)")
        assumptions.append("Trading calendar derived from NAV publication dates (no NSE holiday list supplied).")

    all_dates = pd.concat([pd.Series(nav["date"]), pd.Series(bench["date"])])
    cal = build_calendar(all_dates.min(), all_dates.max(), holidays)
    data_as_of = max(nav["date"]) if len(nav) else None

    # ---------------- expense ratios: dated disclosures if supplied, else one undated snapshot
    ter_rows: list[dict] = []
    if (raw_dir / TER).exists():
        ter, rej = validate_series(pd.read_csv(raw_dir / TER, dtype={"amfi_code": "string"}), "amfi_code",
                                   "as_of_date", "ter_pct", set(master["scheme_code"]), today, TER)
        rejected += rej
        ter_rows = [{"scheme_code": k, "as_of_date": d, "ter": float(v), "as_of_assumed": 0}
                    for k, d, v in ter[["key", "date", "value"]].itertuples(index=False) if v <= 5]
    elif "expense_ratio_pct" in master:
        has_ter_date = "expense_ratio_as_of" in master
        if not has_ter_date:
            assumptions.append(f"Expense ratio as-of date unknown; assumed {data_as_of} (last NAV date).")
        for r in master.to_dict("records"):
            if pd.notna(r.get("expense_ratio_pct")):
                ter_rows.append({"scheme_code": r["scheme_code"],
                                 "as_of_date": _opt_date(r.get("expense_ratio_as_of")) if has_ter_date else data_as_of,
                                 "ter": float(r["expense_ratio_pct"]), "as_of_assumed": int(not has_ter_date)})
    else:
        assumptions.append("No expense ratio data loaded.")

    # ---------------- write a fresh database
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_name(db_path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    eng = create_engine(f"sqlite+pysqlite:///{tmp.as_posix()}")
    m.metadata.create_all(eng)
    with eng.begin() as c:
        scheme_rows = []
        for r in master.to_dict("records"):
            status = str(r.get("status") or "active").lower()
            scheme_rows.append({
                "scheme_code": r["scheme_code"], "isin": r.get("isin") or None,
                "scheme_name": r["scheme_name"], "base_name": base_name(r["scheme_name"]),
                "fund_house": r["fund_house"], "category": r.get("category"), "sub_category": r.get("sub_category"),
                "plan_type": r["plan"], "option_type": (r.get("option_type") if pd.notna(r.get("option_type")) else None)
                or option_type_from_name(r["scheme_name"]),
                "status": status if status in ("active", "closed", "merged") else "active",
                "merged_into": r.get("merged_into") or None,
                "launch_date": _opt_date(r.get("launch_date")), "closure_date": _opt_date(r.get("closure_date")),
                "benchmark_name": r.get("benchmark"), "benchmark_index": name_to_index.get(r.get("benchmark")),
                "risk_category": r.get("risk_category"),
            })
        c.execute(insert(m.schemes), scheme_rows)
        if ter_rows:
            c.execute(insert(m.expense_ratios), ter_rows)
        c.execute(insert(m.nav_history), [{"scheme_code": k, "nav_date": d, "nav": v}
                                          for k, d, v in nav[["key", "date", "value"]].itertuples(index=False)])
        if len(bench):
            c.execute(insert(m.benchmarks), [{"index_name": k, "index_type": t, "date": d, "value": v}
                                             for k, d, v, t in bench[["key", "date", "value", "index_type"]]
                                             .itertuples(index=False)])
        if len(rf):
            c.execute(insert(m.risk_free_rates), [{"date": d, "annual_rate": v, "source": k}
                                                  for k, d, v in rf[["key", "date", "value"]].itertuples(index=False)])
        c.execute(insert(m.trading_calendar), cal.to_dict("records"))

        # series-level checks: the same code the check_data_quality tool runs
        trading = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal["is_trading_day"] == 1, "date"]))
        per_scheme, statuses = {}, {"ok": 0, "warn": 0, "fail": 0}
        for s in scheme_rows:
            g = nav[nav["key"] == s["scheme_code"]]
            issues = run_checks(g.rename(columns={"value": "nav"})[["date", "nav"]], s, trading, today)
            st = status_of(issues)
            statuses[st] += 1
            per_scheme[s["scheme_code"]] = {
                "n": len(g), "start": str(min(g["date"])) if len(g) else None,
                "end": str(max(g["date"])) if len(g) else None, "status": st,
                "issues": {i.check: i.count for i in issues if i.count}}

        rej_by_file = pd.Series([x["source"] for x in rejected]).value_counts().to_dict() if rejected else {}
        finished = datetime.now(UTC).replace(tzinfo=None)
        log_rows = [
            ("schemes", len(scheme_rows), rej_by_file.get(SCHEMES, 0), {"n_schemes": len(scheme_rows)}),
            ("nav_history", len(nav), rej_by_file.get(NAV, 0), {"quality_status": statuses, "per_scheme": per_scheme}),
            ("benchmarks", len(bench), rej_by_file.get(BENCH, 0),
             {n: int(k) for n, k in bench["key"].value_counts().items()}),
            ("expense_ratios", len(ter_rows), rej_by_file.get(TER, 0), {}),
            ("risk_free_rates", len(rf), rej_by_file.get(RISK_FREE, 0), {}),
            ("trading_calendar", len(cal), 0, {"source": calendar_source}),
        ]
        c.execute(insert(m.ingestion_log), [
            {"run_id": run_id, "data_version": version, "started_at": started, "finished_at": finished,
             "table_name": t, "rows_loaded": n, "rows_rejected": r, "coverage": json.dumps(cov)}
            for t, n, r, cov in log_rows])

        present = [f for f in (SCHEMES, NAV, BENCH, TER, RISK_FREE, HOLIDAYS) if (raw_dir / f).exists()]
        default_licence = "synthetic test data" if data_nature == "synthetic" else None
        c.execute(insert(m.data_sources), [
            {"source": file_info.get(f, {}).get("source") or f, "file": f, "url": file_info.get(f, {}).get("url"),
             "retrieved_at": _opt_dt(file_info.get(f, {}).get("retrieved_at")) or
             datetime.fromtimestamp((raw_dir / f).stat().st_mtime, UTC).replace(tzinfo=None),
             "licence": file_info.get(f, {}).get("licence") or default_licence,
             "coverage": json.dumps(file_info.get(f, {}).get("coverage")) if f in file_info else None,
             "sha256": _sha256(raw_dir / f), "data_nature": data_nature} for f in present]
            + [{"source": f"data/reference/{p.name}", "file": p.name, "url": None, "retrieved_at": None,
                "licence": "project reference file (MIT)", "coverage": None, "sha256": _sha256(p),
                "data_nature": "reference"}
               for p in sorted(reference_dir.glob("*")) if p.is_file()])
        retrieved = sorted(v["retrieved_at"] for v in file_info.values() if v.get("retrieved_at"))
        source_summary = "; ".join(f"{f}: {v.get('source')}" for f, v in file_info.items()) or             ("synthetic fixture" if data_nature == "synthetic" else "unrecorded")

        notes_path = reference_dir / "dataset_notes.json"
        notes = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.exists() else {}
        meta_rows = {
            "data_version": str(version), "calc_version": CALC_VERSION, "data_nature": data_nature,
            "data_as_of": str(data_as_of), "built_at": finished.isoformat(timespec="seconds"),
            "calendar_source": calendar_source, "assumptions": json.dumps(assumptions),
            "known_biases": json.dumps(notes.get("known_biases", [])),
            "dataset_findings": json.dumps(notes.get("findings", [])),
            "run_id": run_id, "data_source": source_summary,
            "data_retrieved_at": retrieved[0] if retrieved else "",
        }
        c.execute(insert(m.meta), [{"key": k, "value": v} for k, v in meta_rows.items()])
        cache_rows = _cache_rows(c, version, settings)
        if cache_rows:
            c.execute(insert(m.metrics_cache), cache_rows)
    eng.dispose()

    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    with rejected_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "source", "row", "reason", "raw"])
        for x in rejected:
            w.writerow([run_id, x["source"], x["row"], x["reason"], json.dumps(x["raw"], default=str)])

    os.replace(tmp, db_path)
    summary = {"run_id": run_id, "data_version": version, "db": str(db_path), "schemes": len(scheme_rows),
               "nav_rows": len(nav), "benchmark_rows": len(bench), "rejected": len(rejected),
               "quality_status": statuses, "cache_rows": len(cache_rows), "data_as_of": str(data_as_of)}
    log.info("ingest complete", extra={"fields": summary})
    return summary


# ---------------------------------------------------------------- metrics cache

def _cache_rows(conn, version: int, settings: Settings) -> list[dict]:
    """Compute 1y/3y/5y metric bundles for every scheme as of the latest NAV date."""
    nav = pd.DataFrame(conn.execute(select(m.nav_history)).mappings().all())
    if nav.empty:
        return []
    nav["nav_date"] = pd.to_datetime(nav["nav_date"])
    as_of = nav["nav_date"].max()
    cal = pd.DataFrame(conn.execute(select(m.trading_calendar.c.date)
                                    .where(m.trading_calendar.c.is_trading_day == 1)).all(), columns=["date"])
    trading = pd.DatetimeIndex(pd.to_datetime(cal["date"]))
    bench = pd.DataFrame(conn.execute(select(m.benchmarks)).mappings().all())
    if not bench.empty:
        bench["date"] = pd.to_datetime(bench["date"])
    rf_tab = pd.DataFrame(conn.execute(select(m.risk_free_rates)).mappings().all())
    if rf_tab.empty:
        rf_annual, rf_source, rf_rate = settings.risk_free_annual, settings.risk_free_source, settings.risk_free_annual
    else:
        rf_annual = pd.Series(rf_tab["annual_rate"].to_numpy(), index=pd.to_datetime(rf_tab["date"]))
        rf_source, rf_rate = "risk_free_rates table: " + ", ".join(sorted(set(rf_tab["source"]))), None
    rows = []
    for s in conn.execute(select(m.schemes)).mappings():
        series = nav[nav["scheme_code"] == s["scheme_code"]].set_index("nav_date")["nav"].sort_index()
        b = None
        if s["benchmark_index"] and not bench.empty:
            bb = bench[bench["index_name"] == s["benchmark_index"]]
            b = bb.set_index("date")["value"].sort_index() if len(bb) else None
        for w in PERIOD_YEARS:
            start = window_start(w, as_of)
            win = clip(series, start, as_of)
            suf = window_sufficiency(win, trading, start, as_of)
            row = {"scheme_code": s["scheme_code"], "window": w, "data_version": version, "as_of": as_of.date(),
                   "start_date": win.index.min().date() if len(win) else None,
                   "end_date": win.index.max().date() if len(win) else None,
                   "n_obs": suf.n_obs, "required_obs": suf.required_obs, "coverage_pct": suf.coverage_pct,
                   "benchmark_index": s["benchmark_index"], "rf_annual": rf_rate, "rf_source": rf_source,
                   "calc_version": CALC_VERSION, **{c: None for c in m.METRIC_COLUMNS}}
            if not suf.ok:
                row["status"] = "insufficient_history"
            else:
                rf_d = daily_rf(rf_annual, win.index)
                bundle = metric_bundle(win, rf_d, clip(b, start, as_of) if b is not None else None)
                row.update({c: bundle[c] for c in m.METRIC_COLUMNS})
                row["status"] = "idcw_excluded" if s["option_type"] == "IDCW" else "ok"
            rows.append(row)
    return rows


def rebuild_cache(db_path: Path, settings: Settings) -> int:
    """Recompute metrics_cache for the database's current data_version (e.g. after changing
    the configured risk-free rate). Returns the number of rows written."""
    eng = create_engine(f"sqlite+pysqlite:///{db_path.as_posix()}")
    with eng.begin() as c:
        version = int(c.execute(select(m.meta.c.value).where(m.meta.c.key == "data_version")).scalar_one())
        c.execute(m.metrics_cache.delete().where(m.metrics_cache.c.data_version == version))
        rows = _cache_rows(c, version, settings)
        if rows:
            c.execute(insert(m.metrics_cache), rows)
    eng.dispose()
    return len(rows)
