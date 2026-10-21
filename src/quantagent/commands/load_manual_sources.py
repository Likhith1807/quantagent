"""Convert the hand-downloaded files into staging CSVs and record their provenance.

  * Nifty TRI CSVs from niftyindices.com, placed in data/external/niftyindices/ (one file per index)
  * 91-day T-bill yields (percent p.a.) in data/external/tbill_91d.csv, from RBI DBIE or CCIL

Writes data/staging/benchmark_indices.csv and/or risk_free_rates.csv and updates sources.json.
Either input may be missing: without TRI files benchmark tools report the benchmark as missing;
without T-bill yields the configured QUANTAGENT_RISK_FREE_ANNUAL is used and labelled an assumption.

Usage: quantagent load_manual_sources.py [--tri-dir DIR] [--tbill FILE --tbill-source TEXT]
"""
import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from ..config import get_settings
from ..sources import niftyindices, tbill


def _file_info(paths: list[Path]) -> dict:
    return {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                     "modified": datetime.fromtimestamp(p.stat().st_mtime, UTC).isoformat(timespec="seconds")}
            for p in paths}


def main(argv: list[str] | None = None) -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(s.staging_dir))
    ap.add_argument("--tri-dir", type=Path, default=Path(s.external_dir) / "niftyindices")
    ap.add_argument("--tbill", type=Path, default=Path(s.external_dir) / "tbill_91d.csv")
    ap.add_argument("--tbill-source", default="RBI DBIE: Auctions of 91-Day Government of India Treasury Bills, "
                    "implicit yield at cut-off price (percent, actual/365)")
    ap.add_argument("--tbill-url", default="https://data.rbi.org.in/DBIE/ (Statistics > Financial Market > "
                    "Government Securities Market > Auctions of 91-Day Government of India Treasury Bills)")
    ap.add_argument("--retrieved", default=None, help="download date (YYYY-MM-DD); default: file timestamps")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    sources_path = a.out / "sources.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else {}
    files = sources.setdefault("files", {})
    report = {}

    tri_files = sorted(a.tri_dir.glob("*.csv")) if a.tri_dir.exists() else []
    if tri_files:
        tri = niftyindices.load_dir(a.tri_dir)
        tri.to_csv(a.out / "benchmark_indices.csv", index=False)
        cov = {k: f"{g['date'].min()} to {g['date'].max()} ({len(g)} days)" for k, g in tri.groupby("index_name")}
        files["benchmark_indices.csv"] = {
            "source": "NSE Indices (niftyindices.com) Total Returns Index, downloaded by hand",
            "url": "https://www.niftyindices.com/reports/historical-data",
            "retrieved_at": a.retrieved or max(i["modified"] for i in _file_info(tri_files).values()),
            "licence": "NSE Indices Ltd data; niftyindices.com terms of use (no redistribution)",
            "coverage": cov, "inputs": _file_info(tri_files)}
        report["benchmarks"] = cov
    else:
        report["benchmarks"] = f"no CSVs in {a.tri_dir}"

    if a.tbill.exists():
        rf = tbill.parse_tbill_csv(a.tbill, a.tbill_source)
        rf.to_csv(a.out / "risk_free_rates.csv", index=False)
        files["risk_free_rates.csv"] = {
            "source": a.tbill_source, "url": a.tbill_url,
            "retrieved_at": a.retrieved or _file_info([a.tbill])[a.tbill.name]["modified"],
            "licence": "RBI / CCIL public statistics; their terms of use apply",
            "coverage": f"{rf['date'].min()} to {rf['date'].max()} ({len(rf)} observations)",
            "inputs": _file_info([a.tbill])}
        report["risk_free"] = files["risk_free_rates.csv"]["coverage"]
    else:
        report["risk_free"] = f"{a.tbill} not found; the configured default rate will be used"

    sources_path.write_text(json.dumps(sources, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
