"""Download the real fund universe, NAV histories and expense ratios into a staging directory.

  * universe: AMFI NAVAll.txt -> open-ended Equity Scheme categories, Direct plan, Growth option
  * NAV history: mfapi.in (a free mirror of AMFI's historical NAVs), one request per scheme
  * expense ratios: AMFI TER disclosures (Direct-plan total TER, every change point with its date)

Writes data/staging/{schemes,nav_history,expense_ratios}.csv and sources.json. Raw responses are
cached under data/staging/cache/ so a rerun only downloads what is missing. The downloaded data is
not committed; this script is.

Usage: quantagent fetch_amfi.py [--out data/staging] [--min-years 5] [--ter-months 3] [--limit N]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import get_settings
from ..sources import amfi
from ..sources.http import get, get_json


def _mfapi(code: str, cache: Path, refresh: bool) -> dict:
    f = cache / f"{code}.json"
    if f.exists() and not refresh:
        return json.loads(f.read_text(encoding="utf-8"))
    payload = get_json(amfi.MFAPI_URL.format(code=code))
    f.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(get_settings().staging_dir))
    ap.add_argument("--reference", type=Path, default=Path(get_settings().reference_dir))
    ap.add_argument("--min-years", type=float, default=5.0)
    ap.add_argument("--stale-days", type=int, default=10, help="drop schemes with no NAV this close to the latest")
    ap.add_argument("--ter-months", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N schemes (for a quick try)")
    ap.add_argument("--refresh", action="store_true", help="ignore the download cache")
    a = ap.parse_args(argv)

    out, cache = a.out, a.out / "cache" / "mfapi"
    cache.mkdir(parents=True, exist_ok=True)
    retrieved_at = amfi.utcnow()

    navall_text = get(amfi.NAVALL_URL).decode("utf-8", errors="replace")
    (out / "cache" / "NAVAll.txt").write_text(navall_text, encoding="utf-8")
    universe = amfi.parse_navall(navall_text)[: a.limit]
    print(f"universe: {len(universe)} Direct Growth equity schemes in NAVAll.txt", file=sys.stderr)

    with ThreadPoolExecutor(a.workers) as pool:
        payloads = list(pool.map(lambda u: _mfapi(u.scheme_code, cache, a.refresh), universe))
    navs = {u.scheme_code: amfi.parse_mfapi(p) for u, p in zip(universe, payloads, strict=True)}
    latest = max(n["date"].max() for n in navs.values() if len(n)).date()

    kept, dropped = [], {}
    for u in universe:
        ok, why = amfi.keep_scheme(navs[u.scheme_code], latest, a.min_years, a.stale_days)
        if ok:
            kept.append(u)
        else:
            dropped[why] = dropped.get(why, 0) + 1
    print(f"kept {len(kept)}; dropped {dropped}", file=sys.stderr)

    cat_bench = pd.read_csv(a.reference / "category_benchmarks.csv").set_index("sub_category")["benchmark_name"]
    schemes = pd.DataFrame([{
        "amfi_code": u.scheme_code, "isin": u.isin, "scheme_name": u.scheme_name, "fund_house": u.fund_house,
        "category": "Equity", "sub_category": u.sub_category, "plan": "Direct", "option_type": "Growth",
        "launch_date": navs[u.scheme_code]["date"].min().date(), "benchmark": cat_bench.get(u.sub_category),
        "status": "active"} for u in kept])
    nav_rows = pd.concat([navs[u.scheme_code].assign(amfi_code=u.scheme_code) for u in kept])
    nav_rows["date"] = nav_rows["date"].dt.strftime("%Y-%m-%d")

    months = amfi.recent_months(date.today(), a.ter_months)
    ter_rows = amfi.ter_pages(get_json, months) if a.ter_months else []
    ter, unmatched = amfi.ter_history(ter_rows, kept)
    print(f"TER: {len(ter)} change points for {ter['amfi_code'].nunique()} schemes; "
          f"{unmatched} scheme names without a TER match", file=sys.stderr)

    out.mkdir(parents=True, exist_ok=True)
    schemes.to_csv(out / "schemes.csv", index=False)
    nav_rows[["amfi_code", "date", "nav"]].to_csv(out / "nav_history.csv", index=False)
    ter.to_csv(out / "expense_ratios.csv", index=False)

    sources_path = out / "sources.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else {}
    sources["data_nature"] = "real"
    files = sources.setdefault("files", {})
    files["schemes.csv"] = {
        "source": "AMFI NAVAll.txt (scheme universe, category, fund house, ISIN)", "url": amfi.NAVALL_URL,
        "retrieved_at": retrieved_at, "licence": "Public AMFI disclosure; AMFI website terms of use apply",
        "coverage": f"{len(schemes)} open-ended equity Direct Growth schemes with >= {a.min_years:g}y of NAVs "
                    f"and a NAV within {a.stale_days} days of {latest}",
        "notes": "launch_date is the first NAV in the source, not the official inception date; benchmark is "
                 "assigned by SEBI category from data/reference/category_benchmarks.csv"}
    files["nav_history.csv"] = {
        "source": "mfapi.in historical NAVs (mirror of AMFI daily NAV files)", "url": "https://api.mfapi.in/mf/{code}",
        "retrieved_at": retrieved_at, "licence": "Free public API, no licence stated; underlying data is AMFI's",
        "coverage": f"{len(nav_rows)} NAVs, {nav_rows['date'].min()} to {nav_rows['date'].max()}"}
    files["expense_ratios.csv"] = {
        "source": "AMFI TER disclosures (Direct plan total TER)", "url": amfi.TER_URL.split("?")[0],
        "retrieved_at": retrieved_at, "licence": "Public AMFI disclosure; AMFI website terms of use apply",
        "coverage": f"months {', '.join(months)}; {ter['amfi_code'].nunique()} of {len(schemes)} schemes matched "
                    "by scheme name"}
    sources_path.write_text(json.dumps(sources, indent=2), encoding="utf-8")
    print(json.dumps({"schemes": len(schemes), "nav_rows": len(nav_rows), "ter_rows": len(ter),
                      "latest_nav": str(latest)}, indent=2))


if __name__ == "__main__":
    main()
