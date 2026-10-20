"""Load the staging CSVs (from fetch_amfi.py / load_manual_sources.py) into the database.
This is the only writer. data_nature and provenance come from staging/sources.json.

Usage: quantagent ingest.py [--raw data/staging] [--db data/db/quantagent.db]
"""
import argparse
import json
from pathlib import Path

from ..config import get_settings
from ..data.engine import sqlite_path
from ..data.ingest import ingest
from ..logging_setup import configure


def main(argv: list[str] | None = None) -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path(s.staging_dir))
    ap.add_argument("--reference", type=Path, default=Path(s.reference_dir))
    ap.add_argument("--db", type=Path, default=None, help="defaults to the file in QUANTAGENT_DB_URL")
    ap.add_argument("--rejected", type=Path, default=Path(s.reports_dir) / "rejected_rows.csv")
    ap.add_argument("--data-nature", choices=["synthetic", "real"], default=None,
                    help="override sources.json")
    a = ap.parse_args(argv)
    configure(s.log_level)
    db = a.db or sqlite_path(s.db_url)
    print(json.dumps(ingest(a.raw, db, a.reference, a.rejected, s, a.data_nature), indent=2))


if __name__ == "__main__":
    main()
