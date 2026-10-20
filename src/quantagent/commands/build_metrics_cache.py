"""Recompute metrics_cache for the current data_version (ingest already does this).

Usage: quantagent build-cache [--db data/db/quantagent.db]
"""
import argparse
from pathlib import Path

from ..config import get_settings
from ..data.engine import sqlite_path
from ..data.ingest import rebuild_cache


def main(argv: list[str] | None = None) -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=None)
    a = ap.parse_args(argv)
    print(f"metrics_cache rows written: {rebuild_cache(a.db or sqlite_path(s.db_url), s)}")


if __name__ == "__main__":
    main()
