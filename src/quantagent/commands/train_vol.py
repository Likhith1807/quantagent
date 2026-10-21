"""Train and evaluate the 21-day volatility model on the current database, then register it.

Writes models/registry/<version>/, appends reports/experiments.jsonl and rewrites
reports/vol_model_report.md. Deterministic for a given database and config (fixed seeds).

Usage: quantagent train_vol.py [--no-garch] [--max-schemes N] [--first-test-year 2018]
"""
import argparse
import json
from pathlib import Path

from ..config import get_settings
from ..data.engine import make_readonly_engine
from ..data.repository import Repository
from ..ml.train import run
from ..ml.walkforward import Config


def main(argv: list[str] | None = None) -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=Path(s.model_dir))
    ap.add_argument("--report", type=Path, default=Path(s.reports_dir) / "vol_model_report.md")
    ap.add_argument("--experiments", type=Path, default=Path(s.reports_dir) / "experiments.jsonl")
    ap.add_argument("--no-garch", action="store_true")
    ap.add_argument("--max-schemes", type=int, default=None)
    ap.add_argument("--first-test-year", type=int, default=None)
    ap.add_argument("--gap-days", type=int, default=30)
    ap.add_argument("--holdout-years", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    repo = Repository(make_readonly_engine(s.db_url, timeout_s=600))
    cfg = Config(garch=not a.no_garch, first_test_year=a.first_test_year, gap_days=a.gap_days, seed=a.seed,
                 holdout_years=a.holdout_years)
    out = run(repo, cfg, a.registry, a.report, a.experiments, Path.cwd(), a.max_schemes)
    print(json.dumps({"version": out["version"], "models": out["models"], "selected": out["selected"]}, indent=2))


if __name__ == "__main__":
    main()
