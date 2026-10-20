import csv
import json
import shutil

import pandas as pd
import pytest
from sqlalchemy import create_engine, select

from quantagent.analytics.returns import daily_rf
from quantagent.analytics.windows import clip, metric_bundle, window_start
from quantagent.data import models as m
from quantagent.data.engine import make_readonly_engine
from quantagent.data.ingest import ingest, rebuild_cache
from quantagent.data.repository import Repository
from tests.conftest import RF
from tests.fixtures import make_fixture as fx


def test_ingest_summary_and_rejections(built):
    s = built["summary"]
    assert s["schemes"] == 9 and s["data_version"] == 1
    with built["rejected"].open(encoding="utf-8") as f:
        reasons = sorted(r["reason"] for r in csv.DictReader(f))
    assert reasons == ["bad_scheme_code", "duplicate_date", "future_date", "missing_or_unparseable",
                       "nonpositive_value", "unknown_scheme"]


def test_calendar_excludes_holidays(repo):
    cal = repo.trading_calendar
    for h in fx.HOLIDAYS:
        assert pd.Timestamp(h) not in cal
    assert (cal.dayofweek < 5).all()


def test_schemes_loaded_with_types(repo):
    s = repo.get_scheme(fx.IDCW)
    assert s["option_type"] == "IDCW" and s["benchmark_index"] == fx.BENCH_TRI
    c = repo.get_scheme(fx.CLOSED)
    assert c["status"] == "closed" and str(c["closure_date"]) == "2025-06-30"
    assert repo.get_scheme(fx.GAPPY)["benchmark_index"] is None
    assert {b["index_name"]: b["index_type"] for b in repo.benchmark_names()} == {
        fx.BENCH_TRI: "TRI", fx.BENCH_PRI: "PRI"}
    ter = repo.latest_ter([fx.CONST_DIRECT])[fx.CONST_DIRECT]
    assert ter["ter"] == 0.5 and ter["as_of_assumed"] == 1


def test_ingest_log_and_meta(repo):
    log = repo.last_ingest()
    nav = next(r for r in log if r["table_name"] == "nav_history")
    assert nav["rows_rejected"] == 5
    cov = json.loads(nav["coverage"])["per_scheme"]
    assert cov[fx.GAPPY]["issues"]["calendar_gaps"] == 10
    assert cov[fx.STALE]["issues"] == {"extreme_moves": 1, "stale_runs": 1}
    assert repo.meta["data_nature"] == "synthetic"
    assert "NSE holidays" in repo.meta["calendar_source"]
    assert any("risk-free" in a for a in repo.meta["assumptions"])


def test_cache_equals_fresh_calculation(repo, built):
    cache = repo.metrics_cache("1y", 1).set_index("scheme_code")
    as_of = pd.Timestamp(repo.meta["data_as_of"])
    start = window_start("1y", as_of)
    bench = clip(repo.benchmark_series(fx.BENCH_TRI), start, as_of)
    for code in (fx.DRAWDOWN, fx.STALE, fx.TRACKER):
        nav = clip(repo.nav_series(code), start, as_of)
        b = bench if code != fx.TRACKER else clip(repo.benchmark_series(fx.BENCH_PRI), start, as_of)
        fresh = metric_bundle(nav, daily_rf(RF, nav.index), b)
        for col in m.METRIC_COLUMNS:
            cached = None if pd.isna(cache.loc[code, col]) else cache.loc[code, col]
            assert cached == pytest.approx(fresh[col], rel=1e-12), (code, col)
    assert cache.loc[fx.SHORT, "status"] == "insufficient_history"
    assert cache.loc[fx.IDCW, "status"] == "idcw_excluded"
    assert (repo.metrics_cache("3y", 1)["status"] == "insufficient_history").all()


def test_version_bump_invalidates_old_cache(built, tmp_path):
    db = tmp_path / "v.db"
    shutil.copy(built["db"], db)
    s2 = ingest(built["raw"], db, built["ref"], tmp_path / "rej.csv", built["settings"])
    assert s2["data_version"] == 2
    r = Repository(make_readonly_engine(f"sqlite:///{db.as_posix()}"))
    assert r.meta["data_version"] == 2
    assert r.cache_versions() == [2]
    assert r.metrics_cache("1y", 1).empty


def test_rebuild_cache_with_new_rate(built, tmp_path):
    db = tmp_path / "c.db"
    shutil.copy(built["db"], db)
    settings = built["settings"].model_copy(update={"risk_free_annual": 0.10})
    assert rebuild_cache(db, settings) == 27
    eng = create_engine(f"sqlite:///{db.as_posix()}")
    with eng.connect() as c:
        rates = {r[0] for r in c.execute(select(m.metrics_cache.c.rf_annual))}
    eng.dispose()
    assert rates == {0.10}


def test_risk_free_file_is_used(built, tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(built["raw"], raw)
    pd.DataFrame({"date": ["2024-01-01", "2025-01-01"], "annual_rate": [0.06, 0.07],
                  "source": ["RBI 91d T-bill"] * 2}).to_csv(raw / "risk_free_rates.csv", index=False)
    db = tmp_path / "rf.db"
    ingest(raw, db, built["ref"], tmp_path / "rej.csv", built["settings"])
    r = Repository(make_readonly_engine(f"sqlite:///{db.as_posix()}"))
    series, source = r.risk_free_series()
    assert list(series) == [0.06, 0.07] and "RBI" in source
    assert r.metrics_cache("1y", 1)["rf_annual"].isna().all()
