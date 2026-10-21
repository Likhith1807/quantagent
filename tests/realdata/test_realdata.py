"""Tool outputs on the real database vs an independent pandas calculation from the staging CSVs.

Run after fetching and ingesting (see README):
    python scripts/fetch_amfi.py && python scripts/ingest.py && pytest -m realdata
Schemes are chosen from the data (longest histories), so the tests work on any download date.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.config import Settings
from quantagent.data.engine import make_readonly_engine
from quantagent.data.repository import Repository
from quantagent.tools import _base
from quantagent.tools.benchmark import get_benchmark_comparison
from quantagent.tools.compare import compare_funds
from quantagent.tools.provenance import get_data_provenance
from quantagent.tools.rank import rank_funds
from quantagent.tools.risk import get_risk_metrics

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "db" / "quantagent.db"
STAGING = ROOT / "data" / "staging"
pytestmark = [pytest.mark.realdata,
              pytest.mark.skipif(not (DB.exists() and (STAGING / "nav_history.csv").exists()),
                                 reason="fetch real data and run scripts/ingest.py first")]
RF = 0.065
TOL = 1e-9


@pytest.fixture(scope="module")
def raw():
    """Independent pipeline straight from the staging CSVs (no project code)."""
    nav = pd.read_csv(STAGING / "nav_history.csv", dtype={"amfi_code": str})
    nav["date"] = pd.to_datetime(nav["date"], errors="coerce")
    nav["nav"] = pd.to_numeric(nav["nav"], errors="coerce")
    nav = nav.dropna()
    nav = nav[nav["nav"] > 0].drop_duplicates(["amfi_code", "date"], keep=False)
    navs = nav.pivot(index="date", columns="amfi_code", values="nav").sort_index()
    bench = None
    if (STAGING / "benchmark_indices.csv").exists():
        b = pd.read_csv(STAGING / "benchmark_indices.csv", parse_dates=["date"])
        bench = b.pivot(index="date", columns="index_name", values="close_value").sort_index()
    rf = None
    if (STAGING / "risk_free_rates.csv").exists():
        rf = pd.read_csv(STAGING / "risk_free_rates.csv", parse_dates=["date"]).set_index("date")["annual_rate"]
    return navs, bench, rf


@pytest.fixture(autouse=True)
def services():
    s = Settings(db_url=f"sqlite:///{DB.as_posix()}", risk_free_annual=RF, _env_file=None)
    _base.set_services(_base.Services(s, Repository(make_readonly_engine(s.db_url))))
    yield
    _base.set_services(None)


def _codes(navs, n=5):
    return list(navs.count().sort_values(ascending=False, kind="stable").index[:n])


def _rf_daily(rf, idx):
    if rf is None:
        return pd.Series((1 + RF) ** (1 / 252) - 1, index=idx)
    r = rf.sort_index().reindex(rf.index.union(idx)).ffill().bfill().reindex(idx)
    return (1 + r) ** (1 / 252) - 1


@pytest.mark.parametrize("years", [1, 3])
@pytest.mark.parametrize("i", range(5))
def test_risk_metrics_match_independent_pandas(raw, years, i):
    navs, _, rf = raw
    code = _codes(navs)[i]
    end = navs.index.max()
    s = navs[code][navs.index >= end - pd.DateOffset(years=years)].dropna()
    r = s.pct_change().dropna()
    ex = r - _rf_daily(rf, s.index).reindex(r.index)
    out = get_risk_metrics(scheme_code=code, period=f"{years}y")
    assert out.meta.data_nature == "real" and out.meta.data_retrieved_at
    assert out.n_obs == len(s)
    assert out.volatility == pytest.approx(r.std() * np.sqrt(252), rel=TOL)
    assert out.sharpe == pytest.approx(ex.mean() / ex.std() * np.sqrt(252), rel=TOL)
    assert out.max_drawdown.max_drawdown == pytest.approx((s / s.cummax() - 1).min(), rel=TOL)


def test_compare_matches_independent_shared_window(raw):
    navs, _, _ = raw
    a, b = _codes(navs, 2)
    end = navs.index.max()
    w = navs[[a, b]][navs.index >= end - pd.DateOffset(years=3)].dropna()
    days = (w.index[-1] - w.index[0]).days
    out = compare_funds(scheme_codes=[a, b], period="3y")
    for f in out.funds:
        s = w[f.scheme_code]
        assert f.cagr == pytest.approx((s.iloc[-1] / s.iloc[0]) ** (365.25 / days) - 1, rel=TOL)
        assert f.volatility == pytest.approx(s.pct_change().std() * np.sqrt(252), rel=TOL)


def test_rank_matches_independent_cagr(raw):
    navs, _, _ = raw
    end = navs.index.max()
    out = rank_funds(metric="cagr", period="1y", dedupe="none", top_n=50)
    assert len(out.ranked) == 50
    for f in out.ranked:
        s = navs[f.scheme_code][navs.index >= end - pd.DateOffset(years=1)].dropna()
        days = (s.index[-1] - s.index[0]).days
        assert f.metrics["cagr"] == pytest.approx((s.iloc[-1] / s.iloc[0]) ** (365.25 / days) - 1, rel=TOL)
    cagrs = [f.metrics["cagr"] for f in out.ranked]
    assert cagrs == sorted(cagrs, reverse=True)


def test_benchmark_tracking_error_matches(raw):
    navs, bench, _ = raw
    if bench is None:
        pytest.skip("no Nifty TRI files loaded (scripts/load_manual_sources.py)")
    repo = _base.get_services().repo
    code = next(c for c in _codes(navs, 50) if repo.get_scheme(c)["benchmark_index"] in bench.columns)
    index = repo.get_scheme(code)["benchmark_index"]
    end = navs.index.max()
    j = pd.concat([navs[code], bench[index]], axis=1).dropna()
    j = j[j.index >= end - pd.DateOffset(years=3)]
    r = j.pct_change().dropna()
    out = get_benchmark_comparison(scheme_code=code, period="3y")
    assert out.benchmark.index_type == "TRI"
    assert out.tracking_error == pytest.approx((r[code] - r[index]).std() * np.sqrt(252), rel=TOL)


@pytest.mark.parametrize("i", range(5))
def test_alpha_beta_match_independent_and_agree_across_tools(raw, i):
    navs, bench, rf = raw
    if bench is None:
        pytest.skip("no Nifty TRI files loaded (quantagent load-manual)")
    repo = _base.get_services().repo
    code = [c for c in _codes(navs, 50) if repo.get_scheme(c)["benchmark_index"] in bench.columns][i]
    index = repo.get_scheme(code)["benchmark_index"]
    end = navs.index.max()
    j = pd.concat([navs[code], bench[index]], axis=1).dropna()
    j = j[j.index >= end - pd.DateOffset(years=3)]
    r = j.pct_change().dropna()
    ex = r.sub(_rf_daily(rf, j.index).reindex(r.index), axis=0)
    beta = ex[code].cov(ex[index]) / ex[index].var()
    alpha = (ex[code].mean() - beta * ex[index].mean()) * 252
    risk = get_risk_metrics(scheme_code=code, period="3y")
    comp = get_benchmark_comparison(scheme_code=code, period="3y")
    for out in (risk, comp):
        assert out.beta == pytest.approx(beta, rel=TOL)
        assert out.alpha == pytest.approx(alpha, rel=TOL)


def test_provenance_records_real_sources(raw):
    p = get_data_provenance()
    assert p.data_nature == "real"
    by_file = {s.file: s for s in p.sources}
    assert "mfapi.in" in by_file["nav_history.csv"].url and by_file["nav_history.csv"].retrieved_at
    assert by_file["schemes.csv"].licence and by_file["expense_ratios.csv"].coverage
    sources = json.loads((STAGING / "sources.json").read_text(encoding="utf-8"))
    assert sources["data_nature"] == "real"


def test_every_nav_row_accounted_for(raw):
    staged = len(pd.read_csv(STAGING / "nav_history.csv"))
    p = get_data_provenance()
    nav_log = next(r for r in p.last_ingest if r.table_name == "nav_history")
    assert nav_log.rows_loaded + nav_log.rows_rejected == staged
    assert p.table_counts["nav_history"] == nav_log.rows_loaded
