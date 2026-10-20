from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantagent.tools.benchmark import get_benchmark_comparison
from quantagent.tools.risk import get_risk_metrics
from tests.conftest import RF
from tests.fixtures import make_fixture as fx
from tests.tools.helpers import err, has


# ------------------------------------------------------------ get_risk_metrics
def test_risk_planted_drawdown_and_independent_vol(repo):
    r = get_risk_metrics(scheme_code=fx.DRAWDOWN, period="max")
    dd = r.max_drawdown
    assert dd.max_drawdown == pytest.approx(-0.2)
    assert (dd.peak_date, dd.trough_date, dd.recovery_date) == (
        fx.TRADING[fx.DD_PEAK].date(), fx.TRADING[fx.DD_TROUGH].date(), fx.TRADING[fx.DD_RECOVERY].date())
    nav = repo.nav_series(fx.DRAWDOWN).to_numpy()
    ret = nav[1:] / nav[:-1] - 1
    assert r.volatility == pytest.approx(np.std(ret, ddof=1) * np.sqrt(252), rel=1e-12)
    ex = ret - ((1 + RF) ** (1 / 252) - 1)
    assert r.sharpe == pytest.approx(ex.mean() / ex.std(ddof=1) * np.sqrt(252), rel=1e-12)
    assert r.n_obs == len(fx.TRADING) and r.benchmark.index_name == fx.BENCH_TRI
    assert r.risk_free_annual == RF and r.risk_free_source == "test default"
    assert r.meta.period.start == fx.TRADING[0].date()


def test_risk_tracker_against_itself_and_pri_warning():
    r = get_risk_metrics(scheme_code=fx.TRACKER, period="1y")
    assert r.beta == pytest.approx(1.0) and r.alpha == pytest.approx(0.0, abs=1e-10)
    assert r.r_squared == pytest.approx(1.0)
    assert r.benchmark.index_type == "PRI" and has(r.warnings, "price index")
    assert has(r.warnings, "not economically interpretable")
    t = get_risk_metrics(scheme_code=fx.TRACKER, period="1y", benchmark=fx.BENCH_TRI)
    assert t.benchmark.index_type == "TRI" and not has(t.warnings, "price index")


def test_risk_without_benchmark_and_override_rate():
    r = get_risk_metrics(scheme_code=fx.GAPPY, period="1y", risk_free_annual=0.0)
    assert r.benchmark is None and r.alpha is None and has(r.warnings, "no benchmark series")
    assert r.risk_free_source == "user-supplied risk_free_annual"
    assert r.sharpe != get_risk_metrics(scheme_code=fx.GAPPY, period="1y").sharpe


def test_risk_idcw_withholds_drawdown():
    r = get_risk_metrics(scheme_code=fx.IDCW, period="1y")
    assert r.max_drawdown is None and r.volatility is not None and has(r.warnings, "IDCW")


def test_risk_insufficient_history():
    e = err(get_risk_metrics, scheme_code=fx.SHORT, period="1y")
    assert e.code == "INSUFFICIENT_HISTORY" and e.details["found"] == 30 and e.details["required"] > 200
    e2 = err(get_risk_metrics, scheme_code=fx.SHORT, start_date=date(2024, 1, 1), end_date=date(2025, 12, 31))
    assert e2.code == "INSUFFICIENT_HISTORY" and e2.details == {"found": 30, "required": 61}
    assert err(get_risk_metrics, scheme_code=fx.DRAWDOWN, period="3y").code == "INSUFFICIENT_HISTORY"


def test_risk_short_benchmark_overlap_warns(repo, monkeypatch):
    real = repo.benchmark_series
    monkeypatch.setattr(repo, "benchmark_series", lambda name, *a: real(name).iloc[-50:])
    r = get_risk_metrics(scheme_code=fx.DRAWDOWN, period="1y")
    assert r.beta is None and has(r.warnings, "returns shared with the benchmark")


@pytest.mark.parametrize("kwargs,code", [
    ({"scheme_code": fx.DRAWDOWN, "benchmark": "NOPE"}, "NOT_FOUND"),
    ({"scheme_code": fx.DRAWDOWN, "benchmark": "x; drop"}, "INVALID_INPUT"),
    ({"scheme_code": fx.DRAWDOWN, "period": "1y", "start_date": date(2025, 1, 1)}, "INVALID_INPUT"),
    ({"scheme_code": fx.DRAWDOWN, "risk_free_annual": 0.5}, "INVALID_INPUT"),
    ({"scheme_code": fx.DRAWDOWN, "period": "2y"}, "INVALID_INPUT"),
    ({"scheme_code": "999999"}, "NOT_FOUND"),
])
def test_risk_errors(kwargs, code):
    e = err(get_risk_metrics, **kwargs)
    assert e.code == code
    if code == "NOT_FOUND" and "benchmark" in kwargs:
        assert e.details["available"] == [fx.BENCH_PRI, fx.BENCH_TRI]


# ------------------------------------------------------------ get_benchmark_comparison
def test_benchmark_tracker_identities():
    r = get_benchmark_comparison(scheme_code=fx.TRACKER, period="1y")
    assert r.beta == pytest.approx(1.0) and r.r_squared == pytest.approx(1.0)
    assert r.tracking_error == pytest.approx(0.0, abs=1e-9) and r.information_ratio is None
    assert r.up_capture == pytest.approx(1.0) and r.down_capture == pytest.approx(1.0)
    assert r.excess_cagr == pytest.approx(0.0, abs=1e-12) and has(r.warnings, "price index")
    assert r.up_months + r.down_months == 12


def test_benchmark_independent_check(repo):
    r = get_benchmark_comparison(scheme_code=fx.DRAWDOWN, period="1y")
    end = fx.END
    start = end - pd.DateOffset(years=1)
    f = repo.nav_series(fx.DRAWDOWN, start.date(), end.date())
    b = repo.benchmark_series(fx.BENCH_TRI, start.date(), end.date())
    rf, rb = f.pct_change().dropna(), b.pct_change().dropna()
    active = (rf - rb).to_numpy()
    assert r.tracking_error == pytest.approx(active.std(ddof=1) * np.sqrt(252), rel=1e-12)
    assert r.n_shared == len(f) and r.shared_period.end == end.date()
    assert r.fund.cagr == pytest.approx((f.iloc[-1] / f.iloc[0]) ** (365.25 / (f.index[-1] - f.index[0]).days) - 1)
    assert r.excess_cagr == pytest.approx(r.fund.cagr - r.benchmark_stats.cagr)


def test_benchmark_explicit_for_scheme_without_one_and_idcw():
    assert get_benchmark_comparison(scheme_code=fx.GAPPY, benchmark=fx.BENCH_TRI, period="1y").benchmark.index_name == fx.BENCH_TRI
    i = get_benchmark_comparison(scheme_code=fx.IDCW, period="1y")
    assert i.fund.cagr is None and i.fund.max_drawdown is None and i.excess_cagr is None and has(i.warnings, "IDCW")


def test_benchmark_errors(repo, monkeypatch):
    e = err(get_benchmark_comparison, scheme_code=fx.GAPPY)
    assert e.code == "NOT_FOUND" and fx.BENCH_TRI in e.details["available"]
    assert err(get_benchmark_comparison, scheme_code=fx.DRAWDOWN, benchmark="NOPE").code == "NOT_FOUND"
    assert err(get_benchmark_comparison, scheme_code=fx.SHORT, period="1y").code == "INSUFFICIENT_HISTORY"
    real = repo.benchmark_series
    monkeypatch.setattr(repo, "benchmark_series", lambda name, *a: real(name).iloc[-60:])
    e = err(get_benchmark_comparison, scheme_code=fx.DRAWDOWN, period="1y")
    assert e.code == "INSUFFICIENT_HISTORY" and e.details["required"] == 120
    monkeypatch.setattr(repo, "benchmark_series", lambda name, *a: real(name).iloc[:0])
    assert err(get_benchmark_comparison, scheme_code=fx.DRAWDOWN, period="1y").code == "NO_OVERLAP"


def test_benchmark_short_capture_warning(repo, monkeypatch):
    real = repo.benchmark_series
    monkeypatch.setattr(repo, "benchmark_series", lambda name, *a: real(name).iloc[-130:])
    r = get_benchmark_comparison(scheme_code=fx.DRAWDOWN, period="1y")
    assert r.n_shared == 130
    if r.up_capture is None or r.down_capture is None:
        assert has(r.warnings, "capture ratio is not computed")
