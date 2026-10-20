"""Every metric against an independent NumPy/SciPy calculation, closed forms and invariants."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantagent.analytics import clean_float
from quantagent.analytics.benchmark import (
    align,
    capture_ratios,
    information_ratio,
    regression,
    tracking_error,
)
from quantagent.analytics.returns import (
    cagr,
    correlation_matrix,
    daily_returns,
    daily_rf,
    monthly_returns,
    resample,
    total_return,
)
from quantagent.analytics.risk import max_drawdown, sharpe, sortino, volatility
from quantagent.analytics.windows import metric_bundle, window_start, window_sufficiency

IDX = pd.bdate_range("2023-01-02", periods=600)


def walk(seed, n=600, mu=0.0004, sigma=0.012):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(mu, sigma, n)), index=IDX[:n])


# ------------------------------------------------------------ returns
def test_daily_returns_matches_numpy():
    nav = walk(1)
    np.testing.assert_allclose(daily_returns(nav).to_numpy(), nav.to_numpy()[1:] / nav.to_numpy()[:-1] - 1)


def test_cagr_closed_form():
    nav = pd.Series([100.0, 121.0], index=pd.DatetimeIndex(["2020-01-01", "2021-12-31"]))
    days = (nav.index[1] - nav.index[0]).days
    assert cagr(nav) == pytest.approx(1.21 ** (365.25 / days) - 1, abs=1e-15)
    assert total_return(nav) == pytest.approx(0.21)


def test_cagr_and_total_return_degenerate():
    one = pd.Series([100.0], index=IDX[:1])
    assert cagr(one) is None and total_return(one) is None
    same_day = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2024-01-01", "2024-01-01"]))
    assert cagr(same_day) is None


def test_resample_uses_last_nav_of_period():
    nav = walk(2, 60)
    m = resample(nav, "monthly")
    for d, v in m.items():
        month = nav[(nav.index.year == d.year) & (nav.index.month == d.month)]
        assert d == month.index[-1] and v == month.iloc[-1]
    w = resample(nav, "weekly")
    assert all(d.dayofweek == 4 for d in w.index[:-1])
    assert resample(nav, "daily") is nav
    assert len(monthly_returns(nav)) == len(m) - 1


def test_daily_rf_constant_and_series():
    r = daily_rf(0.0645, IDX[:5])
    assert r.iloc[0] == pytest.approx(1.0645 ** (1 / 252) - 1)
    s = pd.Series([0.05, 0.07], index=pd.DatetimeIndex(["2023-01-01", "2023-01-05"]))
    r2 = daily_rf(s, IDX[:5])  # Jan 2..6
    assert r2.loc["2023-01-04"] == pytest.approx(1.05 ** (1 / 252) - 1)
    assert r2.loc["2023-01-06"] == pytest.approx(1.07 ** (1 / 252) - 1)


# ------------------------------------------------------------ risk
def test_volatility_sharpe_sortino_match_numpy():
    nav = walk(3)
    r = daily_returns(nav)
    rf = daily_rf(0.06, r.index)
    x = r.to_numpy()
    ex = x - rf.to_numpy()
    assert volatility(r) == pytest.approx(np.std(x, ddof=1) * np.sqrt(252), rel=1e-12)
    assert sharpe(r, rf) == pytest.approx(ex.mean() / np.std(ex, ddof=1) * np.sqrt(252), rel=1e-12)
    downside = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
    assert sortino(r, rf) == pytest.approx(ex.mean() / downside * np.sqrt(252), rel=1e-12)


def test_constant_growth_has_zero_vol_and_no_ratio():
    nav = pd.Series(100 * 1.001 ** np.arange(300), index=IDX[:300])
    r = daily_returns(nav)
    rf = daily_rf(0.05, r.index)
    assert volatility(r) == 0.0
    assert sharpe(r, rf) is None
    assert sortino(r, rf) is None  # no downside (growth beats rf every day)
    assert volatility(r.iloc[:1]) is None and sharpe(r.iloc[:1], rf) is None and sortino(r.iloc[:1], rf) is None


def test_planted_drawdown():
    v = np.concatenate([np.linspace(100, 125, 11), np.linspace(125, 100, 11)[1:],
                        np.linspace(100, 125, 11)[1:], [130.0]])
    nav = pd.Series(v, index=IDX[: len(v)])
    dd = max_drawdown(nav)
    assert dd.max_drawdown == pytest.approx(-0.2)
    assert dd.peak_date == str(IDX[10].date())
    assert dd.trough_date == str(IDX[20].date())
    assert dd.recovery_date == str(IDX[30].date())
    assert dd.drawdown_days == (IDX[20] - IDX[10]).days
    assert dd.recovery_days == (IDX[30] - IDX[20]).days


def test_drawdown_not_recovered_and_monotone():
    nav = pd.Series([100.0, 110.0, 90.0, 95.0], index=IDX[:4])
    dd = max_drawdown(nav)
    assert dd.recovery_date is None and dd.recovery_days is None
    assert dd.max_drawdown == pytest.approx(90 / 110 - 1)
    up = max_drawdown(pd.Series([1.0, 2.0, 3.0], index=IDX[:3]))
    assert up.max_drawdown == 0.0 and up.peak_date is None
    assert max_drawdown(pd.Series([1.0], index=IDX[:1])) is None


@pytest.mark.parametrize("seed", range(10))
def test_invariants(seed):
    nav = walk(seed, 600, sigma=0.03)
    r = daily_returns(nav)
    assert max_drawdown(nav).max_drawdown <= 0
    assert volatility(r) >= 0
    reg = regression(r, r)
    assert reg.beta == pytest.approx(1.0) and reg.alpha == pytest.approx(0.0, abs=1e-12)
    assert reg.r_squared == pytest.approx(1.0)
    cap = capture_ratios(nav, nav)
    assert cap.up_capture == pytest.approx(1.0) and cap.down_capture == pytest.approx(1.0)
    assert tracking_error(r, r) == pytest.approx(0.0, abs=1e-15)
    assert information_ratio(r, r) is None


# ------------------------------------------------------------ benchmark
def test_regression_matches_scipy():
    f, b = daily_returns(walk(4)), daily_returns(walk(5))
    f = f + 0.6 * b  # correlated
    lr = stats.linregress(b.to_numpy(), f.to_numpy())
    reg = regression(f, b)
    assert reg.beta == pytest.approx(lr.slope, rel=1e-10)
    assert reg.alpha == pytest.approx(lr.intercept * 252, rel=1e-8)
    assert reg.r_squared == pytest.approx(lr.rvalue**2, rel=1e-10)
    assert reg.n_obs == len(f)


def test_regression_degenerate():
    flat = pd.Series(0.0, index=IDX[:50])
    assert regression(daily_returns(walk(1, 51)), flat).beta is None
    assert regression(flat.iloc[:2], flat.iloc[:2]).n_obs == 2


def test_tracking_error_and_ir_match_numpy():
    f, b = daily_returns(walk(6)), daily_returns(walk(7))
    a = (f - b).to_numpy()
    te = np.std(a, ddof=1) * np.sqrt(252)
    assert tracking_error(f, b) == pytest.approx(te, rel=1e-12)
    assert information_ratio(f, b) == pytest.approx(a.mean() * 252 / te, rel=1e-12)
    assert tracking_error(f.iloc[:1], b.iloc[:1]) is None


def test_capture_ratio_by_hand():
    fund, bench = walk(8), walk(9)
    fm, bm = align(monthly_returns(fund), monthly_returns(bench))
    up = bm > 0
    expected_up = ((1 + fm[up]).prod() - 1) / ((1 + bm[up]).prod() - 1)
    expected_down = ((1 + fm[~up & (bm < 0)]).prod() - 1) / ((1 + bm[bm < 0]).prod() - 1)
    cap = capture_ratios(fund, bench)
    assert cap.up_capture == pytest.approx(expected_up)
    assert cap.down_capture == pytest.approx(expected_down)
    assert cap.up_months + cap.down_months == len(bm[bm != 0])


def test_capture_needs_three_months():
    short = walk(1, 40)
    cap = capture_ratios(short, short)
    assert cap.up_capture is None or cap.down_capture is None


def test_align_and_correlation():
    a, b = walk(1, 10), walk(2, 12)
    x, y = align(a, b)
    assert len(x) == 10 and (x.index == y.index).all()
    c = correlation_matrix(pd.DataFrame({"a": daily_returns(a), "b": daily_returns(a)}))
    assert c["a"]["b"] == pytest.approx(1.0)


# ------------------------------------------------------------ windows
def test_window_sufficiency_and_bundle():
    nav = walk(10, 300)
    end = nav.index[-1]
    start = window_start("1y", end)
    win = nav[nav.index >= start]
    suf = window_sufficiency(win, IDX, start, end)
    assert suf.ok and suf.coverage_pct == pytest.approx(1.0)
    thin = win.iloc[::2]
    assert not window_sufficiency(thin, IDX, start, end).ok
    rf = daily_rf(0.05, win.index)
    b = metric_bundle(win, rf, win * 3)
    assert b["beta"] == pytest.approx(1.0) and b["alpha"] == pytest.approx(0.0, abs=1e-10)
    short = metric_bundle(win.iloc[:50], rf, None)
    assert short["volatility"] is None and short["beta"] is None and short["cagr"] is not None


def test_clean_float():
    assert clean_float(float("nan")) is None and clean_float(float("inf")) is None
    assert clean_float(None) is None and clean_float(1) == 1.0


def test_window_before_calendar_start_is_not_full_coverage():
    cal = pd.bdate_range("2024-01-01", "2025-12-31")
    nav = pd.Series(1.0, index=cal)
    end = cal[-1]
    start = window_start("3y", end)
    suf = window_sufficiency(nav, cal, start, end)
    assert not suf.ok
    assert suf.trading_days == len(pd.bdate_range(start, end))
    assert suf.coverage_pct == pytest.approx(len(cal) / len(pd.bdate_range(start, end)))
    assert not window_sufficiency(nav, pd.DatetimeIndex([]), start, end).ok
