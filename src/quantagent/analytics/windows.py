"""Window resolution, history sufficiency, and the standard metric bundle.

Both the metrics cache and the tools call `metric_bundle`, so a cached value is by
construction the same calculation as a fresh one."""
import math
from dataclasses import dataclass

import pandas as pd

from . import MIN_RETURNS_RISK, MIN_SHARED_OBS_REGRESSION, MIN_WINDOW_COVERAGE, clean_float
from .benchmark import align, regression
from .returns import cagr, daily_returns, total_return
from .risk import max_drawdown, sharpe, sortino, volatility

PERIOD_YEARS = {"1y": 1, "3y": 3, "5y": 5}


def window_start(period: str, end: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(end) - pd.DateOffset(years=PERIOD_YEARS[period])


def clip(s: pd.Series, start, end) -> pd.Series:
    return s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


@dataclass
class Sufficiency:
    ok: bool
    n_obs: int
    trading_days: int
    coverage_pct: float | None
    required_obs: int


def window_sufficiency(nav_window: pd.Series, calendar: pd.DatetimeIndex, start, end) -> Sufficiency:
    """A 1y/3y/5y metric needs NAVs on >= 90% of the window's trading days and >= 60 returns.
    Window days outside the calendar's span (i.e. before the data starts) count as weekdays,
    so a window reaching back before the first NAV is not mistaken for full coverage."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    n_days = int(((calendar >= start) & (calendar <= end)).sum())
    if len(calendar) == 0:
        n_days = len(pd.bdate_range(start, end))
    else:
        if start < calendar[0]:
            n_days += len(pd.bdate_range(start, calendar[0] - pd.Timedelta(days=1)))
        if end > calendar[-1]:
            n_days += len(pd.bdate_range(calendar[-1] + pd.Timedelta(days=1), end))
    n_obs = len(nav_window)
    required = max(math.ceil(n_days * MIN_WINDOW_COVERAGE), MIN_RETURNS_RISK + 1)
    cov = n_obs / n_days if n_days else None
    return Sufficiency(n_obs >= required, n_obs, n_days, clean_float(cov), required)


def metric_bundle(nav: pd.Series, rf_daily: pd.Series, bench: pd.Series | None) -> dict:
    """Standard metrics on one NAV window. Risk ratios need >= 60 returns; alpha/beta need
    >= 120 shared returns with the benchmark. Anything short is None."""
    r = daily_returns(nav)
    enough = len(r) >= MIN_RETURNS_RISK
    dd = max_drawdown(nav)
    out = {
        "total_return": total_return(nav),
        "cagr": cagr(nav),
        "volatility": volatility(r) if enough else None,
        "sharpe": sharpe(r, rf_daily) if enough else None,
        "sortino": sortino(r, rf_daily) if enough else None,
        "max_drawdown": dd.max_drawdown if dd else None,
        "beta": None, "alpha": None, "r_squared": None,
        "n_returns": len(r),
        "n_shared_with_benchmark": None,
    }
    if bench is not None and len(bench) > 1:
        # Align levels before taking returns, so each fund/benchmark return pair spans the same dates.
        fn, bn = align(nav, bench)
        f, b = daily_returns(fn), daily_returns(bn)
        rf = rf_daily.reindex(f.index)
        out["n_shared_with_benchmark"] = len(f)
        if len(f) >= MIN_SHARED_OBS_REGRESSION:
            reg = regression(f - rf, b - rf)
            out.update(beta=reg.beta, alpha=reg.alpha, r_squared=reg.r_squared)
    return out
