import pandas as pd

from . import ANNUALISATION, clean_float


def daily_returns(nav: pd.Series) -> pd.Series:
    """NAV_t / NAV_{t-1} - 1 between consecutive observations."""
    return nav.sort_index().pct_change().dropna()


def total_return(nav: pd.Series) -> float | None:
    if len(nav) < 2:
        return None
    nav = nav.sort_index()
    return clean_float(nav.iloc[-1] / nav.iloc[0] - 1)


def cagr(nav: pd.Series) -> float | None:
    """(NAV_end / NAV_start)^(365.25 / days) - 1."""
    if len(nav) < 2:
        return None
    nav = nav.sort_index()
    days = (nav.index[-1] - nav.index[0]).days
    if days <= 0:
        return None
    return clean_float((nav.iloc[-1] / nav.iloc[0]) ** (365.25 / days) - 1)


def resample(nav: pd.Series, frequency: str) -> pd.Series:
    """Last NAV of each week (Fri-anchored) or calendar month; index = actual NAV date."""
    if frequency == "daily":
        return nav
    rule = {"weekly": "W-FRI", "monthly": "ME"}[frequency]
    nav = nav.sort_index()
    last_dates = nav.index.to_series().resample(rule).last().dropna()
    return nav.loc[pd.DatetimeIndex(last_dates.values)]


def monthly_returns(nav: pd.Series) -> pd.Series:
    return resample(nav, "monthly").pct_change().dropna()


def daily_rf(annual: float | pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Annual risk-free rate -> per-trading-day rate aligned to `index`.
    A Series of annual rates is forward-filled onto the index (last known rate)."""
    if isinstance(annual, pd.Series):
        rates = annual.sort_index().reindex(annual.index.union(index)).ffill().bfill().reindex(index)
    else:
        rates = pd.Series(float(annual), index=index)
    return (1 + rates) ** (1 / ANNUALISATION) - 1


def correlation_matrix(returns: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    c = returns.corr()
    return {a: {b: clean_float(c.loc[a, b]) for b in c.columns} for a in c.index}
