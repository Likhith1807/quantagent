"""Trading calendar: Mon-Fri minus NSE holidays when a holiday list is supplied.

Without a holiday list, ingest derives holidays from NAV publication dates (see
ingest.derive_holidays) and records that as an assumption."""
from collections.abc import Iterable
from datetime import date

import pandas as pd


def build_calendar(start: date, end: date, holidays: Iterable[date] = ()) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D")
    hol = pd.DatetimeIndex(pd.to_datetime(list(holidays)))
    trading = (days.dayofweek < 5) & ~days.isin(hol)
    return pd.DataFrame({"date": days.date, "is_trading_day": trading.astype(int)})

