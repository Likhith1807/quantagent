from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import ANNUALISATION, ZERO, clean_float

SQRT_A = np.sqrt(ANNUALISATION)


def volatility(r: pd.Series) -> float | None:
    """std(r, ddof=1) * sqrt(252)."""
    if len(r) < 2:
        return None
    sd = r.std(ddof=1)
    return 0.0 if sd < ZERO else clean_float(sd * SQRT_A)


def sharpe(r: pd.Series, rf: pd.Series) -> float | None:
    """mean(r - rf) / std(r - rf) * sqrt(252). None when the excess series has zero spread."""
    ex = (r - rf.reindex(r.index)).dropna()
    if len(ex) < 2:
        return None
    sd = ex.std(ddof=1)
    return None if sd < ZERO else clean_float(ex.mean() / sd * SQRT_A)


def sortino(r: pd.Series, rf: pd.Series) -> float | None:
    """mean(r - rf) / sqrt(mean(min(r - rf, 0)^2)) * sqrt(252). None with no downside."""
    ex = (r - rf.reindex(r.index)).dropna()
    if len(ex) < 2:
        return None
    dd = np.sqrt(np.mean(np.minimum(ex.to_numpy(), 0.0) ** 2))
    return None if dd < ZERO else clean_float(ex.mean() / dd * SQRT_A)


@dataclass
class Drawdown:
    max_drawdown: float
    peak_date: str | None
    trough_date: str | None
    recovery_date: str | None
    drawdown_days: int | None  # peak -> trough, calendar days
    recovery_days: int | None  # trough -> recovery, calendar days; None if not recovered

    def as_dict(self) -> dict:
        return asdict(self)


def max_drawdown(nav: pd.Series) -> Drawdown | None:
    """min(NAV / cummax(NAV) - 1) with peak, trough and recovery dates."""
    if len(nav) < 2:
        return None
    nav = nav.sort_index()
    dd = nav / nav.cummax() - 1
    trough = dd.idxmin()
    if dd.loc[trough] >= 0:
        return Drawdown(0.0, None, None, None, None, None)
    peak = nav.loc[:trough].idxmax()
    after = nav.loc[trough:]
    rec = after[after >= nav.loc[peak]]
    recovery = rec.index[0] if len(rec) else None
    return Drawdown(
        max_drawdown=float(dd.loc[trough]),
        peak_date=str(peak.date()),
        trough_date=str(trough.date()),
        recovery_date=str(recovery.date()) if recovery is not None else None,
        drawdown_days=int((trough - peak).days),
        recovery_days=int((recovery - trough).days) if recovery is not None else None,
    )
