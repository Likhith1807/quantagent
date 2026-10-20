from typing import Annotated, Literal

import pandas as pd
from pydantic import Field

from ..analytics.returns import resample
from ..schemas.common import EndDate, SchemeCode, StartDate
from ..schemas.tools import Gap, NavHistoryResult, NavPoint
from ._base import base_warnings, build_meta, get_services, require_scheme, resolve_window, tool_handler

MAX_GAPS = 20


def calendar_gaps(nav_index: pd.DatetimeIndex, calendar: pd.DatetimeIndex) -> tuple[int, list[Gap]]:
    """Trading days with no NAV between the first and last NAV, grouped into runs."""
    if len(nav_index) == 0:
        return 0, []
    span = calendar[(calendar >= nav_index[0]) & (calendar <= nav_index[-1])]
    missing = span.difference(nav_index)
    if missing.empty:
        return 0, []
    pos = span.get_indexer(missing)
    runs, start = [], 0
    for i in range(1, len(pos) + 1):
        if i == len(pos) or pos[i] != pos[i - 1] + 1:
            runs.append(Gap(start=missing[start].date(), end=missing[i - 1].date(), missing_trading_days=i - start))
            start = i
    runs.sort(key=lambda g: (-g.missing_trading_days, g.start))
    return len(missing), runs[:MAX_GAPS]


@tool_handler
def get_nav_history(
    scheme_code: SchemeCode,
    start_date: StartDate = None,
    end_date: EndDate = None,
    frequency: Literal["daily", "weekly", "monthly"] = "daily",
    max_points: Annotated[int, Field(ge=1, le=5000)] = 2000,
) -> NavHistoryResult:
    """Return the NAV time series of one scheme, optionally resampled to weekly or monthly.

    Use for charts or to inspect raw prices. Do NOT use to compute returns or risk yourself:
    get_risk_metrics, get_benchmark_comparison and compare_funds do that with documented methods.
    Without dates, the full history is returned. If more than max_points points exist, the most
    recent max_points are returned and truncated=true. Daily output also reports trading days
    with no NAV (gaps) on the trading calendar.
    """
    svc = get_services()
    scheme = require_scheme(svc.repo, scheme_code)
    start, end, _, warnings = resolve_window(svc.repo, svc.settings, None, start_date, end_date, default="max")
    if not (start_date or end_date):
        start = None
    nav = svc.repo.nav_series(scheme_code, start.date() if start is not None else None, end.date())
    warnings = base_warnings(svc.repo) + warnings
    if nav.empty:
        warnings.append("No NAVs in the requested range.")
    missing, gaps = (calendar_gaps(nav.index, svc.repo.trading_calendar) if frequency == "daily" else (None, []))
    series = resample(nav, frequency)
    truncated = len(series) > max_points
    if truncated:
        series = series.iloc[-max_points:]
        warnings.append(f"Truncated to the latest {max_points} points; use weekly/monthly or a narrower range.")
    points = [NavPoint(date=d.date(), nav=float(v)) for d, v in series.items()]
    return NavHistoryResult(
        meta=build_meta(svc.repo, series.index.min() if len(series) else None,
                        series.index.max() if len(series) else None, len(series)),
        warnings=warnings, scheme_code=scheme_code, scheme_name=scheme["scheme_name"], frequency=frequency,
        points=points, actual_start=points[0].date if points else None,
        actual_end=points[-1].date if points else None, n_points=len(points), missing_trading_days=missing,
        gaps=gaps, truncated=truncated)
