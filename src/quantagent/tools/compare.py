from typing import Annotated

import pandas as pd
from pydantic import Field

from ..analytics import MIN_RETURNS_RISK
from ..analytics.returns import correlation_matrix, daily_returns, daily_rf
from ..analytics.risk import max_drawdown
from ..analytics.windows import metric_bundle, window_sufficiency
from ..config import HARD_MAX_FUNDS
from ..errors import QuantError
from ..schemas.common import DateRange, EndDate, Period, SchemeCode, StartDate
from ..schemas.tools import CompareResult, DrawdownOut, FairComparison, FundMetrics
from ._base import (
    IDCW_WARNING,
    base_warnings,
    build_meta,
    get_services,
    resolve_window,
    risk_free,
    tool_handler,
)

LATE_START_DAYS = 7
SHORT_HISTORY_DAYS = 358  # one year less a week of holidays/weekends


@tool_handler
def compare_funds(
    scheme_codes: Annotated[list[SchemeCode], Field(min_length=2, max_length=HARD_MAX_FUNDS,
                                                    description="2-10 distinct scheme codes.")],
    period: Annotated[Period | None, Field(description="Trailing window; default 3y. Use instead of dates.")] = None,
    start_date: StartDate = None,
    end_date: EndDate = None,
) -> CompareResult:
    """Compare 2-10 schemes side by side on the SAME dates.

    Returns per fund: total return, CAGR, volatility, Sharpe, Sortino, maximum drawdown, n_obs;
    plus the shared period, a daily-return correlation matrix and fair_comparison with reasons
    (mismatched windows, IDCW plans, Direct mixed with Regular, different sub-categories, short
    history). Use when the user names specific funds. Do NOT use to search a whole category for
    the best funds (use rank_funds) or for benchmark-relative stats (get_benchmark_comparison).
    """
    svc = get_services()
    repo = svc.repo
    if len(set(scheme_codes)) != len(scheme_codes):
        raise QuantError("INVALID_INPUT", "scheme_codes must be distinct.")
    if len(scheme_codes) > svc.settings.max_funds:
        raise QuantError("INVALID_INPUT", f"At most {svc.settings.max_funds} funds per comparison.")
    schemes = repo.get_schemes(scheme_codes)
    unknown = [c for c in scheme_codes if c not in schemes]
    if unknown:
        raise QuantError("NOT_FOUND", "Unknown scheme codes.", {"unknown": unknown})
    start, end, label, warnings = resolve_window(repo, svc.settings, period, start_date, end_date)
    warnings = base_warnings(repo) + warnings

    navs = {c: repo.nav_series(c, start.date() if start is not None else None, end.date()) for c in scheme_codes}
    shared = None
    for s in navs.values():
        shared = s.index if shared is None else shared.intersection(s.index)
    if shared is None or len(shared) < 2:
        raise QuantError("NO_OVERLAP", "The funds share fewer than 2 NAV dates in this window.",
                         {c: {"points": len(s), "start": str(s.index.min().date()) if len(s) else None,
                              "end": str(s.index.max().date()) if len(s) else None} for c, s in navs.items()})

    reasons: list[str] = []
    for c, s in navs.items():
        if start is not None and len(s) and (s.index.min() - start).days > LATE_START_DAYS:
            reasons.append(f"{c} has no NAV until {s.index.min().date()}, after the requested start "
                           f"{start.date()}; the shared window is shorter.")
    rf_annual, _, _, rf_assumption = risk_free(repo, svc.settings, None)
    funds, returns = [], {}
    for c in scheme_codes:
        s = schemes[c]
        nav = navs[c].loc[shared]
        idcw = s["option_type"] == "IDCW"
        b = metric_bundle(nav, daily_rf(rf_annual, nav.index), None)
        dd = None if idcw else max_drawdown(nav)
        if idcw:
            warnings.append(IDCW_WARNING.format(code=c))
        funds.append(FundMetrics(
            scheme_code=c, scheme_name=s["scheme_name"], plan_type=s["plan_type"], option_type=s["option_type"],
            total_return=None if idcw else b["total_return"], cagr=None if idcw else b["cagr"],
            volatility=b["volatility"], sharpe=b["sharpe"], sortino=b["sortino"],
            max_drawdown=DrawdownOut(**dd.as_dict()) if dd else None, n_obs=len(nav)))
        returns[c] = daily_returns(nav)

    n_ret = len(shared) - 1
    if n_ret < MIN_RETURNS_RISK:
        warnings.append(f"Only {n_ret} shared returns (< {MIN_RETURNS_RISK}); volatility, Sharpe and Sortino "
                        f"are not computed.")
    span_days = (shared[-1] - shared[0]).days
    if span_days < SHORT_HISTORY_DAYS:
        reasons.append(f"Short shared history: {span_days} calendar days ({n_ret} returns), under 1 year.")
    if label in ("1y", "3y", "5y"):
        suf = window_sufficiency(pd.Series(0.0, index=shared), repo.trading_calendar, start, end)
        if not suf.ok:
            reasons.append(f"Shared dates cover {suf.coverage_pct:.0%} of the {label} window's trading days "
                           f"(need 90%).")
    if any(schemes[c]["option_type"] == "IDCW" for c in scheme_codes):
        reasons.append("Includes IDCW options, which are not total-return series.")
    if len({schemes[c]["plan_type"] for c in scheme_codes}) > 1:
        reasons.append("Mixes Direct and Regular plans: Regular plans carry higher expense ratios.")
    if len({schemes[c]["sub_category"] for c in scheme_codes}) > 1:
        reasons.append("Funds are from different sub-categories with different risk profiles.")

    return CompareResult(
        meta=build_meta(repo, shared[0], shared[-1], len(shared), [rf_assumption] if rf_assumption else None),
        warnings=warnings, funds=funds, shared_period=DateRange(start=shared[0].date(), end=shared[-1].date()),
        correlation=correlation_matrix(pd.DataFrame(returns)),
        fair_comparison=FairComparison(is_fair=not reasons, reasons=reasons))
