from typing import Annotated

from pydantic import Field

from ..analytics import MIN_SHARED_OBS_REGRESSION
from ..analytics.benchmark import align, capture_ratios, information_ratio, regression, tracking_error
from ..analytics.returns import cagr, daily_returns, daily_rf, total_return
from ..analytics.risk import max_drawdown, volatility
from ..analytics.windows import clip
from ..errors import QuantError
from ..schemas.common import BenchmarkName, DateRange, EndDate, Period, SchemeCode, StartDate
from ..schemas.tools import BenchmarkComparisonResult, BenchmarkInfo, SideStats
from ._base import (
    BENCH_SYNTHETIC_WARNING,
    IDCW_WARNING,
    PRI_WARNING,
    base_warnings,
    benchmark_for,
    build_meta,
    get_services,
    load_nav_window,
    require_scheme,
    risk_free,
    tool_handler,
)


def _side(nav, withhold: bool) -> SideStats:
    dd = max_drawdown(nav)
    return SideStats(total_return=None if withhold else total_return(nav), cagr=None if withhold else cagr(nav),
                     volatility=volatility(daily_returns(nav)),
                     max_drawdown=None if withhold or dd is None else dd.max_drawdown)


@tool_handler
def get_benchmark_comparison(
    scheme_code: SchemeCode,
    benchmark: Annotated[BenchmarkName | None,
                         Field(description="Benchmark series; defaults to the scheme's own benchmark.")] = None,
    period: Annotated[Period | None, Field(description="Trailing window; default 3y. Use instead of dates.")] = None,
    start_date: StartDate = None,
    end_date: EndDate = None,
) -> BenchmarkComparisonResult:
    """Compare ONE scheme with its benchmark over the same dates.

    Returns fund vs benchmark CAGR, volatility and drawdown, plus excess CAGR, tracking error,
    information ratio, beta, alpha, R^2 and monthly up/down capture. Use to answer "did this fund
    beat its index?". Do NOT use to compare funds with each other (compare_funds) or for a fund's
    standalone risk (get_risk_metrics). Needs >= 120 shared daily returns.
    """
    svc = get_services()
    repo = svc.repo
    scheme = require_scheme(repo, scheme_code)
    bench = benchmark_for(repo, scheme, benchmark)
    if bench is None:
        raise QuantError("NOT_FOUND", f"Scheme {scheme_code} has no benchmark series; pass benchmark explicitly.",
                         {"available": [b["index_name"] for b in repo.benchmark_names()]})
    nav, start, end, warnings = load_nav_window(repo, svc.settings, scheme_code, period, start_date, end_date)
    warnings = base_warnings(repo) + warnings
    bnav = clip(repo.benchmark_series(bench["index_name"]), start, end)
    f_nav, b_nav = align(nav, bnav)
    if len(f_nav) < 2:
        raise QuantError("NO_OVERLAP", "Fund and benchmark share no dates in this window.",
                         {"fund_points": len(nav), "benchmark_points": len(bnav)})
    if len(f_nav) - 1 < MIN_SHARED_OBS_REGRESSION:
        raise QuantError("INSUFFICIENT_HISTORY", "Too few shared returns with the benchmark.",
                         {"found": len(f_nav) - 1, "required": MIN_SHARED_OBS_REGRESSION})
    if bench["index_type"] == "PRI":
        warnings.append(PRI_WARNING.format(name=bench["index_name"]))
    if repo.meta.get("data_nature") == "synthetic":
        warnings.append(BENCH_SYNTHETIC_WARNING)
    idcw = scheme["option_type"] == "IDCW"
    if idcw:
        warnings.append(IDCW_WARNING.format(code=scheme_code))

    rf_annual, _, _, rf_assumption = risk_free(repo, svc.settings, None)
    rf_r, rb_r = daily_returns(f_nav), daily_returns(b_nav)
    rf_d = daily_rf(rf_annual, rf_r.index)
    reg = regression(rf_r - rf_d, rb_r - rf_d)
    cap = capture_ratios(f_nav, b_nav)
    if cap.up_capture is None or cap.down_capture is None:
        warnings.append("Fewer than 3 up or down benchmark months; that capture ratio is not computed.")
    fund, bstats = _side(f_nav, idcw), _side(b_nav, False)
    return BenchmarkComparisonResult(
        meta=build_meta(repo, f_nav.index[0], f_nav.index[-1], len(f_nav), [rf_assumption] if rf_assumption else None),
        warnings=warnings, scheme_code=scheme_code, scheme_name=scheme["scheme_name"],
        benchmark=BenchmarkInfo(index_name=bench["index_name"], index_type=bench["index_type"]),
        fund=fund, benchmark_stats=bstats,
        excess_cagr=None if fund.cagr is None or bstats.cagr is None else fund.cagr - bstats.cagr,
        tracking_error=tracking_error(rf_r, rb_r), information_ratio=information_ratio(rf_r, rb_r),
        beta=reg.beta, alpha=reg.alpha, r_squared=reg.r_squared, up_capture=cap.up_capture,
        down_capture=cap.down_capture, up_months=cap.up_months, down_months=cap.down_months,
        shared_period=DateRange(start=f_nav.index[0].date(), end=f_nav.index[-1].date()), n_shared=len(f_nav))
