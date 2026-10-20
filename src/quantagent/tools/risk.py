from typing import Annotated

from pydantic import Field

from ..analytics import MIN_SHARED_OBS_REGRESSION
from ..analytics.returns import daily_rf
from ..analytics.risk import max_drawdown
from ..analytics.windows import clip, metric_bundle
from ..schemas.common import BenchmarkName, EndDate, Period, SchemeCode, StartDate
from ..schemas.tools import BenchmarkInfo, DrawdownOut, RiskResult
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


@tool_handler
def get_risk_metrics(
    scheme_code: SchemeCode,
    period: Annotated[Period | None, Field(description="Trailing window; default 3y. Use instead of dates.")] = None,
    start_date: StartDate = None,
    end_date: EndDate = None,
    benchmark: Annotated[BenchmarkName | None,
                         Field(description="Override the scheme's benchmark for alpha/beta.")] = None,
    risk_free_annual: Annotated[float | None, Field(ge=0, le=0.2,
                                                    description="Annual risk-free rate, e.g. 0.065.")] = None,
) -> RiskResult:
    """Risk profile of ONE scheme: volatility, Sharpe, Sortino, alpha, beta, R^2 and maximum drawdown.

    Use for a single fund's risk. Do NOT use to screen many funds (use rank_funds) or to compare
    a few funds side by side (use compare_funds). For excess return, tracking error and capture
    ratios against the benchmark use get_benchmark_comparison.
    Rules: 1y/3y/5y windows need NAVs on >= 90% of trading days; volatility, Sharpe and Sortino
    need >= 60 returns; alpha/beta need >= 120 returns shared with the benchmark.
    """
    svc = get_services()
    repo = svc.repo
    scheme = require_scheme(repo, scheme_code)
    bench = benchmark_for(repo, scheme, benchmark)
    nav, start, end, warnings = load_nav_window(repo, svc.settings, scheme_code, period, start_date, end_date)
    warnings = base_warnings(repo) + warnings
    rf_annual, rf_source, rf_rate, rf_assumption = risk_free(repo, svc.settings, risk_free_annual)

    bench_series = None
    if bench is None:
        warnings.append("Scheme has no benchmark series; alpha, beta and R^2 are not computed.")
    else:
        bench_series = clip(repo.benchmark_series(bench["index_name"]), start, end)
        if bench["index_type"] == "PRI":
            warnings.append(PRI_WARNING.format(name=bench["index_name"]))
        if repo.meta.get("data_nature") == "synthetic":
            warnings.append(BENCH_SYNTHETIC_WARNING)

    b = metric_bundle(nav, daily_rf(rf_annual, nav.index), bench_series)
    if bench is not None and (b["n_shared_with_benchmark"] or 0) < MIN_SHARED_OBS_REGRESSION:
        warnings.append(f"Only {b['n_shared_with_benchmark']} returns shared with the benchmark "
                        f"(need {MIN_SHARED_OBS_REGRESSION}); alpha and beta not computed.")
    dd = max_drawdown(nav)
    if scheme["option_type"] == "IDCW":
        warnings.append(IDCW_WARNING.format(code=scheme_code))
        dd = None
    return RiskResult(
        meta=build_meta(repo, start, end, len(nav), [rf_assumption] if rf_assumption else None),
        warnings=warnings, scheme_code=scheme_code, scheme_name=scheme["scheme_name"],
        benchmark=BenchmarkInfo(index_name=bench["index_name"], index_type=bench["index_type"]) if bench else None,
        volatility=b["volatility"], sharpe=b["sharpe"], sortino=b["sortino"], alpha=b["alpha"], beta=b["beta"],
        r_squared=b["r_squared"], max_drawdown=DrawdownOut(**dd.as_dict()) if dd else None, n_obs=len(nav),
        n_shared_with_benchmark=b["n_shared_with_benchmark"], risk_free_annual=rf_rate, risk_free_source=rf_source)
