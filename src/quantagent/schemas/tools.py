"""Structured outputs for the eight tools."""
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from .common import DateRange, ToolResult


class ExpenseRatio(BaseModel):
    value_pct: float
    as_of: date | None
    as_of_assumed: bool


class Coverage(BaseModel):
    start: date | None
    end: date | None
    n_points: int


class FundSummary(BaseModel):
    scheme_code: str
    scheme_name: str
    fund_house: str
    category: str | None
    sub_category: str | None
    plan_type: str
    option_type: str
    status: str
    benchmark_name: str | None
    benchmark_index: str | None
    latest_expense_ratio: ExpenseRatio | None
    nav_coverage: Coverage | None


class SearchResult(ToolResult):
    items: list[FundSummary]
    total_matches: int
    limit: int
    offset: int


class NavPoint(BaseModel):
    date: date
    nav: float


class Gap(BaseModel):
    start: date
    end: date
    missing_trading_days: int


class NavHistoryResult(ToolResult):
    scheme_code: str
    scheme_name: str
    frequency: Literal["daily", "weekly", "monthly"]
    points: list[NavPoint]
    actual_start: date | None
    actual_end: date | None
    n_points: int
    missing_trading_days: int | None = Field(None, description="Daily frequency only.")
    gaps: list[Gap] = Field(default_factory=list, description="Largest gaps first, at most 20.")
    truncated: bool


class DrawdownOut(BaseModel):
    max_drawdown: float = Field(description="Fraction, <= 0 (e.g. -0.2 = -20%).")
    peak_date: date | None
    trough_date: date | None
    recovery_date: date | None = Field(description="None if not yet recovered.")
    drawdown_days: int | None
    recovery_days: int | None


class FundMetrics(BaseModel):
    scheme_code: str
    scheme_name: str
    plan_type: str
    option_type: str
    total_return: float | None
    cagr: float | None
    volatility: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown: DrawdownOut | None
    n_obs: int


class FairComparison(BaseModel):
    is_fair: bool
    reasons: list[str]


class CompareResult(ToolResult):
    funds: list[FundMetrics]
    shared_period: DateRange
    correlation: dict[str, dict[str, float | None]]
    fair_comparison: FairComparison


class RankedFund(BaseModel):
    rank: int
    scheme_code: str
    scheme_name: str
    fund_house: str
    sub_category: str | None
    plan_type: str
    score: float
    metrics: dict[str, float | None]
    expense_ratio_pct: float | None


class RankResult(ToolResult):
    period: Literal["1y", "3y", "5y"]
    ranked_by: str
    weights: dict[str, float] | None
    ascending: bool | None
    ranked: list[RankedFund]
    universe_size: int
    eligible_count: int
    methodology: str


class BenchmarkInfo(BaseModel):
    index_name: str
    index_type: Literal["TRI", "PRI"]


class RiskResult(ToolResult):
    scheme_code: str
    scheme_name: str
    benchmark: BenchmarkInfo | None
    volatility: float | None
    sharpe: float | None
    sortino: float | None
    alpha: float | None
    beta: float | None
    r_squared: float | None
    max_drawdown: DrawdownOut | None
    n_obs: int
    n_shared_with_benchmark: int | None
    risk_free_annual: float | None = Field(description="None when a dated risk-free series was used.")
    risk_free_source: str


class SideStats(BaseModel):
    total_return: float | None
    cagr: float | None
    volatility: float | None
    max_drawdown: float | None


class BenchmarkComparisonResult(ToolResult):
    scheme_code: str
    scheme_name: str
    benchmark: BenchmarkInfo
    fund: SideStats
    benchmark_stats: SideStats
    excess_cagr: float | None
    tracking_error: float | None
    information_ratio: float | None
    beta: float | None
    alpha: float | None
    r_squared: float | None
    up_capture: float | None
    down_capture: float | None
    up_months: int
    down_months: int
    shared_period: DateRange
    n_shared: int


class QualityIssue(BaseModel):
    check: str
    severity: Literal["fail", "warn"]
    count: int
    examples: list[str]


class SchemeQuality(BaseModel):
    scheme_code: str
    scheme_name: str
    status: Literal["ok", "warn", "fail"]
    issues: list[QualityIssue]


class QualityResult(ToolResult):
    schemes: list[SchemeQuality]
    summary: dict[str, int]
    checks_run: list[str]
    total_schemes: int
    limit: int
    offset: int


class SourceOut(BaseModel):
    source: str
    file: str | None = None
    url: str | None
    retrieved_at: datetime | None
    licence: str | None
    coverage: dict | str | None = None
    sha256: str | None
    data_nature: str


class IngestOut(BaseModel):
    table_name: str
    rows_loaded: int
    rows_rejected: int
    finished_at: datetime | None
    run_id: str


class BenchmarkCoverage(BaseModel):
    index_name: str
    index_type: str
    start: date | None
    end: date | None
    n_points: int


class SchemeProvenance(BaseModel):
    fund: FundSummary
    ingest_quality: dict | None
    benchmark: BenchmarkCoverage | None


class ProvenanceResult(ToolResult):
    data_nature: str
    data_version: int
    calc_version: str
    data_as_of: str
    built_at: str | None
    calendar_source: str | None
    sources: list[SourceOut]
    table_counts: dict[str, int]
    benchmarks: list[BenchmarkCoverage]
    last_ingest: list[IngestOut]
    rejected_rows_total: int
    assumptions: list[str]
    known_biases: list[str]
    dataset_findings: list[str]
    scheme: SchemeProvenance | None


class VolModelInfo(BaseModel):
    version: str
    kind: str = Field(description="har | ridge | gbm: log-volatility regression chosen by walk-forward QLIKE")
    trained_data_version: int
    trained_data_nature: str
    train_range: list[str]
    oos_qlike: float = Field(description="Walk-forward out-of-sample QLIKE of this model kind.")
    oos_qlike_ewma: float
    oos_qlike_har: float
    beats_all_baselines: bool
    dm_p_value_vs_ewma: float | None


class VolForecastResult(ToolResult):
    scheme_code: str
    scheme_name: str
    origin_date: date = Field(description="Last NAV date used; the forecast covers the next horizon_days returns.")
    horizon_days: int
    forecast_vol: float = Field(description="Annualised volatility forecast (fraction; 0.15 = 15%).")
    baselines: dict[str, float] = Field(description="naive (last 22-day RV), ewma, har forecasts, annualised.")
    recent_realised_vol: dict[str, float] = Field(description="rv_22 and rv_63: trailing realised vol, annualised.")
    model: VolModelInfo
