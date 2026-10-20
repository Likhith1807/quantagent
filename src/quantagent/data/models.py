"""Database schema (SQLAlchemy Core). Only quantagent.data.ingest writes to these tables."""
from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)

metadata = MetaData()

schemes = Table(
    "schemes", metadata,
    Column("scheme_code", String(12), primary_key=True),
    Column("isin", String(12)),
    Column("scheme_name", Text, nullable=False),
    Column("base_name", Text, nullable=False),  # name without plan words; pairs Direct/Regular
    Column("fund_house", Text, nullable=False),
    Column("category", Text),
    Column("sub_category", Text),
    Column("plan_type", String(10), nullable=False),  # Direct | Regular
    Column("option_type", String(10), nullable=False),  # Growth | IDCW | NA (ETF)
    Column("status", String(10), nullable=False, server_default="active"),  # active | closed | merged
    Column("merged_into", String(12)),
    Column("launch_date", Date),
    Column("closure_date", Date),
    Column("benchmark_name", Text),  # as stated in the source master
    Column("benchmark_index", Text),  # series in `benchmarks`, via data/reference/benchmark_map.csv
    Column("risk_category", Text),
    CheckConstraint("plan_type IN ('Direct','Regular')", name="ck_plan"),
    CheckConstraint("option_type IN ('Growth','IDCW','NA')", name="ck_option"),
    CheckConstraint("status IN ('active','closed','merged')", name="ck_status"),
)

expense_ratios = Table(
    "expense_ratios", metadata,
    Column("scheme_code", String(12), primary_key=True),
    Column("as_of_date", Date, primary_key=True),
    Column("ter", Float, nullable=False),
    Column("as_of_assumed", Integer, nullable=False, server_default="0"),
    CheckConstraint("ter >= 0 AND ter <= 5", name="ck_ter"),
)

nav_history = Table(
    "nav_history", metadata,
    Column("scheme_code", String(12), primary_key=True),
    Column("nav_date", Date, primary_key=True),
    Column("nav", Float, nullable=False),
    CheckConstraint("nav > 0", name="ck_nav_pos"),
)

benchmarks = Table(
    "benchmarks", metadata,
    Column("index_name", String(60), primary_key=True),
    Column("index_type", String(3), primary_key=True),  # TRI | PRI
    Column("date", Date, primary_key=True),
    Column("value", Float, nullable=False),
    CheckConstraint("index_type IN ('TRI','PRI')", name="ck_index_type"),
    CheckConstraint("value > 0", name="ck_bench_pos"),
)

risk_free_rates = Table(
    "risk_free_rates", metadata,
    Column("date", Date, primary_key=True),
    Column("annual_rate", Float, nullable=False),
    Column("source", Text, nullable=False),
    CheckConstraint("annual_rate >= 0 AND annual_rate <= 0.5", name="ck_rf"),
)

trading_calendar = Table(
    "trading_calendar", metadata,
    Column("date", Date, primary_key=True),
    Column("is_trading_day", Integer, nullable=False),
)

METRIC_COLUMNS = ["total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "beta", "alpha"]

metrics_cache = Table(
    "metrics_cache", metadata,
    Column("scheme_code", String(12), primary_key=True),
    Column("window", String(4), primary_key=True),  # 1y | 3y | 5y
    Column("data_version", Integer, primary_key=True),
    Column("as_of", Date, nullable=False),
    Column("start_date", Date),
    Column("end_date", Date),
    Column("n_obs", Integer, nullable=False),
    Column("required_obs", Integer, nullable=False),
    Column("coverage_pct", Float),
    Column("status", String(24), nullable=False),  # ok | insufficient_history | idcw_excluded
    *[Column(c, Float) for c in METRIC_COLUMNS],
    Column("benchmark_index", Text),
    Column("rf_annual", Float),
    Column("rf_source", Text),
    Column("calc_version", String(16), nullable=False),
)

data_sources = Table(
    "data_sources", metadata,
    Column("source", Text, primary_key=True),
    Column("file", Text),
    Column("url", Text),
    Column("retrieved_at", DateTime),
    Column("licence", Text),
    Column("coverage", Text),  # JSON, as recorded by the fetch script
    Column("sha256", String(64)),
    Column("data_nature", String(10), nullable=False),  # synthetic | real | reference
)

ingestion_log = Table(
    "ingestion_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(36), nullable=False),
    Column("data_version", Integer, nullable=False),
    Column("started_at", DateTime, nullable=False),
    Column("finished_at", DateTime),
    Column("table_name", Text, nullable=False),
    Column("rows_loaded", Integer, nullable=False),
    Column("rows_rejected", Integer, nullable=False),
    Column("coverage", Text),  # JSON
)

meta = Table(
    "meta", metadata,
    Column("key", String(40), primary_key=True),
    Column("value", Text, nullable=False),
)
