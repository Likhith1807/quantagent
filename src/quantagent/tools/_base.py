"""Shared tool plumbing: services, the error/logging decorator, windows, envelopes."""
import contextvars
import functools
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import date

import pandas as pd
from pydantic import ValidationError, validate_call

from ..analytics import CALC_VERSION, MIN_RETURNS_RISK
from ..analytics.windows import PERIOD_YEARS, window_sufficiency
from ..config import Settings, get_settings
from ..data.engine import make_readonly_engine
from ..data.repository import Repository
from ..errors import QuantError
from ..logging_setup import LOGGER_NAME, sanitise
from ..schemas.common import DateRange, Meta
from ..schemas.tools import Coverage, ExpenseRatio, FundSummary

log = logging.getLogger(LOGGER_NAME + ".tools")
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

SYNTHETIC_WARNING = ("Data is synthetic (test fixture): results do not describe real funds. "
                     "See get_data_provenance.")
BENCH_SYNTHETIC_WARNING = ("Benchmark-relative statistics (beta, alpha, R^2, tracking error, capture) are not "
                           "economically interpretable on this synthetic data: fund and benchmark series are "
                           "independent.")
PRI_WARNING = "Benchmark {name} is a price index (PRI): it excludes dividends and understates benchmark returns."
IDCW_WARNING = ("Scheme {code} is an IDCW option: its NAV drops on every payout, so it is not a total-return "
                "series. CAGR, total return and drawdown are withheld.")


# ---------------------------------------------------------------- services
@dataclass
class Services:
    settings: Settings
    repo: Repository


_services: Services | None = None


def get_services() -> Services:
    global _services
    if _services is None:
        s = get_settings()
        _services = Services(s, Repository(make_readonly_engine(s.db_url, s.timeout_s)))
    return _services


def set_services(services: Services | None) -> None:
    global _services
    _services = services


# ---------------------------------------------------------------- decorator
def compact_errors(e: ValidationError) -> list[dict]:
    return [{"field": ".".join(str(p) for p in err["loc"]), "msg": err["msg"]} for err in e.errors()]


def tool_handler(fn):
    """Validate args (also when called directly), map failures to QuantError, log one JSON line."""
    validated = validate_call(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        rid = uuid.uuid4().hex[:12]
        token = _request_id.set(rid)
        t0 = time.perf_counter()
        outcome, code = "ok", None
        try:
            return validated(*args, **kwargs)
        except ValidationError as e:
            outcome, code = "error", "INVALID_INPUT"
            raise QuantError("INVALID_INPUT", f"Invalid arguments for {fn.__name__}.",
                             {"errors": compact_errors(e)}) from None
        except QuantError as e:
            outcome, code = "error", e.code
            raise
        except Exception:
            outcome, code = "error", "INTERNAL"
            log.exception("unexpected error", extra={"fields": {"request_id": rid, "tool": fn.__name__}})
            raise QuantError("INTERNAL", "Unexpected server error; it has been logged.",
                             {"request_id": rid}) from None
        finally:
            log.info("tool call", extra={"fields": {
                "request_id": rid, "tool": fn.__name__, "args": sanitise(kwargs),
                "duration_ms": round((time.perf_counter() - t0) * 1000, 1), "outcome": outcome,
                "error_code": code}})
            _request_id.reset(token)

    return wrapper


# ---------------------------------------------------------------- helpers
def data_as_of(repo: Repository) -> pd.Timestamp:
    return pd.Timestamp(repo.meta["data_as_of"])


def build_meta(repo: Repository, start=None, end=None, n_obs: int | None = None,
               extra_assumptions: list[str] | None = None) -> Meta:
    m = repo.meta
    period = DateRange(start=pd.Timestamp(start).date() if start is not None else None,
                       end=pd.Timestamp(end).date() if end is not None else None) if (start or end) else None
    return Meta(period=period, n_obs=n_obs, data_as_of=m["data_as_of"], data_version=m["data_version"],
                calc_version=CALC_VERSION, data_nature=m.get("data_nature", "synthetic"),
                data_source=m.get("data_source"), data_retrieved_at=m.get("data_retrieved_at") or None,
                assumptions=list(dict.fromkeys([*m["assumptions"], *(extra_assumptions or [])])),
                request_id=_request_id.get())


def base_warnings(repo: Repository) -> list[str]:
    return [SYNTHETIC_WARNING] if repo.meta.get("data_nature") == "synthetic" else []


def require_scheme(repo: Repository, code: str) -> dict:
    s = repo.get_scheme(code)
    if s is None:
        raise QuantError("NOT_FOUND", f"No scheme with code {code}.",
                         {"hint": "Use search_funds to find scheme codes."})
    return s


def resolve_window(repo: Repository, settings: Settings, period: str | None, start_date: date | None,
                   end_date: date | None, default: str = "3y") -> tuple[pd.Timestamp | None, pd.Timestamp, str | None,
                                                                         list[str]]:
    """Returns (start or None for 'max', end, period label or None, warnings)."""
    warnings: list[str] = []
    if period and (start_date or end_date):
        raise QuantError("INVALID_INPUT", "Give either period or start_date/end_date, not both.")
    today = date.today()
    for name, d in (("start_date", start_date), ("end_date", end_date)):
        if d and d > today:
            raise QuantError("INVALID_INPUT", f"{name} {d} is in the future.")
    if start_date and end_date and start_date > end_date:
        raise QuantError("INVALID_INPUT", "start_date is after end_date.")
    if start_date and end_date and (end_date - start_date).days > settings.max_range_years * 366:
        raise QuantError("INVALID_INPUT", f"Range exceeds the {settings.max_range_years}-year limit.")
    as_of = data_as_of(repo)
    if not (start_date or end_date):
        period = period or default
    end = pd.Timestamp(end_date) if end_date else as_of
    if end > as_of:
        warnings.append(f"end_date clipped to the latest data date {as_of.date()}.")
        end = as_of
    if period:
        start = None if period == "max" else end - pd.DateOffset(years=PERIOD_YEARS[period])
        return start, end, period, warnings
    return (pd.Timestamp(start_date) if start_date else None), end, None, warnings


def fund_summaries(repo: Repository, rows: list[dict]) -> list[FundSummary]:
    codes = [r["scheme_code"] for r in rows]
    ters, cov = repo.latest_ter(codes), repo.nav_coverage(codes)
    out = []
    for r in rows:
        t, c = ters.get(r["scheme_code"]), cov.get(r["scheme_code"])
        out.append(FundSummary(
            scheme_code=r["scheme_code"], scheme_name=r["scheme_name"], fund_house=r["fund_house"],
            category=r["category"], sub_category=r["sub_category"], plan_type=r["plan_type"],
            option_type=r["option_type"], status=r["status"], benchmark_name=r["benchmark_name"],
            benchmark_index=r["benchmark_index"],
            latest_expense_ratio=ExpenseRatio(value_pct=t["ter"], as_of=t["as_of_date"],
                                              as_of_assumed=bool(t["as_of_assumed"])) if t else None,
            nav_coverage=Coverage(**c) if c else None))
    return out


def risk_free(repo: Repository, settings: Settings, override: float | None):
    """Returns (annual rate or dated Series, source text, scalar rate or None, assumption or None)."""
    if override is not None:
        return override, "user-supplied risk_free_annual", override, f"Risk-free rate {override:.2%} supplied by caller."
    series, source = repo.risk_free_series()
    if series is not None:
        return series, source, None, None
    r = settings.risk_free_annual
    return r, settings.risk_free_source, r, f"Risk-free rate {r:.2%}: {settings.risk_free_source}."


def benchmark_for(repo: Repository, scheme: dict, requested: str | None) -> dict | None:
    """Resolve a benchmark series. Explicit unknown name -> NOT_FOUND with the available list."""
    available = repo.benchmark_names()
    by_name = {b["index_name"]: b for b in available}
    name = requested or scheme.get("benchmark_index")
    if requested and requested not in by_name:
        raise QuantError("NOT_FOUND", f"Unknown benchmark {requested}.",
                         {"available": sorted(by_name)})
    return by_name.get(name) if name else None


def load_nav_window(repo: Repository, settings: Settings, code: str, period: str | None,
                    start_date: date | None, end_date: date | None, default: str = "3y"):
    """NAV series for the resolved window, enforcing the minimum-history rules.
    Returns (nav, start, end, warnings)."""

    start, end, label, warnings = resolve_window(repo, settings, period, start_date, end_date, default)
    nav = repo.nav_series(code, start.date() if start is not None else None, end.date())
    if label in PERIOD_YEARS:
        suf = window_sufficiency(nav, repo.trading_calendar, start, end)
        if not suf.ok:
            raise QuantError("INSUFFICIENT_HISTORY",
                             f"{label} window needs NAVs on >= 90% of trading days (and >= {MIN_RETURNS_RISK} "
                             f"returns).",
                             {"found": suf.n_obs, "required": suf.required_obs, "window_start": str(start.date()),
                              "window_end": str(end.date()), "first_nav": str(nav.index.min().date())
                              if len(nav) else None})
    elif len(nav) < MIN_RETURNS_RISK + 1:
        raise QuantError("INSUFFICIENT_HISTORY", f"Need at least {MIN_RETURNS_RISK + 1} NAVs in the range.",
                         {"found": len(nav), "required": MIN_RETURNS_RISK + 1})
    return nav, (start if start is not None else nav.index.min()), end, warnings
