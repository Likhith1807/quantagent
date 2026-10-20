from datetime import date
from typing import Annotated, Literal

from pydantic import Field

from ..config import HARD_MAX_RESULTS
from ..data.validation import ALL_CHECKS, run_checks, status_of
from ..errors import QuantError
from ..schemas.common import SchemeCode
from ..schemas.tools import QualityIssue, QualityResult, SchemeQuality
from ._base import base_warnings, build_meta, get_services, tool_handler

CheckName = Literal["duplicate_dates", "nonpositive_nav", "future_dates", "calendar_gaps", "extreme_moves",
                    "stale_runs", "missing_metadata", "idcw_option"]


@tool_handler
def check_data_quality(
    scheme_codes: Annotated[list[SchemeCode] | None,
                            Field(max_length=HARD_MAX_RESULTS, description="Omit to check every scheme.")] = None,
    checks: Annotated[list[CheckName] | None, Field(description="Omit to run all checks.")] = None,
    limit: Annotated[int, Field(ge=1, le=HARD_MAX_RESULTS)] = 20,
    offset: Annotated[int, Field(ge=0, le=100_000)] = 0,
) -> QualityResult:
    """Check NAV data quality per scheme and report ok / warn / fail with examples.

    Use before trusting any metric for a scheme, or to audit the dataset. Do NOT use for
    performance figures. fail checks: duplicate_dates, nonpositive_nav, future_dates.
    warn checks: calendar_gaps (trading days with no NAV), extreme_moves (|daily return| > 15%),
    stale_runs (same NAV 5+ days), missing_metadata, idcw_option (not a total-return series).
    Runs the same code that ingestion uses. Results are paginated with limit/offset.
    """
    svc = get_services()
    if limit > svc.settings.max_results:
        raise QuantError("INVALID_INPUT", f"limit may not exceed {svc.settings.max_results}.")
    if scheme_codes is not None:
        if len(set(scheme_codes)) != len(scheme_codes):
            raise QuantError("INVALID_INPUT", "scheme_codes contains duplicates.")
        found = svc.repo.get_schemes(scheme_codes)
        unknown = [c for c in scheme_codes if c not in found]
        if unknown:
            raise QuantError("NOT_FOUND", "Unknown scheme codes.", {"unknown": unknown})
        rows = [found[c] for c in scheme_codes]
    else:
        rows = svc.repo.all_schemes()
    checks_run = list(dict.fromkeys(checks)) if checks else list(ALL_CHECKS)
    page = rows[offset: offset + limit]
    out, summary = [], {"ok": 0, "warn": 0, "fail": 0}
    for s in page:
        nav = svc.repo.nav_series(s["scheme_code"])
        df = nav.rename("nav").rename_axis("date").reset_index()
        issues = run_checks(df, s, svc.repo.trading_calendar, date.today(), checks_run)
        st = status_of(issues)
        summary[st] += 1
        out.append(SchemeQuality(scheme_code=s["scheme_code"], scheme_name=s["scheme_name"], status=st,
                                 issues=[QualityIssue(**i.as_dict()) for i in issues if i.count]))
    warnings = base_warnings(svc.repo)
    warnings.append("Rows rejected at ingest (duplicates, NAV <= 0, future dates) never reach the database; "
                    "see get_data_provenance for their counts.")
    return QualityResult(meta=build_meta(svc.repo), warnings=warnings, schemes=out, summary=summary,
                         checks_run=checks_run, total_schemes=len(rows), limit=limit, offset=offset)
