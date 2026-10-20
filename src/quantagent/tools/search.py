from typing import Annotated, Literal

from pydantic import Field

from ..config import HARD_MAX_RESULTS
from ..errors import QuantError
from ..schemas.common import OptionType, PlanType, SearchText
from ..schemas.tools import SearchResult
from ._base import base_warnings, build_meta, fund_summaries, get_services, tool_handler


@tool_handler
def search_funds(
    query: Annotated[SearchText | None, Field(description="Text in the scheme name or code, e.g. 'bluechip'.")] = None,
    category: Annotated[SearchText | None, Field(description="Category or sub-category text, e.g. 'Large Cap'.")] = None,
    fund_house: Annotated[SearchText | None, Field(description="Fund house text, e.g. 'HDFC'.")] = None,
    plan_type: PlanType | None = None,
    option_type: OptionType | None = None,
    status: Literal["active", "closed", "merged", "any"] = "active",
    limit: Annotated[int, Field(ge=1, le=HARD_MAX_RESULTS)] = 20,
    offset: Annotated[int, Field(ge=0, le=100_000)] = 0,
) -> SearchResult:
    """Find mutual fund schemes and their codes by name, category, fund house, plan or option.

    Use this first, to get the 6-digit scheme_code that every other tool needs.
    Do NOT use it to compare or rank performance: use rank_funds (screening) or compare_funds.
    At least one of query, category, fund_house, plan_type, option_type is required. Matching is
    case-insensitive substring matching. Results are paginated with limit/offset.
    """
    svc = get_services()
    if not any([query, category, fund_house, plan_type, option_type]):
        raise QuantError("INVALID_INPUT", "Give at least one filter: query, category, fund_house, plan_type "
                                          "or option_type.")
    if limit > svc.settings.max_results:
        raise QuantError("INVALID_INPUT", f"limit may not exceed {svc.settings.max_results}.")
    rows, total = svc.repo.search_schemes(query, category, fund_house, plan_type, option_type, status, limit, offset)
    warnings = base_warnings(svc.repo)
    if total == 0:
        warnings.append("No schemes matched; try a shorter query or fewer filters.")
    elif offset >= total:
        warnings.append(f"offset {offset} is past the last match ({total} total).")
    return SearchResult(meta=build_meta(svc.repo), warnings=warnings, items=fund_summaries(svc.repo, rows),
                        total_matches=total, limit=limit, offset=offset)
