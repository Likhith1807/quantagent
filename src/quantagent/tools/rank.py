from typing import Annotated, Literal

import pandas as pd
from pydantic import Field

from ..analytics.ranking import NATURAL_DIRECTION, rank_single, rank_weighted
from ..config import HARD_MAX_RESULTS
from ..data.cache import current_cache
from ..errors import QuantError
from ..schemas.common import Excluded as ExcludedOut
from ..schemas.common import SearchText
from ..schemas.tools import RankedFund, RankResult
from ._base import base_warnings, build_meta, get_services, tool_handler

RankMetric = Literal["total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "alpha",
                     "expense_ratio"]
SURVIVORSHIP = ("Survivorship bias: closed and merged schemes are absent from the source data, so the universe "
                "contains only surviving funds.")


def _methodology(metric, weights, period, dedupe) -> str:
    how = (f"sorted by {metric} ({'higher' if NATURAL_DIRECTION[metric] else 'lower'} is better unless "
           f"'ascending' overrides)" if metric else
           "score = sum(w_i * direction_i * z_i) / sum(w_i), with z-scores computed across eligible funds")
    return (f"Trailing {period} metrics as of the latest NAV date, from metrics_cache (same code as "
            f"get_risk_metrics). Funds are {how}. Eligible: active schemes, Growth or NA option, enough history "
            f"(>= 90% of trading days), every requested metric available. Dedupe '{dedupe}' keeps one plan per "
            f"scheme (Direct preferred). Alpha uses the scheme's own benchmark.")


@tool_handler
def rank_funds(
    metric: Annotated[RankMetric | None, Field(description="Rank by one metric. Give metric OR weights.")] = None,
    weights: Annotated[dict[RankMetric, Annotated[float, Field(ge=0, le=100)]] | None,
                       Field(description="Composite score, e.g. {'sharpe': 2, 'max_drawdown': 1}.")] = None,
    period: Literal["1y", "3y", "5y"] = "1y",
    category: Annotated[SearchText | None, Field(description="Category or sub-category text.")] = None,
    fund_house: Annotated[SearchText | None, Field(description="Fund house text.")] = None,
    max_expense_ratio: Annotated[float | None, Field(ge=0, le=5, description="In percent, e.g. 1.0.")] = None,
    dedupe: Literal["direct_growth", "none"] = "direct_growth",
    top_n: Annotated[int, Field(ge=1, le=HARD_MAX_RESULTS)] = 10,
    ascending: Annotated[bool | None, Field(description="Override the metric's natural sort order.")] = None,
) -> RankResult:
    """Screen and rank many funds by one metric or a weighted composite over a 1y/3y/5y window.

    Use for "best/top/worst funds in a category" questions. Do NOT use for one fund's details
    (get_risk_metrics) or for a handful of named funds (compare_funds). Funds lacking history or
    the metric are listed in `excluded` with the reason. dedupe='direct_growth' keeps one plan per
    scheme so Direct and Regular versions of the same fund don't both appear.
    """
    svc = get_services()
    repo = svc.repo
    if (metric is None) == (weights is None):
        raise QuantError("INVALID_INPUT", "Give exactly one of metric or weights.")
    if weights is not None and sum(weights.values()) <= 0:
        raise QuantError("INVALID_INPUT", "weights must sum to more than 0.")
    if ascending is not None and weights is not None:
        raise QuantError("INVALID_INPUT", "ascending applies only to single-metric ranking.")
    if top_n > svc.settings.max_results:
        raise QuantError("INVALID_INPUT", f"top_n may not exceed {svc.settings.max_results}.")
    used = [metric] if metric else [m for m, w in weights.items() if w > 0]

    cache = current_cache(repo, period)
    rows, _ = repo.search_schemes(category=category, fund_house=fund_house, status="active",
                                  limit=100_000, offset=0)
    universe = pd.DataFrame(rows)
    warnings = base_warnings(repo) + [SURVIVORSHIP]
    excluded: list[ExcludedOut] = []
    if universe.empty:
        warnings.append("No active schemes match the filters.")
        return RankResult(meta=build_meta(repo), warnings=warnings, period=period,
                          ranked_by=metric or "weighted", weights=weights, ascending=ascending, ranked=[],
                          universe_size=0, eligible_count=0, methodology=_methodology(metric, weights, period, dedupe))
    ters = repo.latest_ter(list(universe["scheme_code"]))
    universe["expense_ratio"] = [ters[c]["ter"] if c in ters else None for c in universe["scheme_code"]]
    df = universe.merge(cache.drop(columns=["benchmark_index"]), on="scheme_code", how="left")

    def drop(mask: pd.Series, reason) -> None:
        nonlocal df
        for _, r in df[mask].iterrows():
            excluded.append(ExcludedOut(id=r["scheme_code"], reason=reason(r) if callable(reason) else reason))
        df = df[~mask]

    drop(df["option_type"] == "IDCW", "IDCW option: not a total-return series")
    drop(df["status_y"] != "ok", lambda r: "no cached metrics" if pd.isna(r["n_obs"]) else
         f"insufficient {period} history: {int(r['n_obs'])} NAVs, {int(r['required_obs'])} required")
    if max_expense_ratio is not None:
        drop(df["expense_ratio"].isna() | (df["expense_ratio"] > max_expense_ratio),
             f"expense ratio above {max_expense_ratio}% or unknown")
    for m in used:
        drop(df[m].isna(), f"{m} unavailable" + (" (no benchmark series)" if m == "alpha" else ""))
    if dedupe == "direct_growth" and not df.empty:
        df = df.assign(_pref=(df["plan_type"] != "Direct").astype(int)).sort_values(["base_name", "_pref"])
        drop(df.duplicated("base_name", keep="first"), "duplicate plan of a scheme already ranked (Direct kept)")

    eligible = len(df)
    if eligible == 0:
        warnings.append("No funds are eligible after exclusions; see `excluded`.")
        ranked_df = df
    else:
        ranked_df = rank_single(df, metric, ascending) if metric else rank_weighted(df, weights)
    metrics_out = sorted(set(used) | {"cagr", "volatility", "sharpe", "max_drawdown"})
    ranked = [RankedFund(rank=i + 1, scheme_code=r["scheme_code"], scheme_name=r["scheme_name"],
                         fund_house=r["fund_house"], sub_category=r["sub_category"], plan_type=r["plan_type"],
                         score=float(r["score"]),
                         metrics={m: (None if pd.isna(r[m]) else float(r[m])) for m in metrics_out},
                         expense_ratio_pct=None if pd.isna(r["expense_ratio"]) else float(r["expense_ratio"]))
              for i, (_, r) in enumerate(ranked_df.head(top_n).iterrows())]
    if "alpha" in used and repo.meta.get("data_nature") == "synthetic":
        warnings.append("Alpha is not economically interpretable on this synthetic data.")
    as_of = cache["as_of"].iloc[0]
    rf_note = cache["rf_source"].iloc[0]
    return RankResult(
        meta=build_meta(repo, None, as_of, None, [f"Cached metrics use risk-free source: {rf_note}."]),
        warnings=warnings, excluded=excluded, period=period, ranked_by=metric or "weighted", weights=weights,
        ascending=ascending, ranked=ranked, universe_size=len(universe), eligible_count=eligible,
        methodology=_methodology(metric, weights, period, dedupe))
