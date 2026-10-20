"""The evaluation question set, generated from whatever database is loaded (real data or the
test fixture). Funds are picked by fixed rules (longest NAV history, then scheme code) so the set
is deterministic for a given database; every expected value comes from Reference."""
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from .expected import Reference

REL_TOL = 1e-9


@dataclass
class Expect:
    kind: str  # equals | contains | error | in
    path: str | None = None
    value: object = None
    tol: float = REL_TOL


@dataclass
class Question:
    id: str
    category: str
    text: str
    tool: str
    args: dict
    expect: list[Expect] = field(default_factory=list)


def _name(ref: Reference, code: str) -> str:
    return ref.schemes.set_index("scheme_code").loc[code, "scheme_name"]


def build(ref: Reference, model_available: bool) -> list[Question]:
    span_years = (ref.as_of - ref.navs.index.min()).days / 365.25
    years = 3 if span_years >= 3.3 else 1
    period = f"{years}y"
    a, b, c = ref.longest(3)
    na, nb, nc = _name(ref, a), _name(ref, b), _name(ref, c)
    sa = ref.schemes.set_index("scheme_code").loc[a]
    sub = sa["sub_category"]
    house = sa["fund_house"]
    active = ref.schemes[ref.schemes["status"] == "active"]
    wa, wb = ref.window_stats(a, years), ref.window_stats(b, years)
    q: list[Question] = []

    def add(id_, cat, text, tool, args, *expect):
        q.append(Question(id_, cat, text, tool, args, list(expect)))

    # ---------------- lookup
    add("search_name", "lookup", f"What is the AMFI scheme code of {na}?", "search_funds",
        {"query": na[:60].split(" - ")[0]}, Expect("contains", "items[*].scheme_code", a))
    n_house = int(active["fund_house"].str.lower().str.contains(house.lower(), regex=False).sum())
    add("search_house", "lookup", f"How many active schemes from {house} are in the database?", "search_funds",
        {"fund_house": house, "limit": 1}, Expect("equals", "total_matches", n_house))
    n_sub = int((active["category"].str.lower().str.contains(sub.lower(), regex=False)
                 | active["sub_category"].str.lower().str.contains(sub.lower(), regex=False)).sum())
    add("search_category", "lookup", f"How many active {sub} schemes are there?", "search_funds",
        {"category": sub, "limit": 1}, Expect("equals", "total_matches", n_sub))
    nav_a = ref.nav(a)
    add("latest_nav", "lookup", f"What is the latest NAV of {na}?", "get_nav_history",
        {"scheme_code": a, "start_date": str((ref.as_of - timedelta(days=31)).date())},
        Expect("equals", "points[-1].nav", float(nav_a.iloc[-1])),
        Expect("equals", "actual_end", str(nav_a.index[-1].date())))
    m_end = (ref.as_of - pd.DateOffset(months=2)).normalize()
    m_start = m_end - pd.DateOffset(months=1)
    add("nav_count", "lookup", f"How many daily NAVs does {na} have between {m_start.date()} and {m_end.date()}?",
        "get_nav_history", {"scheme_code": a, "start_date": str(m_start.date()), "end_date": str(m_end.date())},
        Expect("equals", "n_points", int(((nav_a.index >= m_start) & (nav_a.index <= m_end)).sum())))
    if a in ref.ter.index:
        add("expense_ratio", "lookup", f"What is the latest expense ratio of {na}?", "search_funds",
            {"query": a}, Expect("equals", "items[0].latest_expense_ratio.value_pct", float(ref.ter[a])))

    # ---------------- single-fund risk
    add("vol_a", "risk", f"What was the {period} annualised volatility of {na}?", "get_risk_metrics",
        {"scheme_code": a, "period": period}, Expect("equals", "volatility", wa["volatility"]))
    add("sharpe_a", "risk", f"What was the {period} Sharpe ratio of {na}?", "get_risk_metrics",
        {"scheme_code": a, "period": period}, Expect("equals", "sharpe", wa["sharpe"]))
    add("mdd_a", "risk", f"What was the maximum drawdown of {na} over {period}?", "get_risk_metrics",
        {"scheme_code": a, "period": period}, Expect("equals", "max_drawdown.max_drawdown", wa["max_drawdown"]))
    add("sortino_b", "risk", f"What was the {period} Sortino ratio of {nb}?", "get_risk_metrics",
        {"scheme_code": b, "period": period}, Expect("equals", "sortino", wb["sortino"]))
    add("nobs_b", "risk", f"How many NAV observations were used for {nb}'s {period} risk metrics?",
        "get_risk_metrics", {"scheme_code": b, "period": period}, Expect("equals", "n_obs", wb["n_obs"]))
    end = ref.as_of - pd.DateOffset(months=6)
    start = end - pd.DateOffset(years=1)
    s = nav_a[(nav_a.index >= start) & (nav_a.index <= end)]
    add("vol_dates", "risk", f"What was the volatility of {na} from {start.date()} to {end.date()}?",
        "get_risk_metrics", {"scheme_code": a, "start_date": str(start.date()), "end_date": str(end.date())},
        Expect("equals", "volatility", ref.stats(s)["volatility"]))

    # ---------------- returns
    add("cagr_a", "returns", f"What was the {period} CAGR of {na}?", "compare_funds",
        {"scheme_codes": [a, b], "period": period},
        Expect("equals", "funds[0].cagr", ref.shared_stats([a, b], years)[a]["cagr"]))
    shared = ref.shared_stats([a, b], years)
    add("compare_ab", "compare", f"Compare the {period} CAGR and volatility of {na} and {nb}.", "compare_funds",
        {"scheme_codes": [a, b], "period": period},
        Expect("equals", "funds[0].cagr", shared[a]["cagr"]), Expect("equals", "funds[1].cagr", shared[b]["cagr"]),
        Expect("equals", "funds[1].volatility", shared[b]["volatility"]))
    shared3 = ref.shared_stats([a, b, c], years)
    add("compare_abc", "compare", f"Which of {na}, {nb} and {nc} had the highest {period} Sharpe ratio?",
        "compare_funds", {"scheme_codes": [a, b, c], "period": period},
        *[Expect("equals", f"funds[{i}].sharpe", shared3[x]["sharpe"]) for i, x in enumerate([a, b, c])])
    add("total_return_1y", "returns", f"What was the 1-year total return of {nb}?", "compare_funds",
        {"scheme_codes": [b, c], "period": "1y"},
        Expect("equals", "funds[0].total_return", ref.shared_stats([b, c], 1)[b]["total_return"]))

    # ---------------- ranking
    top = ref.rank("sharpe", years, sub, 3)
    add("rank_sharpe_cat", "rank", f"Which three {sub} funds had the best {period} Sharpe ratio?", "rank_funds",
        {"metric": "sharpe", "period": period, "category": sub, "dedupe": "none", "top_n": 3},
        *[Expect("equals", f"ranked[{i}].scheme_code", x) for i, x in enumerate(top)])
    low = ref.rank("volatility", years, sub, 1)
    add("rank_low_vol", "rank", f"Which {sub} fund had the lowest {period} volatility?", "rank_funds",
        {"metric": "volatility", "period": period, "category": sub, "dedupe": "none", "top_n": 1},
        Expect("equals", "ranked[0].scheme_code", low[0]))
    best = ref.rank("cagr", 1, None, 1)
    add("rank_cagr_all", "rank", "Which fund had the highest 1-year CAGR overall?", "rank_funds",
        {"metric": "cagr", "period": "1y", "dedupe": "none", "top_n": 1},
        Expect("equals", "ranked[0].scheme_code", best[0]))

    # ---------------- provenance / quality / forecast
    add("provenance", "provenance", "Is this real data, which version is loaded and how current is it?",
        "get_data_provenance", {}, Expect("equals", "data_nature", ref.meta["data_nature"]),
        Expect("equals", "data_version", int(ref.meta["data_version"])),
        Expect("equals", "data_as_of", ref.meta["data_as_of"]))
    add("quality", "quality", f"Are there data-quality problems with {na}?", "check_data_quality",
        {"scheme_codes": [a]}, Expect("equals", "schemes[0].scheme_code", a),
        Expect("in", "schemes[0].status", ["ok", "warn"]))
    if model_available:
        add("forecast", "forecast", f"How volatile is {na} likely to be over the next month?", "predict_volatility",
            {"scheme_code": a}, Expect("equals", "recent_realised_vol.rv_22", ref.realised_vol_22(a)),
            Expect("equals", "horizon_days", 21))
    else:
        add("forecast", "forecast", f"How volatile is {na} likely to be over the next month?", "predict_volatility",
            {"scheme_code": a}, Expect("error", value="MODEL_UNAVAILABLE"))

    bidx = sa["benchmark_index"]
    if isinstance(bidx, str) and bidx in ref.bench.columns:
        add("benchmark_te", "benchmark", f"What was the {period} tracking error of {na} against its benchmark?",
            "get_benchmark_comparison", {"scheme_code": a, "period": period},
            Expect("equals", "tracking_error", ref.tracking_error(a, bidx, years)))
    else:
        add("benchmark_missing", "benchmark", f"Did {na} beat its benchmark over {period}?",
            "get_benchmark_comparison", {"scheme_code": a, "period": period}, Expect("error", value="NOT_FOUND"))

    # ---------------- edge cases
    future = str((pd.Timestamp.today() + pd.Timedelta(days=10)).date())
    edge = [
        ("bad_code", "What is the Sharpe ratio of scheme 12AB?", "get_risk_metrics", {"scheme_code": "12AB"},
         "INVALID_INPUT"),
        ("unknown_code", "What is the volatility of scheme 999999?", "get_risk_metrics", {"scheme_code": "999999"},
         "NOT_FOUND"),
        ("future_date", f"What was the volatility of {na} up to {future}?", "get_risk_metrics",
         {"scheme_code": a, "start_date": str(ref.as_of.date()), "end_date": future}, "INVALID_INPUT"),
        ("reversed_dates", f"Volatility of {na} from 2024-06-01 to 2024-01-01?", "get_risk_metrics",
         {"scheme_code": a, "start_date": "2024-06-01", "end_date": "2024-01-01"}, "INVALID_INPUT"),
        ("short_range", f"What was the Sharpe ratio of {na} over the last 20 days?", "get_risk_metrics",
         {"scheme_code": a, "start_date": str((ref.as_of - timedelta(days=20)).date())}, "INSUFFICIENT_HISTORY"),
        ("one_fund_compare", f"Compare {na} with itself.", "compare_funds", {"scheme_codes": [a]}, "INVALID_INPUT"),
        ("metric_and_weights", "Rank funds by Sharpe and also by a 50/50 Sharpe-volatility blend.", "rank_funds",
         {"metric": "sharpe", "weights": {"sharpe": 1, "volatility": 1}}, "INVALID_INPUT"),
        ("unknown_benchmark", f"Compare {na} with the NOPE_TRI index.", "get_benchmark_comparison",
         {"scheme_code": a, "benchmark": "NOPE_TRI"}, "NOT_FOUND"),
        ("bad_period", f"What was the 10-year volatility of {na}?", "get_risk_metrics",
         {"scheme_code": a, "period": "10y"}, "INVALID_INPUT"),
        ("misspelt_arg", f"Risk metrics for {na}.", "get_risk_metrics", {"scheme": a}, "INVALID_INPUT"),
    ]
    for id_, text, tool, args, code in edge:
        add(id_, "edge", text, tool, args, Expect("error", value=code))
    add("empty_search", "edge", "Find the 'Zzqx Nonexistent' fund.", "search_funds", {"query": "Zzqx Nonexistent"},
        Expect("equals", "total_matches", 0))
    add("nav_before_data", "edge", f"Show NAVs of {na} for January 2000.", "get_nav_history",
        {"scheme_code": a, "start_date": "2000-01-01", "end_date": "2000-01-31"}, Expect("equals", "n_points", 0))
    return q
