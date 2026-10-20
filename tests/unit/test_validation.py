"""Each check catches its planted defect and passes clean data."""
from datetime import date

import pandas as pd
import pytest

from quantagent.data.validation import (
    ALL_CHECKS,
    base_name,
    option_type_from_name,
    run_checks,
    status_of,
    validate_schemes,
    validate_series,
)

CAL = pd.bdate_range("2024-01-01", periods=40)
TODAY = date(2025, 1, 1)
META = {"fund_house": "X", "category": "Equity", "sub_category": "Large Cap", "launch_date": date(2020, 1, 1),
        "benchmark_index": "B", "option_type": "Growth"}


def clean_nav(n=40):
    return pd.DataFrame({"date": CAL[:n], "nav": 100 + pd.Series(range(n), dtype=float) * 0.37})


def counts(nav, meta=META, checks=ALL_CHECKS):
    return {i.check: i.count for i in run_checks(nav, meta, CAL, TODAY, checks)}


def test_clean_series_passes_every_check():
    issues = run_checks(clean_nav(), META, CAL, TODAY)
    assert status_of(issues) == "ok"
    assert {i.check for i in issues} == set(ALL_CHECKS)


@pytest.mark.parametrize("check,mutate,expected,status", [
    ("duplicate_dates", lambda d: pd.concat([d, d.iloc[[3]]]), 1, "fail"),
    ("nonpositive_nav", lambda d: d.assign(nav=d["nav"].where(d.index != 5, -1.0)), 1, "fail"),
    ("future_dates", lambda d: pd.concat([d, pd.DataFrame({"date": [pd.Timestamp("2030-01-01")], "nav": [1.0]})]),
     1, "fail"),
    ("calendar_gaps", lambda d: d.drop(index=[10, 11, 12]), 3, "warn"),
    ("extreme_moves", lambda d: d.assign(nav=d["nav"].where(d.index < 20, d["nav"] * 1.3)), 1, "warn"),
    ("stale_runs", lambda d: d.assign(nav=d["nav"].where(~d.index.isin(range(5, 11)), 50.0)), 1, "warn"),
])
def test_each_check_catches_its_defect(check, mutate, expected, status):
    nav = mutate(clean_nav())
    got = counts(nav)
    assert got[check] == expected
    others = {k: v for k, v in got.items() if k != check and k not in ("calendar_gaps", "extreme_moves")}
    assert not any(others.values()), others
    assert status_of(run_checks(nav, META, CAL, TODAY, [check])) == status


def test_metadata_and_idcw_checks():
    assert counts(clean_nav(), {**META, "benchmark_index": None, "category": ""})["missing_metadata"] == 2
    assert counts(clean_nav(), {**META, "option_type": "IDCW"})["idcw_option"] == 1


def test_unknown_check_rejected():
    with pytest.raises(ValueError):
        run_checks(clean_nav(), META, CAL, TODAY, ["nope"])


def test_empty_series_gap_check():
    assert counts(clean_nav().iloc[:0], checks=["calendar_gaps"])["calendar_gaps"] == 0


def test_validate_series_reasons():
    df = pd.DataFrame({"amfi_code": ["100001", "100001", "100001", "100001", "100002", "100001"],
                       "date": ["2024-01-01", "2024-01-01", "bad", "2099-01-01", "2024-01-01", "2024-01-02"],
                       "nav": [10, 11, 10, 10, 10, 0]})
    ok, rej = validate_series(df, "amfi_code", "date", "nav", {"100001"}, TODAY, "f.csv")
    assert len(ok) == 1
    assert sorted(r["reason"] for r in rej) == ["duplicate_date", "future_date", "missing_or_unparseable",
                                                "nonpositive_value", "unknown_scheme"]
    assert all(r["row"] >= 2 for r in rej)


def test_validate_schemes_reasons():
    df = pd.DataFrame({"amfi_code": ["100001", "12AB", "100001", "100003", "100004"],
                       "scheme_name": ["A", "B", "C", "D", None], "fund_house": ["H"] * 5,
                       "plan": ["Direct", "Direct", "Direct", "Hybrid", "Direct"],
                       "expense_ratio_pct": [1.0, 1.0, 1.0, 1.0, 1.0]})
    ok, rej = validate_schemes(df, "m.csv")
    assert list(ok["scheme_code"]) == ["100001"]
    assert sorted(r["reason"] for r in rej) == ["bad_scheme_code", "duplicate_scheme_code",
                                                "missing_required_field", "unknown_plan_type"]
    bad_ter = df.iloc[[0]].assign(expense_ratio_pct=9.0)
    assert validate_schemes(bad_ter, "m.csv")[1][0]["reason"] == "expense_ratio_out_of_range"


def test_name_helpers():
    assert option_type_from_name("X Fund - Regular - IDCW") == "IDCW"
    assert option_type_from_name("X Fund - Direct Plan - Growth") == "Growth"
    assert option_type_from_name("Nippon India ETF Nifty 50 BeES") == "NA"
    assert base_name("SBI Bluechip Fund - Direct Plan - Growth") == base_name("SBI Bluechip Fund - Regular Plan - Growth")
    assert base_name("HDFC Mid-Cap Opportunities Fund - Direct - Growth") == \
        base_name("HDFC Mid-Cap Opportunities Fund - Regular - Growth")
