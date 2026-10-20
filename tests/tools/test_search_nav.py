from datetime import date, timedelta

import pytest

from quantagent.tools.nav import get_nav_history
from quantagent.tools.search import search_funds
from tests.fixtures import make_fixture as fx
from tests.tools.helpers import err, has


# ------------------------------------------------------------ search_funds
def test_search_by_name_and_envelope():
    r = search_funds(query="const")
    assert r.total_matches == 2 and {i.scheme_code for i in r.items} == {fx.CONST_DIRECT, fx.CONST_REGULAR}
    item = next(i for i in r.items if i.scheme_code == fx.CONST_DIRECT)
    assert item.latest_expense_ratio.value_pct == 0.5 and item.latest_expense_ratio.as_of_assumed
    assert item.nav_coverage.n_points == len(fx.TRADING)
    assert r.meta.data_nature == "synthetic" and r.meta.data_version == 1 and r.meta.request_id != "-"
    assert has(r.warnings, "synthetic")


def test_search_by_code_and_filters():
    assert search_funds(query=fx.TRACKER).items[0].scheme_code == fx.TRACKER
    assert {i.scheme_code for i in search_funds(plan_type="Direct").items} == {fx.CONST_DIRECT, fx.TRACKER}
    assert search_funds(option_type="IDCW").items[0].scheme_code == fx.IDCW
    assert search_funds(category="mid cap").total_matches == 1
    assert search_funds(fund_house="index house").total_matches == 1


def test_search_status_filter():
    assert search_funds(query="closed").total_matches == 0
    assert search_funds(query="closed", status="closed").items[0].status == "closed"
    assert search_funds(fund_house="fixture", status="any").total_matches == 8


def test_search_pagination():
    page1 = search_funds(fund_house="fixture", limit=3)
    page2 = search_funds(fund_house="fixture", limit=3, offset=3)
    assert page1.total_matches == 7 and len(page1.items) == 3 and len(page2.items) == 3
    assert not {i.scheme_code for i in page1.items} & {i.scheme_code for i in page2.items}
    assert has(search_funds(fund_house="fixture", offset=50).warnings, "past the last match")


def test_search_empty_is_a_normal_result():
    r = search_funds(query="nothing like this")
    assert r.total_matches == 0 and r.items == [] and has(r.warnings, "No schemes matched")


@pytest.mark.parametrize("kwargs", [{}, {"status": "any"}, {"query": "x", "limit": 0}, {"query": "x", "limit": 51},
                                    {"query": "x", "offset": -1}, {"plan_type": "Both"}, {"query": ""}])
def test_search_invalid_input(kwargs):
    assert err(search_funds, **kwargs).code == "INVALID_INPUT"


def test_search_respects_configured_limit(services):
    services.settings = services.settings.model_copy(update={"max_results": 5})
    assert err(search_funds, query="fund", limit=10).code == "INVALID_INPUT"


# ------------------------------------------------------------ get_nav_history
def test_nav_full_daily_history_has_no_gaps():
    r = get_nav_history(scheme_code=fx.DRAWDOWN)
    assert r.n_points == len(fx.TRADING) and r.missing_trading_days == 0 and r.gaps == []
    assert r.actual_start == fx.TRADING[0].date() and r.actual_end == fx.END.date()
    assert not r.truncated and r.meta.n_obs == r.n_points


def test_nav_gaps_measured_on_trading_calendar():
    r = get_nav_history(scheme_code=fx.GAPPY)
    assert r.missing_trading_days == 10
    g = r.gaps[0]
    assert (g.start, g.end, g.missing_trading_days) == (fx.TRADING[300].date(), fx.TRADING[309].date(), 10)


def test_nav_resampling_and_dates():
    m = get_nav_history(scheme_code=fx.DRAWDOWN, frequency="monthly")
    assert m.n_points == 24 and m.missing_trading_days is None
    w = get_nav_history(scheme_code=fx.DRAWDOWN, frequency="weekly", start_date=date(2025, 1, 1),
                        end_date=date(2025, 3, 31))
    assert all(p.date.weekday() <= 4 for p in w.points) and w.actual_end <= date(2025, 3, 31)


def test_nav_truncation_keeps_latest():
    r = get_nav_history(scheme_code=fx.DRAWDOWN, max_points=10)
    assert r.truncated and r.n_points == 10 and r.actual_end == fx.END.date() and has(r.warnings, "Truncated")


def test_nav_empty_range_and_clipped_end():
    r = get_nav_history(scheme_code=fx.DRAWDOWN, start_date=date(2020, 1, 1), end_date=date(2020, 2, 1))
    assert r.n_points == 0 and has(r.warnings, "No NAVs")
    if date.today() > fx.END.date():
        c = get_nav_history(scheme_code=fx.DRAWDOWN, start_date=date(2025, 12, 1), end_date=date.today())
        assert has(c.warnings, "clipped") and c.actual_end == fx.END.date()


@pytest.mark.parametrize("kwargs,code", [
    ({"scheme_code": "999999"}, "NOT_FOUND"),
    ({"scheme_code": "12ab"}, "INVALID_INPUT"),
    ({"scheme_code": fx.DRAWDOWN, "start_date": date(2025, 2, 1), "end_date": date(2025, 1, 1)}, "INVALID_INPUT"),
    ({"scheme_code": fx.DRAWDOWN, "end_date": date.today() + timedelta(days=5)}, "INVALID_INPUT"),
    ({"scheme_code": fx.DRAWDOWN, "start_date": date(1990, 1, 1), "end_date": date(2025, 1, 1)}, "INVALID_INPUT"),
    ({"scheme_code": fx.DRAWDOWN, "frequency": "hourly"}, "INVALID_INPUT"),
    ({"scheme_code": fx.DRAWDOWN, "max_points": 0}, "INVALID_INPUT"),
])
def test_nav_errors(kwargs, code):
    assert err(get_nav_history, **kwargs).code == code
