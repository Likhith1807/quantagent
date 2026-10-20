import pandas as pd
import pytest

from quantagent.tools.compare import compare_funds
from quantagent.tools.rank import rank_funds
from tests.fixtures import make_fixture as fx
from tests.tools.helpers import err, has


# ------------------------------------------------------------ compare_funds
def test_compare_constant_growth_closed_form(repo):
    r = compare_funds(scheme_codes=[fx.CONST_DIRECT, fx.CONST_REGULAR], period="1y")
    nav = repo.nav_series(fx.CONST_DIRECT)
    win = nav[nav.index >= fx.END - pd.DateOffset(years=1)]
    n, days = len(win) - 1, (win.index[-1] - win.index[0]).days
    d = r.funds[0]
    assert d.cagr == pytest.approx((1 + fx.CONST_STEP) ** (n * 365.25 / days) - 1, rel=1e-9)
    assert d.total_return == pytest.approx((1 + fx.CONST_STEP) ** n - 1, rel=1e-9)
    assert d.volatility == 0.0 and d.sharpe is None and d.max_drawdown.max_drawdown == 0.0
    assert r.funds[0].cagr > r.funds[1].cagr
    assert not r.fair_comparison.is_fair
    assert any("Direct and Regular" in x for x in r.fair_comparison.reasons)
    assert r.meta.n_obs == len(win) and r.shared_period.end == fx.END.date()


def test_compare_late_start_short_history_and_categories():
    r = compare_funds(scheme_codes=[fx.DRAWDOWN, fx.SHORT], period="1y")
    reasons = " ".join(r.fair_comparison.reasons)
    assert "after the requested start" in reasons and "Short shared history" in reasons
    assert "different sub-categories" in reasons and "cover" in reasons
    assert r.funds[0].n_obs == 30 and r.funds[0].volatility is None and has(r.warnings, "shared returns")


def test_compare_idcw_and_correlation():
    r = compare_funds(scheme_codes=[fx.DRAWDOWN, fx.IDCW, fx.STALE], period="max")
    idcw = next(f for f in r.funds if f.scheme_code == fx.IDCW)
    assert idcw.cagr is None and idcw.max_drawdown is None and idcw.volatility is not None
    assert any("IDCW" in x for x in r.fair_comparison.reasons) and has(r.warnings, "IDCW")
    for c in (fx.DRAWDOWN, fx.IDCW, fx.STALE):
        assert r.correlation[c][c] == pytest.approx(1.0)


def test_compare_fair_case():
    r = compare_funds(scheme_codes=[fx.DRAWDOWN, fx.GAPPY], period="1y")
    assert r.fair_comparison.is_fair and r.fair_comparison.reasons == []


@pytest.mark.parametrize("codes,code", [
    ([fx.DRAWDOWN, fx.DRAWDOWN], "INVALID_INPUT"),
    ([fx.DRAWDOWN], "INVALID_INPUT"),
    ([str(900001 + i) for i in range(11)], "INVALID_INPUT"),
    ([fx.DRAWDOWN, "999999"], "NOT_FOUND"),
    ([fx.CLOSED, fx.SHORT], "NO_OVERLAP"),
])
def test_compare_errors(codes, code):
    kwargs = {"scheme_codes": codes, "period": "max" if code == "NO_OVERLAP" else None}
    e = err(compare_funds, **kwargs)
    assert e.code == code
    if code == "NOT_FOUND":
        assert e.details["unknown"] == ["999999"]


def test_compare_configured_max(services):
    services.settings = services.settings.model_copy(update={"max_funds": 2})
    assert err(compare_funds, scheme_codes=[fx.DRAWDOWN, fx.GAPPY, fx.STALE]).code == "INVALID_INPUT"


# ------------------------------------------------------------ rank_funds
def _excluded(r) -> dict:
    return {x.id: x.reason for x in r.excluded}


def test_rank_by_sharpe_matches_cache_and_excludes(repo):
    r = rank_funds(metric="sharpe", period="1y")
    ex = _excluded(r)
    assert "IDCW" in ex[fx.IDCW] and "insufficient 1y history: 30" in ex[fx.SHORT]
    assert "sharpe unavailable" in ex[fx.CONST_DIRECT]  # zero volatility -> no Sharpe
    assert fx.CLOSED not in ex and r.universe_size == 8
    codes = [x.scheme_code for x in r.ranked]
    assert set(codes) == {fx.DRAWDOWN, fx.GAPPY, fx.TRACKER, fx.STALE} and r.eligible_count == 4
    cache = repo.metrics_cache("1y", 1).set_index("scheme_code")["sharpe"]
    assert codes == list(cache[codes].sort_values(ascending=False).index)
    assert [x.rank for x in r.ranked] == [1, 2, 3, 4]
    assert has(r.warnings, "Survivorship") and "metrics_cache" in r.methodology


def test_rank_dedupe_prefers_direct():
    r = rank_funds(metric="cagr", period="1y")
    codes = [x.scheme_code for x in r.ranked]
    assert fx.CONST_DIRECT in codes and fx.CONST_REGULAR not in codes
    assert "Direct kept" in _excluded(r)[fx.CONST_REGULAR]
    both = rank_funds(metric="cagr", period="1y", dedupe="none")
    assert {fx.CONST_DIRECT, fx.CONST_REGULAR} <= {x.scheme_code for x in both.ranked}


def test_rank_direction_and_override():
    asc = rank_funds(metric="volatility", period="1y")
    vols = [x.metrics["volatility"] for x in asc.ranked]
    assert vols == sorted(vols)
    desc = rank_funds(metric="volatility", period="1y", ascending=False)
    assert [x.metrics["volatility"] for x in desc.ranked] == sorted(vols, reverse=True)
    er = rank_funds(metric="expense_ratio", period="1y")
    assert er.ranked[0].expense_ratio_pct == 0.5


def test_rank_alpha_and_filters():
    r = rank_funds(metric="alpha", period="1y")
    assert "no benchmark series" in _excluded(r)[fx.GAPPY] and has(r.warnings, "Alpha is not")
    cheap = rank_funds(metric="cagr", period="1y", max_expense_ratio=0.6)
    assert {x.scheme_code for x in cheap.ranked} == {fx.CONST_DIRECT, fx.TRACKER}
    assert "expense ratio above" in _excluded(cheap)[fx.DRAWDOWN]
    mid = rank_funds(metric="cagr", period="1y", category="Mid Cap")
    assert mid.universe_size == 1 and mid.eligible_count == 0 and has(mid.warnings, "No funds are eligible")
    none = rank_funds(metric="cagr", period="1y", fund_house="nobody")
    assert none.universe_size == 0 and has(none.warnings, "No active schemes")
    top = rank_funds(metric="cagr", period="1y", top_n=2)
    assert len(top.ranked) == 2


def test_rank_weighted():
    r = rank_funds(weights={"sharpe": 2, "max_drawdown": 1}, period="1y")
    assert r.ranked_by == "weighted" and r.weights == {"sharpe": 2, "max_drawdown": 1}
    scores = [x.score for x in r.ranked]
    assert scores == sorted(scores, reverse=True)
    assert set(r.ranked[0].metrics) >= {"sharpe", "max_drawdown", "cagr"}


def test_rank_3y_all_insufficient():
    r = rank_funds(metric="cagr", period="3y")
    assert r.eligible_count == 0 and all("insufficient" in x.reason or "IDCW" in x.reason for x in r.excluded)


@pytest.mark.parametrize("kwargs", [
    {}, {"metric": "cagr", "weights": {"cagr": 1}}, {"weights": {"cagr": 0}},
    {"weights": {"cagr": 1}, "ascending": True}, {"metric": "beta"}, {"weights": {"cagr; drop": 1}},
    {"metric": "cagr", "top_n": 0}, {"metric": "cagr", "period": "max"},
])
def test_rank_invalid(kwargs):
    assert err(rank_funds, **kwargs).code == "INVALID_INPUT"


def test_rank_configured_top_n(services):
    services.settings = services.settings.model_copy(update={"max_results": 3})
    assert err(rank_funds, metric="cagr", top_n=5).code == "INVALID_INPUT"


def test_rank_stale_cache(repo, monkeypatch):
    monkeypatch.setattr(repo, "metrics_cache", lambda window, version: pd.DataFrame())
    e = err(rank_funds, metric="cagr")
    assert e.code == "DATA_QUALITY" and e.details["data_version"] == 1
