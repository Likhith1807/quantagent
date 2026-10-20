import pandas as pd
import pytest

from quantagent.analytics.ranking import NATURAL_DIRECTION, rank_single, rank_weighted

DF = pd.DataFrame({
    "scheme_code": ["a", "b", "c"],
    "sharpe": [1.0, 2.0, 0.5],
    "volatility": [0.10, 0.20, 0.05],
    "max_drawdown": [-0.1, -0.3, -0.2],
})


def test_single_metric_natural_direction():
    assert list(rank_single(DF, "sharpe")["scheme_code"]) == ["b", "a", "c"]
    assert list(rank_single(DF, "volatility")["scheme_code"]) == ["c", "a", "b"]
    assert list(rank_single(DF, "max_drawdown")["scheme_code"]) == ["a", "c", "b"]
    assert list(rank_single(DF, "sharpe", ascending=True)["scheme_code"]) == ["c", "a", "b"]


def test_weighted_equals_manual_zscores():
    out = rank_weighted(DF, {"sharpe": 2, "volatility": 1})
    z = lambda s: (s - s.mean()) / s.std(ddof=0)  # noqa: E731
    expected = (2 * z(DF["sharpe"]) - 1 * z(DF["volatility"])) / 3
    got = out.set_index("scheme_code")["score"]
    for code, val in zip(DF["scheme_code"], expected, strict=True):
        assert got[code] == pytest.approx(val)
    assert list(out["score"]) == sorted(out["score"], reverse=True)


def test_weighted_zero_spread_and_bad_weights():
    flat = DF.assign(sharpe=1.0)
    assert (rank_weighted(flat, {"sharpe": 1})["score"] == 0).all()
    assert rank_weighted(DF, {"sharpe": 1, "volatility": 0})["scheme_code"].iloc[0] == "b"
    with pytest.raises(ValueError):
        rank_weighted(DF, {"sharpe": 0})


def test_every_rank_metric_has_direction():
    assert set(NATURAL_DIRECTION) >= {"cagr", "sharpe", "volatility", "expense_ratio"}
