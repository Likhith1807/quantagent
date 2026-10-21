"""Volatility model: no look-ahead in features, correct target, losses, DM test, walk-forward
purging, registry round trip and the predict_volatility tool."""
import json

import numpy as np
import pandas as pd
import pytest

from quantagent.config import Settings
from quantagent.data.engine import make_readonly_engine
from quantagent.data.ingest import ingest
from quantagent.data.repository import Repository
from quantagent.ml import registry
from quantagent.ml.train import run
from quantagent.ml.volatility import (
    FEATURES,
    diebold_mariano,
    features,
    log_returns,
    panel,
    qlike,
    target,
    target_end,
)
from quantagent.ml.walkforward import TEST_EVERY, Config, walk_forward
from quantagent.tools import _base
from quantagent.tools.volatility import _load, predict_volatility
from tests.fixtures import make_vol_fixture as vf
from tests.tools.helpers import err

R = pd.Series(np.random.default_rng(0).normal(0, 0.01, 600), index=pd.bdate_range("2020-01-01", periods=600))


def test_features_use_only_the_past():
    base = features(R)
    changed = R.copy()
    changed.iloc[400:] *= 5  # rewrite the future
    after = features(changed)
    pd.testing.assert_frame_equal(base.iloc[:400], after.iloc[:400])


def test_target_is_next_21_returns():
    y = target(R)
    t = 300
    expected = np.sqrt(252 / 21 * (R.iloc[t + 1: t + 22] ** 2).sum())
    assert y.iloc[t] == pytest.approx(expected, rel=1e-12)
    assert y.iloc[-21:].isna().all() and y.iloc[-22:-21].notna().all()
    assert target_end(R.index).iloc[t] == R.index[t + 21]


def test_rv_feature_matches_definition():
    f = features(R)
    assert f["rv_m"].iloc[100] == pytest.approx(np.sqrt(252 * (R.iloc[79:101] ** 2).mean()), rel=1e-12)
    assert f["rv_d"].iloc[100] == pytest.approx(abs(R.iloc[100]) * np.sqrt(252), rel=1e-12)


def test_qlike_zero_at_truth_and_asymmetric():
    assert qlike(np.array([0.2]), np.array([0.2]))[0] == pytest.approx(0)
    under, over = qlike(np.array([0.2]), np.array([0.1]))[0], qlike(np.array([0.2]), np.array([0.4]))[0]
    assert under > over > 0  # under-forecasting variance is punished more


def test_diebold_mariano_signs():
    rng = np.random.default_rng(1)
    b = rng.normal(1, 0.1, 200)
    better = diebold_mariano(b - 0.05, b + rng.normal(0, 0.01, 200))
    assert better.statistic < 0 and better.p_value < 0.01
    same = diebold_mariano(b, b + rng.normal(0, 0.01, 200))
    assert same.p_value > 0.01
    assert np.isnan(diebold_mariano(b[:2], b[:2]).statistic)


@pytest.fixture(scope="module")
def vol_db(tmp_path_factory):
    root = tmp_path_factory.mktemp("vol")
    vf.build(root / "raw", root / "ref")
    db = root / "vol.db"
    s = Settings(db_url=f"sqlite:///{db.as_posix()}", model_dir=str(root / "registry"), _env_file=None)
    ingest(root / "raw", db, root / "ref", root / "rej.csv", s)
    return root, s, Repository(make_readonly_engine(s.db_url))


def test_walk_forward_purges_training_targets(vol_db):
    _, _, repo = vol_db
    navs = {c: repo.nav_series(c) for c in vf.CODES[:4]}
    p = panel(navs)
    preds, folds = walk_forward(p, navs, Config(garch=False, first_test_year=2019))
    assert folds and not preds.empty
    for f in folds:
        assert pd.Timestamp(f["train_last_target_end"]) < pd.Timestamp(f["test_start"]) - pd.Timedelta(days=30)
    assert (preds["n_hist"] % TEST_EVERY == 0).all()
    assert preds[["naive", "ewma", "har", "ridge", "gbm"]].notna().all(axis=None)


def test_train_registry_and_tool(vol_db, tmp_path):
    root, s, repo = vol_db
    out = run(repo, Config(first_test_year=2019, holdout_years=2, gbm={"max_iter": 50, "min_samples_leaf": 50}),
              root / "registry",
              tmp_path / "report.md", tmp_path / "exp.jsonl", tmp_path, max_schemes=6, log=lambda *a: None)
    assert set(out["models"]) == {"naive", "ewma", "har", "garch", "ridge", "gbm"}
    assert all(np.isfinite(v["qlike"]) for v in out["models"].values())
    sel = out["selected"]
    assert sel["holdout_start"] == "2021-01-01"
    assert out["selection"]["period"][1] < "2021-01-01" <= out["test"]["period"][0]
    assert sel["model"] == min(sel["selection_qlike"], key=sel["selection_qlike"].get)
    assert out["models"] == out["test"]["models"]
    assert set(out["dm"]) == {f"{sel['model']}_vs_{m}" for m in out["models"] if m != sel["model"]}
    assert out["test_filtered_at_origin"]["n_test"] + out["test_filtered_at_origin"]["excluded"] == out["n_test"]
    objects, config, metrics = registry.load(root / "registry")
    assert config["features"] == FEATURES and config["walk_forward"]["seed"] == 42
    assert json.loads((tmp_path / "exp.jsonl").read_text().splitlines()[0])["version"] == out["version"]
    report = (tmp_path / "report.md").read_text()
    assert "Diebold-Mariano" in report and "Selection period" in report and "Diagnostic only" in report

    _base.set_services(_base.Services(s, repo))
    _load.cache_clear()
    try:
        r = predict_volatility(scheme_code=vf.CODES[0])
        assert 0.02 < r.forecast_vol < 2 and r.horizon_days == 21 and r.model.version == out["version"]
        assert set(r.baselines) == {"naive", "ewma", "har"}
        nav = repo.nav_series(vf.CODES[0])
        expected_rv = np.sqrt(252 * (log_returns(nav).iloc[-22:] ** 2).mean())
        assert r.recent_realised_vol["rv_22"] == pytest.approx(expected_rv, rel=1e-12)
        early = predict_volatility(scheme_code=vf.CODES[0], as_of="2019-06-28")
        assert early.origin_date.isoformat() == "2019-06-28" and early.meta.period.end.isoformat() == "2019-06-28"
        assert err(predict_volatility, scheme_code=vf.CODES[0], as_of="2015-03-02").code == "INSUFFICIENT_HISTORY"
        assert err(predict_volatility, scheme_code="999999").code == "NOT_FOUND"
    finally:
        _base.set_services(None)


def test_registry_rejects_tampered_file(vol_db, tmp_path):
    registry.save(tmp_path, "v1", {"model": {"a": 1}}, {"model_kind": "ridge"}, {})
    assert registry.load(tmp_path)[1]["version"] == "v1"
    (tmp_path / "v1" / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        registry.load(tmp_path)
    with pytest.raises(FileNotFoundError):
        registry.load(tmp_path / "empty")


def test_predict_without_model(services):
    services.settings = services.settings.model_copy(update={"model_dir": "does/not/exist"})
    _load.cache_clear()
    assert err(predict_volatility, scheme_code="900001").code == "MODEL_UNAVAILABLE"


def test_suspect_days_and_clean_targets():
    from quantagent.ml.volatility import suspect_before, suspect_days, target_clean

    r = pd.Series(0.01, index=pd.bdate_range("2024-01-01", periods=60))
    r.iloc[10] = np.log(1.2)  # +20% in a day: NAV error
    r.iloc[30:34] = 0.0  # 5 identical NAVs
    r.iloc[45:47] = 0.0  # 3 identical NAVs: fine
    bad = suspect_days(r)
    assert list(bad[bad].index.map(r.index.get_loc)) == [10, 30, 31, 32, 33]
    clean = target_clean(r, horizon=5)
    assert not clean.iloc[5] and clean.iloc[10] and not clean.iloc[29] and clean.iloc[34] and clean.iloc[40]
    assert not clean.iloc[-1]  # no complete target window
    past = suspect_before(r, lookback=5)
    assert not past.iloc[9] and past.iloc[10] and past.iloc[14] and not past.iloc[15]  # only looks back
