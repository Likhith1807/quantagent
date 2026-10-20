"""Walk-forward evaluation with purging and an embargo gap, so no test target leaks into training.

Folds are calendar years. For a fold starting on date S:
  * training rows: origins whose whole target window ends before S - gap (purge + embargo), and
    whose target window has no suspect NAV (volatility.suspect_days)
  * test rows:     origins dated in [S, next fold start), sampled every TEST_EVERY observations
  * GARCH:         parameters estimated on returns up to S - gap only
Models are refitted from scratch on every fold.
"""
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .volatility import (
    FEATURES,
    HAR_FEATURES,
    diebold_mariano,
    ewma_forecast,
    garch_forecasts,
    log_returns,
    log_x,
    mae,
    naive_forecast,
    qlike,
)

TRAIN_EVERY = 5   # training origins: every 5th observation per scheme
TEST_EVERY = 21   # test origins: every 21st observation (about one per month, non-overlapping)
BASELINES = ["naive", "ewma", "har", "garch"]
MODELS = ["ridge", "gbm"]


@dataclass
class Config:
    horizon: int = 21
    gap_days: int = 30
    first_test_year: int | None = None  # default: 4 years after the first origin
    min_train_rows: int = 500
    seed: int = 42
    garch: bool = True
    ridge_alphas: tuple = (0.01, 0.1, 1.0, 10.0, 100.0)
    gbm: dict = field(default_factory=lambda: {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15,
                                               "min_samples_leaf": 200, "l2_regularization": 1.0})


class LogModel:
    """Regression on log volatility with Duan's smearing correction back to the vol scale."""

    def __init__(self, kind: str, cols: list[str], cfg: Config):
        self.kind, self.cols = kind, cols
        if kind == "har":
            self.est = LinearRegression()
        elif kind == "ridge":
            self.est = make_pipeline(StandardScaler(), RidgeCV(alphas=cfg.ridge_alphas))
        elif kind == "gbm":
            self.est = HistGradientBoostingRegressor(random_state=cfg.seed, **cfg.gbm)
        else:
            raise ValueError(kind)
        self.smear = 1.0

    def fit(self, df: pd.DataFrame) -> "LogModel":
        x, y = log_x(df, self.cols), np.log(df["y"].to_numpy(dtype=float))
        self.est.fit(x, y)
        self.smear = float(np.mean(np.exp(y - self.est.predict(x))))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.est.predict(log_x(df, self.cols))) * self.smear


def make_model(name: str, cfg: Config) -> LogModel:
    return LogModel(name, HAR_FEATURES if name == "har" else FEATURES, cfg)


def fold_starts(p: pd.DataFrame, cfg: Config) -> list[pd.Timestamp]:
    first, last = p["date"].min(), p["date"].max()
    y0 = cfg.first_test_year or first.year + 4
    return [pd.Timestamp(f"{y}-01-01") for y in range(y0, last.year + 1)]


def walk_forward(p: pd.DataFrame, navs: dict[str, pd.Series], cfg: Config,
                 log=lambda *a: None) -> tuple[pd.DataFrame, list[dict]]:
    """Out-of-sample predictions for every test row (one column per model) and per-fold info."""
    p = p.dropna(subset=[*FEATURES]).copy()
    has_y = p["y"].notna()
    train_pool = p[has_y & p["clean"].astype(bool) & (p["n_hist"] % TRAIN_EVERY == 0)]
    test_pool = p[has_y & (p["n_hist"] % TEST_EVERY == 0)]
    starts = fold_starts(p, cfg)
    returns = {c: log_returns(s) for c, s in navs.items()} if cfg.garch else {}
    out, folds = [], []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else pd.Timestamp.max
        cutoff = s - pd.Timedelta(days=cfg.gap_days)
        train = train_pool[train_pool["target_end"] < cutoff]
        test = test_pool[(test_pool["date"] >= s) & (test_pool["date"] < e)].copy()
        if len(train) < cfg.min_train_rows or test.empty:
            continue
        t0 = time.perf_counter()
        test["naive"] = naive_forecast(test)
        test["ewma"] = ewma_forecast(test)
        for name in ["har", *MODELS]:
            test[name] = make_model(name, cfg).fit(train).predict(test)
        if cfg.garch:
            g = []
            for code, rows in test.groupby("scheme_code"):
                f = garch_forecasts(returns[code], pd.DatetimeIndex(rows["date"]), cutoff, cfg.horizon)
                g.append(pd.Series(f.to_numpy(), index=rows.index))
            test["garch"] = pd.concat(g)
        folds.append({"test_start": str(s.date()), "test_end": str(min(e, p["date"].max()).date()),
                      "train_rows": len(train), "train_last_target_end": str(train["target_end"].max().date()),
                      "test_rows": len(test), "schemes": int(test["scheme_code"].nunique()),
                      "seconds": round(time.perf_counter() - t0, 1)})
        log(f"fold {s.date()}: train {len(train)} test {len(test)}")
        out.append(test)
    preds = pd.concat(out) if out else pd.DataFrame()
    return preds, folds


def _scores(rows: pd.DataFrame, models: list[str]) -> dict:
    month = rows["date"].dt.to_period("M")
    res: dict = {"n_test": len(rows), "n_months": int(month.nunique()), "models": {}, "dm": {}}
    losses = {}
    for mdl in models:
        q, a = qlike(rows["y"], rows[mdl]), mae(rows["y"], rows[mdl])
        losses[mdl] = pd.Series(q, index=rows.index).groupby(month).mean()
        res["models"][mdl] = {"qlike": float(np.mean(q)), "qlike_median": float(np.median(q)),
                              "mae": float(np.mean(a))}
    for mdl in [m for m in models if m in MODELS or m == "har"]:
        for b in [b for b in models if b in BASELINES and b != mdl]:
            dm = diebold_mariano(losses[mdl].to_numpy(), losses[b].to_numpy(), lag=1)
            res["dm"][f"{mdl}_vs_{b}"] = {
                "stat": dm.statistic, "p_value": dm.p_value, "mean_qlike_diff": dm.mean_diff, "months": dm.n,
                "months_better_pct": float(100 * (losses[mdl] < losses[b]).mean())}
    return res


def evaluate(preds: pd.DataFrame, models: list[str]) -> dict:
    """QLIKE/MAE per model and DM tests on the monthly cross-sectional mean QLIKE (NW lag 1:
    adjacent months' targets can overlap). Scored on clean target windows (primary) and on all."""
    rows = preds.dropna(subset=models)
    clean = rows[rows["clean"].astype(bool)]
    res = _scores(clean, models)
    res["all_rows"] = _scores(rows, models)
    res["excluded_suspect"] = int(len(rows) - len(clean))
    return res
