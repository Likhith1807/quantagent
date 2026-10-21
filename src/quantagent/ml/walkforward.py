"""Walk-forward evaluation with purging and an embargo gap, so no test target leaks into training.

Folds are calendar years. For a fold starting on date S:
  * training rows: origins whose whole target window ends before S - gap (purge + embargo), and
    whose target window has no suspect NAV (volatility.suspect_days)
  * test rows:     origins dated in [S, next fold start), sampled every TEST_EVERY observations
  * GARCH:         parameters estimated on returns up to S - gap only
Models are refitted from scratch on every fold.

Model selection and testing use different years. Folds before the hold-out start (the last
`holdout_years` calendar years by default) form the selection period: the served model is the one
with the lowest QLIKE there. Folds from the hold-out start on are the test period; they play no part
in any choice, and their scores are the reported results.

The headline scores use every test row. A second view drops test rows whose scheme had a suspect
NAV in the year before the origin, which is known when the forecast is made. Filtering on the
target window itself would use the future, so that view is reported only as a diagnostic.
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
    holdout_years: int = 5  # the last N calendar years are the untouched test period
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


def holdout_start(p: pd.DataFrame, cfg: Config) -> pd.Timestamp:
    return pd.Timestamp(f"{p['date'].max().year - cfg.holdout_years + 1}-01-01")


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


def _scores(rows: pd.DataFrame, models: list[str], ref: str | None = None) -> dict:
    """QLIKE/MAE per model; with `ref`, DM tests of ref against every other model on the monthly
    cross-sectional mean QLIKE (Newey-West lag 1: adjacent months' targets can overlap).
    Negative DM statistic = ref has the lower loss."""
    month = rows["date"].dt.to_period("M")
    res: dict = {"n_test": len(rows), "n_months": int(month.nunique()),
                 "period": [str(rows["date"].min().date()), str(rows["date"].max().date())] if len(rows) else None,
                 "models": {}, "dm": {}}
    losses = {}
    for mdl in models:
        q, a = qlike(rows["y"], rows[mdl]), mae(rows["y"], rows[mdl])
        losses[mdl] = pd.Series(q, index=rows.index).groupby(month).mean()
        res["models"][mdl] = {"qlike": float(np.mean(q)), "qlike_median": float(np.median(q)),
                              "mae": float(np.mean(a))}
    if ref:
        for other in [m for m in models if m != ref]:
            dm = diebold_mariano(losses[ref].to_numpy(), losses[other].to_numpy(), lag=1)
            res["dm"][f"{ref}_vs_{other}"] = {
                "stat": dm.statistic, "p_value": dm.p_value, "mean_qlike_diff": dm.mean_diff, "months": dm.n,
                "months_better_pct": float(100 * (losses[ref] < losses[other]).mean())}
    return res


def evaluate(preds: pd.DataFrame, models: list[str], start: pd.Timestamp) -> dict:
    """Pick the served model on the selection period (test origins before `start`), then score
    every model on the hold-out period (origins from `start`)."""
    rows = preds.dropna(subset=models)
    sel_rows, test_rows = rows[rows["date"] < start], rows[rows["date"] >= start]
    if sel_rows.empty or test_rows.empty:
        raise RuntimeError(f"need test folds both before and after the hold-out start {start.date()}")
    selection = _scores(sel_rows, models)
    best = min(["har", *MODELS], key=lambda m: selection["models"][m]["qlike"])
    test = _scores(test_rows, models, ref=best)
    known = test_rows[~test_rows["suspect_past"].astype(bool)]
    filtered = _scores(known, models, ref=best)
    filtered["excluded"] = int(len(test_rows) - len(known))
    clean = test_rows[test_rows["clean"].astype(bool)]
    diagnostic = _scores(clean, models, ref=best)
    diagnostic["excluded"] = int(len(test_rows) - len(clean))

    others = {b: test["models"][b]["qlike"] for b in models if b in BASELINES and b != best}
    best_other = min(others, key=others.get)
    ml = min(MODELS, key=lambda m: selection["models"][m]["qlike"])
    q = test["models"][best]["qlike"]
    selected = {
        "model": best, "chosen_on": "selection period (folds before the hold-out start)",
        "holdout_start": str(start.date()),
        "selection_qlike": {m: selection["models"][m]["qlike"] for m in ["har", *MODELS]},
        "best_other_baseline": best_other,
        "beats_all_baselines": all(q < v for v in others.values()),
        "ml_beats_har": selection["models"][ml]["qlike"] < selection["models"]["har"]["qlike"],
        "ml_beats_har_test": min(test["models"][m]["qlike"] for m in MODELS) < test["models"]["har"]["qlike"],
        "qlike_vs_ewma_pct": 100 * (q / test["models"]["ewma"]["qlike"] - 1),
        "qlike_vs_best_other_pct": 100 * (q / others[best_other] - 1)}
    return {"protocol": "v2: select on earlier folds, report later folds", "selection": selection, "test": test,
            "test_filtered_at_origin": filtered, "test_clean_target_ex_post": diagnostic, "selected": selected,
            # what the tool and the README read: hold-out scores on all rows
            "models": test["models"], "dm": test["dm"], "n_test": test["n_test"], "n_months": test["n_months"]}
