"""Features, target, baselines and losses for forecasting 21-day realised volatility.

Notation: r_t is the daily log NAV return between consecutive observations. At forecast origin t
everything on the right-hand side uses r_1..r_t only; the target uses r_{t+1}..r_{t+H}.

  target   y_t = sqrt(252/H * sum_{i=1..H} r_{t+i}^2)          annualised realised vol, H = 21
  RV_k(t)    = sqrt(252/k * sum_{i=0..k-1} r_{t-i}^2)          k = 1 (daily), 5, 22, 63, 252
  EWMA(t)    = sqrt(252 * s2_t),  s2_t = lam*s2_{t-1} + (1-lam)*r_t^2,  lam = 0.94 (RiskMetrics)

Volatility is forecast in logs by the regression models and evaluated on variance (QLIKE) and on
volatility (MAE).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..data.validation import EXTREME_MOVE, STALE_RUN_DAYS

HORIZON = 21
ANNUAL = 252
EWMA_LAMBDA = 0.94
RV_WINDOWS = {"rv_d": 1, "rv_w": 5, "rv_m": 22, "rv_q": 63, "rv_y": 252}
FEATURES = [*RV_WINDOWS, "ewma", "ret_21", "ret_63", "dd_252"]
HAR_FEATURES = ["rv_d", "rv_w", "rv_m"]
MIN_HISTORY = 252  # origins need a year of returns for rv_y and dd_252
EPS = 1e-6  # floor on vol before taking logs (a zero daily return gives rv_d = 0)


def log_returns(nav: pd.Series) -> pd.Series:
    nav = nav.sort_index()
    return np.log(nav).diff().dropna()


def features(r: pd.Series) -> pd.DataFrame:
    """Feature rows for every origin t (index = origin date). Uses r up to and including t."""
    r2 = r**2
    out = {name: np.sqrt(ANNUAL * r2.rolling(k).mean()) for name, k in RV_WINDOWS.items()}
    s2 = r2.ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean()
    out["ewma"] = np.sqrt(ANNUAL * s2)
    out["ret_21"] = r.rolling(21).sum()
    out["ret_63"] = r.rolling(63).sum()
    level = r.cumsum()
    out["dd_252"] = level - level.rolling(252, min_periods=1).max()
    f = pd.DataFrame(out)
    f["n_hist"] = np.arange(1, len(r) + 1)
    return f


def target(r: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """y_t from the next `horizon` returns; NaN where fewer than `horizon` remain."""
    r2 = r**2
    fwd = r2[::-1].rolling(horizon).sum()[::-1].shift(-1)
    return np.sqrt(ANNUAL / horizon * fwd)


def target_end(index: pd.DatetimeIndex, horizon: int = HORIZON) -> pd.Series:
    """Date of the last return inside each origin's target window (for purging)."""
    s = pd.Series(index, index=index)
    return s.shift(-horizon)


def suspect_days(r: pd.Series) -> pd.Series:
    """Returns that check_data_quality flags: a daily move above EXTREME_MOVE (15%), or a day inside
    a run of STALE_RUN_DAYS (5) or more identical NAVs. Usually NAV errors, not market moves."""
    jump = np.expm1(r).abs() > EXTREME_MOVE
    zero = r == 0
    run_id = (~zero).cumsum()
    run_len = zero.groupby(run_id).transform("sum")
    stale = zero & (run_len >= STALE_RUN_DAYS - 1)
    return jump | stale


def target_clean(r: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """True when the target window r_{t+1}..r_{t+H} has no suspect day."""
    bad = suspect_days(r).astype(float)
    fwd = bad[::-1].rolling(horizon, min_periods=1).max()[::-1].shift(-1)
    return fwd.fillna(1) == 0


def suspect_before(r: pd.Series, lookback: int = ANNUAL) -> pd.Series:
    """True when a suspect day lies in r_{t-lookback+1}..r_t: known at the forecast origin, so it
    can be used to filter test rows without looking at the target window."""
    return suspect_days(r).astype(float).rolling(lookback, min_periods=1).max() > 0


def panel(navs: dict[str, pd.Series], horizon: int = HORIZON) -> pd.DataFrame:
    """Long panel: one row per (scheme, origin) with features, target, target end date and
    whether the target window is free of suspect NAVs (`clean`, ex post: used only to filter training
    rows, whose targets are in the past at fit time) and whether the past year had one (`suspect_past`,
    known at the origin)."""
    frames = []
    for code, nav in navs.items():
        r = log_returns(nav)
        if len(r) <= MIN_HISTORY:
            continue
        f = features(r)
        f["y"] = target(r, horizon)
        f["target_end"] = target_end(r.index, horizon)
        f["clean"] = target_clean(r, horizon)
        f["suspect_past"] = suspect_before(r)
        f["r_t"] = r
        f["scheme_code"] = code
        frames.append(f.iloc[MIN_HISTORY - 1:])
    if not frames:
        return pd.DataFrame(columns=[*FEATURES, "n_hist", "y", "target_end", "clean", "suspect_past", "r_t", "scheme_code"])
    p = pd.concat(frames)
    p.index.name = "date"
    return p.reset_index()


def log_x(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Design matrix: logs of the vol features (floored), raw return/drawdown features."""
    out = []
    for c in cols:
        v = df[c].to_numpy(dtype=float)
        out.append(np.log(np.maximum(v, EPS)) if c in RV_WINDOWS or c == "ewma" else v)
    return np.column_stack(out)


# ---------------------------------------------------------------- losses and tests

def qlike(y_vol: np.ndarray, f_vol: np.ndarray) -> np.ndarray:
    """Patton (2011) QLIKE on variances, per observation: s/h - log(s/h) - 1 (0 at a perfect forecast)."""
    s = np.maximum(np.asarray(y_vol, float), EPS) ** 2
    h = np.maximum(np.asarray(f_vol, float), EPS) ** 2
    ratio = s / h
    return ratio - np.log(ratio) - 1


def mae(y_vol: np.ndarray, f_vol: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(y_vol, float) - np.asarray(f_vol, float))


@dataclass
class DMResult:
    statistic: float
    p_value: float
    mean_diff: float
    n: int


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 0) -> DMResult:
    """DM test of equal predictive accuracy on a time series of loss differentials d = a - b,
    with a Newey-West long-run variance and the Harvey-Leybourne-Newbold small-sample correction.
    Negative statistic: a has lower loss. Two-sided p-value from Student t(n-1)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return DMResult(float("nan"), float("nan"), float(d.mean()) if n else float("nan"), n)
    dc = d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    if lrv <= 0:
        return DMResult(float("nan"), float("nan"), float(d.mean()), n)
    dm = d.mean() / np.sqrt(lrv / n)
    h = lag + 1
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    return DMResult(stat, float(2 * stats.t.sf(abs(stat), df=n - 1)), float(d.mean()), n)


# ---------------------------------------------------------------- baselines

def naive_forecast(df: pd.DataFrame) -> np.ndarray:
    """Random walk on volatility: next month's vol = last 22 days' realised vol."""
    return df["rv_m"].to_numpy(dtype=float)


def ewma_forecast(df: pd.DataFrame) -> np.ndarray:
    """RiskMetrics EWMA (lambda 0.94); the forecast is flat over the horizon."""
    return df["ewma"].to_numpy(dtype=float)


def garch_forecasts(r: pd.Series, origins: pd.DatetimeIndex, train_end: pd.Timestamp,
                    horizon: int = HORIZON, max_train: int = 2000) -> pd.Series:
    """GARCH(1,1) with constant mean, normal errors. Parameters are estimated on returns up to
    train_end only; the conditional variance is then filtered through later returns (data up to each
    origin, never beyond) to forecast the sum of the next `horizon` daily variances."""
    from arch import arch_model  # optional dependency ([ml] extra)

    pct = 100 * r.sort_index()
    train = pct[pct.index <= train_end].iloc[-max_train:]
    if len(train) < MIN_HISTORY:
        return pd.Series(np.nan, index=origins)
    res = arch_model(train, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(
        disp="off", show_warning=False)
    window = pct[pct.index >= train.index[0]]
    fixed = arch_model(window, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fix(res.params)
    var = fixed.forecast(horizon=horizon, start=window.index[0], reindex=True).variance
    total = var.sum(axis=1).reindex(origins)
    return np.sqrt(ANNUAL / horizon * total) / 100
