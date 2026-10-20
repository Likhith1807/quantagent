"""Benchmark-relative statistics on returns aligned to shared dates."""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import ANNUALISATION, ZERO, clean_float
from .returns import monthly_returns


def align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    return j["a"], j["b"]


@dataclass
class Regression:
    beta: float | None
    alpha: float | None  # Jensen's alpha, annualised
    r_squared: float | None
    n_obs: int

    def as_dict(self) -> dict:
        return asdict(self)


def regression(rf_ex: pd.Series, rb_ex: pd.Series) -> Regression:
    """beta = cov(fund_ex, bench_ex) / var(bench_ex);
    alpha = (mean(fund_ex) - beta * mean(bench_ex)) * 252; R^2 = corr^2."""
    f, b = align(rf_ex, rb_ex)
    n = len(f)
    if n < 3 or b.var(ddof=1) < ZERO**2:
        return Regression(None, None, None, n)
    beta = f.cov(b) / b.var(ddof=1)
    alpha = (f.mean() - beta * b.mean()) * ANNUALISATION
    corr = f.corr(b) if f.std(ddof=1) > ZERO else np.nan
    return Regression(clean_float(beta), clean_float(alpha), clean_float(corr**2), n)


def tracking_error(r_f: pd.Series, r_b: pd.Series) -> float | None:
    """std(r_f - r_b) * sqrt(252)."""
    f, b = align(r_f, r_b)
    if len(f) < 2:
        return None
    return clean_float((f - b).std(ddof=1) * np.sqrt(ANNUALISATION))


def information_ratio(r_f: pd.Series, r_b: pd.Series) -> float | None:
    """Annualised active return / tracking error."""
    f, b = align(r_f, r_b)
    te = tracking_error(f, b)
    if te is None or te < ZERO:
        return None
    return clean_float((f - b).mean() * ANNUALISATION / te)


@dataclass
class Capture:
    up_capture: float | None
    down_capture: float | None
    up_months: int
    down_months: int

    def as_dict(self) -> dict:
        return asdict(self)


def capture_ratios(fund_nav: pd.Series, bench_nav: pd.Series, min_months: int = 3) -> Capture:
    """Monthly returns. Up capture = (prod(1 + r_f) - 1) / (prod(1 + r_b) - 1) over months the
    benchmark rose; down capture likewise over months it fell."""
    f, b = align(monthly_returns(fund_nav), monthly_returns(bench_nav))

    def ratio(mask: pd.Series) -> float | None:
        if mask.sum() < min_months:
            return None
        fb, bb = (1 + f[mask]).prod() - 1, (1 + b[mask]).prod() - 1
        return None if abs(bb) < ZERO else clean_float(fb / bb)

    up, down = b > 0, b < 0
    return Capture(ratio(up), ratio(down), int(up.sum()), int(down.sum()))
