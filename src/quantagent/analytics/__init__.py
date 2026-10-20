"""Pure analytics on pandas Series indexed by Timestamp. No database access."""
import math

CALC_VERSION = "1.0.0"
ANNUALISATION = 252
MIN_RETURNS_RISK = 60  # volatility, Sharpe, Sortino
MIN_SHARED_OBS_REGRESSION = 120  # alpha, beta, R^2
MIN_WINDOW_COVERAGE = 0.90  # share of trading days with a NAV, for 1y/3y/5y windows
ZERO = 1e-9  # standard deviations below this are treated as zero (CSV rounding noise)


def clean_float(x) -> float | None:
    """NaN/inf -> None so results serialise to valid JSON."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None
