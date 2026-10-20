"""Ranking by one metric or by a weighted sum of direction-adjusted z-scores."""
import pandas as pd

# True = higher is better.
NATURAL_DIRECTION = {
    "total_return": True,
    "cagr": True,
    "volatility": False,
    "sharpe": True,
    "sortino": True,
    "max_drawdown": True,  # drawdowns are <= 0, so closer to 0 is better
    "alpha": True,
    "expense_ratio": False,
}
RANK_METRICS = tuple(NATURAL_DIRECTION)


def rank_single(df: pd.DataFrame, metric: str, ascending: bool | None = None) -> pd.DataFrame:
    asc = (not NATURAL_DIRECTION[metric]) if ascending is None else ascending
    out = df.sort_values([metric, "scheme_code"], ascending=[asc, True], kind="stable").copy()
    out["score"] = out[metric]
    return out


def rank_weighted(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """score = sum_i w_i * dir_i * z_i / sum_i w_i, z computed across the eligible universe.
    A metric with zero spread contributes 0."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to more than 0")
    score = pd.Series(0.0, index=df.index)
    for m, w in weights.items():
        if w == 0:
            continue
        col = df[m].astype(float)
        sd = col.std(ddof=0)
        z = (col - col.mean()) / sd if sd > 0 else col * 0.0
        score += w * (1 if NATURAL_DIRECTION[m] else -1) * z
    out = df.assign(score=score / total)
    return out.sort_values(["score", "scheme_code"], ascending=[False, True], kind="stable")
