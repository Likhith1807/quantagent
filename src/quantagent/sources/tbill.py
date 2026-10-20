"""91-day Treasury bill yields (the risk-free proxy) from a CSV downloaded by hand.

Sources that publish the series: RBI DBIE (Financial Market -> Government Securities Market ->
"91-Day Treasury Bill Primary Yield", weekly auction cut-off) and CCIL's T-bill curve. Both sites
need a browser session. Any CSV with a date column and a yield column (percent per annum) works.
"""
import re
from pathlib import Path

import pandas as pd


def _col(df: pd.DataFrame, *needles: str) -> str | None:
    for c in df.columns:
        lc = re.sub(r"[^a-z0-9]", "", str(c).lower())
        if any(n in lc for n in needles):
            return c
    return None


def parse_tbill_csv(path: Path, source: str) -> pd.DataFrame:
    """-> DataFrame[date, annual_rate (fraction), source]. Yields are given in percent."""
    df = pd.read_csv(path)
    dcol = _col(df, "date", "auction")
    vcol = _col(df, "91day", "91d", "yield", "cutoff", "rate")
    if dcol is None or vcol is None or dcol == vcol:
        raise ValueError(f"{path.name}: needs a date column and a yield column; got {list(df.columns)}")
    pct = pd.to_numeric(df[vcol].astype(str).str.replace("%", "").str.strip(), errors="coerce")
    if pct.dropna().between(0, 0.5).all():
        raise ValueError(f"{path.name}: yields look like fractions; give percent per annum (e.g. 6.45)")
    dates = pd.to_datetime(df[dcol].astype(str).str.strip(), format="mixed", dayfirst=True, errors="coerce")
    out = pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d"), "annual_rate": pct / 100, "source": source})
    return out.drop_duplicates("date", keep="last").sort_values("date")
