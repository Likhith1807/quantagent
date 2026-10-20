"""Nifty Total Return Index CSVs downloaded by hand from niftyindices.com.

niftyindices.com blocks scripted downloads, so the files are fetched in a browser:
Reports -> Historical Data -> "Total Returns Index" -> pick the index and range -> Download CSV.
Column names and date formats have varied between site versions; this parser accepts any CSV
with a date column and a total-return value column, and takes the index name from an
"Index Name" column or else from the file name.
"""
import re
from pathlib import Path

import pandas as pd

# niftyindices display name (upper-cased, spaces collapsed) -> series key used in the database
SERIES_KEYS = {
    "NIFTY 50": "NIFTY50_TRI",
    "NIFTY 100": "NIFTY100_TRI",
    "NIFTY 500": "NIFTY500_TRI",
    "NIFTY LARGEMIDCAP 250": "NIFTY_LARGEMIDCAP250_TRI",
    "NIFTY MIDCAP 150": "NIFTY_MIDCAP150_TRI",
    "NIFTY SMALLCAP 250": "NIFTY_SMALLCAP250_TRI",
    "NIFTY500 MULTICAP 50:25:25": "NIFTY500_MULTICAP_TRI",
    "NIFTY 500 MULTICAP 50:25:25": "NIFTY500_MULTICAP_TRI",
}


def _norm(name: str) -> str:
    n = re.sub(r"\s+", " ", name.replace("_", " ").replace("-", " ")).strip().upper()
    return re.sub(r"\s*(TRI|TOTAL RETURNS? INDEX|TOTAL RETURN)$", "", n).strip()


def series_key(name: str) -> str:
    n = _norm(name)
    if n in SERIES_KEYS:
        return SERIES_KEYS[n]
    raise ValueError(f"Unknown Nifty index {name!r}; add it to SERIES_KEYS and data/reference/benchmark_map.csv")


def _col(df: pd.DataFrame, *needles: str) -> str | None:
    for c in df.columns:
        lc = re.sub(r"[^a-z]", "", c.lower())
        if any(n in lc for n in needles):
            return c
    return None


def parse_tri_csv(path: Path) -> pd.DataFrame:
    """-> DataFrame[date (YYYY-MM-DD), index_name (series key), close_value, index_type='TRI']."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    dcol = _col(df, "date")
    vcol = _col(df, "totalreturn", "tri") or _col(df, "close")
    if dcol is None or vcol is None:
        raise ValueError(f"{path.name}: needs a date column and a total-return value column; got {list(df.columns)}")
    ncol = _col(df, "indexname")
    names = df[ncol].astype(str) if ncol else pd.Series(path.stem, index=df.index)
    values = pd.to_numeric(df[vcol].astype(str).str.replace(",", ""), errors="coerce")
    dates = pd.to_datetime(df[dcol].astype(str).str.strip(), format="mixed", dayfirst=True, errors="coerce")
    return pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d"), "index_name": names.map(series_key),
                         "close_value": values, "index_type": "TRI"})


def load_dir(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=["date", "index_name", "close_value", "index_type"])
    out = pd.concat([parse_tri_csv(f) for f in files], ignore_index=True)
    return out.drop_duplicates(["index_name", "date"], keep="last").sort_values(["index_name", "date"])
