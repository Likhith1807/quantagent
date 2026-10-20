"""Independent reference calculations. Reads the SQLite file with the standard library and
re-implements each formula in a few lines of pandas, so a bug in quantagent.analytics cannot
make its own expected answer."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


class Reference:
    def __init__(self, db_path: Path, rf_default: float):
        con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
        try:
            self.meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
            self.schemes = pd.read_sql("SELECT * FROM schemes", con)
            nav = pd.read_sql("SELECT scheme_code, nav_date, nav FROM nav_history", con, parse_dates=["nav_date"])
            self.navs = nav.pivot(index="nav_date", columns="scheme_code", values="nav").sort_index()
            cal = pd.read_sql("SELECT date FROM trading_calendar WHERE is_trading_day = 1", con, parse_dates=["date"])
            self.calendar = pd.DatetimeIndex(cal["date"])
            rf = pd.read_sql("SELECT date, annual_rate FROM risk_free_rates", con, parse_dates=["date"])
            self.rf_series = rf.set_index("date")["annual_rate"].sort_index() if len(rf) else None
            b = pd.read_sql("SELECT index_name, date, value FROM benchmarks", con, parse_dates=["date"])
            self.bench = b.pivot(index="date", columns="index_name", values="value").sort_index()
            ter = pd.read_sql("SELECT scheme_code, as_of_date, ter FROM expense_ratios", con)
            self.ter = ter.sort_values("as_of_date").groupby("scheme_code").last()["ter"]
            self.data_sources = pd.read_sql("SELECT * FROM data_sources", con)
        finally:
            con.close()
        self.rf_default = rf_default
        self.as_of = pd.Timestamp(self.meta["data_as_of"])

    # ------------------------------------------------------------ selections
    def nav(self, code: str) -> pd.Series:
        return self.navs[code].dropna()

    def window(self, code: str, years: int) -> pd.Series:
        s = self.nav(code)
        return s[(s.index >= self.as_of - pd.DateOffset(years=years)) & (s.index <= self.as_of)]

    def coverage(self, code: str, years: int) -> float:
        start = self.as_of - pd.DateOffset(years=years)
        days = ((self.calendar >= start) & (self.calendar <= self.as_of)).sum()
        pre = len(pd.bdate_range(start, self.calendar.min() - pd.Timedelta(days=1))) if start < self.calendar.min() else 0
        return len(self.window(code, years)) / (days + pre)

    def growth_codes(self) -> list[str]:
        return sorted(self.schemes.loc[self.schemes["option_type"] != "IDCW", "scheme_code"])

    def longest(self, n: int = 1, among: list[str] | None = None) -> list[str]:
        codes = among or self.growth_codes()
        counts = self.navs[codes].count()
        return list(counts.sort_values(ascending=False, kind="stable").index[:n])

    # ------------------------------------------------------------ formulas
    def _rf_daily(self, idx: pd.DatetimeIndex) -> pd.Series:
        if self.rf_series is None:
            return pd.Series((1 + self.rf_default) ** (1 / 252) - 1, index=idx)
        r = self.rf_series.reindex(self.rf_series.index.union(idx)).ffill().bfill().reindex(idx)
        return (1 + r) ** (1 / 252) - 1

    def stats(self, s: pd.Series) -> dict:
        r = s.pct_change().dropna()
        ex = r - self._rf_daily(s.index).reindex(r.index)
        downside = np.minimum(ex, 0)
        days = (s.index[-1] - s.index[0]).days
        sd, dsd = ex.std(), np.sqrt((downside**2).mean())
        # METHODOLOGY.md: a ratio is undefined (None) when its denominator is below 1e-9
        return {"volatility": r.std() * np.sqrt(252),
                "sharpe": ex.mean() / sd * np.sqrt(252) if sd >= 1e-9 else None,
                "sortino": ex.mean() / dsd * np.sqrt(252) if dsd >= 1e-9 else None,
                "max_drawdown": (s / s.cummax() - 1).min(), "total_return": s.iloc[-1] / s.iloc[0] - 1,
                "cagr": (s.iloc[-1] / s.iloc[0]) ** (365.25 / days) - 1, "n_obs": len(s)}

    def window_stats(self, code: str, years: int) -> dict:
        return self.stats(self.window(code, years))

    def shared_stats(self, codes: list[str], years: int) -> dict[str, dict]:
        w = self.navs[codes]
        w = w[(w.index >= self.as_of - pd.DateOffset(years=years)) & (w.index <= self.as_of)].dropna()
        return {c: self.stats(w[c]) for c in codes}

    def rank(self, metric: str, years: int, sub_category: str | None, top: int) -> list[str]:
        pool = self.schemes[self.schemes["option_type"] != "IDCW"]
        if sub_category:
            pool = pool[pool["sub_category"].str.lower().str.contains(sub_category.lower(), regex=False)]
        vals = {}
        for c in pool["scheme_code"]:
            if self.coverage(c, years) >= 0.9 and len(self.window(c, years)) >= 61:
                v = self.window_stats(c, years)[metric]
                if v is not None:
                    vals[c] = v
        ascending = metric in ("volatility",)
        ordered = pd.Series(vals).sort_values(ascending=ascending, kind="stable")
        return list(ordered.index[:top])

    def tracking_error(self, code: str, index: str, years: int) -> float:
        j = pd.concat([self.nav(code), self.bench[index]], axis=1).dropna()
        j = j[(j.index >= self.as_of - pd.DateOffset(years=years)) & (j.index <= self.as_of)]
        r = j.pct_change().dropna()
        return float((r.iloc[:, 0] - r.iloc[:, 1]).std() * np.sqrt(252))

    def realised_vol_22(self, code: str) -> float:
        r = np.log(self.nav(code)).diff().dropna()
        return float(np.sqrt(252 * (r.iloc[-22:] ** 2).mean()))
