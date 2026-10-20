"""Read-only data access. SQLAlchemy Core select() with bound parameters only."""
import json
import sqlite3
from datetime import date
from functools import cached_property

import pandas as pd
from sqlalchemy import Engine, and_, func, or_, select, true
from sqlalchemy.exc import OperationalError

from ..errors import QuantError
from . import models as m


def escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _contains(col, value: str):
    return func.lower(col).like(f"%{escape_like(value.lower())}%", escape="\\")


class Repository:
    def __init__(self, engine: Engine):
        self.engine = engine

    def _rows(self, stmt) -> list[dict]:
        try:
            with self.engine.connect() as c:
                return [dict(r) for r in c.execute(stmt).mappings()]
        except OperationalError as e:
            if isinstance(e.orig, sqlite3.OperationalError) and "interrupted" in str(e.orig):
                raise QuantError("INTERNAL", "Query exceeded the configured timeout.", {"reason": "timeout"}) from e
            raise

    # ------------------------------------------------------------ meta
    @cached_property
    def meta(self) -> dict:
        out = {r["key"]: r["value"] for r in self._rows(select(m.meta))}
        for k in ("assumptions", "known_biases", "dataset_findings"):
            out[k] = json.loads(out.get(k, "[]"))
        out["data_version"] = int(out.get("data_version", 0))
        return out

    # ------------------------------------------------------------ schemes
    def search_schemes(self, query=None, category=None, fund_house=None, plan_type=None, option_type=None,
                       status="active", limit=20, offset=0) -> tuple[list[dict], int]:
        s = m.schemes.c
        conds = []
        if query:
            conds.append(or_(_contains(s.scheme_name, query), _contains(s.scheme_code, query)))
        if category:
            conds.append(or_(_contains(s.category, category), _contains(s.sub_category, category)))
        if fund_house:
            conds.append(_contains(s.fund_house, fund_house))
        if plan_type:
            conds.append(s.plan_type == plan_type)
        if option_type:
            conds.append(s.option_type == option_type)
        if status != "any":
            conds.append(s.status == status)
        where = and_(*conds) if conds else true()
        total = self._rows(select(func.count().label("n")).select_from(m.schemes).where(where))[0]["n"]
        rows = self._rows(select(m.schemes).where(where).order_by(s.scheme_name, s.scheme_code)
                          .limit(limit).offset(offset))
        return rows, total

    def get_scheme(self, code: str) -> dict | None:
        rows = self._rows(select(m.schemes).where(m.schemes.c.scheme_code == code))
        return rows[0] if rows else None

    def get_schemes(self, codes: list[str]) -> dict[str, dict]:
        return {r["scheme_code"]: r for r in self._rows(select(m.schemes).where(m.schemes.c.scheme_code.in_(codes)))}

    def all_schemes(self) -> list[dict]:
        return self._rows(select(m.schemes).order_by(m.schemes.c.scheme_code))

    def latest_ter(self, codes: list[str]) -> dict[str, dict]:
        e = m.expense_ratios.c
        latest = (select(e.scheme_code, func.max(e.as_of_date).label("d"))
                  .where(e.scheme_code.in_(codes)).group_by(e.scheme_code).subquery())
        stmt = select(m.expense_ratios).join(
            latest, and_(e.scheme_code == latest.c.scheme_code, e.as_of_date == latest.c.d))
        return {r["scheme_code"]: r for r in self._rows(stmt)}

    def nav_coverage(self, codes: list[str] | None = None) -> dict[str, dict]:
        n = m.nav_history.c
        stmt = select(n.scheme_code, func.min(n.nav_date).label("start"), func.max(n.nav_date).label("end"),
                      func.count().label("n_points")).group_by(n.scheme_code)
        if codes is not None:
            stmt = stmt.where(n.scheme_code.in_(codes))
        return {r.pop("scheme_code"): r for r in self._rows(stmt)}

    # ------------------------------------------------------------ series
    def nav_series(self, code: str, start: date | None = None, end: date | None = None) -> pd.Series:
        n = m.nav_history.c
        stmt = select(n.nav_date, n.nav).where(n.scheme_code == code)
        if start:
            stmt = stmt.where(n.nav_date >= start)
        if end:
            stmt = stmt.where(n.nav_date <= end)
        rows = self._rows(stmt.order_by(n.nav_date))
        return pd.Series([r["nav"] for r in rows], index=pd.DatetimeIndex([r["nav_date"] for r in rows]),
                         name=code, dtype=float)

    def benchmark_names(self) -> list[dict]:
        b = m.benchmarks.c
        return self._rows(select(b.index_name, b.index_type, func.min(b.date).label("start"),
                                 func.max(b.date).label("end"), func.count().label("n_points"))
                          .group_by(b.index_name, b.index_type).order_by(b.index_name))

    def benchmark_series(self, name: str, start: date | None = None, end: date | None = None) -> pd.Series:
        b = m.benchmarks.c
        stmt = select(b.date, b.value).where(b.index_name == name)
        if start:
            stmt = stmt.where(b.date >= start)
        if end:
            stmt = stmt.where(b.date <= end)
        rows = self._rows(stmt.order_by(b.date))
        return pd.Series([r["value"] for r in rows], index=pd.DatetimeIndex([r["date"] for r in rows]),
                         name=name, dtype=float)

    def risk_free_series(self) -> tuple[pd.Series | None, str | None]:
        rows = self._rows(select(m.risk_free_rates).order_by(m.risk_free_rates.c.date))
        if not rows:
            return None, None
        s = pd.Series([r["annual_rate"] for r in rows], index=pd.DatetimeIndex([r["date"] for r in rows]))
        return s, "risk_free_rates table: " + ", ".join(sorted({r["source"] for r in rows}))

    @cached_property
    def trading_calendar(self) -> pd.DatetimeIndex:
        t = m.trading_calendar.c
        rows = self._rows(select(t.date).where(t.is_trading_day == 1).order_by(t.date))
        return pd.DatetimeIndex([r["date"] for r in rows])

    # ------------------------------------------------------------ cache, provenance
    def metrics_cache(self, window: str, version: int) -> pd.DataFrame:
        c = m.metrics_cache.c
        rows = self._rows(select(m.metrics_cache).where(c.window == window, c.data_version == version))
        return pd.DataFrame(rows)

    def cache_versions(self) -> list[int]:
        c = m.metrics_cache.c
        return [r["data_version"] for r in self._rows(select(c.data_version).distinct())]

    def data_sources(self) -> list[dict]:
        return self._rows(select(m.data_sources).order_by(m.data_sources.c.source))

    def last_ingest(self) -> list[dict]:
        log = m.ingestion_log.c
        last = select(func.max(log.data_version)).scalar_subquery()
        return self._rows(select(m.ingestion_log).where(log.data_version == last).order_by(log.id))

    def table_counts(self) -> dict[str, int]:
        tables = [m.schemes, m.expense_ratios, m.nav_history, m.benchmarks, m.risk_free_rates,
                  m.trading_calendar, m.metrics_cache]
        return {t.name: self._rows(select(func.count().label("n")).select_from(t))[0]["n"] for t in tables}
