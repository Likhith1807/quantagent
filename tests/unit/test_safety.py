"""Read-only connections, constrained tool inputs, and no raw SQL anywhere in a tool signature."""
import inspect
import time
import typing

import pytest
from pydantic import TypeAdapter
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from quantagent.data.engine import make_readonly_engine, sqlite_path
from quantagent.data.repository import Repository, escape_like
from quantagent.errors import QuantError
from quantagent.server import TOOLS


@pytest.mark.parametrize("sql", [
    "INSERT INTO schemes (scheme_code, scheme_name, base_name, fund_house, plan_type, option_type, status) "
    "VALUES ('1','x','x','x','Direct','Growth','active')",
    "DELETE FROM nav_history",
    "UPDATE nav_history SET nav = 1",
    "DROP TABLE schemes",
    "CREATE TABLE evil (x int)",
    "PRAGMA query_only=OFF",
])
def test_writes_fail(repo, sql):
    with pytest.raises(OperationalError), repo.engine.begin() as c:
        c.execute(text(sql))
        c.execute(text("INSERT INTO meta (key, value) VALUES ('x','y')"))  # mode=ro blocks writes even if query_only were switched off


def test_data_unchanged_after_write_attempts(repo, built):
    with repo.engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM nav_history")).scalar() == built["summary"]["nav_rows"]
        assert c.execute(text("SELECT count(*) FROM sqlite_master WHERE name = 'evil'")).scalar() == 0


def test_engine_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_readonly_engine(f"sqlite:///{(tmp_path / 'missing.db').as_posix()}")
    with pytest.raises(ValueError):
        make_readonly_engine("mysql://u@h/db")
    with pytest.raises(ValueError):
        sqlite_path("sqlite://")


def test_statement_timeout(built):
    eng = make_readonly_engine(built["settings"].db_url, timeout_s=0.05)
    slow = text("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c")
    t0 = time.monotonic()
    with pytest.raises(QuantError) as e:
        Repository(eng)._rows(slow)
    assert e.value.details["reason"] == "timeout" and time.monotonic() - t0 < 5


def test_like_wildcards_escaped(repo):
    assert escape_like("50%_a\\") == "50\\%\\_a\\\\"
    rows, total = repo.search_schemes(query="%")
    assert total == 0


FORBIDDEN_NAMES = {"sql", "query_sql", "where", "order_by", "column", "columns", "expression", "table", "raw"}


def _string_leaves(tp) -> list:
    """Every str-accepting leaf in an annotation, with its pydantic metadata."""
    origin = typing.get_origin(tp)
    if origin is typing.Annotated:
        base, *meta = typing.get_args(tp)
        if base is str:
            return [meta]
        return _string_leaves(base)
    if origin is typing.Literal:
        return []
    if tp is str:
        return [[]]
    return [leaf for a in typing.get_args(tp) for leaf in _string_leaves(a)]


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t.__name__)
def test_no_tool_accepts_free_text_sql(tool):
    sig = inspect.signature(tool)
    for name, p in sig.parameters.items():
        assert name not in FORBIDDEN_NAMES
        for meta in _string_leaves(p.annotation):
            patterns = [getattr(x, "pattern", None) for f in meta for x in getattr(f, "metadata", [])]
            assert any(patterns), f"{tool.__name__}.{name} accepts unconstrained text"
        schema = TypeAdapter(p.annotation).json_schema()
        assert "';" not in str(schema)


@pytest.mark.parametrize("bad", ["x'; DROP TABLE schemes;--", "a%", "a_b", "1=1 --", "\"q\"", "a;b", "x" * 101])
def test_sql_looking_text_rejected(bad):
    from quantagent.tools.search import search_funds
    with pytest.raises(QuantError) as e:
        search_funds(query=bad)
    assert e.value.code == "INVALID_INPUT"
