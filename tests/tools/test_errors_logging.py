import io
import json
import logging

from quantagent.logging_setup import JsonFormatter, configure, sanitise
from quantagent.tools.search import search_funds
from tests.tools.helpers import err


def test_unexpected_exception_becomes_internal_without_leaking(repo, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret connection string sqlite:///C:/x.db")

    monkeypatch.setattr(repo, "search_schemes", boom)
    e = err(search_funds, query="fund")
    assert e.code == "INTERNAL" and len(e.details["request_id"]) == 12
    assert "secret" not in str(e)


def test_json_log_line_per_call():
    stream = io.StringIO()
    h = logging.StreamHandler(stream)
    h.setFormatter(JsonFormatter())
    log = logging.getLogger("quantagent")
    log.addHandler(h)
    old = log.level
    log.setLevel("INFO")
    try:
        search_funds(query="const")
        err(search_funds)
    finally:
        log.removeHandler(h)
        log.setLevel(old)
    lines = [json.loads(x) for x in stream.getvalue().splitlines()]
    ok, bad = [x for x in lines if x["msg"] == "tool call"]
    assert ok["tool"] == "search_funds" and ok["outcome"] == "ok" and ok["args"] == {"query": "const"}
    assert ok["duration_ms"] >= 0 and ok["request_id"]
    assert bad["outcome"] == "error" and bad["error_code"] == "INVALID_INPUT"


def test_exception_traceback_in_log_only():
    rec = logging.LogRecord("quantagent", logging.ERROR, __file__, 1, "x", None, None)
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        rec.exc_info = sys.exc_info()
    assert "boom" in json.loads(JsonFormatter().format(rec))["traceback"]


def test_configure_is_idempotent_and_sanitise():
    log = configure("WARNING")
    n = len(log.handlers)
    configure("INFO")
    assert len(log.handlers) == n and log.level == logging.INFO and not log.propagate
    assert sanitise("x" * 500).endswith("…") and len(sanitise(list(range(100)))) == 20
    assert sanitise({"k": "v"}) == {"k": "v"} and sanitise(3) == 3
