"""Agent loop over a real in-process MCP session with a scripted LLM: tool calls are logged,
claims are verified against tool outputs, a wrong number triggers one repair round."""
import json

import pytest
from mcp import Client

from quantagent.agent.llm import ScriptedLLM
from quantagent.agent.runner import SUBMIT, Agent, render_report
from quantagent.agent.verify import resolve, verify
from quantagent.server import build_server
from tests.fixtures import make_fixture as fx

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def use(i, name, args):
    return {"type": "tool_use", "id": f"t{i}", "name": name, "input": args}


def submit(i, answer, claims):
    return [use(i, SUBMIT, {"answer": answer, "claims": claims})]


def _vol_from(messages, call_id="c1"):
    for m in reversed(messages):
        if m["role"] == "user" and isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("content", "").startswith(f"call_id={call_id}\n"):
                    return json.loads(b["content"].split("\n", 1)[1])
    raise AssertionError("no result")


async def test_tool_calls_logged_and_answer_verified(tmp_path):
    def answer(messages):
        out = _vol_from(messages)
        v = round(out["volatility"] * 100, 2)
        return submit(2, f"Volatility over 1y was {v}% (period {out['meta']['period']['start']} onwards).",
                      [{"statement": "1y volatility", "value": v, "unit": "percent", "label": "calculated",
                        "call_id": "c1", "field": "volatility"},
                       {"statement": "information ratio", "value": None, "label": "missing", "call_id": "c1",
                        "field": "benchmark.missing_field"}])

    llm = ScriptedLLM([[use(1, "get_risk_metrics", {"scheme_code": fx.DRAWDOWN, "period": "1y"})], answer])
    async with Client(build_server()) as s:
        r = await Agent(llm, s, log_dir=tmp_path).ask("How volatile was the drawdown fund?")
    assert r.verification.ok and not r.repaired and r.error is None
    assert [c.tool for c in r.tool_calls] == ["get_risk_metrics"] and not r.tool_calls[0].is_error
    assert r.verification.counts == {"verified": 1, "confirmed_missing": 1}
    saved = json.loads((tmp_path / f"{r.run_id}.json").read_text())
    assert saved["tool_calls"][0]["args"] == {"scheme_code": fx.DRAWDOWN, "period": "1y"}
    assert "| c1 | get_risk_metrics |" in render_report(r)
    assert any(t["name"] == SUBMIT for t in llm.calls[0]["tools"])


async def test_wrong_number_triggers_one_repair():
    bad = submit(2, "Volatility was 99.9%.", [{"statement": "vol", "value": 99.9, "unit": "percent",
                                               "label": "calculated", "call_id": "c1", "field": "volatility"}])

    def fixed(messages):
        assert "Verification failed" in messages[-1]["content"]
        v = round(_vol_from(messages)["volatility"], 4)
        return submit(3, f"Volatility was {v}.", [{"statement": "vol", "value": v, "unit": "fraction",
                                                   "label": "calculated", "call_id": "c1", "field": "volatility"}])

    llm = ScriptedLLM([[use(1, "get_risk_metrics", {"scheme_code": fx.DRAWDOWN, "period": "1y"})], bad, fixed])
    async with Client(build_server()) as s:
        r = await Agent(llm, s).ask("vol?")
    assert r.repaired and r.verification.ok


async def test_tool_errors_are_passed_back_and_logged():
    llm = ScriptedLLM([[use(1, "get_risk_metrics", {"scheme_code": "999999"})],
                       submit(2, "No such scheme.", [{"statement": "scheme not found", "label": "missing",
                                                      "call_id": None, "field": None}])])
    async with Client(build_server()) as s:
        r = await Agent(llm, s).ask("vol of 999999?")
    assert r.tool_calls[0].is_error and "NOT_FOUND" in r.tool_calls[0].error
    assert r.verification.ok and r.verification.counts == {"unchecked": 1}


async def test_step_limit():
    llm = ScriptedLLM([[{"type": "text", "text": "thinking"}]] * 3)
    async with Client(build_server()) as s:
        r = await Agent(llm, s, max_steps=3).ask("?")
    assert r.answer is None and "3 steps" in r.error


def test_verify_paths_units_and_unlabelled_numbers():
    out = {"c1": {"ranked": [{"metrics": {"sharpe": 1.2345}}], "meta": {"assumptions": ["Risk-free rate 6.50%: x"]}}}
    assert resolve(out["c1"], "ranked[0].metrics.sharpe") == resolve(out["c1"], "ranked.0.metrics.sharpe")
    v = verify("Sharpe 1.23, risk-free 6.5%, and 42.7 elsewhere on 2025-01-31.",
               [{"statement": "s", "value": 1.23, "unit": "number", "label": "calculated", "call_id": "c1",
                 "field": "ranked[0].metrics.sharpe"},
                {"statement": "rf", "value": 6.5, "unit": "percent", "label": "assumed", "call_id": "c1"},
                {"statement": "bad path", "value": 1, "label": "calculated", "call_id": "c1", "field": "nope"},
                {"statement": "bad id", "value": 1, "label": "calculated", "call_id": "c9", "field": "x"}],
               out, {})
    assert [c.status for c in v.checks] == ["verified", "verified", "unresolved", "unresolved"]
    assert v.unlabelled_numbers == ["42.7"] and not v.ok
