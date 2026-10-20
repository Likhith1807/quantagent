"""Agent-mode scoring with a scripted 'oracle' LLM that calls the expected tool and cites every
expected field, and a 'wrong tool' LLM that must score zero on tool selection."""
import json
from pathlib import Path

import pytest
from mcp import Client

from quantagent.agent import llm as llm_mod
from quantagent.agent.llm import ScriptedLLM
from quantagent.agent.runner import SUBMIT
from quantagent.agent.verify import resolve
from quantagent.evaluation.expected import Reference
from quantagent.evaluation.questions import build
from quantagent.evaluation.run import render, run_agent, run_direct, summarise
from quantagent.server import build_server
from quantagent.sources import http

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _last_output(messages):
    for b in messages[-1]["content"]:
        if isinstance(b, dict) and b.get("type") == "tool_result":
            body = b["content"].split("\n", 1)[1]
            return b.get("is_error"), body
    raise AssertionError


def oracle(questions):
    script = []
    for q in questions:
        script.append([{"type": "tool_use", "id": f"{q.id}-1", "name": q.tool, "input": q.args}])

        def answer(messages, q=q):
            is_err, body = _last_output(messages)
            if is_err:
                return [{"type": "tool_use", "id": f"{q.id}-2", "name": SUBMIT,
                         "input": {"answer": "The tool reported an error.", "claims": [
                             {"statement": "error", "label": "missing", "call_id": None, "field": None}]}}]
            out = json.loads(body)
            claims, words = [], []
            for e in q.expect:
                if e.path and "[*]" not in e.path:
                    v = resolve(out, e.path)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        claims.append({"statement": e.path, "value": v, "unit": "number", "label": "calculated",
                                       "call_id": "c1", "field": e.path})
                    else:
                        words.append(str(v))
                elif e.kind == "contains":
                    words.append(str(e.value))
            text = " ".join(words) or "See claims."
            return [{"type": "tool_use", "id": f"{q.id}-2", "name": SUBMIT, "input": {"answer": text,
                                                                                    "claims": claims}}]
        script.append(answer)
    return ScriptedLLM(script)


@pytest.fixture
def questions(built):
    return build(Reference(built["db"], 0.05), model_available=False)


async def test_oracle_agent_scores_like_direct(questions, tmp_path):
    async with Client(build_server()) as s:
        direct = await run_direct(s, questions)
        agent = await run_agent(s, questions, oracle(questions), log_dir=tmp_path)
    d, a = summarise(direct), summarise(agent)
    assert d["answer_accuracy"] == 1.0
    assert a["tool_selection_accuracy"] == 1.0 and a["answer_accuracy"] == 1.0, \
        [(o.id, o.checks) for o in agent if not o.correct]
    report = render({"data_nature": "synthetic", "data_version": 1, "data_as_of": "x", "n_questions": len(questions)},
                    {"direct": direct, "agent": agent}, "fix notes")
    assert "| agent |" in report and "fix notes" in report
    assert len(list(Path(tmp_path).glob("*.json"))) == len(questions)


async def test_wrong_tool_agent_is_penalised(questions):
    q = [x for x in questions if x.category == "risk"][:2]
    script = []
    for _x in q:
        script += [[{"type": "tool_use", "id": "a", "name": "get_data_provenance", "input": {}}],
                   [{"type": "tool_use", "id": "b", "name": SUBMIT, "input": {"answer": "no idea", "claims": []}}]]
    async with Client(build_server()) as s:
        out = await run_agent(s, q, ScriptedLLM(script))
    s_ = summarise(out)
    assert s_["tool_selection_accuracy"] == 0 and s_["answer_accuracy"] == 0
    assert "Failures (agent): 2" in render({"data_nature": "s", "data_version": 1, "data_as_of": "x",
                                            "n_questions": 2}, {"direct": out, "agent": out})


def test_anthropic_llm_requires_key(monkeypatch):
    from quantagent.config import Settings, get_settings

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("quantagent.config.get_settings", lambda: Settings(_env_file=None))
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm_mod.AnthropicLLM()
    s = Settings(_env_file=None, ANTHROPIC_API_KEY="sk-secret-value")
    assert "sk-secret" not in repr(s) and "sk-secret" not in str(s.model_dump()) and get_settings
    with pytest.raises(RuntimeError, match="ran out"):
        ScriptedLLM([]).create("", [], [])


def test_http_get_retries_then_fails(monkeypatch):
    calls = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"ok": 1}'

    def flaky(req, timeout):
        calls.append(req.full_url)
        if len(calls) < 2:
            raise TimeoutError("slow")
        return Resp()

    monkeypatch.setattr(http.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    assert http.get_json("https://example.invalid/x") == {"ok": 1} and len(calls) == 2
    monkeypatch.setattr(http.urllib.request, "urlopen", lambda req, timeout: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        http.get("https://example.invalid/y", retries=2)
