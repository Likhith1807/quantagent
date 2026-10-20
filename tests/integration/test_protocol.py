"""Talks to the server through a real in-process MCP client session."""
import json

import pytest
from mcp import Client

from quantagent.server import TOOLS, build_server, main
from tests.fixtures import make_fixture as fx

pytestmark = pytest.mark.anyio

CALLS = {
    "search_funds": {"query": "fund", "limit": 3},
    "get_nav_history": {"scheme_code": fx.DRAWDOWN, "frequency": "monthly"},
    "compare_funds": {"scheme_codes": [fx.DRAWDOWN, fx.STALE], "period": "1y"},
    "rank_funds": {"metric": "sharpe", "period": "1y", "top_n": 3},
    "get_risk_metrics": {"scheme_code": fx.DRAWDOWN, "period": "1y"},
    "get_benchmark_comparison": {"scheme_code": fx.DRAWDOWN, "period": "1y"},
    "check_data_quality": {"scheme_codes": [fx.STALE]},
    "get_data_provenance": {},
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_lists_everything_with_schemas():
    async with Client(build_server()) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
        assert set(tools) == {t.__name__ for t in TOOLS} == set(CALLS) | {"predict_volatility"}
        for t in tools.values():
            assert t.input_schema["type"] == "object" and t.output_schema and t.description
            assert "Do NOT use" in t.description
            assert t.annotations.read_only_hint is True
        resources = {str(r.uri) for r in (await c.list_resources()).resources}
        assert resources == {"quantagent://methodology", "quantagent://coverage"}
        prompts = (await c.list_prompts()).prompts
        assert [p.name for p in prompts] == ["fund_research"]
        assert "data_nature" in (c.instructions or "")


async def test_every_string_input_is_constrained_in_the_wire_schema():
    def walk(node, path):
        if isinstance(node, dict):
            if node.get("type") == "string" and node.get("format") != "date":
                assert "enum" in node or "pattern" in node or "const" in node, path
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    async with Client(build_server()) as c:
        for t in (await c.list_tools()).tools:
            walk(t.input_schema, t.name)


@pytest.mark.parametrize("name", list(CALLS))
async def test_each_tool_returns_structured_output(name):
    async with Client(build_server()) as c:
        res = await c.call_tool(name, CALLS[name])
        assert res.is_error is False, res.content
        sc = res.structured_content
        assert sc["meta"]["data_nature"] == "synthetic" and sc["meta"]["data_version"] == 1
        assert isinstance(sc["warnings"], list)


@pytest.mark.parametrize("name,args,code", [
    ("search_funds", {"query": "x'; DROP TABLE schemes;--"}, "INVALID_INPUT"),
    ("search_funds", {}, "INVALID_INPUT"),
    ("get_risk_metrics", {"scheme_code": "999999"}, "NOT_FOUND"),
    ("get_risk_metrics", {"scheme_code": fx.SHORT, "period": "1y"}, "INSUFFICIENT_HISTORY"),
    ("compare_funds", {"scheme_codes": [fx.CLOSED, fx.SHORT], "period": "max"}, "NO_OVERLAP"),
    ("compare_funds", {"scheme_codes": [fx.DRAWDOWN]}, "INVALID_INPUT"),
    ("rank_funds", {"metric": "cagr", "extra": 1}, "INVALID_INPUT"),
    ("predict_volatility", {"scheme_code": fx.DRAWDOWN}, "MODEL_UNAVAILABLE"),  # fixture has no trained model
])
async def test_failures_arrive_as_is_error_with_code(name, args, code):
    async with Client(build_server()) as c:
        res = await c.call_tool(name, args)
        assert res.is_error is True
        body = json.loads(res.content[0].text)
        assert body["code"] == code and "message" in body


async def test_unknown_tool_is_error_and_server_survives():
    async with Client(build_server()) as c:
        res = await c.call_tool("drop_tables", {})
        assert res.is_error is True
        ok = await c.call_tool("get_data_provenance", {})
        assert ok.is_error is False


async def test_resources_and_prompt():
    async with Client(build_server()) as c:
        m = await c.read_resource("quantagent://methodology")
        assert "Sharpe" in m.contents[0].text and "INSUFFICIENT_HISTORY" in m.contents[0].text
        cov = json.loads((await c.read_resource("quantagent://coverage")).contents[0].text)
        assert cov["data_version"] == 1 and fx.DRAWDOWN in cov["schemes"]
        p = await c.get_prompt("fund_research", {"question": "Best large cap fund?"})
        text = p.messages[0].content.text
        assert "Best large cap fund?" in text and "check_data_quality" in text


def test_healthcheck_cli(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--healthcheck"])
    assert e.value.code == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
