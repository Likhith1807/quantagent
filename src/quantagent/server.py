"""QuantAgent MCP server: 8 read-only tools, 2 resources, 1 prompt.

Run: quantagent-server                         (stdio, for Claude Desktop / MCP Inspector)
     quantagent-server --transport streamable-http --port 8000
     quantagent-server --healthcheck           (exit 0 if get_data_provenance works)
"""
import argparse
import json
import logging
import sys
from importlib.resources import files
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import ValidationError

from .config import get_settings
from .errors import QuantError
from .logging_setup import configure
from .tools._base import compact_errors, get_services
from .tools.benchmark import get_benchmark_comparison
from .tools.compare import compare_funds
from .tools.nav import get_nav_history
from .tools.provenance import get_data_provenance
from .tools.quality import check_data_quality
from .tools.rank import rank_funds
from .tools.risk import get_risk_metrics
from .tools.search import search_funds
from .tools.volatility import predict_volatility

TOOLS = [search_funds, get_nav_history, compare_funds, rank_funds, get_risk_metrics, get_benchmark_comparison,
         check_data_quality, get_data_provenance, predict_volatility]

INSTRUCTIONS = """Read-only analytics over Indian mutual-fund NAVs (AMFI data via mfapi.in), Nifty TRI
benchmarks and 91-day T-bill yields, as loaded by the operator. meta.data_nature says whether the
database is real or the synthetic test fixture; meta.data_source and data_retrieved_at say where from.
Workflow: search_funds -> check_data_quality -> get_risk_metrics / compare_funds / rank_funds /
get_benchmark_comparison -> verify warnings and meta -> report with period, n_obs and assumptions.
Returns, volatility and drawdowns are fractions (0.12 = 12%). Expense ratios are percentages."""

log = logging.getLogger("quantagent.server")


class QuantMCP(MCPServer):
    """Reports schema-level argument failures with the same INVALID_INPUT payload the tools use."""

    async def call_tool(self, name: str, arguments: dict[str, Any], context=None):
        schema = self._tool_input_schema(name)
        unknown = sorted(set(arguments) - set((schema or {}).get("properties", {}))) if schema else []
        if unknown:  # the SDK ignores extra keys; a misspelt argument should not be silently dropped
            raise QuantError("INVALID_INPUT", f"Unknown arguments for {name}.",
                             {"unknown": unknown, "allowed": sorted(schema.get("properties", {}))})
        try:
            return await super().call_tool(name, arguments, context)
        except ToolError as e:
            if isinstance(e.__cause__, QuantError):  # drop the SDK's "Error executing tool" prefix
                raise e.__cause__ from None
            if isinstance(e.__cause__, ValidationError):
                log.info("invalid arguments", extra={"fields": {"tool": name, "error_code": "INVALID_INPUT",
                                                                "fields": [x["field"] for x in
                                                                           compact_errors(e.__cause__)]}})
                raise QuantError("INVALID_INPUT", f"Invalid arguments for {name}.",
                                 {"errors": compact_errors(e.__cause__)}) from None
            raise


def methodology_text() -> str:
    return files("quantagent").joinpath("resources/METHODOLOGY.md").read_text(encoding="utf-8")


def coverage_json() -> str:
    repo = get_services().repo
    return json.dumps({
        "data_nature": repo.meta.get("data_nature"), "data_version": repo.meta["data_version"],
        "data_as_of": repo.meta["data_as_of"], "table_counts": repo.table_counts(),
        "schemes": repo.nav_coverage(), "benchmarks": repo.benchmark_names(),
    }, default=str, indent=1)


def fund_research(question: str) -> str:
    return f"""You are a careful fund analyst using the QuantAgent tools. Question: {question}

Follow these steps:
1. get_data_provenance: note data_nature, data_as_of, sources and known biases. If the data is
   synthetic, say so prominently; do not present results as facts about real funds.
2. search_funds to resolve every fund the question mentions to a scheme_code. If a name is
   ambiguous, list the candidates instead of guessing.
3. check_data_quality for those schemes; drop or flag any with status 'fail'.
4. Compute: get_risk_metrics (one fund), compare_funds (named funds), rank_funds (screening),
   get_benchmark_comparison (versus its index). Prefer Direct-Growth plans; never compare IDCW
   NAV returns.
5. Verify: read every `warnings`, `excluded` and `fair_comparison` entry; check meta.period and
   n_obs match what you claim; re-run with another period if the result depends on the window.
6. Report: numbers as percentages with their period and as-of date, the assumptions (risk-free
   rate source, PRI benchmarks), limitations, and no investment advice. Label every number as
   calculated (from a tool output), assumed (an input you or the server assumed) or missing."""


def build_server() -> QuantMCP:
    # Our own logs are JSON lines; keep the SDK's plain-text logger to warnings unless debugging.
    sdk_level = "DEBUG" if get_settings().log_level.upper() == "DEBUG" else "WARNING"
    mcp = QuantMCP(name="quantagent", version="1.0.0", instructions=INSTRUCTIONS, log_level=sdk_level)
    read_only = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                                open_world_hint=False)
    for fn in TOOLS:
        mcp.add_tool(fn, annotations=read_only)
    mcp.resource("quantagent://methodology", name="methodology", mime_type="text/markdown",
                 description="Formulas, windows, minimum-history rules and known biases.")(methodology_text)
    mcp.resource("quantagent://coverage", name="coverage", mime_type="application/json",
                 description="Data version, table counts, per-scheme NAV coverage and benchmarks.")(coverage_json)
    mcp.prompt(name="fund_research",
               description="Step-by-step research workflow: provenance, search, quality, metrics, verify, report.",
               )(fund_research)
    return mcp


def healthcheck() -> int:
    try:
        r = get_data_provenance()
        print(json.dumps({"ok": True, "data_version": r.data_version, "data_as_of": r.data_as_of}))
        return 0
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(e)[:500]}), file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="quantagent-server", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--healthcheck", action="store_true")
    a = ap.parse_args(argv)
    configure(get_settings().log_level)
    if a.healthcheck:
        sys.exit(healthcheck())
    get_services()  # fail fast if the database is missing
    server = build_server()
    if a.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
