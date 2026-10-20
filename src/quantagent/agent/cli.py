"""Ask the agent a question. Needs ANTHROPIC_API_KEY and a built database.

  quantagent-ask "Which large-cap fund had the best 3y Sharpe ratio?"
  quantagent-ask --stdio "..."      # run the server as a subprocess over stdio (as a desktop client would)

Prints the verified report (answer, labelled claims, tool-call log) and saves the full run as JSON.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

from ..config import get_settings
from ..server import build_server
from .llm import AnthropicLLM
from .runner import Agent, render_report


async def _run(question: str, stdio: bool, log_dir: Path, model: str | None):
    target = StdioServerParameters(command=sys.executable, args=["-m", "quantagent.server"], env=dict(os.environ)) \
        if stdio else build_server()
    async with Client(target) as session:
        return await Agent(AnthropicLLM(model), session, log_dir=log_dir).ask(question)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="quantagent-ask", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--stdio", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--log-dir", type=Path, default=None, help="default: <reports_dir>/agent_runs")
    ap.add_argument("--report", type=Path, default=None, help="also write the report to this file")
    a = ap.parse_args(argv)
    log_dir = a.log_dir or Path(get_settings().reports_dir) / "agent_runs"
    result = asyncio.run(_run(a.question, a.stdio, log_dir, a.model))
    text = render_report(result)
    print(text)
    if a.report:
        a.report.write_text(text, encoding="utf-8")
    sys.exit(0 if result.verification and result.verification.ok else 1)


if __name__ == "__main__":
    main()
