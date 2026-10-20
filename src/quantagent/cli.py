"""quantagent: one entry point for the whole pipeline.

  quantagent setup [--ter-months N]     fetch AMFI data, load hand-downloaded files if present, ingest
  quantagent fetch | load-manual | ingest | build-cache | train | eval | profile   (see each --help)
  quantagent tool NAME '{"arg": ...}'   call one MCP tool through an in-process MCP session, print JSON
  quantagent ask "question"             agent answer with verified claims (needs ANTHROPIC_API_KEY)
  quantagent serve [--transport ...]    run the MCP server (same as quantagent-server)
"""
import asyncio
import importlib
import json
import sys

COMMANDS = {"fetch": "fetch_amfi", "load-manual": "load_manual_sources", "ingest": "ingest",
            "build-cache": "build_metrics_cache", "train": "train_vol", "eval": "run_eval", "profile": "profile"}


def _command(name: str):
    return importlib.import_module(f"quantagent.commands.{COMMANDS[name]}").main


async def _tool(name: str, args: dict) -> int:
    from mcp import Client

    from .agent.runner import _result_payload
    from .server import build_server

    async with Client(build_server()) as session:
        is_err, out, err = _result_payload(await session.call_tool(name, args))
    print(err if is_err else json.dumps(out, indent=1, default=str))
    return 1 if is_err else 0


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    cmd, rest = argv[0], argv[1:]
    if cmd in COMMANDS:
        _command(cmd)(rest)
    elif cmd == "setup":
        _command("fetch")(rest)
        _command("load-manual")([])
        _command("ingest")([])
    elif cmd == "tool":
        if not rest:
            sys.exit("usage: quantagent tool NAME '{json args}'")
        sys.exit(asyncio.run(_tool(rest[0], json.loads(rest[1]) if len(rest) > 1 else {})))
    elif cmd == "ask":
        from .agent.cli import main as ask

        ask(rest)
    elif cmd == "serve":
        from .server import main as serve

        serve(rest)
    else:
        sys.exit(f"unknown command {cmd!r}; run quantagent --help")


if __name__ == "__main__":
    main()
