"""Run the evaluation question set.

  quantagent run_eval.py                 # direct mode on the configured database
  quantagent run_eval.py --agent         # also agent mode (needs ANTHROPIC_API_KEY)
  quantagent run_eval.py --fixture       # build the synthetic fixture first (what CI runs)

Writes reports/eval_report.md and reports/eval_results.json (or --out-dir). Exits non-zero if
direct-mode accuracy is below --min-accuracy.
"""
import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from mcp import Client

from ..config import Settings, get_settings
from ..data.engine import make_readonly_engine, sqlite_path
from ..data.ingest import ingest
from ..data.repository import Repository
from ..evaluation.expected import Reference
from ..evaluation.questions import build
from ..evaluation.run import render, run_agent, run_direct, summarise, to_json
from ..ml import registry
from ..server import build_server
from ..tools import _base


def _fixture_settings(tmp: Path) -> Settings:
    sys.path.insert(0, str(Path.cwd()))
    from tests.fixtures import make_fixture

    make_fixture.build(tmp / "raw", tmp / "ref")
    db = tmp / "fixture.db"
    s = Settings(db_url=f"sqlite:///{db.as_posix()}", risk_free_annual=0.05, model_dir=str(tmp / "no-model"),
                 _env_file=None)
    ingest(tmp / "raw", db, tmp / "ref", tmp / "rejected.csv", s, data_nature="synthetic")
    return s


async def _main(a) -> int:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        s = _fixture_settings(Path(tmp)) if a.fixture else get_settings()
        _base.set_services(_base.Services(s, Repository(make_readonly_engine(s.db_url, s.timeout_s))))
        model_dir = Path(s.model_dir)
        ref = Reference(sqlite_path(s.db_url), s.risk_free_annual)
        questions = build(ref, registry.current_version(model_dir) is not None)
        results = {}
        async with Client(build_server()) as session:
            results["direct"] = await run_direct(session, questions)
            if a.agent:
                from ..agent.llm import AnthropicLLM

                results["agent"] = await run_agent(session, questions, AnthropicLLM(a.model),
                                                   log_dir=a.out_dir / "agent_runs")
        meta = {"data_nature": ref.meta["data_nature"], "data_version": ref.meta["data_version"],
                "data_as_of": ref.meta["data_as_of"], "n_questions": len(questions)}
        fixes = a.fixes.read_text(encoding="utf-8") if a.fixes and a.fixes.exists() else ""
        a.out_dir.mkdir(parents=True, exist_ok=True)
        (a.out_dir / "eval_report.md").write_text(render(meta, results, fixes), encoding="utf-8")
        (a.out_dir / "eval_results.json").write_text(to_json(meta, results), encoding="utf-8")
        summary = {m: summarise(o) for m, o in results.items()}
        print(json.dumps(summary, indent=2))
        _base.set_services(None)
        return 0 if summary["direct"]["answer_accuracy"] >= a.min_accuracy else 1


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path(get_settings().reports_dir))
    ap.add_argument("--fixes", type=Path, default=Path("docs") / "eval_fixes.md")
    ap.add_argument("--min-accuracy", type=float, default=1.0)
    sys.exit(asyncio.run(_main(ap.parse_args(argv))))


if __name__ == "__main__":
    main()
