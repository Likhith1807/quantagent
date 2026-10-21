"""Run the question set through MCP, either calling the expected tool directly (the deterministic
baseline) or letting the agent choose, and score against the independent expected answers."""
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..agent.runner import Agent, _result_payload
from ..agent.verify import resolve
from .questions import Expect, Question


@dataclass
class Outcome:
    id: str
    category: str
    mode: str
    tool_expected: str
    tools_called: list[str]
    tool_correct: bool
    checks: list[dict]
    correct: bool
    failed: bool  # unexpected error or no answer
    latency_ms: float
    note: str = ""
    error: str | None = None
    extra: dict = field(default_factory=dict)


def _all(obj, path: str) -> list:
    head, _, tail = path.partition("[*]")
    items = resolve(obj, head) if head else obj
    return [resolve(i, tail.lstrip(".")) if tail else i for i in items]


def _close(expected, actual, tol: float) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or actual is None:
            return False
        return abs(actual - expected) <= tol * max(1.0, abs(expected))
    return str(actual) == str(expected)


def score_output(expects: list[Expect], out: dict | None, error: str | None) -> list[dict]:
    res = []
    for e in expects:
        if e.kind == "error":
            got = json.loads(error).get("code") if error and error.strip().startswith("{") else error
            res.append({"kind": "error", "expected": e.value, "actual": got, "ok": got == e.value})
            continue
        if out is None:
            res.append({"kind": e.kind, "path": e.path, "expected": e.value, "actual": error, "ok": False})
            continue
        try:
            actual = _all(out, e.path) if e.kind == "contains" else resolve(out, e.path)
        except (KeyError, IndexError, TypeError, ValueError):
            actual = "<missing>"
        ok = (e.value in actual if e.kind == "contains" and isinstance(actual, list)
              else actual in e.value if e.kind == "in" else _close(e.value, actual, e.tol))
        res.append({"kind": e.kind, "path": e.path, "expected": e.value, "actual": actual, "ok": bool(ok)})
    return res


async def run_direct(session, questions: list[Question]) -> list[Outcome]:
    outs = []
    for q in questions:
        t0 = time.perf_counter()
        res = await session.call_tool(q.tool, q.args)
        ms = (time.perf_counter() - t0) * 1000
        is_err, out, err = _result_payload(res)
        checks = score_output(q.expect, out, err)
        expected_error = any(e.kind == "error" for e in q.expect)
        outs.append(Outcome(q.id, q.category, "direct", q.tool, [q.tool], True, checks,
                            all(c["ok"] for c in checks), is_err and not expected_error, round(ms, 1),
                            error=err if is_err else None))
    return outs


def _agent_checks(q: Question, result) -> list[dict]:
    """An expected number counts as answered when a verified 'calculated' claim cites a tool
    output field whose value equals it; an expected error when the expected tool returned it."""
    checks = []
    verified = [c for c in (result.verification.checks if result.verification else []) if c.status == "verified"
                and c.label == "calculated"]
    errors = {c.tool: c.error for c in result.tool_calls if c.is_error}
    for e in q.expect:
        if e.kind == "error":
            err = errors.get(q.tool)
            got = json.loads(err).get("code") if err and err.startswith("{") else err
            checks.append({"kind": "error", "expected": e.value, "actual": got, "ok": got == e.value})
        elif isinstance(e.value, (int, float)) and not isinstance(e.value, bool):
            ok = any(_close(e.value, c.actual, max(e.tol, 5e-3)) for c in verified)
            checks.append({"kind": "equals", "path": e.path, "expected": e.value, "ok": ok})
        else:
            values = e.value if isinstance(e.value, list) else [e.value]
            ok = bool(result.answer) and any(str(v) in result.answer for v in values)
            if not ok and q.tool in {c.tool for c in result.tool_calls}:
                ok = any(str(v) in (result.answer or "") or _name_in(v, result) for v in values)
            checks.append({"kind": e.kind, "path": e.path, "expected": e.value, "ok": ok})
    return checks


def _name_in(code, result) -> bool:
    """Accept a fund's name in place of its code when a tool output links the two."""
    for c in result.tool_calls:
        text = json.dumps(c.output or {})
        if f'"scheme_code": "{code}"' in text:
            for item in _walk(c.output):
                if isinstance(item, dict) and item.get("scheme_code") == code and item.get("scheme_name"):
                    return item["scheme_name"] in (result.answer or "")
    return False


def _walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


async def run_agent(session, questions: list[Question], llm, log_dir: Path | None = None) -> list[Outcome]:
    outs = []
    for q in questions:
        r = await Agent(llm, session, log_dir=log_dir).ask(q.text)
        called = [c.tool for c in r.tool_calls]
        checks = _agent_checks(q, r)
        outs.append(Outcome(q.id, q.category, "agent", q.tool, called, q.tool in called, checks,
                            all(c["ok"] for c in checks) and r.answer is not None, r.answer is None,
                            r.latency_s * 1000, error=r.error,
                            extra={"run_id": r.run_id, "repaired": r.repaired, "usage": r.usage,
                                   "verification": r.verification.counts if r.verification else None}))
    return outs


def summarise(outcomes: list[Outcome], prices: tuple[float, float] | None = None) -> dict:
    """Accuracy, failure and latency figures; for agent runs also mean tokens per question and, when
    `prices` (USD per million input and output tokens) is given, mean cost per question."""
    if not outcomes:
        return {}
    lat = np.array([o.latency_ms for o in outcomes])
    numeric = [c for o in outcomes for c in o.checks if c["kind"] in ("equals", "in", "contains")]
    out = {"questions": len(outcomes), "answer_accuracy": float(np.mean([o.correct for o in outcomes])),
           "numerical_accuracy": float(np.mean([c["ok"] for c in numeric])) if numeric else None,
           "numeric_checks": len(numeric),
           "tool_selection_accuracy": float(np.mean([o.tool_correct for o in outcomes])),
           "failure_rate": float(np.mean([o.failed for o in outcomes])),
           "latency_p50_ms": float(np.percentile(lat, 50)), "latency_p95_ms": float(np.percentile(lat, 95))}
    usage = [o.extra.get("usage") for o in outcomes if o.extra and o.extra.get("usage")]
    if usage:
        tin = float(np.mean([u.get("input_tokens", 0) for u in usage]))
        tout = float(np.mean([u.get("output_tokens", 0) for u in usage]))
        out.update(input_tokens_per_question=tin, output_tokens_per_question=tout)
        if prices:
            out["cost_usd_per_question"] = (tin * prices[0] + tout * prices[1]) / 1e6
    return out


def render(meta: dict, results: dict[str, list[Outcome]], fixes: str = "",
           prices: tuple[float, float] | None = None) -> str:
    lines = ["# Evaluation report", "",
             f"Database: data_nature **{meta['data_nature']}**, version {meta['data_version']}, data as of "
             f"{meta['data_as_of']}. {meta['n_questions']} questions; expected answers computed independently "
             "from the SQLite file (`quantagent.evaluation.expected`). Generated by `scripts/run_eval.py`.", "",
             "| Mode | Answer accuracy | Numerical accuracy | Tool selection | Failure rate | p50 ms | p95 ms |",
             "|---|---|---|---|---|---|---|"]
    for mode, outs in results.items():
        s = summarise(outs)
        lines.append(f"| {mode} | {s['answer_accuracy']:.1%} | "
                     f"{s['numerical_accuracy']:.1%} ({s['numeric_checks']} checks) | "
                     f"{s['tool_selection_accuracy']:.1%} | {s['failure_rate']:.1%} | {s['latency_p50_ms']:.0f} | "
                     f"{s['latency_p95_ms']:.0f} |")
    for mode, outs in results.items():
        s = summarise(outs, prices)
        if "input_tokens_per_question" in s:
            cost = (f", cost USD {s['cost_usd_per_question']:.4f} per question at {prices[0]}/{prices[1]} USD "
                    "per million input/output tokens") if "cost_usd_per_question" in s else                 " (pass --price-in/--price-out for cost)"
            lines += ["", f"{mode}: {s['input_tokens_per_question']:.0f} input and "
                      f"{s['output_tokens_per_question']:.0f} output tokens per question{cost}."]
    missing = [m for m in ("direct", "agent") if m not in results]
    if "agent" in missing:
        lines += ["", "Agent mode not run for this report (needs ANTHROPIC_API_KEY): `python scripts/run_eval.py "
                  "--agent`."]
    lines += ["", "Direct mode calls the expected tool with fixed arguments: it measures the analytics and the "
              "server, so its tool selection is 100% by construction. Agent mode lets the LLM choose tools from "
              "the question text; a numeric check passes only when a verified claim cites a tool output equal to "
              "the expected value.", ""]
    for mode, outs in results.items():
        bad = [o for o in outs if not o.correct]
        lines += [f"## Failures ({mode}): {len(bad)}", ""]
        if not bad:
            lines += ["None.", ""]
            continue
        lines += ["| Question | Category | Tools called | Failed checks |", "|---|---|---|---|"]
        for o in bad:
            failed = "; ".join(f"{c.get('path') or c['kind']}: expected {c['expected']} got {c.get('actual')}"
                               for c in o.checks if not c["ok"])
            lines.append(f"| {o.id} | {o.category} | {', '.join(o.tools_called) or '-'} | "
                         f"{failed or o.error or o.note} |")
        lines.append("")
    lines += ["## Questions", "", "| id | category | tool | checks |", "|---|---|---|---|"]
    first = next(iter(results.values()))
    for o in first:
        lines.append(f"| {o.id} | {o.category} | {o.tool_expected} | {len(o.checks)} |")
    if fixes:
        lines += ["", "## Failures found during development and their fixes", "", fixes.strip()]
    return "\n".join(lines) + "\n"


def to_json(meta: dict, results: dict[str, list[Outcome]], prices: tuple[float, float] | None = None) -> str:
    return json.dumps({"meta": meta, "summary": {m: summarise(o, prices) for m, o in results.items()},
                       "outcomes": {m: [asdict(x) for x in o] for m, o in results.items()}}, indent=1, default=str)
