"""The agent loop: the LLM picks MCP tools, the runner executes them through an MCP client session,
logs each call, and finally verifies the submitted answer (with one repair round on failure)."""
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .llm import LLM
from .verify import Verification, verify

SUBMIT = "submit_answer"
SUBMIT_TOOL = {
    "name": SUBMIT,
    "description": "Submit the final answer. Call exactly once, after the data tools. Every number in "
                   "`answer` must appear in `claims` with its label and source.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "Final answer in Markdown for the user."},
            "claims": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "value": {"type": ["number", "null"]},
                    "unit": {"type": "string", "enum": ["fraction", "percent", "number"]},
                    "label": {"type": "string", "enum": ["calculated", "assumed", "missing"]},
                    "call_id": {"type": ["string", "null"], "description": "e.g. 'c2', from the tool result."},
                    "field": {"type": ["string", "null"],
                              "description": "Path in that result, e.g. 'ranked[0].metrics.sharpe'."},
                },
                "required": ["statement", "label"]}},
        },
        "required": ["answer", "claims"],
    },
}

SYSTEM = """You answer questions about Indian mutual funds using only the QuantAgent tools.
Rules:
- Resolve fund names with search_funds before using scheme codes; never invent codes.
- Every number you state must come from a tool result. Do not compute new numbers yourself.
- Tool results start with 'call_id=cN'. Cite that id and the JSON field path for each number.
- Label each claim: calculated (read from a tool output), assumed (an input such as the risk-free
  rate, shown in meta.assumptions), or missing (the tool returned null or could not compute it).
- Values in tool outputs are fractions (0.12 = 12%) unless the field says percent (expense ratios).
  If you write 12.3% for 0.1234, give value 12.3 with unit 'percent'.
- Mention the period, the data_as_of date, data_nature and important warnings. No investment advice.
- Finish by calling submit_answer exactly once."""


@dataclass
class ToolCall:
    call_id: str
    tool: str
    args: dict
    is_error: bool
    duration_ms: float
    output: dict | None
    error: str | None = None


@dataclass
class AgentResult:
    run_id: str
    question: str
    model: str
    answer: str | None
    claims: list[dict]
    tool_calls: list[ToolCall]
    verification: Verification | None
    repaired: bool
    steps: int
    latency_s: float
    usage: dict = field(default_factory=dict)
    error: str | None = None

    def to_json(self) -> dict:
        d = asdict(self)
        if self.verification is not None:
            d["verification"]["counts"] = self.verification.counts
            d["verification"]["ok"] = self.verification.ok
        return d


def _tool_defs(tools) -> list[dict]:
    return [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema} for t in tools]


def _result_payload(res) -> tuple[bool, dict | None, str | None]:
    text = " ".join(getattr(c, "text", "") for c in (res.content or []))
    if res.is_error:
        return True, None, text
    if res.structured_content is not None:
        sc = res.structured_content
        return False, sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc, None
    try:
        return False, json.loads(text), None
    except json.JSONDecodeError:
        return False, {"text": text}, None


class Agent:
    def __init__(self, llm: LLM, session, max_steps: int = 12, log_dir: Path | None = None):
        self.llm, self.session, self.max_steps, self.log_dir = llm, session, max_steps, log_dir

    async def ask(self, question: str) -> AgentResult:
        run_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        tools = _tool_defs((await self.session.list_tools()).tools) + [SUBMIT_TOOL]
        messages: list[dict] = [{"role": "user", "content": question}]
        calls: list[ToolCall] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        submitted, verification, repaired, steps, error = None, None, False, 0, None
        while steps < self.max_steps:
            steps += 1
            resp = self.llm.create(SYSTEM, messages, tools)
            for k in usage:
                usage[k] += resp.get("usage", {}).get(k, 0)
            messages.append({"role": "assistant", "content": resp["content"]})
            uses = [b for b in resp["content"] if b["type"] == "tool_use"]
            if not uses:
                messages.append({"role": "user", "content": "Call submit_answer with your final answer."})
                continue
            results = []
            for u in uses:
                if u["name"] == SUBMIT:
                    submitted = u["input"]
                    results.append({"type": "tool_result", "tool_use_id": u["id"], "content": "received"})
                    continue
                cid = f"c{len(calls) + 1}"
                s = time.perf_counter()
                try:
                    res = await self.session.call_tool(u["name"], u["input"])
                    is_err, out, err = _result_payload(res)
                except Exception as e:  # noqa: BLE001 - transport failure becomes a tool error for the LLM
                    is_err, out, err = True, None, f"{type(e).__name__}: {e}"
                calls.append(ToolCall(cid, u["name"], u["input"], is_err, round((time.perf_counter() - s) * 1000, 1),
                                      out, err))
                body = err if is_err else json.dumps(out, default=str)
                results.append({"type": "tool_result", "tool_use_id": u["id"], "is_error": is_err,
                                "content": f"call_id={cid}\n{body}"})
            messages.append({"role": "user", "content": results})
            if submitted is None:
                continue
            outputs = {c.call_id: c.output for c in calls if c.output is not None}
            args = {c.call_id: c.args for c in calls}
            verification = verify(submitted.get("answer", ""), submitted.get("claims", []), outputs, args)
            if (verification.ok and not verification.unlabelled_numbers) or repaired:
                break
            repaired = True
            report = {"failed_claims": [asdict(c) for c in verification.checks
                                        if c.status in ("mismatch", "unresolved", "not_missing")],
                      "unlabelled_numbers": verification.unlabelled_numbers}
            messages.append({"role": "user", "content": "Verification failed:\n" + json.dumps(report, default=str)
                             + "\nFix the claims (or the answer) and call submit_answer again."})
            submitted = None
        else:
            error = f"stopped after {self.max_steps} steps without a verified answer"
        result = AgentResult(run_id, question, self.llm.model, (submitted or {}).get("answer"),
                             (submitted or {}).get("claims", []), calls, verification, repaired, steps,
                             round(time.perf_counter() - t0, 2), usage, error)
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / f"{run_id}.json").write_text(json.dumps(result.to_json(), indent=1, default=str),
                                                          encoding="utf-8")
        return result


def render_report(r: AgentResult) -> str:
    lines = [f"# {r.question}", "", r.answer or f"_No answer: {r.error}_", "", "## Claims", "",
             "| Claim | Value | Label | Source | Check |", "|---|---|---|---|---|"]
    checks = r.verification.checks if r.verification else []
    for c in checks:
        src = f"{c.call_id}:{c.field}" if c.call_id else "-"
        lines.append(f"| {c.statement} | {c.value if c.value is not None else '-'} {c.unit or ''} | {c.label} | "
                     f"{src} | {c.status}{' (' + c.note + ')' if c.note else ''} |")
    if r.verification and r.verification.unlabelled_numbers:
        lines += ["", "Unlabelled numbers in the answer: " + ", ".join(r.verification.unlabelled_numbers)]
    lines += ["", "## Tool calls", "", "| id | tool | arguments | ms | result |", "|---|---|---|---|---|"]
    for c in r.tool_calls:
        lines.append(f"| {c.call_id} | {c.tool} | `{json.dumps(c.args)}` | {c.duration_ms} | "
                     f"{'error: ' + (c.error or '')[:120] if c.is_error else 'ok'} |")
    lines += ["", f"Model {r.model}; {r.steps} LLM steps; {r.latency_s}s; repaired: {r.repaired}; run {r.run_id}."]
    return "\n".join(lines) + "\n"
