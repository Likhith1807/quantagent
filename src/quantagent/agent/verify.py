"""Check an answer's claims against the tool outputs they cite.

Each claim carries a label:
  calculated  a number read from a tool output; must cite call_id + field and match it
  assumed     an input the answer relies on (e.g. the risk-free rate); must appear in a cited
              output's meta.assumptions or call arguments, else it is reported as unchecked
  missing     something the tools could not provide; the cited field must be null/absent
Numbers in the answer text that no claim accounts for are reported as unlabelled.
"""
import re
from dataclasses import dataclass, field

REL_TOL, ABS_TOL = 0.005, 1e-4  # passes values rounded to about 3 significant figures

_TOKEN = re.compile(r"[^.\[\]]+|\[\d+\]")
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?%?")
_DATE_LIKE = re.compile(r"\d{4}-\d{2}-\d{2}")


def resolve(obj, path: str):
    """'ranked[0].metrics.sharpe' or 'ranked.0.metrics.sharpe' -> value; KeyError if absent."""
    cur = obj
    for tok in _TOKEN.findall(path or ""):
        tok = tok.strip("[]")
        if isinstance(cur, list):
            cur = cur[int(tok)]
        elif isinstance(cur, dict):
            cur = cur[tok]
        else:
            raise KeyError(tok)
    return cur


def matches(claimed: float, unit: str, actual) -> bool:
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        return False
    v = claimed / 100 if unit == "percent" else claimed
    return abs(v - actual) <= REL_TOL * abs(actual) + ABS_TOL


@dataclass
class ClaimCheck:
    statement: str
    label: str
    value: float | None
    unit: str | None
    call_id: str | None
    field: str | None
    status: str  # verified | mismatch | unresolved | unchecked | confirmed_missing | not_missing
    actual: object = None
    note: str = ""


@dataclass
class Verification:
    checks: list[ClaimCheck]
    unlabelled_numbers: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.checks:
            out[c.status] = out.get(c.status, 0) + 1
        return out

    @property
    def ok(self) -> bool:
        return not any(c.status in ("mismatch", "unresolved", "not_missing") for c in self.checks)


def _check(claim: dict, outputs: dict[str, dict], args: dict[str, dict]) -> ClaimCheck:
    label, cid, fld = claim.get("label", "calculated"), claim.get("call_id"), claim.get("field")
    c = ClaimCheck(claim.get("statement", ""), label, claim.get("value"), claim.get("unit"), cid, fld, "unresolved")
    out = outputs.get(cid) if cid else None
    if label == "calculated":
        if out is None:
            c.note = f"unknown call_id {cid!r}"
            return c
        try:
            c.actual = resolve(out, fld)
        except (KeyError, IndexError, ValueError, TypeError):
            c.note = f"field {fld!r} not in {cid}"
            return c
        if c.value is None:
            c.note = "calculated claim without a value"
            return c
        c.status = "verified" if matches(float(c.value), c.unit or "number", c.actual) else "mismatch"
        return c
    if label == "missing":
        if out is None:
            c.status, c.note = "unchecked", "no source cited"
            return c
        try:
            c.actual = resolve(out, fld)
        except (KeyError, IndexError, ValueError, TypeError):
            c.actual = None
        c.status = "confirmed_missing" if c.actual is None else "not_missing"
        return c
    # assumed
    haystack = []
    for k in ([cid] if cid else outputs.keys()):
        if k in outputs:
            haystack += outputs[k].get("meta", {}).get("assumptions", []) + [str(args.get(k, {}))]
    text = " ".join(haystack)
    if c.value is not None:
        v = float(c.value)
        forms = {f"{v:g}", f"{v:.2f}", f"{v * 100:.2f}%" if c.unit == "fraction" else f"{v:.2f}%", f"{v / 100:g}"}
        c.status = "verified" if any(f in text for f in forms) else "unchecked"
    else:
        c.status = "unchecked"
    return c


def _identifiers(obj, out: set[str]) -> set[str]:
    """String values in tool outputs/arguments (scheme codes, dates): quoting them is not a claim."""
    if isinstance(obj, dict):
        for v in obj.values():
            _identifiers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _identifiers(v, out)
    elif isinstance(obj, str) and len(obj) <= 40:
        out.add(obj)
    return out


def verify(answer: str, claims: list[dict], outputs: dict[str, dict], args: dict[str, dict]) -> Verification:
    checks = [_check(cl, outputs, args) for cl in claims]
    ids = _identifiers([outputs, args], set())
    known = set()
    for cl in claims:
        if cl.get("value") is not None:
            v = float(cl["value"])
            known |= {round(v, 6), round(v * 100, 6), round(v / 100, 6)}
    unlabelled = []
    stripped = _DATE_LIKE.sub(" ", answer)
    for tok in _NUMBER.findall(stripped):
        raw = tok.rstrip("%").replace(",", "")
        x = float(raw)
        if raw in ids or re.fullmatch(r"(19|20)\d{2}", raw) or (x.is_integer() and abs(x) < 10 and not tok.endswith("%")):
            continue  # years, list numbers, small counts
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if not any(abs(x - k) <= 0.5 * 10 ** -decimals + 1e-9 for k in known):
            unlabelled.append(tok)
    return Verification(checks, unlabelled)
