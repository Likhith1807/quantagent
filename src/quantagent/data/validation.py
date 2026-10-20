"""Row-level validation (used by ingest to reject rows) and series-level quality checks
(used by ingest for its coverage log and by the check_data_quality tool). One implementation
for both, so the tool reports exactly what ingestion enforces."""
import re
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import pandas as pd

SCHEME_CODE_RE = re.compile(r"^\d{6}$")  # AMFI codes, per docs/DATA_PROFILE.md
EXTREME_MOVE = 0.15
STALE_RUN_DAYS = 5
MAX_EXAMPLES = 5

FAIL_CHECKS = ("duplicate_dates", "nonpositive_nav", "future_dates")
WARN_CHECKS = ("calendar_gaps", "extreme_moves", "stale_runs", "missing_metadata", "idcw_option")
ALL_CHECKS = FAIL_CHECKS + WARN_CHECKS
REQUIRED_METADATA = ("fund_house", "category", "sub_category", "launch_date", "benchmark_index")


# ---------------------------------------------------------------- row level

def _reject(df: pd.DataFrame, mask: pd.Series, reason: str, source: str) -> list[dict]:
    return [{"source": source, "row": int(i) + 2, "reason": reason, "raw": r.to_dict()}
            for i, r in df[mask].iterrows()]  # +2: header line and 1-based numbering


def option_type_from_name(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("idcw", "dividend", "payout")):
        return "IDCW"
    if "growth" in n:
        return "Growth"
    return "NA"


def base_name(name: str) -> str:
    """Scheme name without its plan words, so Direct and Regular plans of one scheme match."""
    n = re.sub(r"\s*-?\s*\b(Direct|Regular)\b( Plan)?", "", name, flags=re.I)
    return re.sub(r"\s+", " ", n).strip(" -").lower()


def validate_schemes(df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, list[dict]]:
    df = df.reset_index(drop=True)
    code = df["amfi_code"].astype("string").str.strip()
    rejected: list[dict] = []
    checks = [
        (~code.fillna("").str.fullmatch(SCHEME_CODE_RE.pattern), "bad_scheme_code"),
        (df["scheme_name"].isna() | df["fund_house"].isna(), "missing_required_field"),
        (~df["plan"].isin(["Direct", "Regular"]), "unknown_plan_type"),
        (code.duplicated(keep="first"), "duplicate_scheme_code"),
    ]
    if "expense_ratio_pct" in df:
        ter = pd.to_numeric(df["expense_ratio_pct"], errors="coerce")
        checks.append((ter.notna() & ((ter < 0) | (ter > 5)), "expense_ratio_out_of_range"))
    bad = pd.Series(False, index=df.index)
    for mask, reason in checks:
        mask = mask.fillna(True) & ~bad
        rejected += _reject(df, mask, reason, source)
        bad |= mask
    out = df[~bad].copy()
    out["scheme_code"] = code[~bad]
    return out, rejected


def validate_series(df: pd.DataFrame, key: str, date_col: str, value_col: str, known_keys: set | None,
                    today: date, source: str, date_format: str | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Validate a long-format time series file (NAVs or index levels)."""
    df = df.reset_index(drop=True)
    d = pd.to_datetime(df[date_col], format=date_format, errors="coerce")
    v = pd.to_numeric(df[value_col], errors="coerce")
    k = df[key].astype("string").str.strip()
    checks = [
        (k.isna() | d.isna() | v.isna(), "missing_or_unparseable"),
        (v <= 0, "nonpositive_value"),
        (d.dt.date > today, "future_date"),
    ]
    if known_keys is not None:
        checks.append((~k.isin(known_keys), "unknown_scheme"))
    checks.append((pd.DataFrame({"k": k, "d": d}).duplicated(keep="first"), "duplicate_date"))
    rejected: list[dict] = []
    bad = pd.Series(False, index=df.index)
    for mask, reason in checks:
        mask = mask.fillna(False) & ~bad
        rejected += _reject(df, mask, reason, source)
        bad |= mask
    out = pd.DataFrame({"key": k[~bad], "date": d[~bad].dt.date, "value": v[~bad].astype(float)})
    return out.reset_index(drop=True), rejected


# ---------------------------------------------------------------- series level

@dataclass
class Issue:
    check: str
    severity: str
    count: int
    examples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _runs_of_equal(values: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    runs, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            if i - start >= min_len:
                runs.append((start, i - start))
            start = i
    return runs


def run_checks(nav: pd.DataFrame, scheme: dict, calendar: pd.DatetimeIndex, today: date,
               checks: tuple[str, ...] | list[str] = ALL_CHECKS) -> list[Issue]:
    """nav: columns [date, nav], unsorted, may contain defects. scheme: one `schemes` row."""
    nav = nav.assign(date=pd.to_datetime(nav["date"])).sort_values("date", kind="stable")
    clean = nav.drop_duplicates("date", keep="first")
    clean = clean[clean["nav"] > 0]
    issues: list[Issue] = []

    def add(check: str, count: int, examples) -> None:
        sev = "fail" if check in FAIL_CHECKS else "warn"
        issues.append(Issue(check, sev, int(count), [str(e) for e in list(examples)[:MAX_EXAMPLES]]))

    for check in checks:
        if check == "duplicate_dates":
            dup = nav[nav["date"].duplicated(keep="first")]["date"].dt.date
            add(check, len(dup), dup)
        elif check == "nonpositive_nav":
            bad = nav[nav["nav"].isna() | (nav["nav"] <= 0)]["date"].dt.date
            add(check, len(bad), bad)
        elif check == "future_dates":
            fut = nav[nav["date"].dt.date > today]["date"].dt.date
            add(check, len(fut), fut)
        elif check == "calendar_gaps":
            if clean.empty:
                add(check, 0, [])
                continue
            span = calendar[(calendar >= clean["date"].iloc[0]) & (calendar <= clean["date"].iloc[-1])]
            missing = span.difference(pd.DatetimeIndex(clean["date"]))
            add(check, len(missing), missing.date)
        elif check == "extreme_moves":
            r = clean.set_index("date")["nav"].pct_change().dropna()
            big = r[r.abs() > EXTREME_MOVE]
            add(check, len(big), [f"{d.date()}: {x:+.1%}" for d, x in big.items()])
        elif check == "stale_runs":
            runs = _runs_of_equal(clean["nav"].to_numpy(), STALE_RUN_DAYS)
            dates = clean["date"].dt.date.to_numpy()
            add(check, len(runs), [f"{dates[s]} x{n}" for s, n in runs])
        elif check == "missing_metadata":
            missing = [f for f in REQUIRED_METADATA if scheme.get(f) in (None, "")]
            add(check, len(missing), missing)
        elif check == "idcw_option":
            is_idcw = scheme.get("option_type") == "IDCW"
            add(check, int(is_idcw), ["IDCW NAVs drop on payouts; not a total-return series"] if is_idcw else [])
        else:
            raise ValueError(f"unknown check {check}")
    return issues


def status_of(issues: list[Issue]) -> str:
    if any(i.severity == "fail" and i.count for i in issues):
        return "fail"
    if any(i.severity == "warn" and i.count for i in issues):
        return "warn"
    return "ok"
