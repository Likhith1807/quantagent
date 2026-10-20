"""AMFI scheme universe (NAVAll.txt), full NAV histories (mfapi.in) and expense ratios
(AMFI TER disclosures).

Universe: open-ended Equity Scheme categories, Direct plan, Growth option, with at least
`min_years` of NAV history and a NAV within `stale_days` of the latest date in the file.
"""
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

NAVALL_URL = "https://portal.amfiindia.com/spages/NAVAll.txt"
MFAPI_URL = "https://api.mfapi.in/mf/{code}"
TER_URL = ("https://www.amfiindia.com/api/populate-te-rdata-revised?MF_ID=All&Month={month}"
           "&strCat={cat}&strType=1&page={page}&pageSize=100")
# AMFI TER API sub-category ids for the two "Equity Scheme(s)" families (populate-sub-category).
TER_EQUITY_CATEGORY_IDS = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 55,
                           73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85]

# AMFI writes the same SEBI category several ways; map to one label.
_CATEGORY_ALIASES = {
    "elss": "ELSS", "elss- tax saver fund": "ELSS", "elss tax saver fund": "ELSS",
    "sectoral/ thematic": "Sectoral/Thematic", "sectoral fund": "Sectoral/Thematic",
    "thematic fund": "Sectoral/Thematic",
}


def canonical_category(raw: str) -> str | None:
    """'Open Ended Schemes(Equity Schemes - Large Cap Fund)' -> 'Large Cap'. None if not equity."""
    m = re.match(r"^\s*Open Ended Schemes?\s*\((.*)\)\s*$", raw)
    if not m:
        return None
    family, _, sub = m.group(1).partition(" - ")
    if not family.strip().lower().startswith("equity scheme"):
        return None
    sub = sub.strip()
    alias = _CATEGORY_ALIASES.get(sub.lower())
    if alias:
        return alias
    return re.sub(r"\s+Fund$", "", sub).strip()


@dataclass
class UniverseRow:
    scheme_code: str
    isin: str | None
    scheme_name: str
    base_name: str
    fund_house: str
    sub_category: str


def _is_direct_growth(name: str, plan: str | None, option: str | None) -> bool:
    if plan is not None and option is not None and (plan or option):
        return "direct" in plan.lower() and "growth" in option.lower()
    n = name.lower()
    return "direct" in n and "growth" in n and not re.search(r"idcw|dividend|bonus", n)


def parse_navall(text: str) -> list[UniverseRow]:
    """Direct-plan Growth-option equity schemes from AMFI's NAVAll.txt (8- or 6-column format)."""
    rows: list[UniverseRow] = []
    category: str | None = None
    house = ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("Scheme Code"):
            continue
        if ";" not in line:
            if line.lower().startswith(("open ended", "close ended", "interval")):
                category = canonical_category(line)
            else:
                house = line
            continue
        parts = [p.strip() for p in line.split(";")]
        if category is None or not parts[0].isdigit():
            continue
        if len(parts) >= 8:
            code, isin, _, name, plan, option = parts[:6]
            full = f"{name} - {plan} - {option}"
        else:
            code, isin, _, name = parts[:4]
            plan = option = None
            full = name
        if not _is_direct_growth(name, plan, option):
            continue
        rows.append(UniverseRow(code.zfill(6), isin if isin not in ("", "-") else None, full,
                                name, house, category))
    return rows


def parse_mfapi(payload: dict) -> pd.DataFrame:
    """mfapi.in /mf/{code} JSON -> DataFrame[date, nav] (unvalidated strings kept as given)."""
    data = payload.get("data") or []
    return pd.DataFrame({"date": pd.to_datetime(pd.Series([d.get("date") for d in data], dtype="string"),
                                                format="%d-%m-%Y", errors="coerce"),
                         "nav": [d.get("nav") for d in data]})


def keep_scheme(nav: pd.DataFrame, latest: date, min_years: float, stale_days: int) -> tuple[bool, str]:
    ok = nav.dropna(subset=["date"])
    if ok.empty:
        return False, "no_history"
    first, last = ok["date"].min().date(), ok["date"].max().date()
    if (latest - last).days > stale_days:
        return False, "stale"
    if (latest - first).days < min_years * 365.25:
        return False, "short_history"
    return True, "ok"


_NAME_DROP = re.compile(r"[^a-z0-9]+")


def name_key(name: str) -> str:
    """Normalised scheme name for joining TER disclosures (by name) to NAVAll schemes."""
    n = name.lower().replace("&", " and ")
    n = re.sub(r"\b(the|scheme|plan|option)\b", " ", n)
    return _NAME_DROP.sub("", n)


def ter_pages(fetch_json: Callable[[str], dict], months: Iterable[str]) -> list[dict]:
    """All equity TER rows for the given months ('MM-YYYY')."""
    out: list[dict] = []
    for month in months:
        for cat in TER_EQUITY_CATEGORY_IDS:
            page = 1
            while True:
                d = fetch_json(TER_URL.format(month=month, cat=cat, page=page))
                out += d.get("data") or []
                meta = d.get("meta") or {}
                if page >= int(meta.get("pageCount") or 0):
                    break
                page += 1
    return out


def ter_history(rows: list[dict], universe: list[UniverseRow]) -> tuple[pd.DataFrame, int]:
    """Direct-plan TER change points per scheme: DataFrame[amfi_code, as_of_date, ter_pct].
    Returns (frame, number of universe base names with no TER match)."""
    if not rows:
        return pd.DataFrame(columns=["amfi_code", "as_of_date", "ter_pct"]), len({u.base_name for u in universe})
    t = pd.DataFrame(rows)
    t["key"] = t["Scheme_Name"].map(name_key)
    t["as_of_date"] = pd.to_datetime(t["TER_Date"].str[:10]).dt.date
    t["ter_pct"] = pd.to_numeric(t["D_TER"], errors="coerce")
    t = t.dropna(subset=["ter_pct"]).sort_values(["key", "as_of_date"]).drop_duplicates(["key", "as_of_date"])
    t = t[t["ter_pct"].ne(t.groupby("key")["ter_pct"].shift())]  # keep change points only
    by_key = {k: g[["as_of_date", "ter_pct"]] for k, g in t.groupby("key")}
    out, unmatched = [], set()
    for u in universe:
        g = by_key.get(name_key(u.base_name))
        if g is None:
            unmatched.add(u.base_name)
            continue
        out.append(g.assign(amfi_code=u.scheme_code))
    frame = pd.concat(out)[["amfi_code", "as_of_date", "ter_pct"]] if out else \
        pd.DataFrame(columns=["amfi_code", "as_of_date", "ter_pct"])
    return frame, len(unmatched)


def recent_months(today: date, n: int) -> list[str]:
    """The n most recent months up to and including today's month, as 'MM-YYYY'."""
    p = pd.Period(today, "M")
    return [(p - i).strftime("%m-%Y") for i in range(n)]


def utcnow() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
