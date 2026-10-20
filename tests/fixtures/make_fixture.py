"""SYNTHETIC test data with planted, known answers. Used only by the test suite.

Every scheme exists to test one thing; the constants below are the expected answers.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HOLIDAYS = ["2024-03-25", "2024-08-15", "2025-08-15", "2025-10-02"]
TRADING = pd.bdate_range("2024-01-01", "2025-12-31").difference(pd.DatetimeIndex(HOLIDAYS))
END = TRADING[-1]

CONST_DIRECT, CONST_REGULAR = "900001", "900002"
CONST_STEP = 0.0004  # per-observation growth; zero volatility
DRAWDOWN = "900003"
DD_PEAK, DD_TROUGH, DD_RECOVERY = 100, 150, 200  # indices into TRADING; -20% exactly
SHORT = "900004"  # 30 NAVs
GAPPY = "900005"  # 10-day gap, plus duplicate / negative / future / unknown-code rows rejected at ingest
GAP_SLICE = slice(300, 310)
IDCW = "900006"
CLOSED = "900007"
TRACKER = "900008"  # NAV = PRI / 10 exactly (its own benchmark): beta 1, alpha 0, capture 1
STALE = "900009"  # 6 equal NAVs and one +20% move
STALE_SLICE, JUMP_AT = slice(200, 206), 400
BENCH_TRI, BENCH_PRI = "FIX100_TRI", "FIX100"
PRI_DIVIDEND_YIELD = 0.015


def _walk(seed: int, n: int, mu: float = 0.0004, sigma: float = 0.01, start: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return start * np.cumprod(1 + rng.normal(mu, sigma, n))


def build(raw_dir: Path, reference_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    n = len(TRADING)
    d = TRADING.strftime("%Y-%m-%d")

    tri = _walk(1, n, 0.0005, 0.01, 10_000)
    pri = tri / (1 + PRI_DIVIDEND_YIELD) ** (np.arange(n) / 252)

    dd = np.empty(n)
    dd[: DD_PEAK + 1] = np.linspace(100, 125, DD_PEAK + 1)
    dd[DD_PEAK: DD_TROUGH + 1] = np.linspace(125, 100, DD_TROUGH - DD_PEAK + 1)
    dd[DD_TROUGH: DD_RECOVERY + 1] = np.linspace(100, 125, DD_RECOVERY - DD_TROUGH + 1)
    dd[DD_RECOVERY:] = np.linspace(125, 140, n - DD_RECOVERY)

    idcw = 10 * np.cumprod(np.full(n, 1.0003))
    for i in range(60, n, 60):
        idcw[i:] *= 0.95  # payout drop

    stale = _walk(3, n)
    stale[STALE_SLICE] = stale[STALE_SLICE.start]
    stale[JUMP_AT:] *= 1.20 / (stale[JUMP_AT] / stale[JUMP_AT - 1])

    closed_n = int(np.searchsorted(TRADING, pd.Timestamp("2025-06-30"), side="right"))
    series = {
        CONST_DIRECT: (TRADING, 100 * (1 + CONST_STEP) ** np.arange(n)),
        CONST_REGULAR: (TRADING, 100 * (1 + CONST_STEP * 0.9) ** np.arange(n)),
        DRAWDOWN: (TRADING, dd),
        SHORT: (TRADING[-30:], _walk(4, 30)),
        GAPPY: (TRADING.delete(range(GAP_SLICE.start, GAP_SLICE.stop)), _walk(2, n - 10)),
        IDCW: (TRADING, idcw),
        CLOSED: (TRADING[:closed_n], _walk(5, closed_n)),
        TRACKER: (TRADING, pri / 10),
        STALE: (TRADING, stale),
    }
    rows = [(c, day.strftime("%Y-%m-%d"), round(float(v), 10)) for c, (idx, vals) in series.items()
            for day, v in zip(idx, vals, strict=True)]
    rows += [  # rejected at ingest
        (GAPPY, d[5], 1.0), (GAPPY, "2024-01-02x", 10.0), (GAPPY, d[6], -1.0),
        (GAPPY, "2099-01-01", 10.0), ("999998", d[5], 10.0),
    ]
    pd.DataFrame(rows, columns=["amfi_code", "date", "nav"]).to_csv(raw_dir / "nav_history.csv", index=False)

    def scheme(code, name, plan, sub, bench="Fixture 100 TRI", status="active", closure=""):
        return {"amfi_code": code, "fund_house": "Fixture AMC" if code != TRACKER else "Index House",
                "scheme_name": name, "category": "Equity", "sub_category": sub, "plan": plan,
                "launch_date": "2010-01-04", "benchmark": bench, "expense_ratio_pct": 1.0 if plan == "Regular" else 0.5,
                "status": status, "closure_date": closure}

    master = [
        scheme(CONST_DIRECT, "Const Growth Fund - Direct Plan - Growth", "Direct", "Large Cap"),
        scheme(CONST_REGULAR, "Const Growth Fund - Regular Plan - Growth", "Regular", "Large Cap"),
        scheme(DRAWDOWN, "Drawdown Fund - Regular Plan - Growth", "Regular", "Large Cap"),
        scheme(SHORT, "Short History Fund - Regular Plan - Growth", "Regular", "Mid Cap"),
        scheme(GAPPY, "Gappy Fund - Regular Plan - Growth", "Regular", "Large Cap", bench="Unknown Index"),
        scheme(IDCW, "Income Fund - Regular Plan - IDCW", "Regular", "Large Cap"),
        scheme(CLOSED, "Closed Fund - Regular Plan - Growth", "Regular", "Large Cap", status="closed",
               closure="2025-06-30"),
        scheme(TRACKER, "Index Tracker Fund - Direct Plan - Growth", "Direct", "Index", bench="Fixture 100"),
        scheme(STALE, "Stale Fund - Regular Plan - Growth", "Regular", "Small Cap"),
        scheme("12AB", "Bad Code Fund - Regular Plan - Growth", "Regular", "Large Cap"),  # rejected
    ]
    pd.DataFrame(master).to_csv(raw_dir / "schemes.csv", index=False)

    bench = [(day, BENCH_TRI, round(float(v), 8)) for day, v in zip(d, tri, strict=True)]
    bench += [(day, BENCH_PRI, round(float(v), 8)) for day, v in zip(d, pri, strict=True)]
    pd.DataFrame(bench, columns=["date", "index_name", "close_value"]).to_csv(
        raw_dir / "benchmark_indices.csv", index=False)
    pd.DataFrame({"date": HOLIDAYS}).to_csv(raw_dir / "nse_holidays.csv", index=False)

    pd.DataFrame([
        {"benchmark_name": "Fixture 100 TRI", "index_name": BENCH_TRI, "index_type": "TRI"},
        {"benchmark_name": "Fixture 100", "index_name": BENCH_PRI, "index_type": "PRI"},
    ]).to_csv(reference_dir / "benchmark_map.csv", index=False)
    (reference_dir / "dataset_notes.json").write_text(json.dumps(
        {"findings": ["Fixture: synthetic test data"], "known_biases": ["Fixture bias note"]}), encoding="utf-8")
