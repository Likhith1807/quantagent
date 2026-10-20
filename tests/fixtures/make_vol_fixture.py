"""SYNTHETIC panel for the volatility-model tests: GARCH(1,1) returns, long enough for
several walk-forward folds. Written in the same staging layout as the real fetch."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

TRADING = pd.bdate_range("2015-01-01", "2022-12-30")
N_SCHEMES = 12
CODES = [f"91{i:04d}" for i in range(N_SCHEMES)]


def garch_path(rng: np.random.Generator, n: int, omega: float, alpha: float, beta: float) -> np.ndarray:
    r = np.empty(n)
    var = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(var) * rng.standard_normal()
        var = omega + alpha * r[t] ** 2 + beta * var
    return r


def build(raw_dir: Path, reference_dir: Path, seed: int = 7) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = len(TRADING)
    rows, master = [], []
    for i, code in enumerate(CODES):
        r = garch_path(rng, n, omega=2e-6 * (1 + i % 3), alpha=0.08, beta=0.9)
        nav = 10 * np.exp(np.cumsum(r + 0.0003))
        rows += [(code, d.strftime("%Y-%m-%d"), round(float(v), 8)) for d, v in zip(TRADING, nav, strict=True)]
        master.append({"amfi_code": code, "scheme_name": f"Vol Fixture {i} Fund - Direct Plan - Growth",
                       "fund_house": "Vol House", "category": "Equity", "sub_category": "Large Cap", "plan": "Direct",
                       "option_type": "Growth", "launch_date": "2015-01-01", "benchmark": None, "status": "active"})
    pd.DataFrame(master).to_csv(raw_dir / "schemes.csv", index=False)
    pd.DataFrame(rows, columns=["amfi_code", "date", "nav"]).to_csv(raw_dir / "nav_history.csv", index=False)
    (raw_dir / "sources.json").write_text(json.dumps({"data_nature": "synthetic", "files": {}}), encoding="utf-8")
