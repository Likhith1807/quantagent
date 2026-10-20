"""The whole command pipeline offline: fetch (with faked AMFI/mfapi/TER responses) -> load the
hand-downloaded files -> ingest -> call a tool through the CLI -> evaluation on the fixture."""
import json

import numpy as np
import pandas as pd
import pytest

from quantagent import cli
from quantagent.commands import fetch_amfi
from quantagent.config import get_settings
from quantagent.sources import amfi
from quantagent.tools import _base
from tests.unit.test_sources import NAVALL

DAYS = pd.bdate_range("2019-01-01", "2026-10-16")


def _mfapi(code: str) -> dict:
    n = {"100001": len(DAYS), "100004": 300}[code]  # 100004: too short, dropped
    nav = 10 * np.cumprod(1 + np.random.default_rng(int(code)).normal(0.0004, 0.01, n))
    return {"data": [{"date": d.strftime("%d-%m-%Y"), "nav": f"{v:.4f}"}
                     for d, v in zip(DAYS[-n:][::-1], nav[::-1], strict=True)]}


def fake_get_json(url: str):
    if "mfapi" in url:
        return _mfapi(url.rsplit("/", 1)[1])
    return {"data": [{"Scheme_Name": "Alpha Large Cap Fund", "TER_Date": "2026-10-01T00:00:00.000Z",
                      "D_TER": "0.75"}], "meta": {"pageCount": 1}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k, v in {"STAGING_DIR": tmp_path / "staging", "EXTERNAL_DIR": tmp_path / "external",
                 "REPORTS_DIR": tmp_path / "reports", "MODEL_DIR": tmp_path / "models",
                 "DB_URL": f"sqlite:///{(tmp_path / 'db' / 'q.db').as_posix()}"}.items():
        monkeypatch.setenv(f"QUANTAGENT_{k}", str(v))
    get_settings.cache_clear()
    monkeypatch.setattr(fetch_amfi, "get", lambda url: NAVALL.encode())
    monkeypatch.setattr(fetch_amfi, "get_json", fake_get_json)
    monkeypatch.setattr(amfi, "TER_EQUITY_CATEGORY_IDS", [15])
    yield tmp_path
    get_settings.cache_clear()


def test_fetch_load_ingest_and_query(env, capsys):
    ext = env / "external" / "niftyindices"
    ext.mkdir(parents=True)
    pd.DataFrame({"Date": DAYS.strftime("%d %b %Y"), "Total Returns Index": np.linspace(20000, 40000, len(DAYS))}
                 ).to_csv(ext / "NIFTY 100.csv", index=False)
    pd.DataFrame({"Date": ["2019-01-02", "2024-01-03"], "Yield (%)": [6.8, 6.9]}).to_csv(
        env / "external" / "tbill_91d.csv", index=False)

    _base.set_services(None)  # use the database built here, not the session fixture
    cli.main(["setup", "--ter-months", "1"])
    capsys.readouterr()
    sources = json.loads((env / "staging" / "sources.json").read_text())
    assert sources["data_nature"] == "real" and set(sources["files"]) == {
        "schemes.csv", "nav_history.csv", "expense_ratios.csv", "benchmark_indices.csv", "risk_free_rates.csv"}
    schemes = pd.read_csv(env / "staging" / "schemes.csv", dtype=str)
    assert list(schemes["amfi_code"]) == ["100001"] and schemes["benchmark"][0] == "NIFTY 100 TRI"

    with pytest.raises(SystemExit) as e:
        cli.main(["tool", "get_benchmark_comparison", '{"scheme_code": "100001", "period": "3y"}'])
    assert e.value.code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["meta"]["data_nature"] == "real" and out["benchmark"]["index_name"] == "NIFTY100_TRI"
    assert out["benchmark"]["index_type"] == "TRI" and out["meta"]["data_retrieved_at"]

    with pytest.raises(SystemExit) as e:
        cli.main(["tool", "search_funds", '{"query": "alpha"}'])
    item = json.loads(capsys.readouterr().out)["items"][0]
    assert item["latest_expense_ratio"] == {"value_pct": 0.75, "as_of": "2026-10-01", "as_of_assumed": False}

    with pytest.raises(SystemExit) as e:
        cli.main(["tool", "get_risk_metrics", '{"scheme_code": "999999"}'])
    assert e.value.code == 1 and "NOT_FOUND" in capsys.readouterr().out


def test_cli_help_and_unknown(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert e.value.code == 0 and "quantagent setup" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="unknown command"):
        cli.main(["nope"])
    with pytest.raises(SystemExit, match="usage"):
        cli.main(["tool"])


def test_eval_command_on_fixture(tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main(["eval", "--fixture", "--out-dir", str(tmp_path)])
    assert e.value.code == 0
    res = json.loads((tmp_path / "eval_results.json").read_text())
    assert res["summary"]["direct"]["answer_accuracy"] == 1.0 and res["meta"]["n_questions"] >= 30
    assert "Failures (direct): 0" in (tmp_path / "eval_report.md").read_text()
