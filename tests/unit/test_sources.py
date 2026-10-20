"""Parsers for AMFI NAVAll.txt, mfapi.in, AMFI TER, niftyindices TRI and T-bill CSVs."""
from datetime import date

import pandas as pd
import pytest

from quantagent.sources import amfi, niftyindices, tbill

NAVALL = """Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;Option;Net Asset Value;Date

Open Ended Schemes(Equity Scheme - Large Cap Fund)

Alpha Mutual Fund

100001;INF000A01AA1;-;Alpha Large Cap Fund;Direct Plan;Growth Option;50.1;19-Oct-2026
100002;INF000A01AA2;-;Alpha Large Cap Fund;Direct Plan;IDCW Option;20.1;19-Oct-2026
100003;INF000A01AA3;-;Alpha Large Cap Fund;Regular Plan;Growth;45.1;19-Oct-2026

Open Ended Schemes(Equity Schemes - ELSS- Tax Saver Fund)

Beta Mutual Fund

100004;INF000B01AA1;-;Beta Tax Saver Fund;Direct Plan;GROWTH;80.0;19-Oct-2026

Open Ended Schemes(Debt Scheme - Liquid Fund)

Beta Mutual Fund

100005;INF000B01AA9;-;Beta Liquid Fund;Direct Plan;Growth;1000;19-Oct-2026
"""


def test_parse_navall_keeps_direct_growth_equity_only():
    rows = amfi.parse_navall(NAVALL)
    assert [(r.scheme_code, r.sub_category, r.fund_house) for r in rows] == [
        ("100001", "Large Cap", "Alpha Mutual Fund"), ("100004", "ELSS", "Beta Mutual Fund")]
    assert rows[0].scheme_name == "Alpha Large Cap Fund - Direct Plan - Growth Option"
    assert rows[0].base_name == "Alpha Large Cap Fund" and rows[0].isin == "INF000A01AA1"


def test_parse_navall_six_column_format():
    text = ("Open Ended Schemes(Equity Scheme - Mid Cap Fund)\nGamma MF\n"
            "200001;INF1;-;Gamma Mid Cap Fund - Direct Plan - Growth;12.0;19-Oct-2026\n"
            "200002;INF2;-;Gamma Mid Cap Fund - Direct Plan - IDCW;12.0;19-Oct-2026\n")
    assert [r.scheme_code for r in amfi.parse_navall(text)] == ["200001"]


@pytest.mark.parametrize("raw,expected", [
    ("Open Ended Schemes(Equity Scheme - Sectoral/ Thematic)", "Sectoral/Thematic"),
    ("Open Ended Schemes(Equity Schemes - Thematic Fund)", "Sectoral/Thematic"),
    ("Open Ended Schemes(Equity Scheme - Large & Mid Cap Fund)", "Large & Mid Cap"),
    ("Open Ended Schemes(Hybrid Scheme - Arbitrage Fund)", None),
    ("Close Ended Schemes(Equity Scheme - ELSS)", None),
])
def test_canonical_category(raw, expected):
    assert amfi.canonical_category(raw) == expected


def test_parse_mfapi_and_keep_rules():
    payload = {"data": [{"date": "19-10-2026", "nav": "10.5"}, {"date": "02-01-2018", "nav": "5.0"},
                        {"date": "bad", "nav": "1"}]}
    nav = amfi.parse_mfapi(payload)
    assert nav["date"].isna().sum() == 1 and nav["date"].max() == pd.Timestamp("2026-10-19")
    assert amfi.keep_scheme(nav, date(2026, 10, 19), 5, 10) == (True, "ok")
    assert amfi.keep_scheme(nav, date(2026, 10, 19), 10, 10) == (False, "short_history")
    assert amfi.keep_scheme(nav, date(2026, 12, 31), 5, 10) == (False, "stale")
    assert amfi.keep_scheme(amfi.parse_mfapi({}), date(2026, 1, 1), 5, 10) == (False, "no_history")


def test_ter_history_change_points_and_name_match():
    universe = amfi.parse_navall(NAVALL)
    rows = [{"Scheme_Name": "Alpha Large-Cap Fund", "TER_Date": f"2026-09-0{d}T00:00:00.000Z", "D_TER": ter}
            for d, ter in [(1, "0.90"), (2, "0.90"), (3, "0.85")]]
    frame, unmatched = amfi.ter_history(rows, universe)
    assert list(frame["ter_pct"]) == [0.90, 0.85] and set(frame["amfi_code"]) == {"100001"}
    assert unmatched == 1  # Beta Tax Saver Fund has no rows
    assert amfi.ter_history([], universe)[1] == 2


def test_ter_pages_follows_page_count():
    seen = []

    def fake(url):
        seen.append(url)
        page = int(url.split("page=")[1].split("&")[0])
        return {"data": [{"page": page}], "meta": {"pageCount": 2}}

    rows = amfi.ter_pages(fake, ["09-2026"])
    assert len(rows) == 2 * len(amfi.TER_EQUITY_CATEGORY_IDS) and len(seen) == len(rows)
    assert amfi.recent_months(date(2026, 1, 15), 2) == ["01-2026", "12-2025"]


def test_name_key_normalises():
    assert amfi.name_key("The Alpha & Beta Large-Cap Fund") == amfi.name_key("Alpha and Beta Large Cap Fund")


def test_niftyindices_csv_variants(tmp_path):
    (tmp_path / "NIFTY 100.csv").write_text('Date,Total Returns Index\n"02 Jan 2024","32,100.50"\n03 Jan 2024,32200\n')
    (tmp_path / "x.csv").write_text("Index Name,Date,TotalReturnsIndex\nNIFTY MIDCAP 150,2024-01-02,25000\n")
    df = niftyindices.load_dir(tmp_path)
    assert set(df["index_name"]) == {"NIFTY100_TRI", "NIFTY_MIDCAP150_TRI"}
    assert df.loc[df["index_name"] == "NIFTY100_TRI", "close_value"].tolist() == [32100.5, 32200.0]
    assert (df["index_type"] == "TRI").all() and df["date"].iloc[0] == "2024-01-02"
    assert niftyindices.load_dir(tmp_path / "empty").empty
    assert niftyindices.series_key("Nifty 500 TRI") == "NIFTY500_TRI"
    with pytest.raises(ValueError):
        niftyindices.series_key("NIFTY BANK")
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        niftyindices.parse_tri_csv(tmp_path / "bad.csv")


def test_tbill_csv(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("Auction Date,91 Day Cut-off Yield (%)\n03-01-2024,6.92\n10-01-2024,6.94%\n")
    df = tbill.parse_tbill_csv(f, "RBI DBIE")
    assert df["annual_rate"].tolist() == pytest.approx([0.0692, 0.0694]) and df["date"].iloc[0] == "2024-01-03"
    f.write_text("Date,Yield\n2024-01-03,0.0692\n")
    with pytest.raises(ValueError, match="percent"):
        tbill.parse_tbill_csv(f, "x")
    f.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        tbill.parse_tbill_csv(f, "x")
