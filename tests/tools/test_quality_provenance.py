import pytest

from quantagent.tools.provenance import get_data_provenance
from quantagent.tools.quality import check_data_quality
from tests.fixtures import make_fixture as fx
from tests.tools.helpers import err, has


def test_quality_all_schemes():
    r = check_data_quality(limit=50)
    assert r.total_schemes == 9 and r.summary == {"ok": 6, "warn": 3, "fail": 0}
    by = {s.scheme_code: s for s in r.schemes}
    assert {i.check: i.count for i in by[fx.GAPPY].issues} == {"calendar_gaps": 10, "missing_metadata": 1}
    assert {i.check for i in by[fx.STALE].issues} == {"stale_runs", "extreme_moves"}
    assert by[fx.IDCW].issues[0].check == "idcw_option"
    assert by[fx.DRAWDOWN].status == "ok" and by[fx.DRAWDOWN].issues == []
    assert has(r.warnings, "rejected at ingest")


def test_quality_selected_checks_and_examples():
    r = check_data_quality(scheme_codes=[fx.STALE], checks=["extreme_moves", "extreme_moves"])
    assert r.checks_run == ["extreme_moves"]
    issue = r.schemes[0].issues[0]
    assert issue.count == 1 and "+20.0%" in issue.examples[0]


def test_quality_pagination():
    r = check_data_quality(limit=2, offset=8)
    assert len(r.schemes) == 1 and r.total_schemes == 9


@pytest.mark.parametrize("kwargs,code", [
    ({"scheme_codes": [fx.STALE, fx.STALE]}, "INVALID_INPUT"),
    ({"scheme_codes": ["999999"]}, "NOT_FOUND"),
    ({"checks": ["drop_table"]}, "INVALID_INPUT"),
    ({"limit": 0}, "INVALID_INPUT"),
])
def test_quality_errors(kwargs, code):
    assert err(check_data_quality, **kwargs).code == code


def test_quality_configured_limit(services):
    services.settings = services.settings.model_copy(update={"max_results": 3})
    assert err(check_data_quality, limit=4).code == "INVALID_INPUT"


def test_provenance_dataset():
    r = get_data_provenance()
    assert r.data_nature == "synthetic" and r.data_version == 1
    assert {s.file for s in r.sources} >= {"schemes.csv", "nse_holidays.csv"}
    assert all(s.sha256 and len(s.sha256) == 64 for s in r.sources)
    assert r.rejected_rows_total == 6
    assert r.table_counts["schemes"] == 9 and r.table_counts["metrics_cache"] == 27
    assert r.dataset_findings == ["Fixture: synthetic test data"] and r.known_biases == ["Fixture bias note"]
    assert {b.index_name for b in r.benchmarks} == {fx.BENCH_TRI, fx.BENCH_PRI}
    assert r.scheme is None and any("risk-free" in a for a in r.assumptions)


def test_provenance_scheme():
    r = get_data_provenance(scheme_code=fx.GAPPY)
    assert r.scheme.fund.scheme_code == fx.GAPPY and r.scheme.benchmark is None
    assert r.scheme.ingest_quality["issues"]["calendar_gaps"] == 10
    t = get_data_provenance(scheme_code=fx.TRACKER)
    assert t.scheme.benchmark.index_name == fx.BENCH_PRI and t.scheme.benchmark.index_type == "PRI"


def test_provenance_errors():
    assert err(get_data_provenance, scheme_code="999999").code == "NOT_FOUND"
    assert err(get_data_provenance, scheme_code="1; --").code == "INVALID_INPUT"
