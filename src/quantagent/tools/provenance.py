import json

from ..schemas.common import SchemeCode
from ..schemas.tools import BenchmarkCoverage, IngestOut, ProvenanceResult, SchemeProvenance, SourceOut
from ._base import base_warnings, build_meta, fund_summaries, get_services, require_scheme, tool_handler


@tool_handler
def get_data_provenance(scheme_code: SchemeCode | None = None) -> ProvenanceResult:
    """Explain where the data comes from and how far it can be trusted.

    Use when asked about sources, coverage, data freshness, versions, assumptions or known biases,
    and before presenting conclusions to a user. Pass scheme_code for that scheme's coverage,
    ingest quality result and benchmark coverage. Do NOT use for performance numbers.
    """
    svc = get_services()
    repo = svc.repo
    m = repo.meta
    ingest = repo.last_ingest()
    benches = [BenchmarkCoverage(**b) for b in repo.benchmark_names()]
    scheme = None
    if scheme_code:
        s = require_scheme(repo, scheme_code)
        nav_log = next((r for r in ingest if r["table_name"] == "nav_history"), None)
        quality = json.loads(nav_log["coverage"])["per_scheme"].get(scheme_code) if nav_log else None
        scheme = SchemeProvenance(fund=fund_summaries(repo, [s])[0], ingest_quality=quality,
                                  benchmark=next((b for b in benches if b.index_name == s["benchmark_index"]), None))
    return ProvenanceResult(
        meta=build_meta(repo), warnings=base_warnings(repo),
        data_nature=m.get("data_nature", "synthetic"), data_version=m["data_version"],
        calc_version=m.get("calc_version", ""), data_as_of=m["data_as_of"], built_at=m.get("built_at"),
        calendar_source=m.get("calendar_source"), sources=[SourceOut(**{**s, "coverage": json.loads(s["coverage"]) if s.get("coverage") else None})
                 for s in repo.data_sources()],
        table_counts=repo.table_counts(), benchmarks=benches,
        last_ingest=[IngestOut(**{k: r[k] for k in IngestOut.model_fields}) for r in ingest],
        rejected_rows_total=sum(r["rows_rejected"] for r in ingest),
        assumptions=m["assumptions"], known_biases=m["known_biases"], dataset_findings=m["dataset_findings"],
        scheme=scheme)
