Failures the evaluation and test suites surfaced while this version was built, and what changed:

| Found by | Failure | Cause | Fix |
|---|---|---|---|
| `quantagent eval --fixture` (direct) | 4 of 35 questions failed: Sharpe/Sortino expected `9.7e9`/`inf`, tool returned `null`; ranking order differed | The fixture's zero-volatility scheme has a zero denominator. The tools follow METHODOLOGY.md (ratio is null below 1e-9); the independent reference did not implement that rule | Reference now applies the documented rule and skips null metrics when ranking. Tools unchanged |
| Oracle-agent test (`tests/integration/test_eval_agent.py`) | Every question needed a repair round and the scripted run ran out of turns | The verifier counted 6-digit scheme codes in the answer as unlabelled numbers | Strings that appear in tool outputs or arguments (codes, dates) are no longer treated as numeric claims |
| Real-data ingest | 7 NAV rows rejected | mfapi.in returns `0.00000` NAVs on a few dates | Rejected with reason `nonpositive_value` (see `reports/rejected_rows.csv`); counted in `test_every_nav_row_accounted_for` |
| Real-data fetch | First full run spent minutes parsing dates | `pd.to_datetime` was called once per NAV string | Vectorised parsing in `sources.amfi.parse_mfapi` |
