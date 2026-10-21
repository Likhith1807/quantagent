# QuantAgent calculation methodology (calc_version 1.0.0)

Served as the MCP resource `quantagent://methodology`. Code: `src/quantagent/analytics/`.

## Data

Real data, downloaded by `quantagent fetch` / `quantagent load-manual` (see `docs/DATA_PROFILE.md`
for the exact sources, download dates, coverage and licences):

* **Universe and NAVs:** AMFI `NAVAll.txt` (open-ended Equity Scheme categories, Direct plan,
  Growth option) with full daily NAV histories from mfapi.in; schemes need >= 5 years of NAVs.
* **Expense ratios:** AMFI TER disclosures, Direct-plan total TER, each value with its own date.
* **Benchmarks:** Nifty **Total Return** indices (niftyindices.com), assigned by SEBI category
  (`data/reference/category_benchmarks.csv`).
* **Risk-free:** 91-day Treasury bill yields (RBI DBIE or CCIL) when loaded.

Every response carries `meta.data_nature` (`real`, or `synthetic` for the test fixture),
`meta.data_source` and `meta.data_retrieved_at`.

## Return universe

Only Growth (and ETF, option `NA`) NAVs are total-return series, so the real universe is
Growth-only. IDCW NAVs drop on every payout; if an IDCW scheme is loaded, CAGR, total return and
drawdown are withheld and `rank_funds` excludes it.

## Definitions

| Item | Definition |
|---|---|
| Daily return | `NAV_t / NAV_{t-1} - 1` between consecutive observations |
| Annualisation factor | 252 |
| Total return | `NAV_end / NAV_start - 1` |
| CAGR | `(NAV_end / NAV_start)^(365.25 / calendar days) - 1` |
| Volatility | `std(r, ddof=1) * sqrt(252)` |
| Risk-free rate | dated series from `risk_free_rates` (forward-filled) if loaded; else the configured default with its cited source, flagged in `meta.assumptions`; converted to daily as `(1 + annual)^(1/252) - 1` |
| Sharpe | `mean(r - rf) / std(r - rf) * sqrt(252)`; null if the excess series has zero spread |
| Sortino | `mean(r - rf) / sqrt(mean(min(r - rf, 0)^2)) * sqrt(252)`; null with no downside |
| Beta | `cov(r_f - rf, r_b - rf) / var(r_b - rf)` on shared dates |
| Alpha (Jensen) | `(mean(r_f - rf) - beta * mean(r_b - rf)) * 252` |
| R^2 | squared correlation of the excess returns |
| Tracking error | `std(r_f - r_b) * sqrt(252)` |
| Information ratio | `mean(r_f - r_b) * 252 / tracking error` |
| Up capture | monthly returns (last NAV of each month): `(prod(1 + r_f) - 1) / (prod(1 + r_b) - 1)` over months the benchmark rose; null with fewer than 3 such months |
| Down capture | the same over months the benchmark fell |
| Maximum drawdown | `min(NAV / cummax(NAV) - 1)` with peak, trough, recovery date (null if not recovered), peak-to-trough and trough-to-recovery calendar days |
| Benchmarks | TRI when available. Series without a TRI tag are stored as PRI and every benchmark-relative response warns that PRI understates the benchmark |

## Windows and minimum history

* Trailing windows: `1y`, `3y`, `5y` end at the latest data date (or `end_date`) and start the same
  calendar date 1/3/5 years earlier. `max` uses all history. Explicit dates are inclusive.
* A 1y/3y/5y window needs NAVs on **>= 90% of its trading days**. Trading calendar: Mon-Fri minus
  NSE holidays when a holiday list is loaded; otherwise minus the weekdays on which fewer than half
  of the schemes alive that day published a NAV (derived at ingest and recorded as an assumption).
* Volatility, Sharpe and Sortino need **>= 60 returns**; alpha, beta and R^2 need **>= 120 returns
  shared with the benchmark**; `get_benchmark_comparison` needs 120 shared returns.
* Short of these, tools raise `INSUFFICIENT_HISTORY` with the count found and required, or return
  null for the affected metric with a warning.

## Ranking

`rank_funds` reads `metrics_cache` (built at ingest with the same `metric_bundle` function the
tools call). Single metric: sorted in its natural direction (higher is better except volatility
and expense ratio; drawdowns are negative, so closer to zero ranks higher). Weighted: score =
`sum(w_i * direction_i * z_i) / sum(w_i)` with z-scores across eligible funds. `dedupe =
direct_growth` keeps one plan per scheme, preferring Direct.

## Volatility forecast (`predict_volatility`)

* **Target:** annualised realised volatility of the next 21 daily log NAV returns,
  `sqrt(252/21 * sum r_{t+i}^2)`.
* **Features at origin t** (data up to t only): realised vol over 1, 5, 22, 63 and 252 days, EWMA
  vol (lambda 0.94), 21- and 63-day log returns, drawdown from the 252-day high.
* **Models:** ridge regression and gradient boosting on log volatility. **Baselines:** naive (last
  22-day realised vol), EWMA, HAR-RV (log, pooled OLS on daily/weekly/monthly RV) and GARCH(1,1)
  per scheme.
* **Back-transformation:** the regression models (HAR-RV, ridge, boosting) predict log volatility.
  `exp(E[log vol])` underestimates `E[vol]`, so forecasts are multiplied by Duan's smearing factor,
  the mean of `exp(residual)` on the training rows.
* **Evaluation:** walk-forward by calendar year, models refitted each fold. Training rows are
  purged so their target windows end at least 30 days before the test year (embargo); GARCH
  parameters use returns up to that cut-off only. Training rows whose target window contains a
  suspect NAV (a daily move above 15% or 5+ identical NAVs) are dropped; those windows are in the
  past at fit time.
* **Selection and test periods:** folds before the last five calendar years are the selection
  period, and the served model is the one of HAR-RV, ridge and boosting with the lowest mean QLIKE
  there. The last five years are the test period, used for no choice; reported scores come from it.
* **Test rows:** the headline uses every test row. A second view drops rows whose scheme had a
  suspect NAV in the 252 trading days up to the origin (known at forecast time). Dropping rows by
  their target window uses the future and is shown only as a diagnostic.
* **Losses:** QLIKE on variance and MAE on volatility; Diebold-Mariano tests of the served model
  against each other model on the monthly mean QLIKE differential (Newey-West lag 1, HLN correction).
* The served model, its config, seed and scores live in a hash-checked registry
  (`models/registry/`); `reports/vol_model_report.md` has the latest results.

## Known biases and limitations

* **Survivorship:** only schemes alive on the download date are included; merged and closed
  schemes are missing, which biases category averages upward.
* **Survivor selection in the volatility backtest:** schemes are chosen for having five years of
  history as of the download date, so every test year contains only funds that survived to 2026.
* **Daily data only:** no intraday prices.
* **Expense ratio timing:** TER history covers only the downloaded months; older windows are
  shown with the latest known TER.
* **SEBI 2018 re-categorisation:** a scheme's current category may not describe its earlier portfolio.
* **No transaction costs, exit loads or taxes.**
* **Benchmarks by category:** sectoral/thematic and focused schemes use a broad-market proxy.
* **PRI benchmarks,** if loaded, understate benchmark total returns (responses warn).
* Not investment advice.
