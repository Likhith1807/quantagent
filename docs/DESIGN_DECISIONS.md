# Design decisions

Why the system is built the way it is, what did not work, and what would change with better data.
Numbers quoted here come from `reports/vol_model_report.md` (model `vol21-har-20261021095439`, data
version 3) and `reports/eval_report.md`.

## 1. Why the server is read-only

The server answers questions for an LLM. An LLM chooses the arguments, so every tool argument is
untrusted input. Making the whole server unable to write removes a class of failures instead of
defending against each one.

* **One writer.** Only `quantagent.data.ingest` writes, and it builds a new file and swaps it in
  atomically (`os.replace`). The server never holds a write handle.
* **Three independent locks on SQLite:** the file is opened with `mode=ro`, every connection runs
  `PRAGMA query_only=ON`, and a progress handler aborts statements that run past
  `QUANTAGENT_TIMEOUT_S`. Postgres connections get `default_transaction_read_only=on` and a
  statement timeout.
* **No SQL from the model.** Tools take constrained types (6-digit codes, enums, dates, a
  whitelisted character set for search text, escaped before `LIKE`). Queries are SQLAlchemy Core
  expressions; ruff's S608 rule fails CI on string-built SQL. Unknown argument names are rejected
  rather than silently dropped.
* **Reproducibility.** Every response carries `data_version` and `calc_version`, so an answer can be
  reproduced against the same database. A mutable server would make that impossible.
* **MCP annotations** mark every tool `read_only_hint=True, destructive_hint=False`, so a client can
  run them without confirmation prompts.

## 2. Why HAR-RV (and why it is the served model)

The volatility model forecasts annualised realised volatility over the next 21 trading days.

* **HAR-RV** (Corsi 2009) regresses future realised volatility on daily, weekly and monthly
  realised volatility. It approximates the long memory of volatility with three averages and very
  few parameters. Fitted in logs and pooled across 308 schemes, it is hard to overfit.
* **Ridge and gradient boosting** used HAR's features plus 63- and 252-day realised volatility,
  EWMA, 21- and 63-day returns and drawdown. Both were evaluated with the same walk-forward split.
* **How it was chosen:** the folds from 2014 to 2021 are a selection period. HAR-RV had the lowest
  QLIKE there among the three regression models (0.667, against 0.731 for boosting and 0.777 for ridge),
  so it is served. The folds from 2022 to 2026 are a test period that played no part in the choice.
* **Result on the test period** (308 schemes, 57 months, 16,637 rows, every row scored): HAR-RV QLIKE
  0.252, against 0.280 for EWMA (-10.3%), 0.278 for GARCH (-9.4%) and 0.366 for naive (-31%). Against
  naive the difference is significant (DM p = 0.016); against EWMA (p = 0.20) and GARCH (p = 0.15) it is
  not. EWMA has a lower median QLIKE than HAR-RV, so HAR-RV wins on average rather than row by row.
  Ridge (0.267, p = 0.20) and boosting (0.297, p = 0.026, worse) did not beat HAR-RV.
* **Why GARCH was not a serving candidate:** it needs a per-scheme maximum-likelihood fit at request
  time, while HAR-RV is one pooled regression with four coefficients. On the selection years GARCH's
  QLIKE was slightly lower than HAR-RV's (0.651 vs 0.667); on the test years it was higher. Restricting
  the candidates to the regression models is a serving decision and is stated as one.
* **Log fit and bias correction:** HAR-RV, ridge and boosting are fitted on log volatility, which keeps
  forecasts positive and tames the skew of volatility. `exp(E[log vol])` underestimates `E[vol]`, so
  each forecast is multiplied by Duan's smearing factor, the mean of `exp(residual)` on that fold's
  training rows.
* **PyTorch was not tried.** The tree model, which can fit non-linearities in the same features,
  already lost to a three-feature linear model. A neural network on the same daily features is
  unlikely to change that, and adding it would add a large dependency for no measured benefit.

## 3. How leakage was prevented

* **Features use only the past.** Every feature at origin t is a rolling statistic ending at t.
  A unit test rewrites all returns after t and checks that features up to t are unchanged.
* **Purge and embargo.** Folds are calendar years. A training row is used only if its whole
  21-day target window ends at least 30 days before the test year starts, so no training target
  overlaps a test target or its neighbourhood. A test checks this for every fold.
* **Refit per fold.** Ridge, boosting and HAR are refitted from scratch on each fold. GARCH
  parameters are estimated on returns up to the fold cut-off only; the conditional variance is then
  filtered forward through data that is in the past at each origin.
* **Non-overlapping test origins.** Test rows are sampled every 21 observations per scheme. DM
  tests use the monthly cross-sectional mean loss with Newey-West lag 1, because adjacent months'
  targets can still overlap.
* **Data-quality filter on training rows only.** Training excludes rows whose target window contains
  a NAV flagged by the existing `check_data_quality` rules (a daily move above 15%, or 5+ identical
  NAVs). Every training target ends before the fold's cut-off, so this uses only information
  available at fit time. The thresholds predate the model.
* **No filter on test rows that uses the future.** An earlier version dropped test rows whose target
  window contained a suspect NAV and made that the headline. That uses information from after the
  forecast origin, so the headline is now every test row. A second view drops rows whose scheme had a
  suspect NAV in the year before the origin (known at forecast time); on the test years it removes no
  rows. The ex-post filter is kept only as a labelled diagnostic.
* **Selection and testing on different years.** An earlier version picked the served model on the
  same folds it reported, which flatters the chosen model. Now the model is picked on 2014 to 2021 and
  scored on 2022 to 2026. The final model is then refitted on all clean rows for serving.

## 4. What failed

* **The first training run selected the wrong model.** The rule chose the better of ridge and
  boosting (gradient boosting, QLIKE 0.522) although HAR-RV beat both (0.465), so a worse model would
  have been served. The rule now includes HAR-RV. Both runs are in `reports/experiments.jsonl` and
  both versions stay in the registry.
* **The model was chosen on its own test results, and test rows were filtered with future
  information.** A review caught both. The fixes are the selection/test split and the all-rows
  headline described in section 3. HAR-RV is still the choice, and its advantage over EWMA is still
  not significant.
* **Two tools disagreed on beta.** With real benchmarks loaded, `get_risk_metrics` and
  `get_benchmark_comparison` gave betas of 0.951 and 0.958 for the same fund and window. The first
  computed returns before aligning dates, pairing a two-day fund return with a one-day index return
  whenever one series skipped a day. It now aligns prices first, and a real-data test and an
  evaluation question check both tools against independent pandas.
* **QLIKE was dominated by data errors.** In the first run, 1% of test rows produced 70% of the
  total EWMA QLIKE. One target window implied 579% annualised volatility (a NAV jump) and one
  scheme had stale NAVs (0.7%). No DM test was significant even for large mean differences.
  Dropping the 14 of 34,649 test rows whose target window contained a suspect NAV cut EWMA's QLIKE
  from 0.873 to 0.338, but that filter used the future (see above). All 14 rows fall in 2014 to 2021, so under the current protocol they affect only the
  selection-period scores (EWMA 1.42 there).
* **Real benchmarks could not be downloaded by script.** niftyindices.com and RBI DBIE need a
  browser session, so Nifty TRI and T-bill files are downloaded by hand and loaded with
  `quantagent load-manual`. Without them, benchmark tools return NOT_FOUND and the risk-free rate is a
  labelled assumption. The files used here were downloaded on 2026-10-21.
* **Evaluation found two of its own bugs** (`docs/eval_fixes.md`): the independent reference
  lacked the zero-denominator rule, and the answer verifier treated scheme codes as numeric claims.

## 5. What would change with intraday data

* **Realised volatility from 5-minute returns** instead of squared daily returns. Daily RV is a
  noisy estimate of a day's variance (one squared return), which is why every model here scores a
  QLIKE far above zero. Intraday RV would cut target noise and make model differences testable with
  far fewer months.
* **HAR on intraday RV, with jumps separated** (bipower variation, HAR-RV-J or HAR-CJ), and
  semivariances for the leverage effect (SHAR).
* **Shorter horizons** (1 and 5 days) become meaningful, and microstructure noise needs handling:
  sampling at 5 minutes, or realised kernels.
* **Mutual funds publish one NAV a day,** so intraday data would come from the funds' holdings or
  from tracking indices and ETFs. Fund volatility would then be built bottom-up from the portfolio,
  which also needs the monthly portfolio disclosures.
* **Storage and serving:** intraday data does not fit the SQLite design. It would move to a columnar
  store (Parquet or DuckDB), with the same single-writer, read-only-server split.
