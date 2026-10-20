# Design decisions

Why the system is built the way it is, what did not work, and what would change with better data.
Numbers quoted here come from `reports/vol_model_report.md` and `reports/eval_report.md`.

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
* **Result on real data** (308 schemes, 153 test months, 2014 to 2026, clean target windows):
  HAR-RV QLIKE 0.311, against 0.338 for EWMA (-7.8%) and 0.394 for naive (-20.9%). Against naive
  the difference is significant (DM p = 0.005); against EWMA it is not (p = 0.33). GARCH(1,1) has
  a slightly lower mean QLIKE (0.308), but HAR-RV has the lower median QLIKE (0.104 vs 0.130) and
  MAE, and wins in 65% of months; the DM test cannot separate them (p = 0.69). Ridge (0.326) and
  boosting (0.341) did not beat HAR-RV.
* **Why HAR-RV is served, not GARCH:** the two are statistically tied. HAR-RV is one pooled
  regression with four coefficients, so a forecast is instant and identical for every scheme's
  history. GARCH needs a per-scheme maximum-likelihood fit at request time. The selection rule picks the lowest QLIKE among HAR-RV,
  ridge and boosting, so a future ML model is served only if it actually wins.
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
* **Data-quality filter defined in advance.** Training excludes target windows that contain a NAV
  flagged by the existing `check_data_quality` rules (a daily move above 15%, or 5+ identical
  NAVs). The thresholds predate the model; results are reported both with and without the filter.
* **Selection on the walk-forward results only.** The served model is the winner of the
  out-of-sample comparison, then refitted on all clean rows. There is no separate hold-out year, so
  the reported scores are slightly optimistic for the chosen model. With three candidates and a clear
  winner, the effect is small, but it is stated here rather than hidden.

## 4. What failed

* **The first training run selected the wrong model.** The rule chose the better of ridge and
  boosting (gradient boosting, QLIKE 0.522) although HAR-RV beat both (0.465), so a worse model would
  have been served. The rule now includes HAR-RV. Both runs are in `reports/experiments.jsonl` and
  both versions stay in the registry.
* **QLIKE was dominated by data errors.** In the first run, 1% of test rows produced 70% of the
  total EWMA QLIKE. One target window implied 579% annualised volatility (a NAV jump) and one
  scheme had stale NAVs (0.7%). No DM test was significant even for large mean differences. The
  clean-window rule above removes 14 of 34,649 test rows and cuts EWMA's QLIKE from 0.873 to 0.338.
  The unfiltered numbers are still reported.
* **Real benchmarks could not be downloaded by script.** niftyindices.com and FRED block
  automated requests, so Nifty TRI and T-bill files are downloaded by hand and loaded with
  `quantagent load-manual`. Until they are loaded, benchmark tools return NOT_FOUND and the
  risk-free rate is a labelled assumption.
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
