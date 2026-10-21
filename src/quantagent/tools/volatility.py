from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import pandas as pd
from pydantic import Field

from ..errors import QuantError
from ..ml import registry
from ..ml.volatility import FEATURES, MIN_HISTORY, features, log_returns
from ..schemas.common import SchemeCode
from ..schemas.tools import VolForecastResult, VolModelInfo
from ._base import (
    IDCW_WARNING,
    base_warnings,
    build_meta,
    data_as_of,
    get_services,
    require_scheme,
    tool_handler,
)


@lru_cache(maxsize=4)
def _load(model_dir: str, version: str | None):
    return registry.load(Path(model_dir), version)


@tool_handler
def predict_volatility(
    scheme_code: SchemeCode,
    as_of: Annotated[date | None, Field(description="Forecast origin (YYYY-MM-DD); default the latest NAV date. "
                                                    "Only NAVs up to this date are used.")] = None,
) -> VolForecastResult:
    """Forecast a scheme's annualised volatility over the next 21 trading days.

    Use when asked how volatile a fund is likely to be next month, or for a forward-looking risk
    estimate. The model is a log-volatility regression on realised-volatility features (HAR-RV
    style), chosen on earlier walk-forward years and scored on later held-out years against naive,
    EWMA, HAR-RV and GARCH baselines; those held-out scores are returned in `model`. Do NOT use for historical volatility; use
    get_risk_metrics. Fails with MODEL_UNAVAILABLE if no model has been trained (scripts/train_vol.py).
    """
    svc = get_services()
    repo = svc.repo
    s = require_scheme(repo, scheme_code)
    try:
        objects, config, metrics = _load(svc.settings.model_dir, None)
    except (FileNotFoundError, ValueError) as e:
        raise QuantError("MODEL_UNAVAILABLE", "No usable volatility model is registered.",
                         {"model_dir": svc.settings.model_dir, "reason": str(e)[:200],
                          "hint": "Run scripts/train_vol.py."}) from None
    latest = data_as_of(repo)
    origin = pd.Timestamp(as_of) if as_of else latest
    if origin > latest:
        raise QuantError("INVALID_INPUT", f"as_of {origin.date()} is after the latest data date {latest.date()}.")
    if s["option_type"] == "IDCW":
        raise QuantError("INVALID_INPUT", IDCW_WARNING.format(code=scheme_code))
    nav = repo.nav_series(scheme_code, None, origin.date())
    r = log_returns(nav)
    if len(r) < MIN_HISTORY:
        raise QuantError("INSUFFICIENT_HISTORY", f"Need at least {MIN_HISTORY} daily returns before the origin.",
                         {"found": len(r), "required": MIN_HISTORY})
    row = features(r).iloc[[-1]]
    if row[FEATURES].isna().any(axis=None):
        raise QuantError("INSUFFICIENT_HISTORY", "Feature window incomplete at this origin.")

    m, mods = metrics["models"], config["model_kind"]
    dm = metrics.get("dm", {}).get(f"{mods}_vs_ewma") or {}
    warnings = base_warnings(repo)
    if config["data_version"] != repo.meta["data_version"] or config["data_nature"] != repo.meta.get("data_nature"):
        warnings.append(f"Model was trained on data version {config['data_version']} ({config['data_nature']}); "
                        f"the database is version {repo.meta['data_version']} ({repo.meta.get('data_nature')}).")
    if not metrics["selected"]["beats_all_baselines"]:
        warnings.append("The selected model did not beat every baseline out of sample; compare with `baselines`.")
    origin_date = r.index[-1]
    return VolForecastResult(
        meta=build_meta(repo, r.index[0], origin_date, len(nav),
                        [f"Forecast horizon {config['horizon']} trading days; NAVs after {origin_date.date()} unused."]),
        warnings=warnings, scheme_code=scheme_code, scheme_name=s["scheme_name"], origin_date=origin_date.date(),
        horizon_days=config["horizon"], forecast_vol=float(objects["model"].predict(row)[0]),
        baselines={"naive": float(row["rv_m"].iloc[0]), "ewma": float(row["ewma"].iloc[0]),
                   "har": float(objects["har"].predict(row)[0])},
        recent_realised_vol={"rv_22": float(row["rv_m"].iloc[0]), "rv_63": float(row["rv_q"].iloc[0])},
        model=VolModelInfo(
            version=config["version"], kind=mods, trained_data_version=config["data_version"],
            trained_data_nature=config["data_nature"], train_range=config["train_range"],
            oos_qlike=m[mods]["qlike"], oos_qlike_ewma=m["ewma"]["qlike"], oos_qlike_har=m["har"]["qlike"],
            beats_all_baselines=metrics["selected"]["beats_all_baselines"],
            dm_p_value_vs_ewma=dm.get("p_value")))
