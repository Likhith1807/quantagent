"""Reading the metrics cache for the current data version (built by ingest)."""
import pandas as pd

from ..errors import QuantError
from .repository import Repository


def current_cache(repo: Repository, window: str) -> pd.DataFrame:
    version = repo.meta["data_version"]
    df = repo.metrics_cache(window, version)
    if df.empty:
        raise QuantError("DATA_QUALITY", "Metrics cache is empty or stale for the current data version; "
                         "run scripts/build_metrics_cache.py.",
                         {"data_version": version, "cache_versions": repo.cache_versions()})
    return df
