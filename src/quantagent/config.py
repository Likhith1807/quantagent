"""Runtime settings, read from QUANTAGENT_* environment variables (or a .env file)."""
from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Hard caps baked into the tool schemas; the settings below may only lower them.
HARD_MAX_RESULTS = 50
HARD_MAX_FUNDS = 10


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QUANTAGENT_", env_file=".env", extra="ignore")

    db_url: str = "sqlite:///data/db/quantagent.db"
    risk_free_annual: float = Field(0.065, ge=0, le=0.2)
    risk_free_source: str = "Config default (assumption): approx. RBI 91-day T-bill yield, FY2024-25"
    log_level: str = "INFO"
    max_results: int = Field(HARD_MAX_RESULTS, ge=1, le=HARD_MAX_RESULTS)
    max_funds: int = Field(HARD_MAX_FUNDS, ge=2, le=HARD_MAX_FUNDS)
    max_range_years: int = Field(25, ge=1, le=50)
    timeout_s: float = Field(10.0, gt=0, le=300)
    model_dir: str = "models/registry"
    staging_dir: str = "data/staging"  # downloaded CSVs (fetch -> ingest)
    external_dir: str = "data/external"  # hand-downloaded Nifty TRI / T-bill files
    reference_dir: str = "data/reference"
    reports_dir: str = "reports"
    # Agent only. SecretStr: repr/str/model_dump show '**********', so it cannot leak into logs.
    anthropic_api_key: SecretStr | None = Field(None, validation_alias=AliasChoices("ANTHROPIC_API_KEY"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
