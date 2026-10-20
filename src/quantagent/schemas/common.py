"""Shared input types and the response envelope every tool returns."""
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- inputs (constrained; never raw SQL)
SchemeCode = Annotated[str, Field(pattern=r"^\d{6}$", description="6-digit AMFI scheme code, e.g. '119551'.")]
SearchText = Annotated[str, Field(
    min_length=1, max_length=100, pattern=r"^[A-Za-z0-9 &.,()/+-]+$",
    description="Plain text: letters, digits, spaces and & . , ( ) / + - only.")]
BenchmarkName = Annotated[str, Field(
    pattern=r"^[A-Za-z0-9_]{1,40}$",
    description="Benchmark series name as listed by get_data_provenance, e.g. 'NIFTY100'.")]
Period = Literal["1y", "3y", "5y", "max"]
PlanType = Literal["Direct", "Regular"]
OptionType = Literal["Growth", "IDCW", "NA"]
StartDate = Annotated[date | None, Field(description="Inclusive start date (YYYY-MM-DD). Use instead of period.")]
EndDate = Annotated[date | None, Field(description="Inclusive end date (YYYY-MM-DD). Defaults to the latest data date.")]


# ---------------------------------------------------------------- envelope
class DateRange(BaseModel):
    start: date | None
    end: date | None


class Meta(BaseModel):
    period: DateRange | None = Field(None, description="Date range the numbers were computed over.")
    n_obs: int | None = Field(None, description="NAV observations used.")
    data_as_of: str
    data_version: int
    calc_version: str
    data_nature: Literal["synthetic", "real"]
    data_source: str | None = Field(None, description="Where the series came from (see get_data_provenance).")
    data_retrieved_at: str | None = Field(None, description="When the source data was downloaded.")
    assumptions: list[str]
    request_id: str


class Excluded(BaseModel):
    id: str
    reason: str


class ToolResult(BaseModel):
    meta: Meta
    warnings: list[str] = Field(default_factory=list)
    excluded: list[Excluded] = Field(default_factory=list)
