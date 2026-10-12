from datetime import UTC
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class TimeRange(BaseModel):
    start_utc: AwareDatetime
    end_utc: AwareDatetime

    @model_validator(mode="after")
    def ordered(self):
        if self.end_utc <= self.start_utc:
            raise ValueError("end must be after start")
        self.start_utc, self.end_utc = self.start_utc.astimezone(UTC), self.end_utc.astimezone(UTC)
        return self


class ModelScanCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=1, max_length=100)
    directories: list[str] = Field(min_length=1, max_length=12)
    include_rule_skips: Literal[True] = True
    expected_model_id: UUID
    time_ranges: list[TimeRange] | None = Field(default=None, min_length=1, max_length=16)


class ScanOwned(BaseModel):
    model_config = ConfigDict(extra="forbid")
    worker_id: str = Field(min_length=1, max_length=200)
    attempt_count: int = Field(ge=1)


class ScanClaim(BaseModel):
    worker_id: str = Field(min_length=1, max_length=200)


class Prediction(ScanOwned):
    result: dict


class DiscoveryPage(ScanOwned):
    page_index: int = Field(ge=0)
    payload: dict


class DiscoveryDone(ScanOwned):
    page_count: int = Field(ge=0)
    manifest_sha256: str = Field(pattern="^[0-9a-f]{64}$")


class PlanDone(ScanOwned):
    blocks: list[dict] = Field(max_length=10)


class Materialized(ScanOwned):
    binding: dict


class ScanFail(ScanOwned):
    error_code: str = Field(min_length=1, max_length=100)
    error_message: str = Field(default="", max_length=2000)
