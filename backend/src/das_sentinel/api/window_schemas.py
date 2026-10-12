from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Labels(BaseModel):
    model_config = ConfigDict(extra="forbid")
    earthquake: Literal["present", "absent", "unknown"]


class WindowReviewCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis_id: str = Field(min_length=1, max_length=36)
    input_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    labels: Labels
    note: str = Field(default="", max_length=10000)
    request_id: str = Field(min_length=1, max_length=100)


class WindowFavoriteSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    favorite: bool = Field(strict=True)


class WindowCoverageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis_id: str = Field(min_length=1, max_length=36)
    input_sha256: str = Field(pattern="^[0-9a-f]{64}$")


class WindowClaim(BaseModel):
    worker_id: str = Field(min_length=1, max_length=200)
    supports_empty_skip: bool = False


class WindowOwned(BaseModel):
    worker_id: str = Field(min_length=1, max_length=200)
    attempt_count: int = Field(ge=1)


class WindowSkipCreate(WindowOwned):
    plan_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    reason: Literal["no_qualified_regions"] = "no_qualified_regions"


class WindowFail(WindowOwned):
    error_code: str = Field(min_length=1, max_length=100)
    error_message: str = Field(default="", max_length=10000)
