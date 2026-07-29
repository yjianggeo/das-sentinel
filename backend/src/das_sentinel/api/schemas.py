from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    field_validator,
    model_validator,
)

from das_sentinel.config import ProcessingProfile


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


WorkerId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]
PipelineVersion = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]


class LoginRequest(StrictSchema):
    password: SecretStr = Field(max_length=1024)


class SessionResponse(StrictSchema):
    admin_id: int
    csrf_token: str


class ProfileCreate(ProcessingProfile):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProfilePatch(StrictSchema):
    block_seconds: float | None = None
    overlap_seconds: float | None = None
    device: Literal["auto", "cpu", "cuda"] | None = None
    common_mode_removal: bool | None = None
    bad_channel_rms_z: float | None = None
    stalta_freq_low_hz: float | None = None
    stalta_freq_high_hz: float | None = None
    sta_seconds: float | None = None
    lta_seconds: float | None = None
    stalta_threshold: float | None = None
    stalta_strong_ratio: float | None = None
    min_coincidence: int | None = None
    min_duration_seconds: float | None = None
    energy_bands_hz: tuple[tuple[float, float], ...] | None = None
    energy_window_seconds: float | None = None
    energy_ratio_threshold: float | None = None
    coherence_freq_low_hz: float | None = None
    coherence_freq_high_hz: float | None = None
    coherence_window_seconds: float | None = None
    coherence_threshold: float | None = None
    slowness_max_s_per_m: float | None = None
    slowness_points: int | None = None
    trajectory_freq_low_hz: float | None = None
    trajectory_freq_high_hz: float | None = None
    trajectory_velocity_min_mps: float | None = None
    trajectory_velocity_max_mps: float | None = None
    trajectory_threshold: float | None = None
    template_threshold: float | None = None
    fusion_normal_score: float | None = None
    fusion_strong_score: float | None = None
    merge_gap_seconds: float | None = None
    artifact_pad_seconds: float | None = None
    recursive_discovery: bool | None = None


class SiteCreate(StrictSchema):
    name: str = Field(min_length=1, max_length=200)
    data_path: str = Field(min_length=1)
    timezone: str
    cable_type: Literal["downhole", "urban", "subsea"]
    enabled: bool = True
    profile: ProfileCreate = Field(default_factory=ProfileCreate)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("name must not be blank")
        return stripped

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        return _iana_timezone(value)


class SitePatch(StrictSchema):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    data_path: str | None = Field(default=None, min_length=1)
    timezone: str | None = None
    cable_type: Literal["downhole", "urban", "subsea"] | None = None
    enabled: bool | None = None
    profile: ProfilePatch | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("name must not be blank")
        return stripped

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str | None) -> str | None:
        return None if value is None else _iana_timezone(value)

    @model_validator(mode="after")
    def require_change(self) -> SitePatch:
        if not self.model_fields_set:
            raise ValueError("at least one site field is required")
        return self


class ProfileResponse(StrictSchema):
    id: str
    version: int
    payload: dict[str, Any]
    is_current: bool
    created_at: datetime


class SiteResponse(StrictSchema):
    id: str
    name: str
    data_path: str
    timezone: str
    cable_type: Literal["downhole", "urban", "subsea"]
    enabled: bool
    created_at: datetime
    current_profile: ProfileResponse


class JobEnqueueRequest(StrictSchema):
    source_file_id: str = Field(min_length=1, max_length=36)
    processing_profile_id: str = Field(min_length=1, max_length=36)
    pipeline_version: PipelineVersion


class JobResponse(StrictSchema):
    id: str
    source_file_id: str
    processing_profile_id: str
    pipeline_version: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    attempt_count: int
    worker_id: str | None
    heartbeat_at: datetime | None
    progress_numerator: int
    progress_denominator: int
    error_code: str | None
    error_message: str | None
    retry_eligible: bool
    created_at: datetime
    updated_at: datetime


class WorkerClaimRequest(StrictSchema):
    worker_id: WorkerId


class JobClaimResponse(StrictSchema):
    id: str
    source_file_id: str
    source_path: str
    source_fingerprint: str
    site_id: str
    site_timezone: str
    cable_type: str
    processing_profile_id: str
    profile_version: int
    profile_payload: dict[str, Any]
    pipeline_version: str
    attempt_count: int
    worker_id: str
    heartbeat_at: datetime


class WorkerHeartbeatRequest(StrictSchema):
    worker_id: WorkerId
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_progress(self) -> WorkerHeartbeatRequest:
        if self.numerator > self.denominator:
            raise ValueError("numerator must not exceed denominator")
        return self


ArtifactPath = Annotated[str, Field(min_length=1, max_length=4096)]


class WorkerCompleteRequest(StrictSchema):
    worker_id: WorkerId
    profile_version: int = Field(ge=1)
    pipeline_version: PipelineVersion
    artifact_paths: list[ArtifactPath] = Field(default_factory=list, max_length=1000)


class WorkerFailRequest(StrictSchema):
    worker_id: WorkerId
    error_code: str = Field(min_length=1, max_length=128)
    error_message: str = Field(min_length=1, max_length=4096)
    retry_eligible: bool = True


def _iana_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError("timezone must be a valid IANA timezone") from error
    return value
