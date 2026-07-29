from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from das_sentinel.config import ProcessingProfile


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginRequest(StrictSchema):
    password: SecretStr


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


def _iana_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError("timezone must be a valid IANA timezone") from error
    return value
