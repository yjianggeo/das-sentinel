from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    SecretStr,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from das_sentinel.config import ProcessingProfile


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WaterfallOptionsResponse(StrictSchema):
    automatic_distance_start_m: FiniteFloat
    automatic_distance_stop_m: FiniteFloat
    available_distance_start_m: FiniteFloat
    available_distance_stop_m: FiniteFloat
    primary_detector: str
    primary_detector_display: str
    filter_low_hz: FiniteFloat = Field(ge=0)
    filter_high_hz: FiniteFloat = Field(gt=0)
    distance_unit: Literal["m"] = "m"


class WaterfallPreviewRequest(StrictSchema):
    distance_start_m: Annotated[FiniteFloat, Field(strict=True)] | None = None
    distance_stop_m: Annotated[FiniteFloat, Field(strict=True)] | None = None

    @model_validator(mode="after")
    def validate_order(self) -> WaterfallPreviewRequest:
        if not self.model_fields_set:
            return self
        if self.distance_start_m is None or self.distance_stop_m is None:
            raise ValueError("both distance bounds must be supplied for a manual range")
        if self.distance_stop_m <= self.distance_start_m:
            raise ValueError("distance_stop_m must exceed distance_start_m")
        return self


class WaterfallPreviewResponse(StrictSchema):
    distance_start_m: FiniteFloat
    distance_stop_m: FiniteFloat
    channel_count: StrictInt = Field(ge=1)
    primary_detector: str
    primary_detector_display: str
    filter_low_hz: FiniteFloat = Field(ge=0)
    filter_high_hz: FiniteFloat = Field(gt=0)
    raw_png_base64: str = Field(min_length=8, max_length=16 * 1024 * 1024)
    filtered_png_base64: str = Field(min_length=8, max_length=16 * 1024 * 1024)
    waveform_png_base64: str = Field(min_length=8, max_length=16 * 1024 * 1024)


WorkerId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]
ProgressInteger = Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]
PipelineVersion = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]


class LoginRequest(StrictSchema):
    password: SecretStr = Field(max_length=1024)


class SessionResponse(StrictSchema):
    admin_id: int
    csrf_token: str


class DashboardResponse(StrictSchema):
    job_counts: dict[str, int]
    today_event_count: int
    machine_label_counts: dict[str, int]


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


class ScanRequest(StrictSchema):
    path: Annotated[
        str,
        StringConstraints(min_length=1, max_length=1024),
    ]

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("path must not be blank")
        return value


class ScanDirectoryEntryResponse(StrictSchema):
    name: str
    path: str


class ScanDirectoryResponse(StrictSchema):
    site_id: str
    site_name: str
    path: str
    parent_path: str | None
    directories: list[ScanDirectoryEntryResponse]


class ScanResponse(StrictSchema):
    id: str
    site_id: str
    path: str
    phase: Literal[
        "initial_scan",
        "stability_wait",
        "verification_scan",
        "enqueueing",
        "completed",
        "failed",
    ]
    initial_checked_files: int = Field(ge=0)
    initial_found_files: int = Field(ge=0)
    verification_checked_files: int = Field(ge=0)
    verification_found_files: int = Field(ge=0)
    stable_files: int = Field(ge=0)
    submitted_files: int = Field(ge=0)
    skipped_files: int = Field(ge=0)
    stability_seconds_remaining: int = Field(ge=0)
    error_message: str | None


class JobEnqueueRequest(StrictSchema):
    source_file_id: str = Field(min_length=1, max_length=36)
    processing_profile_id: str = Field(min_length=1, max_length=36)
    pipeline_version: PipelineVersion


class CancelledJobsDeleteResponse(StrictSchema):
    deleted_count: int = Field(ge=0)


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


class JobListItemResponse(JobResponse):
    source_filename: str
    source_directory: str
    site_id: str
    site_name: str


class JobPageResponse(StrictSchema):
    items: list[JobListItemResponse]
    total: int
    global_counts: dict[str, int]


class JobDirectoryResponse(StrictSchema):
    site_id: str
    site_name: str
    directory: str
    job_count: int


class WorkerClaimRequest(StrictSchema):
    worker_id: WorkerId


class JobClaimResponse(StrictSchema):
    id: str
    source_file_id: str
    source_path: str
    source_fingerprint: str
    source_size_bytes: int
    source_mtime_ns: int
    site_id: str
    site_data_root: str
    site_timezone: str
    cable_type: str
    processing_profile_id: str
    profile_version: int
    profile_payload: dict[str, Any]
    pipeline_version: str
    model_version_id: str | None
    model_version: str | None
    model_artifact_path: str | None
    model_bundle_sha256: str | None
    model_feature_names: tuple[str, ...] | None
    attempt_count: int
    worker_id: str
    heartbeat_at: datetime

    @model_validator(mode="after")
    def validate_model_binding(self) -> JobClaimResponse:
        values = (
            self.model_version_id,
            self.model_version,
            self.model_artifact_path,
            self.model_bundle_sha256,
            self.model_feature_names,
        )
        if any(value is None for value in values) and not all(
            value is None for value in values
        ):
            raise ValueError("all model binding fields must be present or absent")
        return self


class WorkerHeartbeatRequest(StrictSchema):
    worker_id: WorkerId
    numerator: ProgressInteger
    denominator: ProgressInteger

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


class WorkerEvidenceResult(StrictSchema):
    detector: str = Field(min_length=1, max_length=64)
    score: FiniteFloat = Field(ge=0, le=1)
    payload: dict[str, Any] = Field(default_factory=dict, max_length=128)

    @field_validator("payload")
    @classmethod
    def validate_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _bounded_json(value, field="evidence payload", maximum=32_768)


class WorkerClassificationResult(StrictSchema):
    label: Literal["earthquake", "blast", "traffic", "noise", "unknown"]
    decision_score: FiniteFloat = Field(ge=0, le=1)
    method: str = Field(min_length=1, max_length=64)
    reasons: list[str] = Field(default_factory=list, max_length=64)

    @field_validator("method")
    @classmethod
    def validate_method(cls, value: str) -> str:
        return _nonblank(value, field="classification method")

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, value: list[str]) -> list[str]:
        return [
            _nonblank(item, field="classification reason", maximum=512)
            for item in value
        ]


class WorkerEventResult(StrictSchema):
    event_uid: str = Field(min_length=1, max_length=128)
    start_sample: StrictInt = Field(ge=0, le=2**63 - 2)
    end_sample: StrictInt = Field(ge=1, le=2**63 - 1)
    channel_start: StrictInt = Field(ge=0, le=2**31 - 2)
    channel_stop: StrictInt = Field(ge=1, le=2**31 - 1)
    features: dict[str, FiniteFloat | str] = Field(default_factory=dict, max_length=512)
    evidence: list[WorkerEvidenceResult] = Field(
        default_factory=list,
        max_length=64,
    )
    classification: WorkerClassificationResult

    @field_validator("event_uid")
    @classmethod
    def validate_event_uid(cls, value: str) -> str:
        return _nonblank(value, field="event UID", maximum=128)

    @field_validator("features")
    @classmethod
    def validate_features(cls, value: dict[str, float | str]) -> dict[str, float | str]:
        for key, item in value.items():
            _nonblank(key, field="feature name", maximum=128)
            if isinstance(item, str):
                _nonblank(item, field="feature value", maximum=256)
        return value

    @model_validator(mode="after")
    def validate_ranges(self) -> WorkerEventResult:
        if self.end_sample <= self.start_sample:
            raise ValueError("event end_sample must exceed start_sample")
        if self.channel_stop <= self.channel_start:
            raise ValueError("event channel_stop must exceed channel_start")
        return self


class WorkerArtifactResult(StrictSchema):
    event_uid: str = Field(min_length=1, max_length=128)
    kind: Literal[
        "waterfall_raw",
        "waterfall_filtered",
        "waveform",
        "psd",
        "stft",
        "evidence",
    ]
    path: str = Field(min_length=1, max_length=4096)
    media_type: Literal["image/png", "application/json"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("event_uid", "path")
    @classmethod
    def validate_nonblank_fields(cls, value: str) -> str:
        return _nonblank(value, field="artifact field", maximum=4096)


class WorkerResultRequest(StrictSchema):
    worker_id: WorkerId
    source_file_id: str = Field(min_length=1, max_length=36)
    processing_profile_id: str = Field(min_length=1, max_length=36)
    model_version_id: str | None = Field(default=None, max_length=36)
    profile_version: StrictInt = Field(ge=1, le=2**31 - 1)
    pipeline_version: PipelineVersion
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_start_time_utc: datetime
    sample_rate_hz: FiniteFloat = Field(gt=0, le=10_000_000)
    requested_backend: Literal["auto", "cpu", "cuda"]
    actual_backend: Literal["cpu", "cuda"]
    backend_fallback_reason: Literal[
        "cuda_unavailable",
        "cuda_out_of_memory",
    ] | None = None
    timings_ms: dict[str, FiniteFloat] = Field(default_factory=dict, max_length=64)
    events: list[WorkerEventResult] = Field(default_factory=list, max_length=10_000)
    artifacts: list[WorkerArtifactResult] = Field(
        default_factory=list,
        max_length=60_000,
    )

    @field_validator("source_start_time_utc")
    @classmethod
    def validate_source_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("source_start_time_utc must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("timings_ms")
    @classmethod
    def validate_timings(cls, value: dict[str, float]) -> dict[str, float]:
        for name, duration in value.items():
            _nonblank(name, field="timing name", maximum=64)
            if duration < 0:
                raise ValueError("timings must be nonnegative")
        return value

    @model_validator(mode="after")
    def validate_total_bounds(self) -> WorkerResultRequest:
        maximum_offset_seconds = 366 * 24 * 60 * 60
        if any(
            event.end_sample / self.sample_rate_hz > maximum_offset_seconds
            for event in self.events
        ):
            raise ValueError("event sample offsets exceed the bounded source duration")
        encoded = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        if len(encoded) > 16 * 1024 * 1024:
            raise ValueError("worker result exceeds the maximum encoded size")
        return self


class ReviewCreate(StrictSchema):
    label: Literal["earthquake", "blast", "traffic", "noise"]
    note: str | None = Field(default=None, max_length=2000)
    request_id: UUID | None = None

    @field_validator("note")
    @classmethod
    def validate_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class ReviewResponse(StrictSchema):
    id: int
    label: Literal["earthquake", "blast", "traffic", "noise"]
    note: str | None
    created_at: datetime


class EventSummaryResponse(StrictSchema):
    id: str
    event_uid: str
    site_id: str
    site_timezone: str
    occurred_at: datetime
    start_sample: int
    end_sample: int
    channel_start: int
    channel_stop: int
    machine_label: Literal[
        "earthquake",
        "blast",
        "traffic",
        "noise",
        "unknown",
    ]
    latest_review: ReviewResponse | None


class EventEvidenceResponse(StrictSchema):
    detector: str
    score: float
    payload: dict[str, Any]


class MachineClassificationResponse(StrictSchema):
    id: int
    label: Literal[
        "earthquake",
        "blast",
        "traffic",
        "noise",
        "unknown",
    ]
    decision_score: float
    method: str
    reasons: list[str]
    model_version_id: str | None
    created_at: datetime


class EventArtifactResponse(StrictSchema):
    id: str
    kind: str
    media_type: str


class EventRunResponse(StrictSchema):
    id: str
    source_file_id: str
    source_fingerprint: str
    processing_profile_id: str
    requested_backend: str
    actual_backend: str
    backend_fallback_reason: str | None
    profile_version: int
    pipeline_version: str
    model_version_id: str | None
    timings_ms: dict[str, float]
    started_at: datetime | None
    finished_at: datetime | None


class EventDetailResponse(EventSummaryResponse):
    features: dict[str, Any]
    evidence: list[EventEvidenceResponse]
    machine_history: list[MachineClassificationResponse]
    review_history: list[ReviewResponse]
    artifacts: list[EventArtifactResponse]
    run: EventRunResponse


class ModelResponse(StrictSchema):
    id: str
    site_id: str
    version: str
    status: Literal["candidate", "active", "retired", "invalid"]
    dataset_sha256: str | None
    bundle_sha256: str
    metrics: dict[str, Any]
    created_at: datetime
    published_at: datetime | None


def _iana_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError("timezone must be a valid IANA timezone") from error
    return value


def _nonblank(
    value: str,
    *,
    field: str,
    maximum: int = 64,
) -> str:
    stripped = value.strip()
    if not stripped or len(stripped) > maximum:
        raise ValueError(f"{field} must be nonblank and at most {maximum} characters")
    return stripped


def _bounded_json(
    value: dict[str, Any],
    *,
    field: str,
    maximum: int,
) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain finite JSON values") from error
    if len(encoded.encode()) > maximum:
        raise ValueError(f"{field} is too large")
    return value
