import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessingProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    block_seconds: float = 20.0
    overlap_seconds: float = 5.0
    device: Literal["auto", "cpu", "cuda"] = "auto"
    common_mode_removal: bool = True
    bad_channel_rms_z: float = 8.0
    stalta_freq_low_hz: float = 2.0
    stalta_freq_high_hz: float = 40.0
    sta_seconds: float = 0.5
    lta_seconds: float = 10.0
    stalta_threshold: float = 4.0
    stalta_strong_ratio: float = 8.0
    min_coincidence: int = 4
    min_duration_seconds: float = 0.2
    energy_bands_hz: tuple[tuple[float, float], ...] = (
        (0.5, 5.0),
        (5.0, 20.0),
        (20.0, 80.0),
    )
    energy_window_seconds: float = 2.0
    energy_ratio_threshold: float = 4.0
    coherence_freq_low_hz: float = 1.0
    coherence_freq_high_hz: float = 40.0
    coherence_window_seconds: float = 2.0
    coherence_threshold: float = 0.08
    slowness_max_s_per_m: float = 0.002
    slowness_points: int = 21
    trajectory_freq_low_hz: float = 1.0
    trajectory_freq_high_hz: float = 30.0
    trajectory_velocity_min_mps: float = 2.0
    trajectory_velocity_max_mps: float = 60.0
    trajectory_threshold: float = 0.65
    template_threshold: float = 0.75
    fusion_normal_score: float = 0.60
    fusion_strong_score: float = 0.85
    merge_gap_seconds: float = 1.0
    artifact_pad_seconds: float = 5.0
    recursive_discovery: bool = True

    @model_validator(mode="after")
    def validate_windows_and_bands(self) -> "ProcessingProfile":
        numeric_values = (
            value
            for name, value in self.model_dump().items()
            if name not in {"device", "common_mode_removal", "recursive_discovery"}
            and name != "energy_bands_hz"
        )
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError("all numeric profile values must be finite")
        if any(
            not math.isfinite(value)
            for band in self.energy_bands_hz
            for value in band
        ):
            raise ValueError("all energy-band values must be finite")

        if self.block_seconds <= 0 or not 0 <= self.overlap_seconds < self.block_seconds:
            raise ValueError("overlap_seconds must be nonnegative and smaller than block_seconds")
        if self.sta_seconds <= 0 or self.lta_seconds <= self.sta_seconds:
            raise ValueError("lta_seconds must be greater than sta_seconds")
        for low, high in (
            (self.stalta_freq_low_hz, self.stalta_freq_high_hz),
            (self.coherence_freq_low_hz, self.coherence_freq_high_hz),
            (self.trajectory_freq_low_hz, self.trajectory_freq_high_hz),
        ):
            if low <= 0 or high <= low:
                raise ValueError("every frequency range must satisfy 0 < low < high")
        if self.energy_window_seconds <= 0 or self.coherence_window_seconds <= 0:
            raise ValueError("sampling windows must be positive")
        for value in (
            self.bad_channel_rms_z,
            self.stalta_threshold,
            self.stalta_strong_ratio,
            self.min_duration_seconds,
            self.energy_ratio_threshold,
            self.slowness_max_s_per_m,
        ):
            if value <= 0:
                raise ValueError("profile counts, ratios, and slowness must be positive")
        if self.min_coincidence <= 0:
            raise ValueError("min_coincidence must be positive")
        if self.slowness_points < 3 or self.slowness_points % 2 == 0:
            raise ValueError("slowness_points must be odd and at least 3")
        for value in (
            self.coherence_threshold,
            self.trajectory_threshold,
            self.template_threshold,
            self.fusion_normal_score,
            self.fusion_strong_score,
        ):
            if not 0 < value <= 1:
                raise ValueError("probability-like thresholds must satisfy 0 < value <= 1")
        if self.trajectory_velocity_min_mps <= 0 or not (
            self.trajectory_velocity_min_mps < self.trajectory_velocity_max_mps
        ):
            raise ValueError("trajectory velocities must be positive and ordered")
        if self.fusion_normal_score > self.fusion_strong_score:
            raise ValueError("fusion_normal_score must not exceed fusion_strong_score")
        if self.merge_gap_seconds < 0 or self.artifact_pad_seconds < 0:
            raise ValueError("merge and artifact padding must be nonnegative")
        if len(self.energy_bands_hz) < 2:
            raise ValueError("energy_bands_hz must contain at least two bands")
        previous_high = 0.0
        for low, high in self.energy_bands_hz:
            if low <= 0 or high <= low:
                raise ValueError("every frequency band must satisfy 0 < low < high")
            if low < previous_high:
                raise ValueError("energy bands must be ordered and non-overlapping")
            previous_high = high
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DAS_", env_file=".env")

    data_root: Path = Path("/data")
    app_data_dir: Path = Path("/app-data")
    database_url: str = "sqlite+pysqlite:////app-data/das-sentinel.sqlite3"
    api_url: str = "http://127.0.0.1:8000"
    session_secret: str = Field(min_length=32)
    worker_token: str = Field(min_length=32)
    secure_cookies: bool = True
    pipeline_version: str = "v1"
    worker_device: Literal["auto", "cpu", "cuda"] = "auto"
