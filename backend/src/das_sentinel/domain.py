from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import numpy as np


class Label(StrEnum):
    EARTHQUAKE = "earthquake"
    BLAST = "blast"
    TRAFFIC = "traffic"
    NOISE = "noise"
    UNKNOWN = "unknown"


class ComputeBackend(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True)
class DasMetadata:
    source_path: str
    start_time_utc: datetime
    sample_rate_hz: float
    channel_distance_m: np.ndarray
    unit: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DasBlock:
    data: np.ndarray
    metadata: DasMetadata
    sample_offset: int

    def __post_init__(self) -> None:
        if self.data.dtype != np.float32:
            raise ValueError("DasBlock data must be float32")
        if self.data.ndim != 2:
            raise ValueError("DasBlock data must have shape [time, channel]")
        if self.data.shape[1] != len(self.metadata.channel_distance_m):
            raise ValueError("channel distance count does not match data")


@dataclass(frozen=True)
class Candidate:
    detector: str
    start_sample: int
    end_sample: int
    peak_sample: int
    channel_start: int
    channel_stop: int
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    start_sample: int
    end_sample: int
    peak_sample: int
    channel_start: int
    channel_stop: int
    evidence: tuple[Candidate, ...]
    features: dict[str, float] = field(default_factory=dict)
    label: Label = Label.UNKNOWN
    decision_score: float = 0.0
    reasons: tuple[str, ...] = ()
