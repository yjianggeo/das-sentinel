from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import Candidate
from das_sentinel.signal.preprocess import PreprocessedBlock


def _normalize(values: np.ndarray) -> np.ndarray:
    centered = values.astype(np.float64, copy=False) - np.mean(values)
    norm = np.linalg.norm(centered)
    return centered / norm if norm > np.finfo(np.float64).eps else np.zeros_like(centered)


def _valid_text(value: str) -> bool:
    return bool(value and value.strip())


def _valid_sample_rate(sample_rate_hz: float) -> bool:
    return bool(np.isfinite(sample_rate_hz) and sample_rate_hz > 0)


@dataclass(frozen=True)
class Template:
    """A reviewed, site- and profile-scoped normalized beam waveform."""

    waveform: np.ndarray
    spectrum: np.ndarray
    site_id: str
    profile_version: str
    source_event_id: str
    sample_rate_hz: float

    @classmethod
    def from_event(
        cls,
        block: PreprocessedBlock,
        event_id: str,
        *,
        site_id: str = "unknown-site",
        profile_version: str = "unknown-profile",
    ) -> Template:
        if not _valid_text(event_id):
            raise ValueError("templates require a non-empty event_id")
        if not _valid_text(site_id) or not _valid_text(profile_version):
            raise ValueError("templates require non-empty site_id and profile_version")
        if not _valid_sample_rate(block.sample_rate_hz):
            raise ValueError("templates require a finite positive sample rate")
        waveform = _beam_waveform(block)
        if waveform.size == 0:
            raise ValueError("templates require at least one good channel")
        spectrum = np.abs(np.fft.rfft(waveform))
        spectrum_norm = np.linalg.norm(spectrum)
        if spectrum_norm > np.finfo(np.float64).eps:
            spectrum = spectrum / spectrum_norm
        return cls(
            waveform=waveform,
            spectrum=spectrum,
            site_id=site_id,
            profile_version=profile_version,
            source_event_id=event_id,
            sample_rate_hz=block.sample_rate_hz,
        )


def _beam_waveform(block: PreprocessedBlock) -> np.ndarray:
    good = np.flatnonzero(~block.bad_channel_mask)
    if good.size == 0:
        return np.empty(0, dtype=np.float64)
    return _normalize(np.asarray(block.data)[:, good].mean(axis=1))


def score_template(block: PreprocessedBlock, template: Template) -> float:
    """Return normalized waveform correlation, or zero for incompatible inputs."""
    if not _valid_text(template.source_event_id):
        return 0.0
    if not _valid_text(template.site_id) or not _valid_text(template.profile_version):
        return 0.0
    if not _valid_sample_rate(template.sample_rate_hz) or not _valid_sample_rate(block.sample_rate_hz):
        return 0.0
    if getattr(block, "site_id", template.site_id) != template.site_id:
        return 0.0
    if getattr(block, "profile_version", template.profile_version) != template.profile_version:
        return 0.0
    if not np.isclose(block.sample_rate_hz, template.sample_rate_hz):
        return 0.0
    waveform = _beam_waveform(block)
    if waveform.shape != template.waveform.shape or waveform.size == 0:
        return 0.0
    return float(np.clip(np.dot(waveform, template.waveform), 0.0, 1.0))


def _energy_image(
    block: PreprocessedBlock, channel_indices: np.ndarray, profile: ProcessingProfile
) -> np.ndarray:
    data = np.asarray(block.data)[:, channel_indices].astype(np.float64, copy=False)
    frequency = np.fft.rfftfreq(data.shape[0], d=1.0 / block.sample_rate_hz)
    band = (frequency >= profile.trajectory_freq_low_hz) & (
        frequency <= min(profile.trajectory_freq_high_hz, block.sample_rate_hz / 2.0)
    )
    filtered = np.fft.irfft(np.fft.rfft(data, axis=0) * band[:, None], n=data.shape[0], axis=0)
    energy = np.square(filtered)
    return np.clip(energy - np.median(energy, axis=1, keepdims=True), 0.0, None)


def _line_energy(
    energy: np.ndarray, distances_m: np.ndarray, sample_rate_hz: float, velocity_mps: float
) -> tuple[float, int, np.ndarray]:
    delays = np.rint((distances_m - distances_m[0]) / velocity_mps * sample_rate_hz).astype(int)
    sample_count = energy.shape[0]
    best_energy, best_start, best_indices = -1.0, 0, np.empty(0, dtype=int)
    for start in range(sample_count):
        indices = start + delays
        valid = (indices >= 0) & (indices < sample_count)
        if not valid.all():
            continue
        value = float(energy[indices, np.arange(energy.shape[1])].sum())
        if value > best_energy:
            best_energy, best_start, best_indices = value, start, indices
    return best_energy, best_start, best_indices


def detect_trajectories(block: PreprocessedBlock, profile: ProcessingProfile) -> list[Candidate]:
    """Score bounded, bidirectional distance-time energy lines with Hough-style voting."""
    good = np.flatnonzero(~block.bad_channel_mask)
    if good.size < profile.min_coincidence or block.data.shape[0] == 0:
        return []
    if profile.trajectory_velocity_min_mps <= 0 or (
        profile.trajectory_velocity_max_mps <= profile.trajectory_velocity_min_mps
    ):
        raise ValueError("trajectory velocity bounds must be positive and ordered")

    channel_step = max(1, int(np.ceil(good.size / 64)))
    channels = good[::channel_step]
    time_step = max(1, int(np.ceil(block.sample_rate_hz / (2.0 * profile.trajectory_freq_high_hz))))
    energy = _energy_image(block, channels, profile)[::time_step]
    total_energy = float(energy.sum())
    reference_energy = float(energy.max(axis=0).sum())
    if reference_energy <= np.finfo(np.float64).eps:
        return []
    distance = block.channel_distance_m[channels]
    velocities = np.linspace(
        profile.trajectory_velocity_min_mps, profile.trajectory_velocity_max_mps, 25
    )
    scored: list[tuple[float, float, float, int, np.ndarray]] = []
    for magnitude in velocities:
        for velocity in (-float(magnitude), float(magnitude)):
            line_energy, start, line_samples = _line_energy(
                energy, distance, block.sample_rate_hz / time_step, velocity
            )
            strength = line_energy / reference_energy
            if strength >= profile.trajectory_threshold:
                scored.append((strength, velocity, line_energy, start, line_samples))

    selected: list[tuple[float, float, float, int, np.ndarray]] = []
    for candidate in sorted(scored, reverse=True, key=lambda result: result[0]):
        _, _, _, start, line_samples = candidate
        if all(abs(start - prior[3]) > max(1, np.ptp(prior[4]) // 2) for prior in selected):
            selected.append(candidate)

    candidates: list[Candidate] = []
    for strength, velocity, line_energy, start, line_samples in selected:
        full_line_samples = line_samples * time_step
        peak = int(
            full_line_samples[np.argmax(energy[line_samples, np.arange(channels.size)])]
        )
        candidates.append(
            Candidate(
                detector="trajectory",
                start_sample=block.sample_offset + int(full_line_samples.min()),
                end_sample=block.sample_offset + int(full_line_samples.max()) + time_step,
                peak_sample=block.sample_offset + peak,
                channel_start=int(channels[0]),
                channel_stop=int(channels[-1]) + 1,
                score=float(strength),
                metrics={
                    "apparent_velocity_mps": abs(velocity),
                    "direction": float(np.sign(velocity)),
                    "trajectory_strength": float(strength),
                    "line_energy": line_energy,
                    "reference_line_energy": reference_energy,
                    "total_energy": total_energy,
                },
            )
        )
    return candidates
