import pytest
from das_sentinel.config import ProcessingProfile, Settings
from pydantic import ValidationError


def test_profile_has_approved_block_defaults():
    profile = ProcessingProfile()

    assert profile.block_seconds == 20.0
    assert profile.overlap_seconds == 5.0
    assert profile.device == "auto"


@pytest.mark.parametrize("field", ["scan_distance_start_m", "scan_distance_stop_m"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "300"])
def test_scan_distance_rejects_invalid_bound(field, value):
    with pytest.raises(ValidationError):
        ProcessingProfile(**{field: value})


def test_scan_distance_supports_open_bounds_and_rejects_reversed_range():
    assert ProcessingProfile().scan_distance_start_m is None
    assert ProcessingProfile(scan_distance_start_m=300).scan_distance_stop_m is None
    assert ProcessingProfile(scan_distance_stop_m=900).scan_distance_start_m is None
    for stop in (299, 300):
        with pytest.raises(ValidationError):
            ProcessingProfile(scan_distance_start_m=300, scan_distance_stop_m=stop)


def test_profile_rejects_overlap_not_smaller_than_block():
    with pytest.raises(ValidationError):
        ProcessingProfile(block_seconds=5.0, overlap_seconds=5.0)


@pytest.mark.parametrize(
    ("low_field", "high_field", "low", "high"),
    [
        ("stalta_freq_low_hz", "stalta_freq_high_hz", 0.0, 40.0),
        ("stalta_freq_low_hz", "stalta_freq_high_hz", 40.0, 40.0),
        ("coherence_freq_low_hz", "coherence_freq_high_hz", 0.0, 40.0),
        ("coherence_freq_low_hz", "coherence_freq_high_hz", 40.0, 40.0),
        ("trajectory_freq_low_hz", "trajectory_freq_high_hz", 0.0, 30.0),
        ("trajectory_freq_low_hz", "trajectory_freq_high_hz", 30.0, 30.0),
    ],
)
def test_profile_rejects_invalid_frequency_order(low_field, high_field, low, high):
    with pytest.raises(ValidationError):
        ProcessingProfile(**{low_field: low, high_field: high})


@pytest.mark.parametrize("window_field", ["energy_window_seconds", "coherence_window_seconds"])
def test_profile_rejects_nonpositive_sampling_windows(window_field):
    with pytest.raises(ValidationError):
        ProcessingProfile(**{window_field: 0.0})


@pytest.mark.parametrize(
    "field",
    [
        "block_seconds",
        "overlap_seconds",
        "bad_channel_rms_z",
        "stalta_freq_low_hz",
        "stalta_freq_high_hz",
        "sta_seconds",
        "lta_seconds",
        "stalta_threshold",
        "stalta_strong_ratio",
        "min_coincidence",
        "min_duration_seconds",
        "energy_window_seconds",
        "energy_ratio_threshold",
        "coherence_freq_low_hz",
        "coherence_freq_high_hz",
        "coherence_window_seconds",
        "coherence_threshold",
        "slowness_max_s_per_m",
        "slowness_points",
        "trajectory_freq_low_hz",
        "trajectory_freq_high_hz",
        "trajectory_velocity_min_mps",
        "trajectory_velocity_max_mps",
        "trajectory_threshold",
        "template_threshold",
        "fusion_normal_score",
        "fusion_strong_score",
        "merge_gap_seconds",
        "artifact_pad_seconds",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_profile_rejects_non_finite_numeric_fields(field, value):
    with pytest.raises(ValidationError):
        ProcessingProfile(**{field: value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_coincidence": 0},
        {"min_duration_seconds": 0.0},
        {"bad_channel_rms_z": 0.0},
        {"stalta_threshold": 0.0},
        {"stalta_strong_ratio": 0.0},
        {"energy_ratio_threshold": 0.0},
        {"slowness_max_s_per_m": 0.0},
        {"coherence_threshold": 0.0},
        {"coherence_threshold": 1.01},
        {"trajectory_threshold": 0.0},
        {"trajectory_threshold": 1.01},
        {"template_threshold": 0.0},
        {"template_threshold": 1.01},
        {"fusion_normal_score": 0.0},
        {"fusion_strong_score": 1.01},
        {"fusion_normal_score": 0.9, "fusion_strong_score": 0.8},
        {"trajectory_velocity_min_mps": 0.0},
        {"trajectory_velocity_min_mps": 60.0, "trajectory_velocity_max_mps": 60.0},
        {"merge_gap_seconds": -0.1},
        {"artifact_pad_seconds": -0.1},
        {"energy_bands_hz": ()},
        {"energy_bands_hz": ((5.0, 10.0), (1.0, 2.0))},
        {"energy_bands_hz": ((1.0, 5.0), (4.0, 8.0))},
        {"energy_bands_hz": ((1.0, float("nan")),)},
    ],
)
def test_profile_rejects_non_runnable_values(overrides):
    with pytest.raises(ValidationError):
        ProcessingProfile(**overrides)


def test_profile_rejects_single_energy_band():
    with pytest.raises(ValidationError):
        ProcessingProfile(energy_bands_hz=((1.0, 5.0),))


def test_settings_default_to_current_pipeline(monkeypatch):
    monkeypatch.delenv("DAS_PIPELINE_VERSION", raising=False)
    settings = Settings(_env_file=None, session_secret="s" * 32, worker_token="w" * 32)
    assert settings.pipeline_version == "v2"


@pytest.mark.parametrize("version", ["v1", "pipeline-v1", "v3"])
def test_settings_reject_noncurrent_pipeline(version):
    with pytest.raises(ValidationError, match="pipeline_version"):
        Settings(_env_file=None, session_secret="s" * 32, worker_token="w" * 32,
                 pipeline_version=version)


def test_settings_reject_legacy_pipeline_environment(monkeypatch):
    monkeypatch.setenv("DAS_PIPELINE_VERSION", "v1")
    with pytest.raises(ValidationError, match="pipeline_version"):
        Settings(_env_file=None, session_secret="s" * 32, worker_token="w" * 32)
