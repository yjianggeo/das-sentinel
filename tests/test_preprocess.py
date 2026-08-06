from datetime import UTC, datetime

import numpy as np
from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import DasBlock, DasMetadata
from das_sentinel.signal.backend import select_backend
from das_sentinel.signal.preprocess import (
    fft_bandpass,
    filter_prepared_signal,
    find_bad_channels,
    prepare_signal,
    preprocess_block,
)


def _shared_block() -> DasBlock:
    rng = np.random.default_rng(42)
    data = rng.normal(0.0, 0.1, size=(400, 8)).astype(np.float32)
    data[:, 7] = 0.0
    data[10, 6] = np.nan
    return DasBlock(
        data=data,
        metadata=DasMetadata(
            source_path="synthetic.h5",
            start_time_utc=datetime(2026, 8, 6, tzinfo=UTC),
            sample_rate_hz=100.0,
            channel_distance_m=np.arange(8, dtype=np.float32) * 4.0,
            unit="strain_rate",
        ),
        sample_offset=1_000,
    )


def test_cpu_fft_bandpass_preserves_ten_hz_and_removes_dc():
    fs = 100.0
    time = np.arange(1000) / fs
    block = (4.0 + np.sin(2 * np.pi * 10 * time))[:, None].astype(np.float32)

    filtered = fft_bandpass(block, fs, 5.0, 20.0, select_backend("cpu"))

    assert abs(float(filtered.mean())) < 1e-3
    assert np.corrcoef(filtered[:, 0], np.sin(2 * np.pi * 10 * time))[0, 1] > 0.99


def test_constant_and_nonfinite_channels_are_bad():
    block = np.random.default_rng(4).normal(size=(100, 4)).astype(np.float32)
    block[:, 1] = 0
    block[2, 3] = np.nan

    assert find_bad_channels(block).tolist() == [False, True, False, True]


def test_nonzero_constant_channel_is_bad():
    time = np.arange(100, dtype=np.float32)
    variable = 5 * np.sqrt(2) * np.sin(2 * np.pi * time / 20)
    block = np.column_stack((variable, np.full_like(variable, 5.0), -variable)).astype(np.float32)

    assert find_bad_channels(block).tolist() == [False, True, False]


def test_rms_outlier_is_bad_when_valid_channel_mad_is_zero():
    time = np.arange(100, dtype=np.float32)
    unit_rms = np.sin(2 * np.pi * time / 20)
    block = np.column_stack((unit_rms, -unit_rms, unit_rms, 20 * unit_rms)).astype(np.float32)

    assert find_bad_channels(block).tolist() == [False, False, False, True]


def test_preprocess_returns_backend_data_without_mutating_input():
    samples = 200
    channels = 3
    data = np.random.default_rng(7).normal(size=(samples, channels)).astype(np.float32)
    data[:, 1] = 0
    original = data.copy()
    metadata = DasMetadata(
        source_path="/data/example.h5",
        start_time_utc=datetime.now(UTC),
        sample_rate_hz=100.0,
        channel_distance_m=np.arange(channels, dtype=np.float32),
        unit="strain_rate",
    )
    block = DasBlock(data=data, metadata=metadata, sample_offset=17)

    result = preprocess_block(block, ProcessingProfile(), select_backend("cpu"))

    assert result.backend_name == "cpu"
    assert result.sample_offset == 17
    assert result.sample_rate_hz == 100.0
    assert result.bad_channel_mask.tolist() == [False, True, False]
    assert isinstance(result.data, np.ndarray)
    np.testing.assert_array_equal(block.data, original)


def test_prepare_then_filter_matches_preprocess_block():
    """Catches a split that changes the compatibility preprocessing result."""
    block = _shared_block()
    profile = ProcessingProfile()
    backend = select_backend("cpu")

    expected = preprocess_block(block, profile, backend)
    prepared = prepare_signal(block, profile, backend)
    actual = filter_prepared_signal(
        prepared,
        profile.stalta_freq_low_hz,
        profile.stalta_freq_high_hz,
        backend,
    )

    np.testing.assert_allclose(actual.data, expected.data, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual.bad_channel_mask, expected.bad_channel_mask)
    assert actual.sample_offset == expected.sample_offset
    assert actual.sample_rate_hz == expected.sample_rate_hz
    np.testing.assert_array_equal(
        actual.channel_distance_m, expected.channel_distance_m
    )


def test_filter_prepared_signal_keeps_prepared_values_independent_and_unchanged():
    """Catches filtering that aliases or mutates preparation state."""
    block = _shared_block()
    backend = select_backend("cpu")
    prepared = prepare_signal(block, ProcessingProfile(), backend)
    prepared_data = prepared.data.copy()
    prepared_spectrum = prepared.spectrum.copy()

    block.data[0, 0] += 1.0
    filter_prepared_signal(prepared, 2.0, 40.0, backend)

    np.testing.assert_array_equal(prepared.data, prepared_data)
    np.testing.assert_array_equal(prepared.spectrum, prepared_spectrum)
