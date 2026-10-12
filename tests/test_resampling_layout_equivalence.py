"""Observable channel/UTC/FIR regressions for grouped resampling."""

from datetime import timedelta

import h5py
import numpy as np
import pytest
from das_sentinel.readers.continuous_low_frequency import read_low_frequency_window

from tests.test_continuous_low_frequency import BASE, _write


@pytest.mark.parametrize("fs,rate,seam", [(100,100,.003), (500,100,-.001),
                                         (1000,100,.0005), (100,5,.003),
                                         (500,5,-.001), (1000,5,.0005)])
def test_short_core_preserves_distinct_tail_channels_and_utc_phase(tmp_path, fs, rate, seam):
    # A wrong channel offset, dropped tail group, halo crop or UTC phase fails
    # against this analytic signal, without deriving expected data from the reader.
    channels = 103
    paths = []
    for index, start in enumerate((0., 12. + seam)):
        path = _write(tmp_path / f"part-{index}.h5", start, duration=12,
                      fs=fs, channels=channels, high_frequency=False)
        t = start + np.arange(12 * fs) / fs
        amplitudes = 0.5 + np.arange(channels) / 100
        offsets = np.arange(channels) / 10
        with h5py.File(path, "r+") as handle:
            handle["Acquisition/Raw[0]/RawData"][:] = (
                np.sin(2 * np.pi * .4 * t[:, None]) * amplitudes + offsets
            ).astype("f4")
        paths.append(path)
    result = read_low_frequency_window(
        paths, BASE + timedelta(seconds=10.013), BASE + timedelta(seconds=13.183),
        4, 408, margin_seconds=0, sample_rate_hz=rate,
    )
    first = 10.02 if rate == 100 else 10.2
    count = 317 if rate == 100 else 15
    assert result.block.metadata.start_time_utc == BASE + timedelta(seconds=first)
    assert result.block.data.shape == (count, 102)
    np.testing.assert_array_equal(result.block.metadata.channel_distance_m,
                                  np.arange(1, 103) * 4)
    t = first + np.arange(count) / rate
    expected = (np.sin(2 * np.pi * .4 * t[:, None]) *
                (0.5 + np.arange(1, 103) / 100) + np.arange(1, 103) / 10)
    np.testing.assert_allclose(result.block.data, expected, atol=.003, rtol=0)
    assert result.sources[1]["previous_seam_seconds"] == pytest.approx(seam)


@pytest.mark.parametrize("fs", [500, 1000])
def test_100hz_grouped_reader_rejects_above_nyquist_energy(tmp_path, fs):
    # Removing/changing FIR downsampling leaves the 73Hz component in the
    # returned waveform; low-frequency phase and distinct channels remain visible.
    path = _write(tmp_path / "source.h5", 0, duration=24, fs=fs, channels=103,
                  high_frequency=False)
    t = np.arange(24 * fs) / fs
    low = np.sin(2 * np.pi * .4 * t)
    amplitude = 0.5 + np.arange(103) / 100
    with h5py.File(path, "r+") as handle:
        handle["Acquisition/Raw[0]/RawData"][:] = (
            (low + .5 * np.sin(2 * np.pi * 73 * t))[:, None] * amplitude
        ).astype("f4")
    result = read_low_frequency_window(
        [path], BASE + timedelta(seconds=10), BASE + timedelta(seconds=13.17),
        margin_seconds=0, sample_rate_hz=100,
    )
    expected = np.sin(2 * np.pi * .4 * (10 + np.arange(317) / 100))[:, None] * amplitude
    np.testing.assert_allclose(result.block.data, expected, atol=.004, rtol=0)
