from datetime import UTC, datetime, timedelta

import h5py
import numpy as np
import pytest
from das_sentinel.readers import continuous_low_frequency as continuous
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.services.discovery import fingerprint

BASE = datetime(2026, 6, 16, 9, 0, tzinfo=UTC)
RAW = "Acquisition/Raw[0]"


def _wave(seconds):
    return np.sin(2 * np.pi * 0.04 * seconds) + 0.2 * np.sin(
        2 * np.pi * 0.25 * seconds
    )


def _files(tmp_path, *, fs=100, transpose=False, seam=0.0):
    paths = []
    for index, start in enumerate((0.42, 60.42 + seam)):
        seconds = start + np.arange(60 * fs) / fs
        data = (_wave(seconds)[:, None] * np.arange(1, 6)).astype(np.float32)
        path = tmp_path / f"part-{index}.h5"
        with h5py.File(path, "w") as handle:
            raw = handle.create_group(RAW)
            raw.create_dataset("RawData", data=data.T if transpose else data)
            raw.create_dataset(
                "RawDataTime",
                data=np.rint(BASE.timestamp() * 1_000_000 + seconds * 1_000_000).astype(
                    np.int64
                ),
            )
            raw.attrs["OutputDataRate"] = fs
            raw.attrs["SpatialSamplingInterval"] = 4.0
            raw.attrs["StartLocusIndex"] = 0
            raw.attrs["RawDataUnit"] = "strain_rate"
        paths.append(path)
    return paths


def _read(paths, *, sample_rate_hz=100):
    return continuous.read_low_frequency_window(
        paths[::-1],
        BASE + timedelta(seconds=30),
        BASE + timedelta(seconds=100),
        4,
        8,
        margin_seconds=10,
        sample_rate_hz=sample_rate_hz,
    )


def _record_reads(monkeypatch):
    original = ZdHdf5Reader.read_block
    reads = []

    def counted(self, path, first, last, channel_slice=slice(None)):
        block = original(self, path, first, last, channel_slice)
        reads.append((path, first, last, channel_slice, block.data.nbytes))
        return block

    monkeypatch.setattr(ZdHdf5Reader, "read_block", counted)
    return reads


def _assert_wave_and_grid(result, sample_rate_hz=100):
    assert result.block.metadata.start_time_utc == BASE + timedelta(seconds=20)
    assert result.block.metadata.sample_rate_hz == sample_rate_hz
    assert result.block.sample_offset == 0
    assert result.block.data.shape == (90 * sample_rate_hz, 2)
    assert result.block.data.dtype == np.float32
    assert (result.analysis_start, result.analysis_stop) == (
        10 * sample_rate_hz,
        80 * sample_rate_hz,
    )
    np.testing.assert_array_equal(result.block.metadata.channel_distance_m, [4, 8])
    seconds = 20 + np.arange(90 * sample_rate_hz) / sample_rate_hz
    expected = _wave(seconds)[:, None] * [2, 3]
    np.testing.assert_allclose(result.block.data, expected, atol=0.003)


@pytest.mark.parametrize("fs", [100, 500, 1000])
@pytest.mark.parametrize("transpose", [False, True], ids=["normal", "transposed"])
def test_100hz_adjacent_cores_read_each_raw_halo_only_once(
    tmp_path, monkeypatch, fs, transpose
):
    # A repeated full-halo read should exceed this physical IO budget, even if
    # its resampled output looks right. The half-sample seam also catches reuse
    # by UTC offset instead of by the source's absolute native sample index.
    seam = 0.5 / fs
    paths = _files(tmp_path, fs=fs, transpose=transpose, seam=seam)
    reads = _record_reads(monkeypatch)
    result = _read(paths)

    _assert_wave_and_grid(result)
    assert len(result.sources) == 2
    assert result.sources[1]["previous_seam_seconds"] == pytest.approx(seam)
    assert result.sources[1]["previous_source_end_utc"] == (
        BASE + timedelta(seconds=60.42)
    ).isoformat()
    assert result.sources[1]["boundary_utc"] == (
        BASE + timedelta(seconds=60.42 + seam)
    ).isoformat()
    assert any("时钟" in warning for warning in result.warnings)

    for source, path in zip(result.sources, paths, strict=True):
        source_reads = [read for read in reads if read[0] == path.resolve()]
        assert source["fingerprint"] == fingerprint(path)
        assert source["source_verified_unchanged"] is True
        assert source["sample_start"] == min(read[1] for read in source_reads)
        assert source["sample_stop"] == max(read[2] for read in source_reads)
        assert (source["channel_start"], source["channel_stop"]) == (1, 3)
        source_start = datetime.fromisoformat(source["source_start_utc"])
        assert datetime.fromisoformat(source["read_start_utc"]) == (
            source_start + timedelta(seconds=source["sample_start"] / fs)
        )
        assert datetime.fromisoformat(source["read_stop_utc"]) == (
            source_start + timedelta(seconds=source["sample_stop"] / fs)
        )

    # 90 seconds of output plus two 10-second halos, with at most eight
    # interpolation neighbours across the two sources. Measure real returned
    # float32 values, rather than asserting a mocked read's call count.
    assert sum(read[4] for read in reads) <= (110 * fs + 8) * 2 * 4


def test_multiple_io_channel_groups_fall_back_without_changing_values(
    tmp_path, monkeypatch
):
    paths = _files(tmp_path)
    monkeypatch.setattr(continuous, "_MAX_IO_VALUES", 6_000)
    reads = _record_reads(monkeypatch)
    result = _read(paths)

    _assert_wave_and_grid(result)
    assert reads
    assert max(read[4] for read in reads) <= 6_000 * 4
    assert sum(read[4] for read in reads) > (150 * 100) * 2 * 4


def test_5hz_keeps_existing_full_halo_reads(tmp_path, monkeypatch):
    paths = _files(tmp_path)
    reads = _record_reads(monkeypatch)
    result = _read(paths, sample_rate_hz=5)

    _assert_wave_and_grid(result, sample_rate_hz=5)
    assert sum(read[4] for read in reads) > (150 * 100) * 2 * 4


def test_nonfinite_value_in_reused_halo_is_rejected(tmp_path):
    paths = _files(tmp_path)
    with h5py.File(paths[0], "r+") as handle:
        # This sample lies in both the first and second core's raw halo.
        handle[f"{RAW}/RawData"][4_000, 1] = np.nan

    with pytest.raises(ValueError, match="非有限采样值"):
        _read(paths)


def test_timestamp_change_between_windows_is_validated_again(tmp_path):
    paths = _files(tmp_path)
    _assert_wave_and_grid(_read(paths))
    with h5py.File(paths[0], "r+") as handle:
        times = handle[f"{RAW}/RawDataTime"]
        # Keep source start/end intact; only a full clock validation catches
        # this internal jump after an earlier call read the same halo.
        times[4_000:4_010] = times[4_000:4_010] + 100_000

    with pytest.raises(ValueError, match="RawDataTime.*时标"):
        _read(paths)


def test_raw_values_are_not_reused_between_windows(tmp_path, monkeypatch):
    paths = _files(tmp_path)
    reads = _record_reads(monkeypatch)
    first = _read(paths)
    _assert_wave_and_grid(first)
    for path in paths:
        with h5py.File(path, "r+") as handle:
            data = handle[f"{RAW}/RawData"]
            data[:] = data[:] + np.float32(1)
    reads.clear()
    second = _read(paths)

    assert reads
    np.testing.assert_allclose(second.block.data, first.block.data + 1, atol=1e-5)
