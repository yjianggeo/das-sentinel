import importlib
import importlib.util
import os
from collections import Counter
from datetime import UTC, datetime, timedelta

import h5py
import numpy as np
import pytest
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.services.discovery import fingerprint

BASE = datetime(2026, 6, 16, 9, 0, tzinfo=UTC)


def _reader():
    name = "das_sentinel.readers.continuous_low_frequency"
    assert importlib.util.find_spec(name) is not None, (
        "continuous low-frequency reader is missing"
    )
    return importlib.import_module(name).read_low_frequency_window


def _write(
    path,
    start,
    *,
    duration=60,
    fs=100,
    channels=3,
    unit="strain_rate",
    high_frequency=True,
    high_frequency_hz=19,
):
    n = round(duration * fs)
    t = start + np.arange(n) / fs
    wave = np.sin(2 * np.pi * 0.01 * t) + 0.2 * np.sin(2 * np.pi * 0.4 * t)
    if high_frequency:
        wave += 0.5 * np.sin(2 * np.pi * high_frequency_hz * t)
    with h5py.File(path, "w") as h:
        raw = h.create_group("Acquisition/Raw[0]")
        raw.create_dataset(
            "RawData", data=np.repeat(wave[:, None], channels, axis=1).astype("f4")
        )
        raw.create_dataset(
            "RawDataTime", data=np.rint(BASE.timestamp() * 1e6 + t * 1e6).astype("i8")
        )
        raw.attrs["OutputDataRate"] = fs
        raw.attrs["SpatialSamplingInterval"] = 4.0
        raw.attrs["StartLocusIndex"] = 0
        raw.attrs["RawDataUnit"] = unit
    return path


def _files(tmp_path, shift=0, **kwargs):
    return [
        _write(tmp_path / f"part-{i}.h5", 0.42 + 60 * i + (shift if i else 0), **kwargs)
        for i in range(4)
    ]


def test_continuous_resampling_preserves_absolute_utc_and_rejects_alias_energy(
    tmp_path,
):
    paths = _files(tmp_path)
    result = _reader()(
        paths[::-1],
        BASE + timedelta(seconds=30),
        BASE + timedelta(seconds=150),
        4,
        8,
        margin_seconds=10,
    )
    block = result.block
    assert block.metadata.start_time_utc == BASE + timedelta(seconds=20)
    assert block.metadata.sample_rate_hz == 5
    assert block.sample_offset == 0
    assert block.data.shape == (700, 2)
    assert block.data.dtype == np.float32
    assert (result.analysis_start, result.analysis_stop) == (50, 650)
    np.testing.assert_array_equal(block.metadata.channel_distance_m, [4, 8])
    t = 20 + np.arange(700) / 5
    expected = np.sin(2 * np.pi * 0.01 * t) + 0.2 * np.sin(2 * np.pi * 0.4 * t)
    np.testing.assert_allclose(block.data[:, 0], expected, atol=0.002)
    assert len(result.sources) == 3
    assert all(
        s["fingerprint"] == fingerprint(type(paths[0])(s["path"]))
        for s in result.sources
    )
    assert all(s["sample_stop"] > s["sample_start"] >= 0 for s in result.sources)


def test_100hz_resampling_uses_absolute_utc_grid_and_rejects_alias_energy(tmp_path):
    paths = _files(tmp_path, fs=1000, high_frequency_hz=119)
    result = _reader()(
        paths,
        BASE + timedelta(seconds=30.003),
        BASE + timedelta(seconds=80.007),
        4,
        8,
        margin_seconds=10,
        sample_rate_hz=100,
    )
    block = result.block
    assert block.metadata.start_time_utc == BASE + timedelta(seconds=20.01)
    assert block.metadata.sample_rate_hz == 100
    assert block.data.shape == (7000, 2)
    assert (result.analysis_start, result.analysis_stop) == (1000, 6000)
    t = 20.01 + np.arange(7000) / 100
    expected = (
        np.sin(2 * np.pi * 0.01 * t)
        + 0.2 * np.sin(2 * np.pi * 0.4 * t)
    )
    np.testing.assert_allclose(block.data[:, 0], expected, atol=0.003)
    assert any("0.01秒" in warning and "100Hz" in warning for warning in result.warnings)


@pytest.mark.parametrize("sample_rate_hz", [0, 10, 101])
def test_rejects_unsupported_output_sample_rate(tmp_path, sample_rate_hz):
    with pytest.raises(ValueError, match="5.*100|100.*5"):
        _reader()(
            _files(tmp_path),
            BASE + timedelta(seconds=30),
            BASE + timedelta(seconds=80),
            margin_seconds=10,
            sample_rate_hz=sample_rate_hz,
        )


@pytest.mark.parametrize("shift", [-0.011, 0.011])
def test_100hz_seam_cannot_exceed_one_native_sample_period(tmp_path, shift):
    with pytest.raises(ValueError, match="0.01|连续"):
        _reader()(
            _files(tmp_path, shift, high_frequency=False),
            BASE + timedelta(seconds=30),
            BASE + timedelta(seconds=150),
            margin_seconds=10,
            sample_rate_hz=100,
        )


@pytest.mark.parametrize("shift", [-0.03, 0.03, 0.05])
def test_small_clock_seam_is_reported_and_mapped_without_zero_fill(tmp_path, shift):
    result = _reader()(
        _files(tmp_path, shift, high_frequency=False),
        BASE + timedelta(seconds=30),
        BASE + timedelta(seconds=150),
        margin_seconds=10,
    )
    assert any("时钟" in w for w in result.warnings)
    t = 20 + np.arange(700) / 5
    expected = np.sin(2 * np.pi * 0.01 * t) + 0.2 * np.sin(2 * np.pi * 0.4 * t)
    # Pure low-frequency data isolates UTC remapping from unrecoverable
    # high-frequency energy lost inside a physical gap.
    np.testing.assert_allclose(result.block.data[:, 0], expected, atol=0.002)
    assert result.sources[1]["previous_seam_seconds"] == pytest.approx(shift)
    assert (
        result.sources[1]["previous_source_end_utc"]
        == (BASE + timedelta(seconds=60.42)).isoformat()
    )
    assert (
        result.sources[1]["boundary_utc"]
        == (BASE + timedelta(seconds=60.42 + shift)).isoformat()
    )


def test_high_frequency_energy_missing_in_a_small_gap_is_explicitly_flagged(tmp_path):
    result = _reader()(
        _files(tmp_path, 0.05),
        BASE + timedelta(seconds=30),
        BASE + timedelta(seconds=150),
        margin_seconds=10,
    )
    assert result.sources[1]["previous_seam_seconds"] == pytest.approx(0.05)
    assert any("低频残差" in warning for warning in result.warnings)
    assert any("09:01:00.470000" in warning for warning in result.warnings)


@pytest.mark.parametrize("shift", [-0.051, 0.051, 1.0])
def test_large_clock_gap_or_overlap_is_rejected(tmp_path, shift):
    with pytest.raises(ValueError, match="0.05|连续"):
        _reader()(
            _files(tmp_path, shift),
            BASE + timedelta(seconds=30),
            BASE + timedelta(seconds=150),
            margin_seconds=10,
        )


@pytest.mark.parametrize("edge", ["前", "后"])
def test_requires_real_margin_and_resampling_halo_on_both_sides(tmp_path, edge):
    paths = _files(tmp_path)
    start, stop = (10, 100) if edge == "前" else (30, 230)
    with pytest.raises(ValueError, match=f"{edge}.*上下文"):
        _reader()(
            paths,
            BASE + timedelta(seconds=start),
            BASE + timedelta(seconds=stop),
            margin_seconds=10,
        )


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("OutputDataRate", 101),
        ("SpatialSamplingInterval", 5),
        ("RawDataUnit", "strain"),
    ],
)
def test_refuses_inconsistent_selected_file_metadata(tmp_path, attribute, value):
    paths = _files(tmp_path)
    with h5py.File(paths[1], "r+") as h:
        h["Acquisition/Raw[0]"].attrs[attribute] = value
    with pytest.raises(ValueError, match="一致|时标"):
        _reader()(
            paths,
            BASE + timedelta(seconds=30),
            BASE + timedelta(seconds=150),
            margin_seconds=10,
        )


def test_rejects_internal_time_jump_even_with_unchanged_first_and_last_times(tmp_path):
    paths = _files(tmp_path)
    with h5py.File(paths[1], "r+") as h:
        times = h["Acquisition/Raw[0]/RawDataTime"]
        times[2000:2100] = times[2000:2100] + 100_000
    with pytest.raises(ValueError, match="文件内.*时标|RawDataTime"):
        _reader()(
            paths,
            BASE + timedelta(seconds=30),
            BASE + timedelta(seconds=150),
            margin_seconds=10,
        )


def test_matching_utc8_filename_uses_existing_reader_conversion_once(tmp_path):
    path = tmp_path / "ZD_UTC8_202606160900.h5"
    _write(path, 0.42, duration=180)
    start = BASE - timedelta(hours=8) + timedelta(seconds=30)
    result = _reader()([path], start, start + timedelta(seconds=60), margin_seconds=10)
    assert result.block.metadata.start_time_utc == BASE - timedelta(
        hours=8
    ) + timedelta(seconds=20)
    assert result.block.data.shape == (400, 3)
    assert any("UTC" in w for w in result.warnings)


def test_detects_source_change_during_data_read(tmp_path, monkeypatch):
    paths = _files(tmp_path)
    original = ZdHdf5Reader.read_block
    changed = False

    def read_and_change(self, path, *args, **kwargs):
        nonlocal changed
        block = original(self, path, *args, **kwargs)
        if not changed:
            changed = True
            stat = paths[0].stat()
            os.utime(paths[0], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return block

    monkeypatch.setattr(ZdHdf5Reader, "read_block", read_and_change)
    with pytest.raises(ValueError, match="源文件.*变化"):
        _reader()(
            paths,
            BASE + timedelta(seconds=30),
            BASE + timedelta(seconds=150),
            margin_seconds=10,
        )


@pytest.mark.parametrize("fs", [100, 1000])
def test_each_raw_piece_is_read_once_across_compute_channel_groups(
    tmp_path, monkeypatch, fs
):
    paths = _files(tmp_path, channels=150, fs=fs)
    read = _reader()
    module = importlib.import_module(read.__module__)
    original = ZdHdf5Reader.read_block
    original_resample = module.resample_poly
    reads = []
    compute_shapes = []

    def checked_read(self, path, start, stop, channel_slice=slice(None)):
        reads.append((path, start, stop, channel_slice.stop - channel_slice.start))
        return original(self, path, start, stop, channel_slice)

    def checked_resample(values, *args, **kwargs):
        compute_shapes.append(values.shape)
        return original_resample(values, *args, **kwargs)

    monkeypatch.setattr(ZdHdf5Reader, "read_block", checked_read)
    monkeypatch.setattr(module, "resample_poly", checked_resample)
    result = read(
        paths,
        BASE + timedelta(seconds=30),
        BASE + timedelta(seconds=150),
        margin_seconds=10,
    )
    assert result.block.data.shape == (700, 150)
    assert max(stop - start for _, start, stop, _ in reads) <= 42 * fs
    assert max((stop - start) * c for _, start, stop, c in reads) <= 33_554_432
    assert (
        max(Counter((path, start, stop) for path, start, stop, _ in reads).values())
        == 1
    )
    assert max(c for _, c in compute_shapes) <= min(64, 2_000_000 // (40 * fs + 4))
    assert max(n * c for n, c in compute_shapes) <= 2_000_000


def test_raw_io_batches_obey_value_budget_and_keep_identical_resampling(
    tmp_path, monkeypatch
):
    paths = _files(tmp_path, channels=15)
    # Distinct amplitudes and a nonzero selected first channel expose incorrect
    # channel offsets when an IO batch contains multiple compute slices.
    for path in paths:
        with h5py.File(path, "r+") as handle:
            data = handle["Acquisition/Raw[0]/RawData"]
            data[:] = data[:] * np.arange(1, 16, dtype=np.float32)
    read = _reader()
    module = importlib.import_module(read.__module__)
    start, stop = BASE + timedelta(seconds=30), BASE + timedelta(seconds=150)
    expected = read(paths, start, stop, 4, 52, margin_seconds=10).block.data
    monkeypatch.setattr(module, "_MAX_IO_VALUES", 20_000, raising=False)
    original = ZdHdf5Reader.read_block
    reads = []

    def checked_read(self, path, first, last, channel_slice=slice(None)):
        count = (last - first) * (channel_slice.stop - channel_slice.start)
        reads.append(count)
        assert count <= 20_000
        return original(self, path, first, last, channel_slice)

    monkeypatch.setattr(ZdHdf5Reader, "read_block", checked_read)
    result = read(paths, start, stop, 4, 52, margin_seconds=10)
    assert reads
    np.testing.assert_array_equal(result.block.data, expected)


def test_ignores_noncovering_file_metadata_and_rejects_empty_distance_range(tmp_path):
    paths = _files(tmp_path)
    irrelevant = _write(tmp_path / "earlier.h5", -500, unit="other")
    result = _reader()(
        [irrelevant, *paths],
        BASE + timedelta(seconds=30),
        BASE + timedelta(seconds=150),
        margin_seconds=10,
    )
    assert len(result.sources) == 3
    with pytest.raises(ValueError, match="距离"):
        _reader()(
            paths,
            BASE + timedelta(seconds=30),
            BASE + timedelta(seconds=150),
            100,
            200,
            margin_seconds=10,
        )


def test_noncovering_low_sample_rate_file_does_not_block_valid_window(tmp_path):
    paths = _files(tmp_path)
    earlier = _write(tmp_path / "low-rate-earlier.h5", -500, fs=2)
    result = _reader()(
        [earlier, *paths],
        BASE + timedelta(seconds=30),
        BASE + timedelta(seconds=150),
        margin_seconds=10,
    )
    assert result.block.data.shape == (700, 3)


def test_subsample_analysis_interval_is_rejected_even_with_large_margins(tmp_path):
    with pytest.raises(ValueError, match="5Hz样本"):
        _reader()(
            _files(tmp_path),
            BASE + timedelta(seconds=30.01),
            BASE + timedelta(seconds=30.09),
            margin_seconds=10,
        )


def test_non_grid_bounds_select_absolute_utc_samples_inside_interval(tmp_path):
    result = _reader()(
        _files(tmp_path),
        BASE + timedelta(seconds=30.03),
        BASE + timedelta(seconds=80.07),
        margin_seconds=10,
    )
    assert result.block.metadata.start_time_utc == BASE + timedelta(seconds=20.2)
    assert result.block.data.shape == (350, 3)
    assert (result.analysis_start, result.analysis_stop) == (50, 300)
    assert any("0.2秒" in warning for warning in result.warnings)


def test_awkward_resampling_ratio_is_rejected_before_raw_data_allocation(
    tmp_path, monkeypatch
):
    path = _write(tmp_path / "awkward-rate.h5", 0.42, duration=180, fs=100.314159265)

    def no_raw_read(*args, **kwargs):
        pytest.fail("An unbounded rational filter reached raw data allocation")

    monkeypatch.setattr(ZdHdf5Reader, "read_block", no_raw_read)
    with pytest.raises(ValueError, match="采样率.*重采样"):
        _reader()(
            [path],
            BASE + timedelta(seconds=30),
            BASE + timedelta(seconds=100),
            margin_seconds=10,
        )


def test_output_memory_budget_is_checked_before_raw_reads(tmp_path, monkeypatch):
    read = _reader()
    module = importlib.import_module(read.__module__)
    monkeypatch.setattr(module, "_MAX_OUTPUT_BYTES", 1000)

    def no_raw_read(*args, **kwargs):
        pytest.fail("Oversized output reached raw data allocation")

    monkeypatch.setattr(ZdHdf5Reader, "read_block", no_raw_read)
    with pytest.raises(ValueError, match="内存上限"):
        read(
            _files(tmp_path),
            BASE + timedelta(seconds=30),
            BASE + timedelta(seconds=150),
            margin_seconds=10,
        )

def test_native_distance_none_reads_negative_coordinates(tmp_path):
    paths=_files(tmp_path)
    for path in paths:
        with h5py.File(path,'r+') as h:
            h['Acquisition/Raw[0]'].attrs['StartLocusIndex']=-1
    whole=_reader()(paths,BASE+timedelta(seconds=30),BASE+timedelta(seconds=90),None,None,margin_seconds=10,sample_rate_hz=100)
    np.testing.assert_array_equal(whole.block.metadata.channel_distance_m,[-4,0,4])
    default=_reader()(paths,BASE+timedelta(seconds=30),BASE+timedelta(seconds=90),margin_seconds=10,sample_rate_hz=100)
    np.testing.assert_array_equal(default.block.metadata.channel_distance_m,[0,4])
    with pytest.raises(ValueError,match='距离范围'):
        _reader()(paths,BASE+timedelta(seconds=30),BASE+timedelta(seconds=90),-4,4,margin_seconds=10,sample_rate_hz=100)
