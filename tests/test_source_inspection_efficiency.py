from datetime import UTC, datetime, timedelta

import h5py
import numpy as np
import pytest
from das_sentinel.readers import continuous_low_frequency as continuous
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.whole_window import inspect_sources

from tests.fixtures import write_zd_h5

BASE = datetime(2026, 7, 26, tzinfo=UTC)


@pytest.fixture
def hdf5_reads(monkeypatch):
    real_file = h5py.File
    reads = []

    def watched_file(path, mode="r", *args, **kwargs):
        handle = real_file(path, mode, *args, **kwargs)
        if mode == "r":
            reads.append(handle)
        return handle

    monkeypatch.setattr(h5py, "File", watched_file)
    return reads


@pytest.mark.parametrize("transpose", [False, True])
def test_source_inspection_opens_one_hdf5_handle(tmp_path, hdf5_reads, transpose):
    path = tmp_path / "source.h5"
    write_zd_h5(path, n_time=237, n_channel=3, transpose=transpose)

    sources = inspect_sources([path])

    assert len(hdf5_reads) == 1
    assert all(not handle.id.valid for handle in hdf5_reads)
    assert len(sources) == 1
    assert sources[0]["start_utc"] == BASE.isoformat()
    assert sources[0]["end_utc"] == (BASE + timedelta(seconds=2.37)).isoformat()
    assert sources[0]["distance_m"] == [8.0, 12.0, 16.0]
    assert sources[0]["unit"] == "strain_rate"


@pytest.mark.parametrize("transpose", [False, True])
def test_continuous_inspection_opens_one_handle_before_full_clock_check(
    tmp_path, monkeypatch, hdf5_reads, transpose
):
    path = tmp_path / "source.h5"
    write_zd_h5(path, n_time=12000, n_channel=3, transpose=transpose)
    real_check_clock = continuous._check_clock
    inspection_opens = []

    def watched_clock(source, warnings):
        inspection_opens.append(len(hdf5_reads))
        return real_check_clock(source, warnings)

    monkeypatch.setattr(continuous, "_check_clock", watched_clock)
    result = continuous.read_low_frequency_window(
        [path], BASE + timedelta(seconds=30), BASE + timedelta(seconds=50),
        margin_seconds=0, sample_rate_hz=100,
    )

    assert inspection_opens == [1]
    assert all(not handle.id.valid for handle in hdf5_reads)
    assert result.block.data.shape == (2000, 3)
    np.testing.assert_array_equal(result.block.metadata.channel_distance_m, [8, 12, 16])
    np.testing.assert_array_equal(result.block.data[0], [9000, 9001, 9002])
    assert result.sources[0]["source_verified_unchanged"] is True


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("held_stream", [False, True])
def test_combined_inspection_has_one_fresh_handle_and_preserves_stream(
    tmp_path, hdf5_reads, transpose, held_stream
):
    path = tmp_path / "source.h5"
    write_zd_h5(path, n_time=237, n_channel=3, transpose=transpose)
    reader = ZdHdf5Reader()

    if held_stream:
        with path.open("rb") as stream:
            metadata, count = reader.read_metadata_and_count(stream)
            assert not stream.closed
    else:
        metadata, count = reader.read_metadata_and_count(path)

    assert len(hdf5_reads) == 1
    assert not hdf5_reads[0].id.valid
    assert count == 237
    assert metadata.start_time_utc == BASE
    assert metadata.sample_rate_hz == 100
    assert metadata.unit == "strain_rate"
    assert metadata.warnings == ()
    np.testing.assert_array_equal(metadata.channel_distance_m, [8, 12, 16])

    write_zd_h5(path, n_time=311, n_channel=2, fs=200, transpose=transpose)
    fresh, count = reader.read_metadata_and_count(path)
    assert len(hdf5_reads) == 2
    assert all(not handle.id.valid for handle in hdf5_reads)
    assert count == 311
    assert fresh.sample_rate_hz == 200
    np.testing.assert_array_equal(fresh.channel_distance_m, [8, 12])


@pytest.mark.parametrize(
    "filename,spacing_source,expected_start,expected_warnings",
    [
        ("source.h5", "raw", BASE, ()),
        ("20260726T000001Z.h5", "raw", BASE,
         ("filename timestamp disagrees with RawDataTime",)),
        ("ZD_3400m_8m_4m_1000Hz_UTC8_202607260000.h5", "filename",
         BASE - timedelta(hours=8), ("channel spacing inferred from filename",)),
        ("jjk__3400m_8m_4_500Hz_UTC8_202607260000.h5", "acquisition",
         BASE - timedelta(hours=8), ()),
    ],
)
def test_combined_metadata_preserves_time_units_spacing_and_warnings(
    tmp_path, filename, spacing_source, expected_start, expected_warnings
):
    path = tmp_path / filename
    write_zd_h5(path, n_time=237, n_channel=3)
    with h5py.File(path, "r+") as handle:
        raw = handle["Acquisition/Raw[0]"]
        raw.attrs["RawDataUnit"] = np.bytes_("strain")
        if spacing_source != "raw":
            del raw.attrs["SpatialSamplingInterval"]
        if spacing_source == "acquisition":
            handle["Acquisition"].attrs["SpatialSamplingInterval"] = 4.25

    reader = ZdHdf5Reader()
    metadata, count = reader.read_metadata_and_count(path)
    legacy = reader.read_metadata(path)
    assert count == reader.sample_count(path) == 237
    assert metadata.start_time_utc == legacy.start_time_utc == expected_start
    assert metadata.unit == legacy.unit == "strain"
    assert metadata.warnings == legacy.warnings == expected_warnings
    expected_distance = [8.5, 12.75, 17] if spacing_source == "acquisition" else [8, 12, 16]
    np.testing.assert_array_equal(metadata.channel_distance_m, expected_distance)
    np.testing.assert_array_equal(legacy.channel_distance_m, expected_distance)


@pytest.mark.parametrize(
    "data_shape,time_shape,reason",
    [
        ((0, 3), (0,), "empty"),
        ((237, 0), (237,), "empty"),
        ((237,), (237,), "two-dimensional"),
        ((237, 3), (237, 1), "one-dimensional"),
        ((5, 3), (237,), "do not match"),
    ],
)
def test_combined_inspection_rejects_invalid_dimensions_and_closes_handle(
    tmp_path, hdf5_reads, data_shape, time_shape, reason
):
    path = tmp_path / "invalid.h5"
    write_zd_h5(path)
    with h5py.File(path, "r+") as handle:
        raw = handle["Acquisition/Raw[0]"]
        del raw["RawData"]
        del raw["RawDataTime"]
        raw.create_dataset("RawData", shape=data_shape, dtype="f4")
        raw.create_dataset("RawDataTime", shape=time_shape, dtype="i8")

    reader = ZdHdf5Reader()
    for read in (reader.read_metadata_and_count, reader.read_metadata, reader.sample_count):
        with pytest.raises(ValueError, match=reason):
            read(path)
    assert len(hdf5_reads) == 3
    assert all(not handle.id.valid for handle in hdf5_reads)


@pytest.mark.parametrize("consumer", ["whole_window", "continuous"])
def test_source_mutation_after_combined_inspection_is_rejected(
    tmp_path, monkeypatch, consumer
):
    path = tmp_path / "source.h5"
    write_zd_h5(path, n_time=12000, n_channel=3)
    real_read = ZdHdf5Reader.read_metadata_and_count

    def read_and_change(self, source):
        result = real_read(self, source)
        with h5py.File(source, "r+") as handle:
            handle.attrs["mutation"] = "changed after metadata read"
        return result

    monkeypatch.setattr(ZdHdf5Reader, "read_metadata_and_count", read_and_change)
    with pytest.raises(ValueError, match="source changed|源文件.*变化"):
        if consumer == "whole_window":
            inspect_sources([path])
        else:
            continuous.read_low_frequency_window(
                [path], BASE + timedelta(seconds=30), BASE + timedelta(seconds=50),
                margin_seconds=0, sample_rate_hz=100,
            )
