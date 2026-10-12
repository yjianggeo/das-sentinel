from datetime import UTC, datetime

import h5py
import numpy as np
import pytest
from das_sentinel.readers.base import ReaderRegistry
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.services.discovery import _open_anchored_source

from tests.fixtures import write_zd_h5


@pytest.mark.parametrize("n_time,n_channel", [(0, 8), (200, 0), (0, 0)])
@pytest.mark.parametrize("transpose", [False, True])
def test_empty_recording_is_rejected_before_accessing_samples(tmp_path, n_time, n_channel, transpose):
    path = tmp_path / "empty.h5"
    write_zd_h5(path, n_time=n_time, n_channel=n_channel, transpose=transpose)
    reader = ZdHdf5Reader()
    assert not reader.probe(path).supported
    for read in (reader.read_metadata, reader.sample_count):
        with pytest.raises(ValueError, match="empty"):
            read(path)



def test_reads_metadata_and_slice(tmp_path):
    path = tmp_path / "sample.h5"
    write_zd_h5(path)

    reader = ZdHdf5Reader()

    assert reader.probe(path).supported
    block = reader.read_block(path, 10, 30, slice(1, 4))

    assert block.data.shape == (20, 3)
    assert block.data.dtype == np.float32
    np.testing.assert_allclose(block.data[0], [81, 82, 83])
    np.testing.assert_allclose(block.metadata.channel_distance_m, [12, 16, 20])
    assert block.sample_offset == 10
    assert block.metadata.start_time_utc.isoformat() == "2026-07-26T00:00:00.100000+00:00"


def test_interprets_matching_utc8_filename_clock_as_station_local_time(tmp_path):
    path = tmp_path / "ZD_3400m_8m_4m_5000Hz_1000Hz_UTC8_202607260000.h5"
    write_zd_h5(path)

    reader = ZdHdf5Reader()
    metadata = reader.read_metadata(path)
    block = reader.read_block(path, 10, 11)

    assert metadata.start_time_utc == datetime(2026, 7, 25, 16, 0, tzinfo=UTC)
    assert block.metadata.start_time_utc == datetime(
        2026, 7, 25, 16, 0, 0, 100_000, tzinfo=UTC
    )


def test_does_not_apply_filename_offset_when_wall_clocks_disagree(tmp_path):
    path = tmp_path / "ZD_3400m_8m_4m_5000Hz_1000Hz_UTC8_202607260100.h5"
    write_zd_h5(path)

    assert ZdHdf5Reader().read_metadata(path).start_time_utc == datetime(
        2026, 7, 26, 0, 0, tzinfo=UTC
    )


def test_reads_repeatedly_from_one_held_binary_stream(tmp_path):
    path = tmp_path / "sample.h5"
    write_zd_h5(path, n_time=237)
    streams = []

    with _open_anchored_source(tmp_path, path) as source:
        streams.append(source.stream)
        reader = ReaderRegistry([ZdHdf5Reader()]).resolve(source.stream)
        metadata = reader.read_metadata(source.stream)
        count = reader.sample_count(source.stream)
        block = reader.read_block(source.stream, 10, 12, slice(1, 4))

        assert metadata.sample_rate_hz == 100.0
        assert count == 237
        assert block.data.shape == (2, 3)
        assert not source.stream.closed

    assert all(stream.closed for stream in streams)


def test_reports_total_sample_count_without_reading_outside_file(tmp_path):
    path = tmp_path / "sample.h5"
    write_zd_h5(path, n_time=237)

    assert ZdHdf5Reader().sample_count(path) == 237


def test_normalizes_transposed_data_to_time_channel_order(tmp_path):
    path = tmp_path / "transposed.h5"
    write_zd_h5(path, transpose=True)

    block = ZdHdf5Reader().read_block(path, 10, 12, slice(1, 4))

    assert block.data.shape == (2, 3)
    np.testing.assert_allclose(block.data, [[81, 82, 83], [89, 90, 91]])


def test_rejects_file_missing_required_dataset(tmp_path):
    path = tmp_path / "missing-time.h5"
    with h5py.File(path, "w") as handle:
        raw = handle.create_group("Acquisition").create_group("Raw[0]")
        raw.create_dataset("RawData", data=np.zeros((2, 2), dtype=np.float32))

    result = ZdHdf5Reader().probe(path)

    assert not result.supported
    with pytest.raises(ValueError, match="RawDataTime"):
        ZdHdf5Reader().read_metadata(path)


@pytest.mark.parametrize(
    ("corruption", "expected_reason"),
    [
        ("missing_attribute", "OutputDataRate"),
        ("dataset_is_group", "RawData"),
        ("time_axis_mismatch", "dimensions"),
        ("raw_is_dataset", "Acquisition/Raw[0]"),
    ],
)
def test_probe_rejects_invalid_required_structure(tmp_path, corruption, expected_reason):
    path = tmp_path / f"{corruption}.h5"
    write_zd_h5(path)
    with h5py.File(path, "r+") as handle:
        acquisition = handle["Acquisition"]
        raw = acquisition["Raw[0]"]
        if corruption == "missing_attribute":
            del raw.attrs["OutputDataRate"]
        elif corruption == "dataset_is_group":
            del raw["RawData"]
            raw.create_group("RawData")
        elif corruption == "time_axis_mismatch":
            del raw["RawData"]
            raw.create_dataset("RawData", data=np.zeros((3, 4), dtype=np.float32))
        else:
            del acquisition["Raw[0]"]
            acquisition.create_dataset("Raw[0]", data=np.zeros(1, dtype=np.float32))

    result = ZdHdf5Reader().probe(path)

    assert not result.supported
    assert expected_reason in result.reason


def test_compares_filename_time_with_source_start_not_block_start(tmp_path):
    matching_path = tmp_path / "20260726T000000Z.h5"
    mismatched_path = tmp_path / "20260726T000001Z.h5"
    write_zd_h5(matching_path)
    write_zd_h5(mismatched_path)

    block = ZdHdf5Reader().read_block(matching_path, 10, 11)
    metadata = ZdHdf5Reader().read_metadata(mismatched_path)

    assert block.metadata.warnings == ()
    assert metadata.warnings == ("filename timestamp disagrees with RawDataTime",)


@pytest.mark.parametrize(
    ("sample_start", "sample_stop", "channel_slice"),
    [(-1, 1, slice(None)), (1, 201, slice(None)), (2, 1, slice(None)), (0, 1, slice(-1, 2))],
)
def test_rejects_out_of_range_block_bounds(tmp_path, sample_start, sample_stop, channel_slice):
    path = tmp_path / "sample.h5"
    write_zd_h5(path)

    with pytest.raises(ValueError, match="bounds"):
        ZdHdf5Reader().read_block(path, sample_start, sample_stop, channel_slice)


def _drop_spacing_attr(path) -> None:
    with h5py.File(path, "r+") as handle:
        del handle["Acquisition/Raw[0]"].attrs["SpatialSamplingInterval"]


def test_rejects_legacy_gauge_length_as_channel_spacing(tmp_path):
    path = tmp_path / "jjk__3400m_4m_2_1000Hz_500Hz_UTC8_20230919173.h5"
    write_zd_h5(path, fs=500.0)
    _drop_spacing_attr(path)

    reader = ZdHdf5Reader()

    assert not reader.probe(path).supported
    with pytest.raises(ValueError, match="SpatialSamplingInterval"):
        reader.read_metadata(path)


@pytest.mark.parametrize(("layout", "spacing"), [("8m_4", 4.089979648590088), ("4m_2", 2.044989824295044)])
def test_reads_legacy_spacing_from_acquisition_on_held_stream(tmp_path, layout, spacing):
    path = tmp_path / f"jjk__3400m_{layout}_5000Hz_500Hz_UTC8_202607260000.h5"
    write_zd_h5(path)
    _drop_spacing_attr(path)
    with h5py.File(path, "r+") as handle:
        handle["Acquisition"].attrs["SpatialSamplingInterval"] = spacing

    with _open_anchored_source(tmp_path, path) as source:
        reader = ZdHdf5Reader()
        assert reader.probe(source.stream).supported
        metadata = reader.read_metadata(source.stream)
        block = reader.read_block(source.stream, 10, 12, slice(1, 4))

    np.testing.assert_allclose(metadata.channel_distance_m[:3], np.array([2, 3, 4]) * spacing)
    np.testing.assert_allclose(block.metadata.channel_distance_m, np.array([3, 4, 5]) * spacing)
    assert "channel spacing inferred from filename" not in metadata.warnings
    assert metadata.start_time_utc == datetime(2026, 7, 25, 16, 0, tzinfo=UTC)


@pytest.mark.parametrize("spacing", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_legacy_acquisition_spacing(tmp_path, spacing):
    path = tmp_path / "jjk__3400m_8m_4_5000Hz_500Hz_UTC8_202607260000.h5"
    write_zd_h5(path)
    _drop_spacing_attr(path)
    with h5py.File(path, "r+") as handle:
        handle["Acquisition"].attrs["SpatialSamplingInterval"] = spacing
    reader = ZdHdf5Reader()
    assert not reader.probe(path).supported
    with pytest.raises(ValueError, match="SpatialSamplingInterval"):
        reader.read_metadata(path)


def test_preserves_explicit_nominal_spacing_with_acquisition_attribute(tmp_path):
    path = tmp_path / "ZD_3400m_8m_4m_10000Hz_1000Hz_UTC8_202607260000.h5"
    write_zd_h5(path)
    _drop_spacing_attr(path)
    with h5py.File(path, "r+") as handle:
        handle["Acquisition"].attrs["SpatialSamplingInterval"] = 4.089979648590088
    np.testing.assert_allclose(ZdHdf5Reader().read_metadata(path).channel_distance_m[:3], [8, 12, 16])


def test_infers_channel_spacing_from_filename_on_held_stream(tmp_path):
    path = tmp_path / "ZD_3400m_8m_4m_10000Hz_1000Hz_UTC8_202405191333.h5"
    write_zd_h5(path)
    _drop_spacing_attr(path)

    with _open_anchored_source(tmp_path, path) as source:
        reader = ZdHdf5Reader()

        assert reader.probe(source.stream).supported
        metadata = reader.read_metadata(source.stream)

    np.testing.assert_allclose(metadata.channel_distance_m[:3], [8, 12, 16])
    assert "channel spacing inferred from filename" in metadata.warnings


def test_rejects_file_without_spacing_attribute_or_filename_hint(tmp_path):
    path = tmp_path / "plain-acquisition.h5"
    write_zd_h5(path)
    _drop_spacing_attr(path)

    result = ZdHdf5Reader().probe(path)

    assert not result.supported
    assert "SpatialSamplingInterval" in result.reason
