import h5py
import numpy as np
import pytest
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader

from tests.fixtures import write_zd_h5


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
