import hashlib
from datetime import timedelta

import h5py
import numpy as np
import pytest
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.whole_window import inspect_sources, plan_windows

from tests.test_whole_window import BASE


@pytest.mark.parametrize("rate", [100, 1000])
def test_cached_cores_preserve_exact_arrays_and_reduce_reads(tmp_path, monkeypatch, rate):
    from das_sentinel.model_scan_input import iter_scan_inputs
    path = tmp_path / "long.h5"
    time = np.arange(420 * rate) / rate - 60
    wave = (np.sin(2 * np.pi * 8 * time) + .2 * np.sin(2 * np.pi * 72 * time)).astype("f4")
    with h5py.File(path, "w") as stream:
        raw = stream.create_group("Acquisition/Raw[0]")
        raw.create_dataset("RawData", data=np.tile(wave[:, None], (1, 8)))
        raw.create_dataset("RawDataTime", data=np.rint(BASE.timestamp() * 1e6 + time * 1e6).astype("i8"))
        raw.attrs.update(OutputDataRate=rate, SpatialSamplingInterval=100., StartLocusIndex=0, RawDataUnit="strain_rate")
    plans = [p for p in plan_windows(inspect_sources([path])) if BASE.isoformat() <= p["start_utc"] < (BASE + timedelta(seconds=300)).isoformat()]
    distance = np.arange(10, dtype="<f8") * 100
    metadata = {"unit": "strain_rate", "distance_start_m": 300, "training_grid": {
        "version": "native-channel-union-v1", "spacing_m": 100., "start_locus": 0, "channels": 10,
        "grid_sha256": hashlib.sha256(distance.tobytes()).hexdigest()}}
    reads = []
    original = ZdHdf5Reader.read_block
    def counted(self, *args, **kwargs):
        reads.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(ZdHdf5Reader, "read_block", counted)
    expected = list(iter_scan_inputs(plans, metadata, tmp_path / "cache", read_mode="single"))
    old_reads = len(reads)
    reads.clear()
    actual = list(iter_scan_inputs(plans, metadata, tmp_path / "cache", read_mode="block"))
    assert len(reads) < old_reads
    for (p, first, before), (q, second, after) in zip(expected, actual, strict=True):
        assert p == q
        assert before["model_input_sha256"] == after["model_input_sha256"]
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])


def test_core_cache_is_bounded_and_source_identity_is_part_of_key():
    from das_sentinel.readers.continuous_low_frequency import CoreCache
    cache = CoreCache(max_bytes=80)
    cache.put(("old-source",), np.ones(10, dtype="f4"))
    assert cache.get(("changed-source",)) is None
    cache.put(("second",), np.ones(15, dtype="f4"))
    assert cache.bytes <= 80 and cache.get(("old-source",)) is None
