from datetime import UTC, datetime

import h5py
import numpy as np


def write_zd_h5(
    path, *, n_time: int = 200, n_channel: int = 8, fs: float = 100.0, transpose: bool = False
) -> None:
    start = datetime(2026, 7, 26, tzinfo=UTC)
    times = start.timestamp() * 1_000_000 + np.arange(n_time) * 1_000_000 / fs
    data = np.arange(n_time * n_channel, dtype=np.float32).reshape(n_time, n_channel)
    if transpose:
        data = data.T

    with h5py.File(path, "w") as handle:
        raw = handle.create_group("Acquisition").create_group("Raw[0]")
        raw.create_dataset("RawData", data=data)
        raw.create_dataset("RawDataTime", data=times.astype(np.int64))
        raw.attrs["OutputDataRate"] = fs
        raw.attrs["SpatialSamplingInterval"] = 4.0
        raw.attrs["StartLocusIndex"] = 2
        raw.attrs["RawDataUnit"] = "strain_rate"
