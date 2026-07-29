import re
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np

from das_sentinel.domain import DasBlock, DasMetadata
from das_sentinel.readers.base import ProbeResult


class ZdHdf5Reader:
    """Reader for ZD/PRODML HDF5 acquisition files."""

    FORMAT_NAME = "ZD/PRODML HDF5"
    RAW_GROUP = "Acquisition/Raw[0]"
    DATASET_DATA = "RawData"
    DATASET_TIME = "RawDataTime"
    ATTR_SAMPLE_RATE = "OutputDataRate"
    ATTR_CHANNEL_SPACING = "SpatialSamplingInterval"
    ATTR_START_LOCUS = "StartLocusIndex"
    ATTR_UNIT = "RawDataUnit"

    def probe(self, path: Path) -> ProbeResult:
        try:
            with h5py.File(path, "r") as handle:
                raw = handle.get(self.RAW_GROUP)
                if not isinstance(raw, h5py.Group):
                    return ProbeResult(False, self.FORMAT_NAME, f"{self.RAW_GROUP} is not a group")
                data = raw.get(self.DATASET_DATA)
                times = raw.get(self.DATASET_TIME)
                if not isinstance(data, h5py.Dataset):
                    return ProbeResult(False, self.FORMAT_NAME, f"{self.DATASET_DATA} is not a dataset")
                if not isinstance(times, h5py.Dataset):
                    return ProbeResult(False, self.FORMAT_NAME, f"{self.DATASET_TIME} is not a dataset")
                if data.ndim != 2 or times.ndim != 1:
                    return ProbeResult(False, self.FORMAT_NAME, "RawData dimensions are invalid")
                n_time = times.shape[0]
                if data.shape[0] != n_time and data.shape[1] != n_time:
                    return ProbeResult(False, self.FORMAT_NAME, "RawData dimensions do not match RawDataTime")
                for attribute_name in (
                    self.ATTR_SAMPLE_RATE,
                    self.ATTR_CHANNEL_SPACING,
                    self.ATTR_START_LOCUS,
                    self.ATTR_UNIT,
                ):
                    if attribute_name not in raw.attrs:
                        return ProbeResult(False, self.FORMAT_NAME, f"missing {attribute_name}")
        except (OSError, TypeError, ValueError) as error:
            return ProbeResult(False, self.FORMAT_NAME, str(error))
        return ProbeResult(True, self.FORMAT_NAME)

    def read_metadata(self, path: Path) -> DasMetadata:
        with h5py.File(path, "r") as handle:
            raw, data, times = self._required_datasets(handle)
            n_time, n_channel, _ = self._dimensions(data, times)
            del n_time
            start_time = self._time_at(times, 0)
            spacing = float(self._required_attr(raw, self.ATTR_CHANNEL_SPACING))
            start_locus = int(self._required_attr(raw, self.ATTR_START_LOCUS))
            return self._metadata(
                path,
                start_time,
                float(self._required_attr(raw, self.ATTR_SAMPLE_RATE)),
                (start_locus + np.arange(n_channel)) * spacing,
                self._required_attr(raw, self.ATTR_UNIT),
            )

    def sample_count(self, path: Path) -> int:
        with h5py.File(path, "r") as handle:
            _, data, times = self._required_datasets(handle)
            n_time, _, _ = self._dimensions(data, times)
            return n_time

    def read_block(
        self,
        path: Path,
        sample_start: int,
        sample_stop: int,
        channel_slice: slice = slice(None),
    ) -> DasBlock:
        with h5py.File(path, "r") as handle:
            raw, data, times = self._required_datasets(handle)
            n_time, n_channel, transposed = self._dimensions(data, times)
            self._validate_sample_bounds(sample_start, sample_stop, n_time)
            channel_start, channel_stop = self._channel_bounds(channel_slice, n_channel)
            if transposed:
                values = data[channel_start:channel_stop, sample_start:sample_stop].T
            else:
                values = data[sample_start:sample_stop, channel_start:channel_stop]

            spacing = float(self._required_attr(raw, self.ATTR_CHANNEL_SPACING))
            start_locus = int(self._required_attr(raw, self.ATTR_START_LOCUS))
            metadata = self._metadata(
                path,
                self._time_at(times, sample_start),
                float(self._required_attr(raw, self.ATTR_SAMPLE_RATE)),
                (start_locus + np.arange(channel_start, channel_stop)) * spacing,
                self._required_attr(raw, self.ATTR_UNIT),
                source_start_time=self._time_at(times, 0),
            )
        return DasBlock(np.asarray(values, dtype=np.float32), metadata, sample_start)

    def _required_datasets(self, handle: h5py.File):
        raw = handle.get(self.RAW_GROUP)
        if raw is None:
            raise ValueError(f"missing required group {self.RAW_GROUP}")
        try:
            return raw, raw[self.DATASET_DATA], raw[self.DATASET_TIME]
        except KeyError as error:
            raise ValueError(f"missing required dataset {error.args[0]}") from error

    def _dimensions(self, data: h5py.Dataset, times: h5py.Dataset) -> tuple[int, int, bool]:
        if data.ndim != 2:
            raise ValueError("RawData must be two-dimensional")
        if times.ndim != 1:
            raise ValueError("RawDataTime must be one-dimensional")
        n_time = len(times)
        if data.shape[0] == n_time:
            return n_time, data.shape[1], False
        if data.shape[1] == n_time:
            return n_time, data.shape[0], True
        raise ValueError("RawData dimensions do not match RawDataTime")

    @staticmethod
    def _required_attr(raw: h5py.Group, name: str):
        if name not in raw.attrs:
            raise ValueError(f"missing required attribute {name}")
        return raw.attrs[name]

    @staticmethod
    def _time_at(times: h5py.Dataset, index: int) -> datetime:
        return datetime.fromtimestamp(int(times[index]) / 1_000_000, tz=UTC)

    def _metadata(
        self,
        path: Path,
        start_time: datetime,
        sample_rate_hz: float,
        channel_distance_m: np.ndarray,
        unit: str | bytes,
        source_start_time: datetime | None = None,
    ) -> DasMetadata:
        if isinstance(unit, bytes):
            unit = unit.decode()
        return DasMetadata(
            source_path=str(path),
            start_time_utc=start_time,
            sample_rate_hz=sample_rate_hz,
            channel_distance_m=np.asarray(channel_distance_m),
            unit=str(unit),
            warnings=self._filename_warnings(path, source_start_time or start_time),
        )

    @staticmethod
    def _validate_sample_bounds(start: int, stop: int, n_time: int) -> None:
        if start < 0 or stop < start or stop > n_time:
            raise ValueError("sample bounds are outside RawData")

    @staticmethod
    def _channel_bounds(channel_slice: slice, n_channel: int) -> tuple[int, int]:
        if channel_slice.step not in (None, 1):
            raise ValueError("channel bounds must use a unit step")
        start = 0 if channel_slice.start is None else channel_slice.start
        stop = n_channel if channel_slice.stop is None else channel_slice.stop
        if start < 0 or stop < start or stop > n_channel:
            raise ValueError("channel bounds are outside RawData")
        return start, stop

    @staticmethod
    def _filename_warnings(path: Path, data_time: datetime) -> tuple[str, ...]:
        match = re.search(r"(?<!\d)(\d{8}T\d{6})Z?(?!\d)", path.name)
        if match is None:
            return ()
        filename_time = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(
            tzinfo=UTC
        )
        if filename_time == data_time:
            return ()
        return ("filename timestamp disagrees with RawDataTime",)
