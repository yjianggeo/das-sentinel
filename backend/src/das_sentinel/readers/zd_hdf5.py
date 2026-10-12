import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO

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

    def probe(self, path: Path | BinaryIO) -> ProbeResult:
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
                self._dimensions(data, times)
                for attribute_name in (
                    self.ATTR_SAMPLE_RATE,
                    self.ATTR_START_LOCUS,
                    self.ATTR_UNIT,
                ):
                    if attribute_name not in raw.attrs:
                        return ProbeResult(False, self.FORMAT_NAME, f"missing {attribute_name}")
                self._spacing(raw, path)
        except (OSError, TypeError, ValueError) as error:
            return ProbeResult(False, self.FORMAT_NAME, str(error))
        return ProbeResult(True, self.FORMAT_NAME)

    def read_metadata(self, path: Path | BinaryIO) -> DasMetadata:
        metadata, _ = self.read_metadata_and_count(path)
        return metadata

    def read_metadata_and_count(self, path: Path | BinaryIO) -> tuple[DasMetadata, int]:
        with h5py.File(path, "r") as handle:
            raw, data, times = self._required_datasets(handle)
            n_time, n_channel, _ = self._dimensions(data, times)
            start_time = self._time_at(times, 0, path)
            spacing, inferred = self._spacing(raw, path)
            start_locus = int(self._required_attr(raw, self.ATTR_START_LOCUS))
            metadata = self._metadata(
                path,
                start_time,
                float(self._required_attr(raw, self.ATTR_SAMPLE_RATE)),
                (start_locus + np.arange(n_channel)) * spacing,
                self._required_attr(raw, self.ATTR_UNIT),
                spacing_inferred=inferred,
            )
            return metadata, n_time

    def sample_count(self, path: Path | BinaryIO) -> int:
        with h5py.File(path, "r") as handle:
            _, data, times = self._required_datasets(handle)
            n_time, _, _ = self._dimensions(data, times)
            return n_time

    def read_block(
        self,
        path: Path | BinaryIO,
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

            spacing, inferred = self._spacing(raw, path)
            start_locus = int(self._required_attr(raw, self.ATTR_START_LOCUS))
            metadata = self._metadata(
                path,
                self._time_at(times, sample_start, path),
                float(self._required_attr(raw, self.ATTR_SAMPLE_RATE)),
                (start_locus + np.arange(channel_start, channel_stop)) * spacing,
                self._required_attr(raw, self.ATTR_UNIT),
                source_start_time=self._time_at(times, 0, path),
                spacing_inferred=inferred,
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
        if n_time == 0 or 0 in data.shape:
            raise ValueError("empty RawData or RawDataTime")
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
    def _source_name(path: Path | BinaryIO) -> str:
        name = path.name if isinstance(path, Path) else getattr(path, "name", "")
        return Path(name).name if isinstance(name, (str, Path)) else ""

    @classmethod
    def _filename_utc_offset(
        cls, path: Path | BinaryIO, raw_start: datetime
    ) -> timedelta | None:
        match = re.search(
            r"_UTC(?P<offset>[+-]?\d{1,2})_(?P<stamp>\d{12}(?:\d{2})?)(?!\d)",
            cls._source_name(path),
            re.IGNORECASE,
        )
        if match is None:
            return None
        stamp = match.group("stamp")
        format_string = "%Y%m%d%H%M%S" if len(stamp) == 14 else "%Y%m%d%H%M"
        filename_clock = datetime.strptime(stamp, format_string).replace(tzinfo=UTC)
        raw_clock = raw_start.replace(microsecond=0)
        if len(stamp) == 12:
            raw_clock = raw_clock.replace(second=0)
        if filename_clock != raw_clock:
            return None
        return timedelta(hours=int(match.group("offset")))

    @classmethod
    def _time_at(
        cls, times: h5py.Dataset, index: int, path: Path | BinaryIO
    ) -> datetime:
        raw_start = datetime.fromtimestamp(int(times[0]) / 1_000_000, tz=UTC)
        raw_time = datetime.fromtimestamp(int(times[index]) / 1_000_000, tz=UTC)
        return raw_time - (cls._filename_utc_offset(path, raw_start) or timedelta())

    def _metadata(
        self,
        path: Path | BinaryIO,
        start_time: datetime,
        sample_rate_hz: float,
        channel_distance_m: np.ndarray,
        unit: str | bytes,
        source_start_time: datetime | None = None,
        spacing_inferred: bool = False,
    ) -> DasMetadata:
        if isinstance(unit, bytes):
            unit = unit.decode()
        warnings = self._filename_warnings(path, source_start_time or start_time)
        if spacing_inferred:
            warnings = (*warnings, "channel spacing inferred from filename")
        return DasMetadata(
            source_path=str(path),
            start_time_utc=start_time,
            sample_rate_hz=sample_rate_hz,
            channel_distance_m=np.asarray(channel_distance_m),
            unit=str(unit),
            warnings=warnings,
        )

    def _spacing(self, raw: h5py.Group, path: Path | BinaryIO) -> tuple[float, bool]:
        """Channel spacing in metres, plus whether it was filename-inferred."""
        if self.ATTR_CHANNEL_SPACING in raw.attrs:
            return float(raw.attrs[self.ATTR_CHANNEL_SPACING]), False
        hint = self._spacing_from_filename(path)
        if hint is None:
            if self.ATTR_CHANNEL_SPACING in raw.parent.attrs:
                spacing = float(raw.parent.attrs[self.ATTR_CHANNEL_SPACING])
                if not np.isfinite(spacing) or spacing <= 0:
                    raise ValueError(f"{self.ATTR_CHANNEL_SPACING} must be finite and positive")
                return spacing, False
            raise ValueError(
                f"missing required attribute {self.ATTR_CHANNEL_SPACING} "
                "and no filename spacing hint"
            )
        return hint, True

    @classmethod
    def _spacing_from_filename(cls, path: Path | BinaryIO) -> float | None:
        """Infer channel spacing from ZD names like ``site__3400m_8m_4m_...``.

        The interrogator naming convention embeds ``<gauge length>m_<spacing>m``;
        when several ``Xm`` tokens exist the spacing is the last one.
        """
        name = cls._source_name(path)
        # Legacy names end in gauge-length metres followed by a bare spacing
        # token; the last ``Xm`` is not the channel spacing. Use acquisition
        # metadata for these files instead of stretching their distance axis.
        if re.search(r"_\d+(?:\.\d+)?m_\d+(?:\.\d+)?_\d+(?:\.\d+)?Hz_", name, re.IGNORECASE):
            return None
        matches = re.findall(r"_(\d+(?:\.\d+)?)m(?=_)", name)
        if not matches:
            return None
        return float(matches[-1])

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
    def _filename_warnings(
        path: Path | BinaryIO,
        data_time: datetime,
    ) -> tuple[str, ...]:
        name = path.name if isinstance(path, Path) else ""
        match = re.search(r"(?<!\d)(\d{8}T\d{6})Z?(?!\d)", name)
        if match is None:
            return ()
        filename_time = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(
            tzinfo=UTC
        )
        if filename_time == data_time:
            return ()
        return ("filename timestamp disagrees with RawDataTime",)
