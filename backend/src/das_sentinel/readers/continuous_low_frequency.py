"""Bounded, read-only resampling of continuous ZD data onto a UTC grid."""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from math import ceil, floor
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import resample_poly

from das_sentinel.domain import DasBlock, DasMetadata
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.services.discovery import fingerprint

_RATE = 5
_STEP_US = 200_000
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_CHUNK_SECONDS = 20
_HALO_SECONDS = 10
_MAX_RAW_VALUES = 2_000_000
_MAX_IO_VALUES = 128 * 1024 * 1024 // np.dtype("float32").itemsize
_MAX_OUTPUT_BYTES = 256 * 1024 * 1024


class CoreCache:
    """Bounded reuse of unchanged, fully validated 20-second resampling cores."""
    def __init__(self, max_bytes=64 * 1024 * 1024):
        self.max_bytes, self.bytes, self.hits = max_bytes, 0, 0
        self.values = OrderedDict()

    def get(self, key):
        value = self.values.get(key)
        if value is not None:
            self.values.move_to_end(key)
            self.hits += 1
        return value

    def put(self, key, value):
        if value.nbytes > self.max_bytes:
            return
        previous = self.values.pop(key, None)
        if previous is not None:
            self.bytes -= previous.nbytes
        while self.values and self.bytes + value.nbytes > self.max_bytes:
            self.bytes -= self.values.popitem(last=False)[1].nbytes
        self.values[key] = value.copy()
        self.bytes += value.nbytes


@dataclass(frozen=True)
class ContinuousLowFrequencyWindow:
    block: DasBlock
    analysis_start: int
    analysis_stop: int
    sources: tuple[dict, ...]
    warnings: tuple[str, ...]


@dataclass
class _Source:
    path: Path
    metadata: DasMetadata
    count: int
    signature: tuple[int, ...]
    fingerprint: str
    sample_start: int
    sample_stop: int = 0

    @property
    def end(self) -> datetime:
        return self.metadata.start_time_utc + timedelta(
            seconds=self.count / self.metadata.sample_rate_hz
        )


def _microseconds(value: datetime) -> int:
    elapsed = value - _EPOCH
    return (elapsed.days * 86400 + elapsed.seconds) * 1_000_000 + elapsed.microseconds


def _grid_ceil(value: datetime, step_us: int = _STEP_US) -> int:
    return -(-_microseconds(value) // step_us)


def _signature(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino


def _verify_source(source: _Source) -> None:
    try:
        unchanged = (
            _signature(source.path) == source.signature
            and fingerprint(source.path) == source.fingerprint
        )
    except OSError as error:
        raise ValueError(
            f"源文件在连续读取期间发生变化或不可用：{source.path}"
        ) from error
    if not unchanged:
        raise ValueError(f"源文件在连续读取期间发生变化：{source.path}")


def _check_clock(source: _Source, warnings: list[str]) -> None:
    """Check every time sample using bounded timestamp-only reads."""
    fs = source.metadata.sample_rate_hz
    with h5py.File(source.path, "r") as handle:
        times = handle[f"{ZdHdf5Reader.RAW_GROUP}/{ZdHdf5Reader.DATASET_TIME}"]
        raw_start = int(times[0])
        previous = None
        for first in range(0, source.count, max(1, int(fs * _CHUNK_SECONDS))):
            stop = min(source.count, first + max(1, int(fs * _CHUNK_SECONDS)))
            values = np.asarray(times[first:stop], dtype=np.int64)
            expected = np.arange(first, stop, dtype=np.float64) * (1_000_000 / fs)
            if (
                np.any(np.diff(values) <= 0)
                or (previous is not None and int(values[0]) <= previous)
                or np.any(np.abs((values - raw_start) - expected) > 2.0)
            ):
                raise ValueError(
                    f"文件内RawDataTime时标不连续或与采样率不一致：{source.path}"
                )
            previous = int(values[-1])
        offset_us = raw_start - _microseconds(source.metadata.start_time_utc)
        if offset_us:
            warnings.append(
                f"{source.path.name}：文件名UTC偏移已由ZD读取器校正"
                f"（RawDataTime减去{offset_us / 1_000_000:g}秒）"
            )
    _verify_source(source)


def _raw_pieces(sources: list[_Source], start: datetime, stop: datetime):
    """Include interpolation neighbours, giving later files overlap ownership."""
    pieces = []
    for index, source in enumerate(sources):
        fs = source.metadata.sample_rate_hz
        begin = source.metadata.start_time_utc
        first = max(0, floor((start - begin).total_seconds() * fs) - 1)
        last = min(source.count, ceil((stop - begin).total_seconds() * fs) + 1)
        if index + 1 < len(sources):
            next_start = sources[index + 1].metadata.start_time_utc
            last = min(last, ceil((next_start - begin).total_seconds() * fs - 1e-7))
        if last <= first:
            continue
        times = (begin - start).total_seconds() + np.arange(first, last) / fs
        pieces.append((source, first, last, times))
        source.sample_start = min(source.sample_start, first)
        source.sample_stop = max(source.sample_stop, last)
    return pieces


def read_low_frequency_window(
    paths: Sequence[Path],
    start_utc: datetime,
    end_utc: datetime,
    distance_start_m: float | None = 0,
    distance_stop_m: float | None = None,
    *,
    margin_seconds: float = 300.0,
    sample_rate_hz: int = 5,
    core_cache: CoreCache | None = None,
) -> ContinuousLowFrequencyWindow:
    """Return [start-margin, end+margin) at 5 or 100 Hz with slice indices.

    Source coverage must extend a further 10 real seconds on both sides for
    anti-alias FIR context. Non-grid bounds select UTC-epoch grid samples inside
    the requested interval. Distances are inclusive. The function only
    resamples; the caller applies its low-frequency bandpass to the whole block.
    A None distance start selects the native first channel (including negative
    coordinates); the default zero retains the existing nonnegative selection.

    At 5 Hz, file clock seams up to 0.05 s retain the existing compatibility
    policy. At 100 Hz, seams may span at most one native sample period, capped
    at 0.05 s. Accepted seams are reported and mapped on the native-rate UTC
    grid before anti-alias resampling. No gaps are zero filled. Larger seams,
    internal clock defects and metadata changes fail.
    Raw reads use 20-second cores and 10-second halos with an at-most-128-MiB
    float32 cache, split by channel only when necessary. Resampling slices the
    cache into at most 64 channels and two million native values. Output is
    capped at 256 MiB; callers must split longer scans into overlapping jobs.
    At 100 Hz, one preceding raw core is retained within this call to reuse
    overlapping halos, only when all selected channels fit the 128-MiB cache.
    """
    if sample_rate_hz not in (5, 100):
        raise ValueError("输出采样率仅支持5Hz或100Hz")
    step_us = 1_000_000 // sample_rate_hz
    if (
        start_utc.tzinfo is None
        or start_utc.utcoffset() is None
        or end_utc.tzinfo is None
        or end_utc.utcoffset() is None
        or end_utc <= start_utc
    ):
        raise ValueError("起止时间必须包含时区，且结束时间晚于开始时间")
    if not np.isfinite(margin_seconds) or margin_seconds < 0:
        raise ValueError("保护区秒数必须为有限非负数")
    if (
        (distance_start_m is not None and (not np.isfinite(distance_start_m) or distance_start_m < 0))
        or (
            distance_stop_m is not None
            and (not np.isfinite(distance_stop_m) or distance_stop_m < 0
                 or (distance_start_m is not None and distance_stop_m < distance_start_m))
        )
    ):
        raise ValueError("距离范围必须为有效非负范围")
    if not paths:
        raise ValueError("未提供连续低频源文件")
    start_utc, end_utc = start_utc.astimezone(UTC), end_utc.astimezone(UTC)
    if _grid_ceil(end_utc, step_us) <= _grid_ceil(start_utc, step_us):
        raise ValueError(f"分析时间范围内没有{sample_rate_hz}Hz样本")
    requested_start = start_utc - timedelta(seconds=margin_seconds)
    requested_stop = end_utc + timedelta(seconds=margin_seconds)
    grid_start = _grid_ceil(requested_start, step_us)
    grid_stop = _grid_ceil(requested_stop, step_us)
    if grid_stop <= grid_start:
        raise ValueError(f"时间范围内没有{sample_rate_hz}Hz样本")
    block_start = _EPOCH + timedelta(microseconds=grid_start * step_us)
    block_stop = _EPOCH + timedelta(microseconds=grid_stop * step_us)
    read_start = min(requested_start, block_start) - timedelta(seconds=_HALO_SECONDS)
    read_stop = max(requested_stop, block_stop) + timedelta(seconds=_HALO_SECONDS)
    warnings: list[str] = []
    if block_start != requested_start or block_stop != requested_stop:
        warnings.append(
            f"非整{1 / sample_rate_hz:g}秒边界已映射到UTC绝对"
            f"{sample_rate_hz}Hz网格，分析范围按[start,end)取样"
        )

    reader = ZdHdf5Reader()
    selected: list[_Source] = []
    for path in dict.fromkeys(Path(p).resolve(strict=True) for p in paths):
        signature = _signature(path)
        source_fingerprint = fingerprint(path)
        metadata, count = reader.read_metadata_and_count(path)
        if not np.isfinite(metadata.sample_rate_hz) or metadata.sample_rate_hz <= 0:
            raise ValueError(f"原始采样率必须为有限正数：{path}")
        if count <= 0:
            raise ValueError(f"源文件不含采样数据：{path}")
        source = _Source(path, metadata, count, signature, source_fingerprint, count)
        _verify_source(source)
        if source.end > read_start and metadata.start_time_utc < read_stop:
            selected.append(source)
    selected.sort(key=lambda s: s.metadata.start_time_utc)
    if not selected or selected[0].metadata.start_time_utc > read_start:
        raise ValueError("前侧真实连续上下文不足：需要完整保护区及额外10秒抗混叠上下文")
    if selected[-1].end < read_stop:
        raise ValueError("后侧真实连续上下文不足：需要完整保护区及额外10秒抗混叠上下文")

    reference = selected[0].metadata
    fs = reference.sample_rate_hz
    if fs < sample_rate_hz:
        raise ValueError(f"选中源文件的原始采样率必须不低于{sample_rate_hz}Hz")
    seam_limit = 0.05 if sample_rate_hz == 5 else min(1 / fs, 0.05)
    distances = reference.channel_distance_m
    if (
        distances.ndim != 1
        or not len(distances)
        or not np.isfinite(distances).all()
        or np.any(np.diff(distances) <= 0)
    ):
        raise ValueError("源文件距离数组必须有限且严格递增")
    for index, source in enumerate(selected):
        metadata = source.metadata
        if (
            metadata.sample_rate_hz != fs
            or metadata.unit != reference.unit
            or not np.array_equal(metadata.channel_distance_m, distances)
        ):
            raise ValueError(f"连续文件的采样率、距离数组或单位不一致：{source.path}")
        warnings.extend(metadata.warnings)
        if index:
            seam = (metadata.start_time_utc - selected[index - 1].end).total_seconds()
            if abs(seam) > seam_limit + 1e-9:
                raise ValueError(
                    f"文件间不连续：时钟间隙/重叠{seam:+.6f}秒超过"
                    f"{seam_limit:g}秒：{source.path}"
                )
            if seam:
                warnings.append(
                    f"{source.path.name}（{metadata.start_time_utc.isoformat()}）："
                    f"文件边界时钟{'间隙' if seam > 0 else '重叠'}"
                    f"{seam:+.6f}秒，已按真实时标映射；小间隙线性插值，重叠取后文件"
                )
                if seam > 0:
                    warnings.append(
                        f"{metadata.start_time_utc.isoformat()}：小缺测内的高频能量不可恢复，"
                        "插值可能产生局部低频残差；跨此接缝的检测候选须复核"
                    )
        _check_clock(source, warnings)

    channel_start = 0 if distance_start_m is None else int(np.searchsorted(distances, distance_start_m, side="left"))
    channel_stop = (
        len(distances)
        if distance_stop_m is None
        else int(np.searchsorted(distances, distance_stop_m, side="right"))
    )
    channels = channel_stop - channel_start
    if channels <= 0:
        raise ValueError("距离范围内没有源通道")
    output_count = grid_stop - grid_start
    if output_count * channels * np.dtype("float32").itemsize > _MAX_OUTPUT_BYTES:
        raise ValueError("连续低频输出超过256MiB内存上限，请缩短时间窗或距离范围")
    native_length = ceil((_CHUNK_SECONDS + 2 * _HALO_SECONDS) * fs) + 4
    if native_length > _MAX_RAW_VALUES:
        raise ValueError("原始采样率超过连续低频分块内存上限")
    group_size = min(64, max(1, _MAX_RAW_VALUES // native_length))
    ratio = (Fraction(sample_rate_hz) / Fraction(str(fs))).limit_denominator(20_000)
    if not np.isclose(float(ratio), sample_rate_hz / fs, rtol=1e-10, atol=0):
        raise ValueError("原始采样率无法在有界滤波器长度内进行精确重采样")
    output = np.empty((output_count, channels), dtype=np.float32)
    previous_raw = None
    previous_ranges = {}
    for out_first in range(0, output_count, _CHUNK_SECONDS * sample_rate_hz):
        out_stop = min(output_count, out_first + _CHUNK_SECONDS * sample_rate_hz)
        core_start = block_start + timedelta(seconds=out_first / sample_rate_hz)
        core_stop = block_start + timedelta(seconds=out_stop / sample_rate_hz)
        chunk_start = core_start - timedelta(seconds=_HALO_SECONDS)
        chunk_stop = core_stop + timedelta(seconds=_HALO_SECONDS)
        pieces = _raw_pieces(selected, chunk_start, chunk_stop)
        cache_key = (core_start, core_stop, sample_rate_hz, fs, channel_start, channel_stop,
            tuple((str(source.path), source.signature, source.fingerprint, first, last)
                  for source, first, last, _ in pieces))
        cached = core_cache.get(cache_key) if core_cache is not None else None
        if cached is not None:
            output[out_first:out_stop] = cached
            continue
        raw_times = np.concatenate([part[3] for part in pieces])
        native_times = (
            np.arange(ceil((chunk_stop - chunk_start).total_seconds() * fs)) / fs
        )
        if (
            not len(raw_times)
            or raw_times[0] > native_times[0] + 2e-6
            or raw_times[-1] < native_times[-1] - 2e-6
            or np.any(np.diff(raw_times) <= 0)
        ):
            raise ValueError("真实连续上下文不足，无法在不补零的情况下映射原始采样网格")
        if len(raw_times) > _MAX_IO_VALUES:
            raise ValueError("原始片段超过连续低频读取缓存上限")
        io_group_size = max(1, _MAX_IO_VALUES // len(raw_times))
        reuse_raw = sample_rate_hz == 100 and io_group_size >= channels
        if not reuse_raw:
            previous_raw = None
            previous_ranges = {}
        for io_first in range(channel_start, channel_stop, io_group_size):
            io_stop = min(channel_stop, io_first + io_group_size)
            # Reuse one physical HDF5 read per piece across compute groups.
            # Allocate the whole cache once; release each piece after copying
            # so cross-file chunks never retain all piece arrays as well.
            values = np.empty((len(raw_times), io_stop - io_first), dtype=np.float32)
            cache_first = 0
            next_ranges = {}
            for source, first, last, _ in pieces:
                cache_stop = cache_first + last - first
                read_first = first
                prior = previous_ranges.get(source.path) if reuse_raw else None
                if previous_raw is not None and prior is not None:
                    old_first, old_last, old_offset = prior
                    if old_first <= first < old_last:
                        read_first = min(last, old_last)
                        count = read_first - first
                        offset = old_offset + first - old_first
                        values[cache_first:cache_first + count] = previous_raw[offset:offset + count]
                if read_first < last:
                    block = reader.read_block(
                        source.path, read_first, last, slice(io_first, io_stop)
                    )
                    if not np.isfinite(block.data).all():
                        raise ValueError(f"源文件包含非有限采样值：{source.path}")
                    values[cache_first + read_first - first:cache_stop] = block.data
                    del block
                if reuse_raw:
                    next_ranges[source.path] = (first, last, cache_first)
                cache_first = cache_stop
            for first_channel in range(io_first, io_stop, group_size):
                last_channel = min(io_stop, first_channel + group_size)
                aligned = np.empty(
                    (len(native_times), last_channel - first_channel), dtype=np.float32, order="F"
                )
                for channel in range(aligned.shape[1]):
                    aligned[:, channel] = np.interp(
                        native_times,
                        raw_times,
                        values[:, first_channel - io_first + channel],
                    )
                reduced = resample_poly(
                    aligned, ratio.numerator, ratio.denominator, axis=0
                )
                first = _HALO_SECONDS * sample_rate_hz
                output[
                    out_first:out_stop,
                    first_channel - channel_start : last_channel - channel_start,
                ] = reduced[first : first + out_stop - out_first]
            if reuse_raw:
                # Retain one bounded core, never clocks, metadata or cross-call values.
                # Source identity is checked again before returning, as for CoreCache.
                previous_raw, previous_ranges = values, next_ranges
            del values
        if core_cache is not None:
            # Every raw value in this core's halo was checked finite; source signatures are
            # checked again at return, including on cache hits, and all clocks were checked above.
            core_cache.put(cache_key, output[out_first:out_stop])

    provenance = []
    for index, source in enumerate(selected):
        _verify_source(source)
        if source.sample_stop <= source.sample_start:
            continue
        provenance.append(
            {
                "path": str(source.path),
                "fingerprint": source.fingerprint,
                "size_bytes": source.signature[0],
                "mtime_ns": source.signature[1],
                "sample_start": source.sample_start,
                "sample_stop": source.sample_stop,
                "sample_rate_hz": fs,
                "channel_start": channel_start,
                "channel_stop": channel_stop,
                "source_start_utc": source.metadata.start_time_utc.isoformat(),
                "source_stop_utc": source.end.isoformat(),
                "read_start_utc": (
                    source.metadata.start_time_utc
                    + timedelta(seconds=source.sample_start / fs)
                ).isoformat(),
                "read_stop_utc": (
                    source.metadata.start_time_utc
                    + timedelta(seconds=source.sample_stop / fs)
                ).isoformat(),
                "unit": source.metadata.unit,
                "source_verified_unchanged": True,
                "boundary_utc": source.metadata.start_time_utc.isoformat(),
                "previous_source_end_utc": selected[index - 1].end.isoformat()
                if index
                else None,
                "previous_seam_seconds": (
                    (
                        source.metadata.start_time_utc - selected[index - 1].end
                    ).total_seconds()
                    if index
                    else None
                ),
            }
        )
    unique_warnings = tuple(dict.fromkeys(warnings))
    metadata = DasMetadata(
        source_path=str(selected[0].path),
        start_time_utc=block_start,
        sample_rate_hz=float(sample_rate_hz),
        channel_distance_m=distances[channel_start:channel_stop].copy(),
        unit=reference.unit,
        warnings=unique_warnings,
    )
    return ContinuousLowFrequencyWindow(
        block=DasBlock(output, metadata, sample_offset=0),
        analysis_start=_grid_ceil(start_utc, step_us) - grid_start,
        analysis_stop=_grid_ceil(end_utc, step_us) - grid_start,
        sources=tuple(provenance),
        warnings=unique_warnings,
    )
