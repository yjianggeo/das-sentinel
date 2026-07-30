from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import BinaryIO, Protocol

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from das_sentinel.config import ProcessingProfile as ProfileConfig
from das_sentinel.persistence.models import ProcessingProfile, Site
from das_sentinel.persistence.repositories import SourceFileRepository
from das_sentinel.readers.base import ProbeResult
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.services.jobs import JobService, JobSnapshot


class DiscoveryConfigurationError(ValueError):
    """The site cannot be scanned with its current persisted configuration."""


class DiscoveryPathError(ValueError):
    """A discovery path is missing, unsafe, or outside the configured root."""


@dataclass(frozen=True)
class DiscoverySite:
    """Immutable site/profile snapshot used while polling outside a transaction."""

    site_id: str
    data_root: Path
    profile_id: str
    profile_version: int
    recursive_discovery: bool


@dataclass(frozen=True)
class DiscoveredFile:
    """Canonical metadata captured for one stable source file."""

    path: Path
    fingerprint: str
    size_bytes: int
    mtime_ns: int

    @classmethod
    def from_path(cls, path: Path) -> DiscoveredFile:
        return _snapshot_file(path).discovered()


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    size_bytes: int
    mtime_ns: int

    @property
    def signature(self) -> tuple[Path, int, int]:
        return (self.path, self.size_bytes, self.mtime_ns)

    def discovered(self) -> DiscoveredFile:
        payload = json.dumps(
            (str(self.path), self.size_bytes, self.mtime_ns),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        return DiscoveredFile(
            path=self.path,
            fingerprint=hashlib.sha256(payload).hexdigest(),
            size_bytes=self.size_bytes,
            mtime_ns=self.mtime_ns,
        )


@dataclass
class _Observation:
    snapshot: _FileSnapshot
    since: float


class _ProbeReader(Protocol):
    def probe(self, source: BinaryIO) -> ProbeResult: ...


class StableFileTracker:
    """Emit each supported fingerprint once after a monotonic stability window."""

    def __init__(
        self,
        *,
        reader: _ProbeReader | None = None,
        stable_seconds: float = 10.0,
    ) -> None:
        if (
            isinstance(stable_seconds, bool)
            or not isinstance(stable_seconds, (int, float))
            or not math.isfinite(stable_seconds)
            or stable_seconds <= 0
        ):
            raise ValueError("stable seconds must be a positive finite number")
        self.reader = reader or ZdHdf5Reader()
        self.stable_seconds = float(stable_seconds)
        self._observations: dict[str, _Observation] = {}
        self._emitted: set[str] = set()

    def observe(
        self,
        path: Path,
        *,
        now: float,
        root: Path | None = None,
    ) -> DiscoveredFile | None:
        observed_at = _monotonic_value(now)
        key = _path_key(path)
        try:
            anchor_root = _validated_root(root or Path(path).parent)
            with _open_anchored_source(
                anchor_root,
                Path(path),
            ) as source_context:
                current = source_context.snapshot()
                previous = self._observations.get(key)
                if (
                    previous is None
                    or previous.snapshot.signature != current.signature
                ):
                    self._observations[key] = _Observation(
                        current,
                        observed_at,
                    )
                    return None
                if observed_at < previous.since:
                    raise ValueError(
                        "monotonic observation time moved backwards"
                    )
                if observed_at - previous.since < self.stable_seconds:
                    return None

                candidate = current.discovered()
                if candidate.fingerprint in self._emitted:
                    return None

                try:
                    probe = self.reader.probe(source_context.stream)
                except (OSError, TypeError, ValueError):
                    self._observations[key] = _Observation(
                        current,
                        observed_at,
                    )
                    return None
                if not probe.supported:
                    self._observations[key] = _Observation(
                        current,
                        observed_at,
                    )
                    return None

                after_probe = source_context.snapshot()
                if after_probe.signature != current.signature:
                    self._observations[key] = _Observation(
                        after_probe,
                        observed_at,
                    )
                    return None
                if not _path_still_targets_source(
                    anchor_root,
                    Path(path),
                    source_context.stat_result(),
                ):
                    self._observations.pop(key, None)
                    return None

                self._emitted.add(candidate.fingerprint)
                return candidate
        except (DiscoveryPathError, FileNotFoundError, OSError):
            self._observations.pop(key, None)
            return None

    def retain(self, paths: Iterable[Path]) -> None:
        """Reset baselines for files absent from the current directory poll."""

        present = {_path_key(path) for path in paths}
        for key in tuple(self._observations):
            if key not in present:
                del self._observations[key]


def fingerprint(path: Path) -> str:
    """Return the approved SHA-256 of canonical path, size, and mtime_ns."""

    return DiscoveredFile.from_path(path).fingerprint


def load_discovery_site(session: Session, site_id: str) -> DiscoverySite:
    """Load an enabled site's unique current profile into a detached value."""

    row = session.get(Site, site_id)
    if row is None:
        raise DiscoveryConfigurationError(f"site not found: {site_id}")
    if not row.enabled:
        raise DiscoveryConfigurationError("site is disabled")
    profiles = list(
        session.scalars(
            select(ProcessingProfile).where(
                ProcessingProfile.site_id == row.id,
                ProcessingProfile.is_current.is_(True),
            )
        )
    )
    if len(profiles) != 1:
        raise DiscoveryConfigurationError(
            "site must have exactly one current profile"
        )
    profile = profiles[0]
    try:
        validated_profile = ProfileConfig.model_validate_json(
            profile.payload_json,
            strict=True,
        )
    except (TypeError, ValidationError) as error:
        raise DiscoveryConfigurationError(
            "current processing profile, including recursive_discovery, is invalid"
        ) from error
    recursive = validated_profile.recursive_discovery
    root = Path(row.data_path)
    _require_lexical_absolute(root, field="site data root")
    return DiscoverySite(
        site_id=row.id,
        data_root=root,
        profile_id=profile.id,
        profile_version=profile.version,
        recursive_discovery=recursive,
    )


def discover_batch(
    site: DiscoverySite,
    path: Path,
    *,
    tracker: StableFileTracker,
    now: float,
) -> tuple[DiscoveredFile, ...]:
    """Poll one file or directory without opening a database transaction."""

    root = _validated_root(site.data_root)
    scan_path = _validate_under_root(root, path)
    candidates = _candidate_files(
        root,
        scan_path,
        recursive=site.recursive_discovery,
    )
    tracker.retain(candidates)
    stable: list[DiscoveredFile] = []
    for candidate in candidates:
        discovered = tracker.observe(candidate, now=now, root=root)
        if discovered is not None:
            _validate_discovered(root, discovered)
            stable.append(discovered)
    return tuple(stable)


def enqueue_discovered(
    session: Session,
    site: DiscoverySite,
    files: Sequence[DiscoveredFile],
    pipeline_version: str,
    *,
    artifact_root: Path = Path("."),
) -> tuple[JobSnapshot, ...]:
    """Persist stable files and enqueue jobs; flush while the caller owns commit."""

    if not isinstance(pipeline_version, str):
        raise DiscoveryConfigurationError("pipeline version must be a string")
    pipeline_version = pipeline_version.strip()
    if not pipeline_version or len(pipeline_version) > 128:
        raise DiscoveryConfigurationError(
            "pipeline version must contain between 1 and 128 characters"
        )

    current = load_discovery_site(session, site.site_id)
    if (
        current.profile_id != site.profile_id
        or current.profile_version != site.profile_version
    ):
        raise DiscoveryConfigurationError(
            "current processing profile changed after discovery"
        )
    root = _validated_root(current.data_root)

    validated: list[DiscoveredFile] = []
    for item in files:
        if not isinstance(item, DiscoveredFile):
            raise DiscoveryPathError(
                "discovered files must be frozen DiscoveredFile values"
            )
        _validate_discovered(root, item)
        validated.append(item)

    sources = SourceFileRepository(session)
    jobs = JobService(session, artifact_root)
    snapshots: list[JobSnapshot] = []
    for item in validated:
        source = sources.find(site_id=current.site_id, fingerprint=item.fingerprint)
        if source is None:
            source = sources.add(
                site_id=current.site_id,
                path=str(item.path),
                fingerprint=item.fingerprint,
                size_bytes=item.size_bytes,
                mtime_ns=item.mtime_ns,
            )
        snapshots.append(
            jobs.enqueue(source.id, current.profile_id, pipeline_version)
        )
    return tuple(snapshots)


def watch_site(
    site: DiscoverySite,
    path: Path,
    *,
    tracker: StableFileTracker,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_discovered: Callable[[tuple[DiscoveredFile, ...]], None] | None = None,
    poll_interval: float = 5.0,
) -> None:
    """Poll deterministically and invoke persistence only after each file probe."""

    if (
        isinstance(poll_interval, bool)
        or not isinstance(poll_interval, (int, float))
        or not math.isfinite(poll_interval)
        or poll_interval <= 0
    ):
        raise ValueError("poll interval must be a positive finite number")
    monotonic = clock or time.monotonic
    wait = sleep or time.sleep
    cancelled = should_stop or (lambda: False)
    emit = on_discovered or (lambda _files: None)

    while not cancelled():
        files = discover_batch(
            site,
            path,
            tracker=tracker,
            now=monotonic(),
        )
        if files:
            emit(files)
        if cancelled():
            return
        wait(float(poll_interval))


@dataclass(frozen=True)
class _AnchoredSource:
    path: Path
    stream: BinaryIO

    def stat_result(self) -> os.stat_result:
        return os.fstat(self.stream.fileno())

    def snapshot(self) -> _FileSnapshot:
        result = self.stat_result()
        if not stat.S_ISREG(result.st_mode):
            raise DiscoveryPathError("source handle must identify a regular file")
        return _FileSnapshot(
            path=self.path,
            size_bytes=result.st_size,
            mtime_ns=result.st_mtime_ns,
        )


class _SourceHandle:
    def __init__(self, value: int, close: Callable[[int], object]) -> None:
        self.value: int | None = value
        self._close = close

    def close(self) -> None:
        if self.value is not None:
            value = self.value
            self.value = None
            self._close(value)


@contextmanager
def _open_anchored_source(
    root: Path,
    path: Path,
) -> Iterator[_AnchoredSource]:
    candidate = Path(path)
    _require_lexical_absolute(candidate, field="source path")
    if candidate.suffix.lower() != ".h5":
        raise DiscoveryPathError("source path must have an .h5 extension")
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise DiscoveryPathError(
            "source path must remain beneath the configured site root"
        ) from error
    if not relative.parts:
        raise DiscoveryPathError("source path must identify a regular file")

    with ExitStack() as stack:
        try:
            parent_handle: _SourceHandle | None = None
            for root_component in _absolute_path_chain(root):
                opened = _open_source_component(
                    Path(root_component),
                    parent_handle,
                    expect_directory=True,
                )
                stack.callback(opened.close)
                parent_handle = opened
            current = root
            final_handle: _SourceHandle | None = None
            for index, component in enumerate(relative.parts):
                current /= component
                expect_directory = index < len(relative.parts) - 1
                opened = _open_source_component(
                    current,
                    parent_handle,
                    expect_directory=expect_directory,
                )
                stack.callback(opened.close)
                parent_handle = opened
                final_handle = opened
            if final_handle is None:
                raise OSError("source path is missing a final component")
            stream = stack.enter_context(_source_stream(final_handle))
            source = _AnchoredSource(candidate, stream)
            source.snapshot()
        except OSError as error:
            raise DiscoveryPathError(
                "source path must use an existing no-follow regular file chain"
            ) from error
        yield source


def _absolute_path_chain(path: PurePath) -> tuple[PurePath, ...]:
    if not path.is_absolute() or not path.anchor or ".." in path.parts:
        raise DiscoveryPathError("filesystem anchor path must be absolute and normalized")
    current = type(path)(path.anchor)
    components = [current]
    for component in path.parts[1:]:
        current /= component
        components.append(current)
    return tuple(components)


@contextmanager
def _source_stream(handle: _SourceHandle) -> Iterator[BinaryIO]:
    if handle.value is None:
        raise OSError("source handle is already closed")
    if os.name == "nt":
        import msvcrt

        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        descriptor = msvcrt.open_osfhandle(handle.value, flags)
        handle.value = None
    else:
        descriptor = os.dup(handle.value)
    try:
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    try:
        yield stream
    finally:
        stream.close()


def _path_still_targets_source(
    root: Path,
    path: Path,
    expected: os.stat_result,
) -> bool:
    try:
        with _open_anchored_source(root, path) as reopened:
            return os.path.samestat(expected, reopened.stat_result())
    except (DiscoveryPathError, FileNotFoundError, OSError):
        return False


def _open_source_component(
    path: Path,
    parent_handle: _SourceHandle | None,
    *,
    expect_directory: bool,
) -> _SourceHandle:
    if os.name == "nt":
        return _open_windows_source_component(path, expect_directory)
    return _open_posix_source_component(
        path,
        parent_handle,
        expect_directory,
    )


def _open_posix_source_component(
    path: Path,
    parent_handle: _SourceHandle | None,
    expect_directory: bool,
) -> _SourceHandle:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if expect_directory:
        flags |= os.O_DIRECTORY
    opened: str | Path = path
    directory_descriptor = None
    if parent_handle is not None:
        if parent_handle.value is None:
            raise OSError("source parent handle is closed")
        opened = path.name
        directory_descriptor = parent_handle.value
    descriptor = os.open(opened, flags, dir_fd=directory_descriptor)
    try:
        mode = os.fstat(descriptor).st_mode
        valid = stat.S_ISDIR(mode) if expect_directory else stat.S_ISREG(mode)
        if not valid:
            raise OSError("source component has the wrong file type")
        return _SourceHandle(descriptor, os.close)
    except BaseException:
        os.close(descriptor)
        raise


def _open_windows_source_component(
    path: Path,
    expect_directory: bool,
) -> _SourceHandle:
    import ctypes
    from ctypes import wintypes

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("creation_time", wintypes.FILETIME),
            ("last_access_time", wintypes.FILETIME),
            ("last_write_time", wintypes.FILETIME),
            ("volume_serial_number", wintypes.DWORD),
            ("file_size_high", wintypes.DWORD),
            ("file_size_low", wintypes.DWORD),
            ("number_of_links", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ByHandleFileInformation),
    ]
    get_information.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    generic_read = 0x80000000
    share_read_write = 0x00000001 | 0x00000002
    open_existing = 3
    file_flag_backup_semantics = 0x02000000
    file_flag_open_reparse_point = 0x00200000
    handle = create_file(
        str(path),
        generic_read,
        share_read_write,
        None,
        open_existing,
        file_flag_backup_semantics | file_flag_open_reparse_point,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        information = ByHandleFileInformation()
        if not get_information(handle, ctypes.byref(information)):
            raise ctypes.WinError(ctypes.get_last_error())
        file_attribute_directory = 0x00000010
        file_attribute_reparse_point = 0x00000400
        is_directory = bool(
            information.file_attributes & file_attribute_directory
        )
        is_reparse = bool(
            information.file_attributes & file_attribute_reparse_point
        )
        if is_reparse or is_directory != expect_directory:
            raise OSError(
                "source component has the wrong type or is a reparse point"
            )
        result = _SourceHandle(handle, close_handle)
        handle = None
        return result
    finally:
        if handle is not None:
            close_handle(handle)


def _validate_discovered(root: Path, item: DiscoveredFile) -> None:
    current_path = _validate_under_root(root, item.path, regular_file=True)
    current = _snapshot_file(current_path).discovered()
    if current != item:
        raise DiscoveryPathError("source file changed after stable discovery")


def _candidate_files(
    root: Path,
    scan_path: Path,
    *,
    recursive: bool,
) -> tuple[Path, ...]:
    if scan_path.is_file():
        return (scan_path,) if scan_path.suffix.lower() == ".h5" else ()
    if not scan_path.is_dir():
        raise DiscoveryPathError("scan path must be a regular file or directory")

    candidates: list[Path] = []
    pending = [scan_path]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(
                    iterator,
                    key=lambda entry: (entry.name.casefold(), entry.name),
                )
        except OSError as error:
            raise DiscoveryPathError(
                f"cannot enumerate scan directory: {directory}"
            ) from error
        for entry in entries:
            entry_path = Path(entry.path)
            if _is_reparse(entry_path):
                raise DiscoveryPathError(
                    f"scan path contains a symlink or reparse point: {entry_path}"
                )
            try:
                if entry.is_file(follow_symlinks=False):
                    if entry_path.suffix.lower() == ".h5":
                        candidates.append(
                            _validate_under_root(
                                root,
                                entry_path,
                                regular_file=True,
                            )
                        )
                elif recursive and entry.is_dir(follow_symlinks=False):
                    pending.append(
                        _validate_under_root(
                            root,
                            entry_path,
                            directory=True,
                        )
                    )
            except OSError as error:
                raise DiscoveryPathError(
                    f"cannot inspect discovery entry: {entry_path}"
                ) from error
    return tuple(
        sorted(candidates, key=lambda value: (str(value).casefold(), str(value)))
    )

def _validated_root(value: Path) -> Path:
    root = Path(value)
    _require_lexical_absolute(root, field="site data root")
    _ensure_no_reparse_components(root)
    try:
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise DiscoveryPathError(
            "site data root must be an existing directory"
        ) from error
    if not resolved.is_dir():
        raise DiscoveryPathError("site data root must be an existing directory")
    return resolved


def _validate_under_root(
    root: Path,
    value: Path,
    *,
    regular_file: bool = False,
    directory: bool = False,
) -> Path:
    candidate = Path(value)
    _require_lexical_absolute(candidate, field="scan path")
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise DiscoveryPathError(
            "scan path must remain beneath the configured site root"
        ) from error
    current = root
    for part in relative.parts:
        current /= part
        if _is_reparse(current):
            raise DiscoveryPathError(
                f"scan path contains a symlink or reparse point: {current}"
            )
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise DiscoveryPathError("scan path must exist") from error
    if not resolved.is_relative_to(root):
        raise DiscoveryPathError(
            "scan path must remain beneath the configured site root"
        )
    if regular_file and not resolved.is_file():
        raise DiscoveryPathError("discovery candidate must be a regular file")
    if directory and not resolved.is_dir():
        raise DiscoveryPathError("discovery path component must be a directory")
    return resolved


def _snapshot_file(value: Path) -> _FileSnapshot:
    path = Path(value)
    _require_lexical_absolute(path, field="source path")
    if path.suffix.lower() != ".h5":
        raise DiscoveryPathError("source path must have an .h5 extension")
    if _is_reparse(path):
        raise DiscoveryPathError(
            f"source path contains a symlink or reparse point: {path}"
        )
    try:
        resolved = path.resolve(strict=True)
        result = resolved.stat()
    except FileNotFoundError:
        raise
    except (OSError, RuntimeError) as error:
        raise DiscoveryPathError("source path must be readable") from error
    if not stat.S_ISREG(result.st_mode):
        raise DiscoveryPathError("source path must be a regular file")
    return _FileSnapshot(
        path=resolved,
        size_bytes=result.st_size,
        mtime_ns=result.st_mtime_ns,
    )


def _ensure_no_reparse_components(path: Path) -> None:
    parts = path.parts
    if not parts:
        raise DiscoveryPathError("path has no filesystem components")
    current = Path(parts[0])
    if _is_reparse(current):
        raise DiscoveryPathError(
            f"path contains a symlink or reparse point: {current}"
        )
    for part in parts[1:]:
        current /= part
        if _is_reparse(current):
            raise DiscoveryPathError(
                f"path contains a symlink or reparse point: {current}"
            )


def _is_reparse(path: Path) -> bool:
    try:
        result = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise DiscoveryPathError(f"cannot inspect path component: {path}") from error
    attributes = getattr(result, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(result.st_mode) or bool(attributes & reparse_flag)


def _require_lexical_absolute(path: Path, *, field: str) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise DiscoveryPathError(f"{field} must be absolute and normalized")


def _path_key(path: Path) -> str:
    return os.path.normcase(str(Path(path).absolute()))


def _monotonic_value(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("observation time must be a finite monotonic number")
    return float(value)
