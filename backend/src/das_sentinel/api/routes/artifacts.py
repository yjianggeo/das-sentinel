from __future__ import annotations

import hashlib
import os
import stat
from contextlib import ExitStack
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response, status

from das_sentinel.api.auth import AuthenticatedAdmin
from das_sentinel.persistence.models import Artifact

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


class _ArtifactAccessError(RuntimeError):
    pass


class _ArtifactHandle:
    def __init__(self, value: int, close) -> None:
        self.value = value
        self._close = close

    def close(self) -> None:
        if self.value is not None:
            value = self.value
            self.value = None
            self._close(value)

    def read_all(self) -> bytes:
        if os.name == "nt":
            return _read_windows_handle(self.value)
        os.lseek(self.value, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(self.value, 64 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)


@router.get("/{artifact_id}")
def download_artifact(
    artifact_id: str,
    request: Request,
    admin: AuthenticatedAdmin,
) -> Response:
    del admin
    with request.app.state.session_factory() as session:
        row = session.get(Artifact, artifact_id)
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Artifact not found",
            )
        stored_path = row.path
        expected_sha256 = row.sha256
        media_type = row.media_type
    if media_type not in {"image/png", "application/json"}:
        raise _conflict()
    try:
        content = _read_registered_artifact(
            request.app.state.settings.app_data_dir / "artifacts",
            stored_path,
        )
    except _ArtifactAccessError as error:
        raise _conflict() from error
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise _conflict()
    return Response(content=content, media_type=media_type)


def _read_registered_artifact(root: Path, stored_path: str) -> bytes:
    anchor_root = Path(root)
    relative = Path(stored_path)
    if (
        not anchor_root.is_absolute()
        or ".." in anchor_root.parts
        or relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative == Path(".")
    ):
        raise _ArtifactAccessError("unsafe artifact registration")
    anchor_root = anchor_root.absolute()
    full_parts = list(anchor_root.parts) + list(relative.parts)
    if not full_parts:
        raise _ArtifactAccessError("empty artifact path")

    anchor = Path(anchor_root.anchor)
    if not anchor.anchor:
        raise _ArtifactAccessError("artifact root has no trusted anchor")
    with ExitStack() as handles:
        try:
            parent_handle = _open_artifact_component(
                anchor,
                None,
                expect_directory=True,
            )
            handles.callback(parent_handle.close)
            current = anchor
            root_components = list(anchor_root.parts[1:])
            all_components = root_components + list(relative.parts)
            for index, component in enumerate(all_components):
                current = current / component
                expect_directory = index < len(all_components) - 1
                opened = _open_artifact_component(
                    current,
                    parent_handle,
                    expect_directory=expect_directory,
                )
                handles.callback(opened.close)
                parent_handle = opened
            return parent_handle.read_all()
        except (OSError, ValueError) as error:
            raise _ArtifactAccessError("artifact cannot be opened safely") from error


def _open_artifact_component(
    path: Path,
    parent_handle: _ArtifactHandle | None,
    *,
    expect_directory: bool,
) -> _ArtifactHandle:
    if os.name == "nt":
        return _open_windows_artifact_component(path, expect_directory)
    return _open_posix_artifact_component(
        path,
        parent_handle,
        expect_directory,
    )


def _open_posix_artifact_component(
    path: Path,
    parent_handle: _ArtifactHandle | None,
    expect_directory: bool,
) -> _ArtifactHandle:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    if expect_directory:
        flags |= os.O_DIRECTORY
    opened = path if parent_handle is None else path.name
    descriptor = os.open(
        opened,
        flags,
        dir_fd=None if parent_handle is None else parent_handle.value,
    )
    try:
        mode = os.fstat(descriptor).st_mode
        valid = stat.S_ISDIR(mode) if expect_directory else stat.S_ISREG(mode)
        if not valid:
            raise OSError("artifact component has the wrong file type")
        return _ArtifactHandle(descriptor, os.close)
    except BaseException:
        os.close(descriptor)
        raise


def _open_windows_artifact_component(
    path: Path,
    expect_directory: bool,
) -> _ArtifactHandle:
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

    handle = create_file(
        str(path),
        0x80000000,
        0x00000001 | 0x00000002,
        None,
        3,
        0x02000000 | 0x00200000,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        information = ByHandleFileInformation()
        if not get_information(handle, ctypes.byref(information)):
            raise ctypes.WinError(ctypes.get_last_error())
        is_directory = bool(information.file_attributes & 0x00000010)
        is_reparse = bool(information.file_attributes & 0x00000400)
        if is_reparse or is_directory != expect_directory:
            raise OSError("artifact component is a reparse point or wrong type")
        result = _ArtifactHandle(handle, close_handle)
        handle = None
        return result
    finally:
        if handle is not None:
            close_handle(handle)


def _read_windows_handle(handle: int) -> bytes:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_pointer = kernel32.SetFilePointerEx
    set_pointer.argtypes = [
        wintypes.HANDLE,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_longlong),
        wintypes.DWORD,
    ]
    set_pointer.restype = wintypes.BOOL
    read_file = kernel32.ReadFile
    read_file.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    read_file.restype = wintypes.BOOL
    if not set_pointer(handle, 0, None, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    chunks = []
    while True:
        buffer = ctypes.create_string_buffer(64 * 1024)
        count = wintypes.DWORD()
        if not read_file(handle, buffer, len(buffer), ctypes.byref(count), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if count.value == 0:
            return b"".join(chunks)
        chunks.append(buffer.raw[: count.value])


def _conflict() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Artifact is unavailable or failed integrity validation",
    )
