from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from sqlalchemy.orm import Session, sessionmaker

from das_sentinel.persistence.db import AsyncWriteCoordinator
from das_sentinel.services.discovery import (
    DiscoveryConfigurationError,
    DiscoveryPathError,
    DiscoveryProgress,
    DiscoverySite,
    StableFileTracker,
    discover_batch,
    enqueue_discovered,
    resolve_scan_directory,
)

logger = logging.getLogger(__name__)

SCAN_READ_ERROR = "无法读取所选目录，请检查 NAS 连接和目录权限后重试"
SCAN_CONFIG_ERROR = "站点配置已变化或不可用，请检查站点设置后重试"
SCAN_UNKNOWN_ERROR = "扫描失败，请检查服务日志后重试"

ScanPhase = Literal[
    "initial_scan",
    "stability_wait",
    "verification_scan",
    "enqueueing",
    "completed",
    "failed",
]


@dataclass(frozen=True)
class ScanSnapshot:
    id: str
    site_id: str
    path: str
    phase: ScanPhase
    initial_checked_files: int = 0
    initial_found_files: int = 0
    verification_checked_files: int = 0
    verification_found_files: int = 0
    stable_files: int = 0
    submitted_files: int = 0
    skipped_files: int = 0
    stability_seconds_remaining: int = 0
    error_message: str | None = None


class ScanCoordinator:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        artifact_root: Path,
        pipeline_version: str,
        write_coordinator: AsyncWriteCoordinator,
        *,
        stable_seconds: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.sessions = sessions
        self.artifact_root = Path(artifact_root)
        self.pipeline_version = pipeline_version
        self.write_coordinator = write_coordinator
        self.stable_seconds = float(stable_seconds)
        self.clock = clock
        self.sleep = sleep
        self._lock = Lock()
        self._snapshots: dict[str, ScanSnapshot] = {}
        self._latest_by_site: dict[str, str] = {}
        self._tasks_by_site: dict[str, asyncio.Task[None]] = {}

    def start(
        self,
        site: DiscoverySite,
        scan_path: Path,
        relative_path: str,
    ) -> tuple[ScanSnapshot, bool]:
        with self._lock:
            active_task = self._tasks_by_site.get(site.site_id)
            latest_id = self._latest_by_site.get(site.site_id)
            if active_task is not None and latest_id is not None:
                current = self._snapshots[latest_id]
                if current.phase not in {"completed", "failed"}:
                    return current, False

            previous_id = self._latest_by_site.get(site.site_id)
            if previous_id is not None:
                self._snapshots.pop(previous_id, None)
            scan_id = str(uuid4())
            snapshot = ScanSnapshot(
                id=scan_id,
                site_id=site.site_id,
                path=relative_path,
                phase="initial_scan",
            )
            self._snapshots[scan_id] = snapshot
            self._latest_by_site[site.site_id] = scan_id
            task = asyncio.create_task(
                self._run(scan_id, site, Path(scan_path), relative_path)
            )
            self._tasks_by_site[site.site_id] = task
            return snapshot, True

    def get(self, scan_id: str) -> ScanSnapshot | None:
        with self._lock:
            return self._snapshots.get(scan_id)

    def active(self, site_id: str) -> ScanSnapshot | None:
        with self._lock:
            task = self._tasks_by_site.get(site_id)
            scan_id = self._latest_by_site.get(site_id)
            if task is None or scan_id is None:
                return None
            snapshot = self._snapshots[scan_id]
            if snapshot.phase in {"completed", "failed"}:
                return None
            return snapshot

    async def close(self) -> None:
        with self._lock:
            tasks = tuple(self._tasks_by_site.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(
        self,
        scan_id: str,
        site: DiscoverySite,
        scan_path: Path,
        relative_path: str,
    ) -> None:
        task = asyncio.current_task()
        try:
            resolved_path = await _to_thread(
                resolve_scan_directory,
                site,
                relative_path,
            )
            if resolved_path != scan_path:
                raise DiscoveryPathError("scan path changed before discovery")

            tracker = StableFileTracker(stable_seconds=self.stable_seconds)
            await _to_thread(
                discover_batch,
                site,
                scan_path,
                tracker=tracker,
                now=self.clock(),
                recursive=True,
                on_progress=lambda value: self._set_pass_progress(
                    scan_id, "initial", value
                ),
            )

            deadline = self.clock() + self.stable_seconds
            while True:
                seconds_left = deadline - self.clock()
                if seconds_left <= 0:
                    break
                self._replace(
                    scan_id,
                    phase="stability_wait",
                    stability_seconds_remaining=math.ceil(seconds_left),
                )
                await self.sleep(min(1.0, seconds_left))

            self._replace(
                scan_id,
                phase="verification_scan",
                stability_seconds_remaining=0,
            )
            files = await _to_thread(
                discover_batch,
                site,
                scan_path,
                tracker=tracker,
                now=self.clock(),
                recursive=True,
                on_progress=lambda value: self._set_pass_progress(
                    scan_id, "verification", value
                ),
            )
            verification = self.get(scan_id)
            if verification is None:
                return
            self._replace(
                scan_id,
                phase="enqueueing",
                stable_files=len(files),
                skipped_files=max(
                    0,
                    verification.verification_found_files - len(files),
                ),
            )
            async with self.write_coordinator.lock:
                submitted = await _to_thread(
                    _enqueue,
                    self.sessions,
                    self.artifact_root,
                    site,
                    files,
                    self.pipeline_version,
                )
            self._replace(
                scan_id,
                phase="completed",
                submitted_files=len(submitted),
            )
        except DiscoveryPathError:
            logger.exception("scan %s could not read its selected path", scan_id)
            self._replace(
                scan_id,
                phase="failed",
                stability_seconds_remaining=0,
                error_message=SCAN_READ_ERROR,
            )
        except DiscoveryConfigurationError:
            logger.exception("scan %s encountered invalid site configuration", scan_id)
            self._replace(
                scan_id,
                phase="failed",
                stability_seconds_remaining=0,
                error_message=SCAN_CONFIG_ERROR,
            )
        except Exception:
            logger.exception("scan %s failed unexpectedly", scan_id)
            self._replace(
                scan_id,
                phase="failed",
                stability_seconds_remaining=0,
                error_message=SCAN_UNKNOWN_ERROR,
            )
        finally:
            with self._lock:
                if self._tasks_by_site.get(site.site_id) is task:
                    del self._tasks_by_site[site.site_id]

    def _set_pass_progress(
        self,
        scan_id: str,
        pass_name: Literal["initial", "verification"],
        progress: DiscoveryProgress,
    ) -> None:
        if pass_name == "initial":
            self._replace(
                scan_id,
                initial_checked_files=progress.checked_files,
                initial_found_files=progress.found_files,
            )
        else:
            self._replace(
                scan_id,
                verification_checked_files=progress.checked_files,
                verification_found_files=progress.found_files,
            )

    def _replace(self, scan_id: str, **changes: Any) -> None:
        with self._lock:
            current = self._snapshots.get(scan_id)
            if current is not None:
                self._snapshots[scan_id] = replace(current, **changes)


def _enqueue(
    sessions: sessionmaker[Session],
    artifact_root: Path,
    site: DiscoverySite,
    files,
    pipeline_version: str,
):
    if not files:
        return ()
    with sessions.begin() as session:
        return enqueue_discovered(
            session,
            site,
            files,
            pipeline_version,
            artifact_root=artifact_root,
        )


async def _to_thread[ResultT](
    function: Callable[..., ResultT],
    /,
    *args: Any,
    **kwargs: Any,
) -> ResultT:
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
        worker.result()
        raise
