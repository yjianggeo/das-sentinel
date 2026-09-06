from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Event

import das_sentinel.services.scans as scans_module
import pytest
from das_sentinel.persistence import (
    AsyncWriteCoordinator,
    Repositories,
    create_local_engine,
    session_factory,
    upgrade_database,
)
from das_sentinel.persistence.models import Job, SourceFile
from das_sentinel.services.discovery import (
    DiscoveryPathError,
    load_discovery_site,
)
from das_sentinel.services.scans import (
    SCAN_READ_ERROR,
    ScanCoordinator,
    ScanSnapshot,
)
from sqlalchemy import func, select

from tests.fixtures import write_zd_h5


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        await asyncio.sleep(0)


class PausedClock(FakeClock):
    def __init__(self) -> None:
        super().__init__()
        self.sleeping = asyncio.Event()
        self.release = asyncio.Event()

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.sleeping.set()
        await self.release.wait()


@pytest.fixture
def scan_database(tmp_path: Path):
    app_data = tmp_path / "app-data"
    app_data.mkdir()
    database_url = f"sqlite+pysqlite:///{app_data / 'scans.sqlite3'}"
    upgrade_database(database_url)
    engine = create_local_engine(database_url)
    sessions = session_factory(engine)
    try:
        yield sessions, app_data / "artifacts"
    finally:
        engine.dispose()


def seed_site(sessions, data_root: Path):
    with sessions.begin() as session:
        repositories = Repositories(session)
        site = repositories.sites.add(
            name=f"site-{data_root.name}",
            data_path=str(data_root),
            timezone="Asia/Shanghai",
            cable_type="urban",
            enabled=True,
        )
        profile = repositories.profiles.add(
            site_id=site.id,
            version=1,
            payload={"recursive_discovery": True},
            is_current=True,
        )
        repositories.sites.set_current_profile(site.id, profile.id)
        site_id = site.id
    with sessions() as session:
        return load_discovery_site(session, site_id)


@pytest.fixture
def scan_context(scan_database, tmp_path: Path):
    sessions, artifact_root = scan_database
    root = tmp_path / "nas"
    selected = root / "jjk" / "day"
    selected.mkdir(parents=True)
    write_zd_h5(selected / "stable.h5")
    site = seed_site(sessions, root)
    clock = FakeClock()
    coordinator = ScanCoordinator(
        sessions,
        artifact_root,
        "pipeline-v1",
        AsyncWriteCoordinator(),
        stable_seconds=2.0,
        clock=clock,
        sleep=clock.sleep,
    )
    return coordinator, site, selected, sessions


async def wait_for_terminal(
    coordinator: ScanCoordinator,
    scan_id: str,
) -> ScanSnapshot:
    for _ in range(500):
        snapshot = coordinator.get(scan_id)
        assert snapshot is not None
        if snapshot.phase in {"completed", "failed"}:
            await asyncio.sleep(0)
            return snapshot
        await asyncio.sleep(0.01)
    raise AssertionError("scan did not reach a terminal phase")


def test_scan_runs_two_passes_and_enqueues_without_second_start(scan_context):
    async def scenario():
        coordinator, site, selected, sessions = scan_context
        first, created = coordinator.start(site, selected, "jjk/day")
        assert created is True
        terminal = await wait_for_terminal(coordinator, first.id)
        assert terminal.phase == "completed"
        assert terminal.initial_found_files == 1
        assert terminal.verification_found_files == 1
        assert terminal.stable_files == 1
        assert terminal.submitted_files == 1
        assert terminal.skipped_files == 0
        assert coordinator.active(site.site_id) is None
        with sessions() as session:
            assert session.scalar(select(func.count()).select_from(SourceFile)) == 1
            assert session.scalar(select(func.count()).select_from(Job)) == 1
        await coordinator.close()

    asyncio.run(scenario())


def test_stability_wait_exposes_countdown_and_reuses_active_site_scan(scan_database, tmp_path: Path):
    async def scenario():
        sessions, artifact_root = scan_database
        root = tmp_path / "nas"
        selected = root / "jjk" / "day"
        selected.mkdir(parents=True)
        write_zd_h5(selected / "stable.h5")
        site = seed_site(sessions, root)
        clock = PausedClock()
        coordinator = ScanCoordinator(
            sessions,
            artifact_root,
            "pipeline-v1",
            AsyncWriteCoordinator(),
            stable_seconds=2.0,
            clock=clock,
            sleep=clock.sleep,
        )

        first, created = coordinator.start(site, selected, "jjk/day")
        assert created is True
        await asyncio.wait_for(clock.sleeping.wait(), timeout=2)
        waiting = coordinator.get(first.id)
        assert waiting is not None
        assert waiting.phase == "stability_wait"
        assert waiting.stability_seconds_remaining == 2

        duplicate, duplicate_created = coordinator.start(site, selected, "jjk/day")
        assert duplicate_created is False
        assert duplicate.id == first.id

        clock.release.set()
        assert (await wait_for_terminal(coordinator, first.id)).phase == "completed"
        await coordinator.close()

    asyncio.run(scenario())


def test_different_sites_get_independent_active_scans(scan_database, tmp_path: Path):
    async def scenario():
        sessions, artifact_root = scan_database
        clock = PausedClock()
        coordinator = ScanCoordinator(
            sessions,
            artifact_root,
            "pipeline-v1",
            AsyncWriteCoordinator(),
            stable_seconds=1.0,
            clock=clock,
            sleep=clock.sleep,
        )
        sites = []
        selected_paths = []
        for name in ("north", "south"):
            root = tmp_path / name
            selected = root / "jjk" / "day"
            selected.mkdir(parents=True)
            write_zd_h5(selected / "stable.h5")
            sites.append(seed_site(sessions, root))
            selected_paths.append(selected)

        first, first_created = coordinator.start(
            sites[0], selected_paths[0], "jjk/day"
        )
        second, second_created = coordinator.start(
            sites[1], selected_paths[1], "jjk/day"
        )
        assert first_created is True
        assert second_created is True
        assert first.id != second.id
        assert coordinator.active(sites[0].site_id).id == first.id
        assert coordinator.active(sites[1].site_id).id == second.id
        await coordinator.close()

    asyncio.run(scenario())


def test_unsupported_candidate_is_skipped_without_failing_stable_file(scan_context):
    async def scenario():
        coordinator, site, selected, _sessions = scan_context
        (selected / "unsupported.h5").write_bytes(b"not an HDF5 file")

        first, _created = coordinator.start(site, selected, "jjk/day")
        terminal = await wait_for_terminal(coordinator, first.id)
        assert terminal.phase == "completed"
        assert terminal.verification_found_files == 2
        assert terminal.stable_files == 1
        assert terminal.submitted_files == 1
        assert terminal.skipped_files == 1
        await coordinator.close()

    asyncio.run(scenario())


def test_enumeration_error_fails_with_controlled_message_and_releases_site(
    scan_context,
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_discovery(*args, **kwargs):
        del args, kwargs
        raise DiscoveryPathError("secret NAS path")

    monkeypatch.setattr(scans_module, "discover_batch", fail_discovery)

    async def scenario():
        coordinator, site, selected, _sessions = scan_context
        first, _created = coordinator.start(site, selected, "jjk/day")
        terminal = await wait_for_terminal(coordinator, first.id)
        assert terminal.phase == "failed"
        assert terminal.error_message == SCAN_READ_ERROR
        assert "secret" not in terminal.error_message
        assert coordinator.active(site.site_id) is None
        await coordinator.close()

    asyncio.run(scenario())


def test_concurrent_close_waits_for_an_in_flight_discovery_thread(
    scan_context,
    monkeypatch: pytest.MonkeyPatch,
):
    worker_started = Event()
    release_worker = Event()
    worker_finished = Event()

    def blocking_discovery(*args, **kwargs):
        del args, kwargs
        worker_started.set()
        release_worker.wait(timeout=2)
        worker_finished.set()
        return ()

    monkeypatch.setattr(scans_module, "discover_batch", blocking_discovery)

    async def scenario():
        coordinator, site, selected, _sessions = scan_context
        coordinator.start(site, selected, "jjk/day")
        assert await asyncio.to_thread(worker_started.wait, 2)

        first_close = asyncio.create_task(coordinator.close())
        await asyncio.sleep(0)
        second_close = asyncio.create_task(coordinator.close())
        await asyncio.sleep(0.05)
        try:
            assert first_close.done() is False
            assert second_close.done() is False
        finally:
            release_worker.set()
        await asyncio.wait_for(
            asyncio.gather(first_close, second_close),
            timeout=2,
        )
        assert worker_finished.is_set()

    asyncio.run(scenario())


def test_scan_snapshot_is_immutable():
    snapshot = ScanSnapshot(
        id="scan-1",
        site_id="site-1",
        path="jjk/day",
        phase="initial_scan",
    )

    with pytest.raises(FrozenInstanceError):
        snapshot.phase = "completed"  # type: ignore[misc]
