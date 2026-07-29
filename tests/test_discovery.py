from __future__ import annotations

import os
import subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import BinaryIO

import h5py
import pytest
from das_sentinel.config import ProcessingProfile as ProfileConfig
from das_sentinel.persistence import (
    Repositories,
    create_local_engine,
    session_factory,
    upgrade_database,
)
from das_sentinel.persistence.models import Job, SourceFile
from das_sentinel.pipeline import _source_fingerprint
from das_sentinel.readers.base import ProbeResult
from das_sentinel.services.discovery import (
    DiscoveredFile,
    DiscoveryConfigurationError,
    DiscoveryPathError,
    DiscoverySite,
    StableFileTracker,
    discover_batch,
    enqueue_discovered,
    fingerprint,
    load_discovery_site,
    watch_site,
)
from sqlalchemy import func, select

from tests.fixtures import write_zd_h5


@pytest.fixture
def discovery_database(tmp_path: Path):
    app_data = tmp_path / "app-data"
    app_data.mkdir()
    database_url = f"sqlite+pysqlite:///{app_data / 'discovery.sqlite3'}"
    upgrade_database(database_url)
    engine = create_local_engine(database_url)
    sessions = session_factory(engine)
    try:
        yield sessions, app_data / "artifacts"
    finally:
        engine.dispose()


def _seed_site(
    sessions,
    data_root: Path,
    *,
    enabled: bool = True,
    current: bool = True,
    recursive: bool = True,
) -> tuple[str, str | None]:
    with sessions.begin() as session:
        repositories = Repositories(session)
        site = repositories.sites.add(
            name=f"site-{data_root.name}",
            data_path=str(data_root),
            timezone="Asia/Shanghai",
            cable_type="urban",
            enabled=enabled,
        )
        profile_id = None
        if current:
            profile = repositories.profiles.add(
                site_id=site.id,
                version=1,
                payload={"recursive_discovery": recursive},
                is_current=True,
            )
            profile_id = profile.id
        return site.id, profile_id


def _site(data_root: Path, *, recursive: bool = True) -> DiscoverySite:
    return DiscoverySite(
        site_id="site-1",
        data_root=data_root,
        profile_id="profile-1",
        profile_version=1,
        recursive_discovery=recursive,
    )


def _stabilize(
    site: DiscoverySite,
    path: Path,
    tracker: StableFileTracker,
) -> tuple[DiscoveredFile, ...]:
    assert discover_batch(site, path, tracker=tracker, now=0) == ()
    return discover_batch(site, path, tracker=tracker, now=10)


def test_fingerprint_matches_pipeline_canonical_algorithm_and_value_is_frozen(
    tmp_path: Path,
):
    path = tmp_path / "含空格-a.h5"
    write_zd_h5(path)
    resolved = path.resolve(strict=True)
    stat_result = resolved.stat()

    discovered = DiscoveredFile.from_path(path)

    assert discovered.path == resolved
    assert discovered.size_bytes == stat_result.st_size
    assert discovered.mtime_ns == stat_result.st_mtime_ns
    assert discovered.fingerprint == _source_fingerprint(path)
    with pytest.raises(FrozenInstanceError):
        discovered.size_bytes = 0


def test_changed_file_metadata_gets_new_fingerprint(tmp_path: Path):
    path = tmp_path / "a.h5"
    write_zd_h5(path)
    before = fingerprint(path)
    before_mtime = path.stat().st_mtime_ns

    os.utime(path, ns=(before_mtime + 1_000_000, before_mtime + 1_000_000))

    assert fingerprint(path) != before


def test_file_emits_after_ten_stable_seconds_and_each_fingerprint_only_once(
    tmp_path: Path,
):
    path = tmp_path / "a.h5"
    write_zd_h5(path)
    tracker = StableFileTracker()

    assert tracker.observe(path, now=0) is None
    assert tracker.observe(path, now=5) is None
    stable = tracker.observe(path, now=10)

    assert stable is not None
    assert stable.fingerprint == fingerprint(path)
    assert tracker.observe(path, now=15) is None
    assert tracker.observe(path, now=30) is None


def test_metadata_change_resets_stability_baseline(tmp_path: Path):
    path = tmp_path / "a.h5"
    write_zd_h5(path)
    tracker = StableFileTracker()
    assert tracker.observe(path, now=0) is None
    old_mtime = path.stat().st_mtime_ns
    os.utime(path, ns=(old_mtime + 1_000_000, old_mtime + 1_000_000))

    assert tracker.observe(path, now=9) is None
    assert tracker.observe(path, now=18) is None
    assert tracker.observe(path, now=19) is not None


class _MutatingProbe:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.calls = 0

    def probe(self, source: BinaryIO) -> ProbeResult:
        del source
        self.calls += 1
        old_mtime = self.path.stat().st_mtime_ns
        os.utime(
            self.path,
            ns=(old_mtime + 1_000_000, old_mtime + 1_000_000),
        )
        return ProbeResult(True, "test")


def test_mutation_during_probe_resets_baseline_and_emits_nothing(tmp_path: Path):
    path = tmp_path / "a.h5"
    write_zd_h5(path)
    reader = _MutatingProbe(path)
    tracker = StableFileTracker(reader=reader)

    assert tracker.observe(path, now=0) is None
    assert tracker.observe(path, now=10) is None
    assert reader.calls == 1
    assert tracker.observe(path, now=19) is None


def test_partial_hdf5_resets_baseline_until_repaired(tmp_path: Path):
    path = tmp_path / "partial.h5"
    path.write_bytes(b"not-hdf5")
    tracker = StableFileTracker()

    assert tracker.observe(path, now=0) is None
    assert tracker.observe(path, now=10) is None
    write_zd_h5(path)
    assert tracker.observe(path, now=20) is None
    assert tracker.observe(path, now=29) is None
    assert tracker.observe(path, now=30) is not None


def test_disappearance_and_reappearance_require_a_new_stability_window(tmp_path: Path):
    path = tmp_path / "a.h5"
    write_zd_h5(path)
    tracker = StableFileTracker()

    assert tracker.observe(path, now=0) is None
    path.unlink()
    assert tracker.observe(path, now=10) is None
    write_zd_h5(path)
    assert tracker.observe(path, now=20) is None
    assert tracker.observe(path, now=29) is None
    assert tracker.observe(path, now=30) is not None


def test_batch_honors_flat_discovery_and_filters_non_h5_files(tmp_path: Path):
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    top = root / "top.h5"
    uppercase = root / "UPPER.H5"
    deep = nested / "deep.h5"
    write_zd_h5(top)
    write_zd_h5(uppercase)
    write_zd_h5(deep)
    (root / "notes.txt").write_text("ignore", encoding="utf-8")
    (root / "directory.h5").mkdir()
    site = _site(root, recursive=False)
    tracker = StableFileTracker()

    files = _stabilize(site, root, tracker)

    assert tuple(item.path.name for item in files) == ("top.h5", "UPPER.H5")


def test_batch_honors_recursive_discovery(tmp_path: Path):
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    top = root / "top.h5"
    deep = nested / "deep.h5"
    write_zd_h5(top)
    write_zd_h5(deep)
    site = _site(root, recursive=True)
    tracker = StableFileTracker()

    files = _stabilize(site, root, tracker)

    assert tuple(item.path.relative_to(root) for item in files) == (
        Path("nested/deep.h5"),
        Path("top.h5"),
    )


@pytest.mark.parametrize("submitted", ["relative", "dotdot", "outside"])
def test_batch_rejects_noncanonical_or_outside_scan_paths(
    tmp_path: Path,
    submitted: str,
):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    paths = {
        "relative": Path("root"),
        "dotdot": root / ".." / "root",
        "outside": outside,
    }

    with pytest.raises(DiscoveryPathError):
        discover_batch(
            _site(root),
            paths[submitted],
            tracker=StableFileTracker(),
            now=0,
        )


def test_batch_rejects_symlink_escape(tmp_path: Path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    write_zd_h5(outside / "external.h5")
    link = root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink unavailable: {error}")

    with pytest.raises(DiscoveryPathError, match="reparse|symlink"):
        discover_batch(
            _site(root),
            link,
            tracker=StableFileTracker(),
            now=0,
        )


def test_batch_rejects_symlinked_site_root(tmp_path: Path):
    actual_root = tmp_path / "actual-root"
    actual_root.mkdir()
    link_root = tmp_path / "linked-root"
    try:
        link_root.symlink_to(actual_root, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlink unavailable: {error}")

    with pytest.raises(DiscoveryPathError, match="reparse|symlink"):
        discover_batch(
            _site(link_root),
            link_root,
            tracker=StableFileTracker(),
            now=0,
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_batch_rejects_junction_escape(tmp_path: Path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    junction = root / "junction"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip(f"junction unavailable: {created.stderr or created.stdout}")

    with pytest.raises(DiscoveryPathError, match="reparse|symlink"):
        discover_batch(
            _site(root),
            junction,
            tracker=StableFileTracker(),
            now=0,
        )


def test_load_site_requires_enabled_site_and_exactly_one_current_profile(
    discovery_database,
    tmp_path: Path,
):
    sessions, _ = discovery_database
    enabled_root = tmp_path / "enabled"
    disabled_root = tmp_path / "disabled"
    missing_root = tmp_path / "missing"
    enabled_root.mkdir()
    disabled_root.mkdir()
    missing_root.mkdir()
    enabled_id, profile_id = _seed_site(
        sessions, enabled_root, recursive=False
    )
    disabled_id, _ = _seed_site(
        sessions, disabled_root, enabled=False
    )
    missing_id, _ = _seed_site(
        sessions, missing_root, current=False
    )

    with sessions() as session:
        context = load_discovery_site(session, enabled_id)
        assert context.profile_id == profile_id
        assert context.profile_version == 1
        assert context.recursive_discovery is False
        with pytest.raises(DiscoveryConfigurationError, match="disabled"):
            load_discovery_site(session, disabled_id)
        with pytest.raises(
            DiscoveryConfigurationError, match="exactly one current"
        ):
            load_discovery_site(session, missing_id)


def test_load_site_rejects_malformed_non_discovery_profile_field(
    discovery_database,
    tmp_path: Path,
):
    sessions, _ = discovery_database
    root = tmp_path / "invalid-profile"
    root.mkdir()
    with sessions.begin() as session:
        repositories = Repositories(session)
        site = repositories.sites.add(
            name="invalid-profile-site",
            data_path=str(root),
            timezone="UTC",
            cable_type="urban",
        )
        repositories.profiles.add(
            site_id=site.id,
            version=1,
            payload={
                "recursive_discovery": True,
                "block_seconds": -1,
            },
            is_current=True,
        )
        site_id = site.id

    with sessions() as session, pytest.raises(
        DiscoveryConfigurationError, match="processing profile"
    ):
        load_discovery_site(session, site_id)


def test_load_site_accepts_complete_json_profile_with_energy_band_arrays(
    discovery_database,
    tmp_path: Path,
):
    sessions, _ = discovery_database
    root = tmp_path / "complete-profile"
    root.mkdir()
    with sessions.begin() as session:
        repositories = Repositories(session)
        site = repositories.sites.add(
            name="complete-profile-site",
            data_path=str(root),
            timezone="UTC",
            cable_type="urban",
        )
        repositories.profiles.add(
            site_id=site.id,
            version=1,
            payload=ProfileConfig().model_dump(mode="json"),
            is_current=True,
        )
        site_id = site.id

    with sessions() as session:
        loaded = load_discovery_site(session, site_id)

    assert loaded.recursive_discovery is True


class _ReplacingParentProbe:
    def __init__(
        self,
        parent: Path,
        moved: Path,
        outside: Path,
    ) -> None:
        self.parent = parent
        self.moved = moved
        self.outside = outside
        self.replaced = False
        self.opened_markers: list[str] = []

    def probe(self, source: Path | BinaryIO) -> ProbeResult:
        try:
            self.parent.rename(self.moved)
            self.parent.symlink_to(self.outside, target_is_directory=True)
            self.replaced = True
        except OSError:
            self.replaced = False
        with h5py.File(source, "r") as handle:
            self.opened_markers.append(str(handle.attrs["discovery_marker"]))
        return ProbeResult(self.replaced, "test")


class _FailingProbe:
    def probe(self, source: BinaryIO) -> ProbeResult:
        del source
        raise ValueError("probe failed")


def test_probe_cannot_be_redirected_by_parent_replacement_after_enumeration(
    tmp_path: Path,
):
    root = tmp_path / "root"
    parent = root / "incoming"
    moved = root / "incoming-original"
    outside = tmp_path / "outside"
    parent.mkdir(parents=True)
    outside.mkdir()
    internal = parent / "a.h5"
    external = outside / "a.h5"
    write_zd_h5(internal)
    write_zd_h5(external)
    with h5py.File(internal, "r+") as handle:
        handle.attrs["discovery_marker"] = "internal"
    with h5py.File(external, "r+") as handle:
        handle.attrs["discovery_marker"] = "external"

    preflight = root / "symlink-preflight"
    try:
        preflight.symlink_to(outside, target_is_directory=True)
        preflight.unlink()
    except OSError as error:
        pytest.skip(f"directory symlink unavailable: {error}")

    reader = _ReplacingParentProbe(parent, moved, outside)
    tracker = StableFileTracker(reader=reader)
    site = _site(root)
    assert discover_batch(site, root, tracker=tracker, now=0) == ()

    files = discover_batch(site, root, tracker=tracker, now=10)

    assert files == ()
    assert reader.opened_markers == ["internal"]
    assert "external" not in reader.opened_markers


def test_probe_failure_releases_all_anchored_handles(tmp_path: Path):
    root = tmp_path / "root"
    parent = root / "incoming"
    parent.mkdir(parents=True)
    write_zd_h5(parent / "a.h5")
    tracker = StableFileTracker(reader=_FailingProbe())

    assert discover_batch(_site(root), root, tracker=tracker, now=0) == ()
    assert discover_batch(_site(root), root, tracker=tracker, now=10) == ()

    parent.rename(root / "released")


def test_enqueue_is_idempotent_across_repeated_scans_and_tracker_restart(
    discovery_database,
    tmp_path: Path,
):
    sessions, artifact_root = discovery_database
    root = tmp_path / "root"
    root.mkdir()
    path = root / "a.h5"
    write_zd_h5(path)
    site_id, _ = _seed_site(sessions, root)

    with sessions.begin() as session:
        site = load_discovery_site(session, site_id)
    first = _stabilize(site, root, StableFileTracker())
    restarted = _stabilize(site, root, StableFileTracker())

    with sessions.begin() as session:
        first_jobs = enqueue_discovered(
            session,
            site,
            first,
            "pipeline-v1",
            artifact_root=artifact_root,
        )
    with sessions.begin() as session:
        second_jobs = enqueue_discovered(
            session,
            site,
            restarted,
            "pipeline-v1",
            artifact_root=artifact_root,
        )

    assert second_jobs[0].id == first_jobs[0].id
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(SourceFile)) == 1
        assert session.scalar(select(func.count()).select_from(Job)) == 1


def test_profile_and_pipeline_run_key_changes_create_new_jobs(
    discovery_database,
    tmp_path: Path,
):
    sessions, artifact_root = discovery_database
    root = tmp_path / "root"
    root.mkdir()
    path = root / "a.h5"
    write_zd_h5(path)
    site_id, _ = _seed_site(sessions, root)
    with sessions.begin() as session:
        site_v1 = load_discovery_site(session, site_id)
    files = _stabilize(site_v1, root, StableFileTracker())
    with sessions.begin() as session:
        first = enqueue_discovered(
            session,
            site_v1,
            files,
            "pipeline-v1",
            artifact_root=artifact_root,
        )
        second = enqueue_discovered(
            session,
            site_v1,
            files,
            "pipeline-v2",
            artifact_root=artifact_root,
        )
    with sessions.begin() as session:
        repositories = Repositories(session)
        profile_v2 = repositories.profiles.add(
            site_id=site_id,
            version=2,
            payload={"recursive_discovery": True},
            is_current=False,
        )
        repositories.sites.set_current_profile(site_id, profile_v2.id)
        site_v2 = load_discovery_site(session, site_id)
        third = enqueue_discovered(
            session,
            site_v2,
            files,
            "pipeline-v1",
            artifact_root=artifact_root,
        )

    assert len({first[0].id, second[0].id, third[0].id}) == 3


def test_enqueue_revalidates_enabled_site_and_current_profile(
    discovery_database,
    tmp_path: Path,
):
    sessions, artifact_root = discovery_database
    root = tmp_path / "root"
    root.mkdir()
    write_zd_h5(root / "a.h5")
    site_id, _ = _seed_site(sessions, root)
    with sessions.begin() as session:
        stale_context = load_discovery_site(session, site_id)
    files = _stabilize(stale_context, root, StableFileTracker())

    with sessions.begin() as session:
        repositories = Repositories(session)
        profile_v2 = repositories.profiles.add(
            site_id=site_id,
            version=2,
            payload={"recursive_discovery": True},
            is_current=False,
        )
        repositories.sites.set_current_profile(site_id, profile_v2.id)
    with sessions.begin() as session, pytest.raises(
        DiscoveryConfigurationError, match="changed"
    ):
        enqueue_discovered(
            session,
            stale_context,
            files,
            "pipeline-v1",
            artifact_root=artifact_root,
        )
    with sessions.begin() as session:
        repositories = Repositories(session)
        repositories.sites.set_enabled(site_id, False)
    with sessions.begin() as session:
        current = DiscoverySite(
            site_id=site_id,
            data_root=root,
            profile_id=profile_v2.id,
            profile_version=2,
            recursive_discovery=True,
        )
        with pytest.raises(DiscoveryConfigurationError, match="disabled"):
            enqueue_discovered(
                session,
                current,
                files,
                "pipeline-v1",
                artifact_root=artifact_root,
            )


def test_enqueue_flushes_without_committing(discovery_database, tmp_path: Path):
    sessions, artifact_root = discovery_database
    root = tmp_path / "root"
    root.mkdir()
    write_zd_h5(root / "a.h5")
    site_id, _ = _seed_site(sessions, root)
    with sessions.begin() as session:
        site = load_discovery_site(session, site_id)
    files = _stabilize(site, root, StableFileTracker())

    session = sessions()
    try:
        jobs = enqueue_discovered(
            session,
            site,
            files,
            "rollback-pipeline",
            artifact_root=artifact_root,
        )
        assert session.get(Job, jobs[0].id) is not None
        session.rollback()
    finally:
        session.close()

    with sessions() as verification:
        count = verification.scalar(
            select(func.count())
            .select_from(Job)
            .where(Job.pipeline_version == "rollback-pipeline")
        )
        assert count == 0


def test_enqueue_rejects_file_changed_after_discovery_without_writes(
    discovery_database,
    tmp_path: Path,
):
    sessions, artifact_root = discovery_database
    root = tmp_path / "root"
    root.mkdir()
    path = root / "a.h5"
    write_zd_h5(path)
    site_id, _ = _seed_site(sessions, root)
    with sessions.begin() as session:
        site = load_discovery_site(session, site_id)
    files = _stabilize(site, root, StableFileTracker())
    old_mtime = path.stat().st_mtime_ns
    os.utime(path, ns=(old_mtime + 1_000_000, old_mtime + 1_000_000))

    with sessions.begin() as session, pytest.raises(
        DiscoveryPathError, match="changed"
    ):
        enqueue_discovered(
            session,
            site,
            files,
            "pipeline-v1",
            artifact_root=artifact_root,
        )

    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(SourceFile)) == 0
        assert session.scalar(select(func.count()).select_from(Job)) == 0


def test_watch_uses_injected_clock_sleep_and_emits_after_stability(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    write_zd_h5(root / "a.h5")
    now = [0.0]
    polls = [0]
    batches: list[tuple[DiscoveredFile, ...]] = []

    def stop() -> bool:
        return polls[0] >= 3

    def sleep(seconds: float) -> None:
        assert seconds == 5
        now[0] += seconds
        polls[0] += 1

    watch_site(
        _site(root),
        root,
        tracker=StableFileTracker(),
        clock=lambda: now[0],
        sleep=sleep,
        should_stop=stop,
        on_discovered=batches.append,
        poll_interval=5,
    )

    assert len(batches) == 1
    assert batches[0][0].path.name == "a.h5"
    assert now[0] == 15


def test_watch_cancellation_before_start_does_not_poll_or_sleep(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    calls: list[str] = []

    watch_site(
        _site(root),
        root,
        tracker=StableFileTracker(),
        clock=lambda: calls.append("clock") or 0,
        sleep=lambda _: calls.append("sleep"),
        should_stop=lambda: True,
        on_discovered=lambda _: calls.append("emit"),
    )

    assert calls == []


def test_watch_cancellation_after_poll_does_not_perform_an_extra_poll(
    tmp_path: Path,
):
    root = tmp_path / "root"
    root.mkdir()
    write_zd_h5(root / "a.h5")
    stopped = [False]
    clock_calls = [0]

    def clock() -> float:
        clock_calls[0] += 1
        return 0

    def sleep(_: float) -> None:
        stopped[0] = True

    watch_site(
        _site(root),
        root,
        tracker=StableFileTracker(),
        clock=clock,
        sleep=sleep,
        should_stop=lambda: stopped[0],
    )

    assert clock_calls[0] == 1


def test_watch_rejects_nonpositive_or_nonfinite_poll_interval(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()

    for interval in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="poll interval"):
            watch_site(
                _site(root),
                root,
                tracker=StableFileTracker(),
                should_stop=lambda: True,
                poll_interval=interval,
            )


def test_profile_payload_must_contain_boolean_recursive_discovery(
    discovery_database,
    tmp_path: Path,
):
    sessions, _ = discovery_database
    root = tmp_path / "root"
    root.mkdir()
    with sessions.begin() as session:
        repositories = Repositories(session)
        site = repositories.sites.add(
            name="malformed-profile-site",
            data_path=str(root),
            timezone="UTC",
            cable_type="urban",
        )
        repositories.profiles.add(
            site_id=site.id,
            version=1,
            payload={"recursive_discovery": "yes"},
            is_current=True,
        )
        site_id = site.id

    with sessions() as session, pytest.raises(
        DiscoveryConfigurationError, match="recursive_discovery"
    ):
        load_discovery_site(session, site_id)
