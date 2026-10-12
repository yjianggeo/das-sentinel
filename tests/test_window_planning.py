from __future__ import annotations

import importlib
import importlib.util
import json
from datetime import UTC, datetime, timedelta

import h5py
import pytest
from das_sentinel.persistence import (
    Repositories,
    create_local_engine,
    session_factory,
    upgrade_database,
)
from das_sentinel.persistence.models import Job
from das_sentinel.services.discovery import DiscoveredFile, load_discovery_site
from sqlalchemy import func, select

from tests.fixtures import write_zd_h5


def module():
    assert importlib.util.find_spec("das_sentinel.services.window_planning") is not None
    return importlib.import_module("das_sentinel.services.window_planning")


def write_minute(path, start):
    write_zd_h5(path, n_time=6000)
    with h5py.File(path, "r+") as handle:
        times = handle["Acquisition/Raw[0]/RawDataTime"]
        times[:] = times[:] - times[0] + round(start.timestamp() * 1_000_000)


@pytest.fixture
def planning_env(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    url = f"sqlite+pysqlite:///{tmp_path / 'test.sqlite3'}"
    upgrade_database(url)
    engine = create_local_engine(url)
    sessions = session_factory(engine)
    with sessions.begin() as session:
        repos = Repositories(session)
        site = repos.sites.add(
            name="site",
            data_path=str(raw),
            timezone="Asia/Shanghai",
            cable_type="borehole",
        )
        repos.profiles.add(site_id=site.id, version=1, payload={}, is_current=True)
        site_id = site.id
    with sessions() as session:
        site = load_discovery_site(session, site_id)
    yield sessions, site, raw, tmp_path / "window-artifacts"
    engine.dispose()


def test_planning_uses_registered_cross_day_context_and_creates_no_legacy_jobs(
    planning_env,
):
    implementation = module()
    sessions, site, raw, artifacts = planning_env
    start = datetime(2023, 9, 27, tzinfo=UTC)
    paths = []
    for index in (-1, 0, 1):
        folder = raw / ("previous" if index < 0 else "today")
        folder.mkdir(exist_ok=True)
        path = folder / f"minute{index}.h5"
        write_minute(path, start + timedelta(minutes=index))
        paths.append(path)
    with sessions.begin() as session:
        for path in (paths[0], paths[2]):
            source = DiscoveredFile.from_path(path)
            Repositories(session).source_files.add(
                site_id=site.site_id,
                path=str(path),
                fingerprint=source.fingerprint,
                size_bytes=source.size_bytes,
                mtime_ns=source.mtime_ns,
            )
    selected = [DiscoveredFile.from_path(paths[1])]
    first = implementation.enqueue_window_files(sessions, site, selected, artifacts)
    second = implementation.enqueue_window_files(sessions, site, selected, artifacts)
    assert len(first) == 2
    assert [json.loads(row.plan_json)["start_utc"] for row in first] == [
        start.isoformat(), (start + timedelta(seconds=30)).isoformat()
    ]
    assert [row.id for row in first] == [row.id for row in second]
    assert first[0].status == "queued"
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(Job)) == 0


def test_missing_context_is_visible_failure_not_noise_or_queued(planning_env):
    implementation = module()
    sessions, site, raw, artifacts = planning_env
    path = raw / "single.h5"
    write_minute(path, datetime(2023, 9, 27, tzinfo=UTC))
    jobs = implementation.enqueue_window_files(
        sessions, site, [DiscoveredFile.from_path(path)], artifacts
    )
    assert len(jobs) == 2
    assert all(job.status == "failed" for job in jobs)
    assert jobs[0].status == "failed"
    assert jobs[0].error_code == "coverage_incomplete"


def test_changed_selected_source_rejects_stale_discovery_snapshot(planning_env):
    implementation = module()
    sessions, site, raw, artifacts = planning_env
    path = raw / "single.h5"
    write_minute(path, datetime(2023, 9, 27, tzinfo=UTC))
    source = DiscoveredFile.from_path(path)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed|变化"):
        implementation.enqueue_window_files(sessions, site, [source], artifacts)


def test_unregistered_neighbor_files_supply_context_without_creating_extra_windows(
    planning_env,
):
    implementation = module()
    sessions, site, raw, artifacts = planning_env
    start = datetime(2023, 9, 27, tzinfo=UTC)
    selected = None
    for offset in (-1, 0, 1):
        stamp = start + timedelta(minutes=offset)
        directory = raw / stamp.strftime("%Y%m%d%H")
        directory.mkdir(exist_ok=True)
        path = directory / f"DAS_UTC_{stamp:%Y%m%d%H%M}.h5"
        write_minute(path, stamp)
        if offset == 0:
            selected = DiscoveredFile.from_path(path)
    plans = implementation.prepare_window_files(sessions, site, [selected], artifacts)
    assert len(plans) == 2
    assert plans[0]["coverage"] == "complete"
    assert len(plans[0]["sources"]) == 3


def test_neighbor_filename_is_only_hint_actual_timestamp_controls_coverage(
    planning_env,
):
    implementation = module()
    sessions, site, raw, artifacts = planning_env
    start = datetime(2023, 9, 27, tzinfo=UTC)
    selected = raw / "DAS_202309270000.h5"
    write_minute(selected, start)
    # The name looks like a neighbor but the numeric clock is a day later.
    write_minute(raw / "DAS_202309270001.h5", start + timedelta(days=1))
    plans = implementation.prepare_window_files(
        sessions, site, [DiscoveredFile.from_path(selected)], artifacts
    )
    assert len(plans) == 2
    assert plans[0]["coverage"] == "incomplete"


def test_identical_copies_do_not_make_continuous_windows_incomplete(planning_env):
    import shutil

    implementation = module()
    sessions, site, raw, artifacts = planning_env
    start = datetime(2023, 9, 27, tzinfo=UTC)
    selected = []
    for offset in (-1, 0, 1, 2):
        path = raw / f"original{offset}.h5"
        write_minute(path, start + timedelta(minutes=offset))
        copy = raw / f"copy{offset}.h5"
        shutil.copyfile(path, copy)
        selected += [DiscoveredFile.from_path(path), DiscoveredFile.from_path(copy)]
    plans = implementation.prepare_window_files(sessions, site, selected, artifacts)
    target = next(plan for plan in plans if plan['start_utc'] == start.isoformat())
    assert target['coverage'] == 'complete'
    assert len(target['sources']) == 3
    # Same-clock files with different content remain ambiguous, never silently dropped.
    with h5py.File(raw / 'copy0.h5', 'r+') as handle:
        data = handle['Acquisition/Raw[0]/RawData']
        data[0, 0] += 1
    selected = [DiscoveredFile.from_path(source.path) for source in selected]
    plans = implementation.prepare_window_files(sessions, site, selected, artifacts)
    target = next(plan for plan in plans if plan['start_utc'] == start.isoformat())
    assert target['coverage'] == 'incomplete'


def test_unrelated_registered_copies_are_not_hashed(planning_env, monkeypatch):
    import shutil

    implementation = module()
    sessions, site, raw, artifacts = planning_env
    start = datetime(2023, 9, 27, tzinfo=UTC)
    selected = []
    for offset in (-1, 0, 1):
        path = raw / f'minute{offset}.h5'
        write_minute(path, start + timedelta(minutes=offset))
        selected.append(DiscoveredFile.from_path(path))
    unrelated = raw / 'unrelated.h5'
    write_minute(unrelated, start + timedelta(days=10))
    copy = raw / 'unrelated-copy.h5'
    shutil.copyfile(unrelated, copy)
    with sessions.begin() as session:
        for path in (unrelated, copy):
            source = DiscoveredFile.from_path(path)
            Repositories(session).source_files.add(site_id=site.site_id, path=str(path),
                fingerprint=source.fingerprint, size_bytes=source.size_bytes, mtime_ns=source.mtime_ns)

    def unexpected(*args):
        pytest.fail('Unrelated recording was read in full')

    monkeypatch.setattr(implementation.hashlib, 'file_digest', unexpected)
    plans = implementation.prepare_window_files(sessions, site, selected, artifacts)
    assert next(p for p in plans if p['start_utc'] == start.isoformat())['coverage'] == 'complete'


@pytest.mark.parametrize("changed_offset", [0, 1])
def test_changed_source_after_planning_rejected_before_enqueue(
    planning_env, changed_offset
):
    implementation = module()
    sessions, site, raw, artifacts = planning_env
    start = datetime(2023, 9, 27, tzinfo=UTC)
    paths = {}
    for offset in (-1, 0, 1):
        stamp = start + timedelta(minutes=offset)
        path = raw / f"DAS_{stamp:%Y%m%d%H%M}.h5"
        write_minute(path, stamp)
        paths[offset] = path
    selected = [DiscoveredFile.from_path(paths[0])]
    plans = implementation.prepare_window_files(sessions, site, selected, artifacts)
    with paths[changed_offset].open("ab") as stream:
        stream.write(b"changed after planning")
    with sessions.begin() as session, pytest.raises(ValueError, match="changed"):
        implementation.persist_window_files(session, site, selected, plans)


@pytest.mark.parametrize("corruption", [
    "invalid_json", "identity_only", "missing_distance", "naive_time", "backwards_time",
    "bad_rate", "nan_rate", "empty_distance", "nan_distance", "reversed_distance", "empty_unit",
])
def test_metadata_cache_incomplete_or_invalid_entry_is_rebuilt(tmp_path, corruption):
    implementation = module()
    path = tmp_path / "source.h5"
    write_minute(path, datetime(2023, 9, 27, tzinfo=UTC))
    source = DiscoveredFile.from_path(path)
    cache = tmp_path / "metadata"
    expected = implementation._inspect(source, tmp_path, cache)
    damaged = dict(expected)
    if corruption == "identity_only":
        damaged = {key: damaged[key] for key in ("path", "fingerprint", "size_bytes", "mtime_ns")}
    elif corruption == "missing_distance":
        del damaged["distance_m"]
    elif corruption == "naive_time":
        damaged["start_utc"] = "2023-09-27T00:00:00"
    elif corruption == "backwards_time":
        damaged["end_utc"] = damaged["start_utc"]
    elif corruption == "bad_rate":
        damaged["sample_rate_hz"] = 10
    elif corruption == "nan_rate":
        damaged["sample_rate_hz"] = float("nan")
    elif corruption == "empty_distance":
        damaged["distance_m"] = []
    elif corruption == "nan_distance":
        damaged["distance_m"] = [float("nan")]
    elif corruption == "reversed_distance":
        damaged["distance_m"] = list(reversed(damaged["distance_m"]))
    elif corruption == "empty_unit":
        damaged["unit"] = ""
    cached = cache / f"{source.fingerprint}.json"
    cached.write_text("{" if corruption == "invalid_json" else json.dumps(damaged), encoding="utf8")
    assert implementation._inspect(source, tmp_path, cache) == expected
    assert json.loads(cached.read_text(encoding="utf8")) == expected


def test_metadata_cache_rejects_source_change_during_cache_hit(tmp_path, monkeypatch):
    from pathlib import Path
    implementation = module()
    path = tmp_path / "source.h5"
    write_minute(path, datetime(2023, 9, 27, tzinfo=UTC))
    source = DiscoveredFile.from_path(path)
    cache = tmp_path / "metadata"
    implementation._inspect(source, tmp_path, cache)
    cached = cache / f"{source.fingerprint}.json"
    read_text = Path.read_text

    def read_and_change(self, *args, **kwargs):
        value = read_text(self, *args, **kwargs)
        if self == cached:
            with path.open("ab") as stream:
                stream.write(b"changed while reading metadata cache")
        return value

    monkeypatch.setattr(Path, "read_text", read_and_change)
    with pytest.raises(ValueError, match="changed"):
        implementation._inspect(source, tmp_path, cache)


def test_metadata_cache_changed_source_reinspects_new_identity(tmp_path):
    from datetime import timedelta
    implementation = module()
    path = tmp_path / "source.h5"
    start = datetime(2023, 9, 27, tzinfo=UTC)
    write_minute(path, start)
    cache = tmp_path / "metadata"
    first = DiscoveredFile.from_path(path)
    implementation._inspect(first, tmp_path, cache)
    write_minute(path, start + timedelta(minutes=1))
    second = DiscoveredFile.from_path(path)
    assert second.fingerprint != first.fingerprint
    value = implementation._inspect(second, tmp_path, cache)
    assert value["start_utc"] == "2023-09-27T00:01:00+00:00"
