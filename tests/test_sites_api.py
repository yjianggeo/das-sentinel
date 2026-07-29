from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from threading import Event, Thread

import pytest
from argon2 import PasswordHasher
from das_sentinel.api import create_app
from das_sentinel.config import Settings
from das_sentinel.persistence import (
    Repositories,
    create_local_engine,
    session_factory,
    upgrade_database,
)
from das_sentinel.persistence.models import ProcessingProfile, Site
from das_sentinel.persistence.repositories import ProfileRepository, SiteRepository
from fastapi.testclient import TestClient
from sqlalchemy import func, select

PASSWORD = "correct horse"


def _settings(tmp_path: Path) -> Settings:
    data_root = tmp_path / "raw"
    app_data = tmp_path / "app-data"
    data_root.mkdir()
    app_data.mkdir()
    return Settings(
        data_root=data_root,
        app_data_dir=app_data,
        database_url=f"sqlite+pysqlite:///{app_data / 'das.sqlite3'}",
        api_url="http://testserver",
        session_secret="s" * 32,
        worker_token="w" * 32,
        secure_cookies=False,
    )


@pytest.fixture
def csrf_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, str, Settings]]:
    settings = _settings(tmp_path)
    upgrade_database(settings.database_url)
    engine = create_local_engine(settings.database_url)
    try:
        sessions = session_factory(engine)
        with sessions.begin() as session:
            Repositories(session).admins.add(
                password_hash=PasswordHasher().hash(PASSWORD),
            )
    finally:
        engine.dispose()
    monkeypatch.setattr(
        "das_sentinel.api.app.check_database_startup",
        lambda engine: None,
    )
    with TestClient(create_app(settings)) as client:
        login = client.post("/api/v1/session", json={"password": PASSWORD})
        assert login.status_code == 204
        yield client, login.headers["x-csrf-token"], settings


def _site_payload(
    settings: Settings,
    *,
    name: str = "city-main",
    data_path: str = "city-main",
) -> dict:
    path = settings.data_root / data_path
    path.mkdir(exist_ok=True)
    return {
        "name": name,
        "data_path": data_path,
        "timezone": "Asia/Shanghai",
        "cable_type": "urban",
        "enabled": True,
        "profile": {
            "device": "cpu",
            "block_seconds": 20.0,
            "overlap_seconds": 5.0,
        },
    }


def _create_site(
    client: TestClient,
    token: str,
    settings: Settings,
    **overrides,
):
    payload = _site_payload(settings)
    payload.update(overrides)
    return client.post(
        "/api/v1/sites",
        json=payload,
        headers={"X-CSRF-Token": token},
    )


def test_site_mutation_requires_csrf(
    csrf_client: tuple[TestClient, str, Settings],
):
    client, _, settings = csrf_client

    response = client.post("/api/v1/sites", json=_site_payload(settings))

    assert response.status_code == 403


def test_site_mutation_rejects_wrong_csrf(
    csrf_client: tuple[TestClient, str, Settings],
):
    client, token, settings = csrf_client

    response = client.post(
        "/api/v1/sites",
        json=_site_payload(settings),
        headers={"X-CSRF-Token": token + "x"},
    )

    assert response.status_code == 403


def test_create_site_persists_resolved_path_and_current_profile_v1(
    csrf_client: tuple[TestClient, str, Settings],
):
    client, token, settings = csrf_client

    response = _create_site(client, token, settings)

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "city-main"
    assert body["data_path"] == str((settings.data_root / "city-main").resolve())
    assert body["timezone"] == "Asia/Shanghai"
    assert body["cable_type"] == "urban"
    assert body["enabled"] is True
    assert body["current_profile"]["version"] == 1
    assert body["current_profile"]["is_current"] is True
    assert body["current_profile"]["payload"]["device"] == "cpu"
    assert body["current_profile"]["payload"]["recursive_discovery"] is True
    assert "password" not in response.text


def test_list_and_detail_include_disabled_sites_for_admin(
    csrf_client: tuple[TestClient, str, Settings],
):
    client, token, settings = csrf_client
    created = _create_site(client, token, settings, enabled=False)
    site_id = created.json()["id"]

    listed = client.get("/api/v1/sites")
    detailed = client.get(f"/api/v1/sites/{site_id}")

    assert listed.status_code == 200
    assert [site["id"] for site in listed.json()] == [site_id]
    assert listed.json()[0]["enabled"] is False
    assert detailed.status_code == 200
    assert detailed.json()["enabled"] is False


def test_site_reads_require_enabled_admin_session(
    csrf_client: tuple[TestClient, str, Settings],
):
    client, _, _ = csrf_client
    client.cookies.clear()

    assert client.get("/api/v1/sites").status_code == 401
    assert client.get("/api/v1/sites/missing").status_code == 401


@pytest.mark.parametrize("cable_type", ["downhole", "urban", "subsea"])
def test_approved_cable_types_are_accepted(
    csrf_client: tuple[TestClient, str, Settings],
    cable_type: str,
):
    client, token, settings = csrf_client
    data_path = f"path-{cable_type}"
    payload = _site_payload(
        settings,
        name=f"site-{cable_type}",
        data_path=data_path,
    )
    payload["cable_type"] = cable_type

    response = client.post(
        "/api/v1/sites",
        json=payload,
        headers={"X-CSRF-Token": token},
    )

    assert response.status_code == 201


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cable_type", "aerial"),
        ("timezone", "Mars/Olympus"),
    ],
)
def test_site_rejects_invalid_cable_type_or_timezone(
    csrf_client: tuple[TestClient, str, Settings],
    field: str,
    value: str,
):
    client, token, settings = csrf_client
    payload = _site_payload(settings)
    payload[field] = value

    response = client.post(
        "/api/v1/sites",
        json=payload,
        headers={"X-CSRF-Token": token},
    )

    assert response.status_code == 422


def test_site_and_profile_schemas_reject_unknown_or_invalid_fields(
    csrf_client: tuple[TestClient, str, Settings],
):
    client, token, settings = csrf_client
    unknown = _site_payload(settings)
    unknown["unexpected"] = True
    invalid_profile = {
        **_site_payload(settings, name="other", data_path="other"),
        "profile": {
            "block_seconds": 5.0,
            "overlap_seconds": 5.0,
        },
    }

    unknown_response = client.post(
        "/api/v1/sites",
        json=unknown,
        headers={"X-CSRF-Token": token},
    )
    invalid_response = client.post(
        "/api/v1/sites",
        json=invalid_profile,
        headers={"X-CSRF-Token": token},
    )

    assert unknown_response.status_code == 422
    assert invalid_response.status_code == 422


def test_site_path_must_exist_be_directory_and_stay_under_data_root(
    csrf_client: tuple[TestClient, str, Settings],
    tmp_path: Path,
):
    client, token, settings = csrf_client
    outside = tmp_path / "outside"
    outside.mkdir()
    source_file = settings.data_root / "source.h5"
    source_file.write_bytes(b"raw")

    for name, data_path in (
        ("outside", str(outside)),
        ("missing", "missing"),
        ("file", "source.h5"),
        ("traversal", "../outside"),
    ):
        payload = _site_payload(
            settings,
            name=f"site-{name}",
            data_path=f"valid-{name}",
        )
        payload["data_path"] = data_path
        response = client.post(
            "/api/v1/sites",
            json=payload,
            headers={"X-CSRF-Token": token},
        )
        assert response.status_code == 422, name


def test_site_path_rejects_symlink_escape(
    csrf_client: tuple[TestClient, str, Settings],
    tmp_path: Path,
):
    client, token, settings = csrf_client
    outside = tmp_path / "outside-link-target"
    outside.mkdir()
    link = settings.data_root / "escape-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    payload = _site_payload(settings, data_path="valid-link-test")
    payload["data_path"] = "escape-link"

    response = client.post(
        "/api/v1/sites",
        json=payload,
        headers={"X-CSRF-Token": token},
    )

    assert response.status_code == 422


def test_site_validation_never_modifies_raw_directory(
    csrf_client: tuple[TestClient, str, Settings],
):
    client, token, settings = csrf_client
    raw_site = settings.data_root / "read-only-site"
    raw_site.mkdir()
    marker = raw_site / "source.h5"
    marker.write_bytes(b"immutable raw bytes")
    before = tuple(raw_site.iterdir()), marker.read_bytes()
    payload = {
        **_site_payload(settings, data_path="read-only-site"),
        "name": "read-only-site",
    }

    response = client.post(
        "/api/v1/sites",
        json=payload,
        headers={"X-CSRF-Token": token},
    )

    assert response.status_code == 201
    assert (tuple(raw_site.iterdir()), marker.read_bytes()) == before


def test_patch_creates_new_immutable_current_profile_and_updates_site(
    csrf_client: tuple[TestClient, str, Settings],
):
    client, token, settings = csrf_client
    created = _create_site(client, token, settings)
    original = created.json()

    response = client.patch(
        f"/api/v1/sites/{original['id']}",
        json={
            "name": "city-renamed",
            "enabled": False,
            "profile": {"device": "cuda"},
        },
        headers={"X-CSRF-Token": token},
    )

    assert response.status_code == 200
    updated = response.json()
    assert updated["name"] == "city-renamed"
    assert updated["enabled"] is False
    assert updated["current_profile"]["version"] == 2
    assert updated["current_profile"]["payload"]["device"] == "cuda"
    assert updated["current_profile"]["payload"]["block_seconds"] == 20.0

    engine = create_local_engine(settings.database_url)
    try:
        sessions = session_factory(engine)
        with sessions() as session:
            profiles = Repositories(session).profiles.list_for_site(original["id"])
            assert [profile.version for profile in profiles] == [1, 2]
            assert [profile.is_current for profile in profiles] == [False, True]
            assert profiles[0].payload["device"] == "cpu"
    finally:
        engine.dispose()


def test_patch_enabled_only_still_creates_next_profile_version(
    csrf_client: tuple[TestClient, str, Settings],
):
    client, token, settings = csrf_client
    created = _create_site(client, token, settings).json()

    response = client.patch(
        f"/api/v1/sites/{created['id']}",
        json={"enabled": False},
        headers={"X-CSRF-Token": token},
    )

    assert response.status_code == 200
    assert response.json()["current_profile"]["version"] == 2
    assert response.json()["current_profile"]["payload"] == (
        created["current_profile"]["payload"]
    )


def test_patch_invalid_profile_returns_validation_error_without_new_version(
    csrf_client: tuple[TestClient, str, Settings],
):
    client, token, settings = csrf_client
    created = _create_site(client, token, settings).json()

    response = client.patch(
        f"/api/v1/sites/{created['id']}",
        json={
            "profile": {
                "block_seconds": 5.0,
                "overlap_seconds": 5.0,
            }
        },
        headers={"X-CSRF-Token": token},
    )

    assert response.status_code == 422
    unchanged = client.get(f"/api/v1/sites/{created['id']}").json()
    assert unchanged["current_profile"]["version"] == 1


def test_patch_requires_csrf_and_missing_site_returns_not_found(
    csrf_client: tuple[TestClient, str, Settings],
):
    client, token, settings = csrf_client
    created = _create_site(client, token, settings).json()

    assert client.patch(
        f"/api/v1/sites/{created['id']}",
        json={"enabled": False},
    ).status_code == 403
    assert client.patch(
        "/api/v1/sites/00000000-0000-0000-0000-000000000000",
        json={"enabled": False},
        headers={"X-CSRF-Token": token},
    ).status_code == 404


def test_duplicate_name_returns_conflict_and_rolls_back_create(
    csrf_client: tuple[TestClient, str, Settings],
):
    client, token, settings = csrf_client
    assert _create_site(client, token, settings).status_code == 201
    other = settings.data_root / "other"
    other.mkdir()

    duplicate = client.post(
        "/api/v1/sites",
        json={
            **_site_payload(settings, name="unused", data_path="other"),
            "name": "city-main",
        },
        headers={"X-CSRF-Token": token},
    )

    assert duplicate.status_code == 409
    engine = create_local_engine(settings.database_url)
    try:
        sessions = session_factory(engine)
        with sessions() as session:
            assert session.scalar(select(func.count()).select_from(Site)) == 1
            assert (
                session.scalar(
                    select(func.count()).select_from(ProcessingProfile)
                )
                == 1
            )
    finally:
        engine.dispose()


def test_failed_patch_rolls_back_site_and_profile_switch(
    csrf_client: tuple[TestClient, str, Settings],
):
    client, token, settings = csrf_client
    first = _create_site(client, token, settings).json()
    _site_payload(settings, name="second", data_path="second")
    second = _create_site(
        client,
        token,
        settings,
        name="second",
        data_path="second",
    ).json()

    response = client.patch(
        f"/api/v1/sites/{second['id']}",
        json={
            "name": first["name"],
            "enabled": False,
            "profile": {"device": "cuda"},
        },
        headers={"X-CSRF-Token": token},
    )

    assert response.status_code == 409
    unchanged = client.get(f"/api/v1/sites/{second['id']}").json()
    assert unchanged["name"] == "second"
    assert unchanged["enabled"] is True
    assert unchanged["current_profile"]["version"] == 1
    assert unchanged["current_profile"]["payload"]["device"] == "cpu"


@pytest.mark.parametrize("operation", ["list", "detail"])
def test_site_reads_do_not_block_event_loop(
    csrf_client: tuple[TestClient, str, Settings],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
):
    client, token, settings = csrf_client
    created = _create_site(client, token, settings).json()
    entered = Event()
    release = Event()
    ticked = Event()

    @client.app.get(f"/api/v1/site-read-tick-{operation}")
    async def tick():
        ticked.set()
        return {"ok": True}

    target = "list" if operation == "list" else "get"
    original = getattr(SiteRepository, target)

    def blocking(repository, *args, **kwargs):
        entered.set()
        release.wait(5)
        return original(repository, *args, **kwargs)

    monkeypatch.setattr(SiteRepository, target, blocking)
    url = "/api/v1/sites" if operation == "list" else f"/api/v1/sites/{created['id']}"
    request = Thread(target=lambda: client.get(url))
    request.start()
    assert entered.wait(2)
    tick_request = Thread(
        target=lambda: client.get(f"/api/v1/site-read-tick-{operation}")
    )
    tick_request.start()
    try:
        assert ticked.wait(0.5)
    finally:
        release.set()
        request.join(5)
        tick_request.join(5)


@pytest.mark.parametrize("operation", ["create", "patch"])
def test_site_writes_run_complete_sync_transaction_off_event_loop(
    csrf_client: tuple[TestClient, str, Settings],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
):
    client, token, settings = csrf_client
    created = (
        _create_site(client, token, settings).json()
        if operation == "patch"
        else None
    )
    entered = Event()
    release = Event()
    ticked = Event()

    @client.app.get(f"/api/v1/site-write-tick-{operation}")
    async def tick():
        ticked.set()
        return {"ok": True}

    original = ProfileRepository.add

    def blocking_add(repository, **kwargs):
        entered.set()
        release.wait(5)
        return original(repository, **kwargs)

    monkeypatch.setattr(ProfileRepository, "add", blocking_add)
    if operation == "create":
        payload = _site_payload(
            settings,
            name="threaded-create",
            data_path="threaded-create",
        )
        request_call = lambda: client.post(
            "/api/v1/sites",
            json=payload,
            headers={"X-CSRF-Token": token},
        )
    else:
        request_call = lambda: client.patch(
            f"/api/v1/sites/{created['id']}",
            json={"enabled": False},
            headers={"X-CSRF-Token": token},
        )

    request = Thread(target=request_call)
    request.start()
    assert entered.wait(2)
    tick_request = Thread(
        target=lambda: client.get(f"/api/v1/site-write-tick-{operation}")
    )
    tick_request.start()
    try:
        assert ticked.wait(0.5)
    finally:
        release.set()
        request.join(5)
        tick_request.join(5)


@pytest.mark.parametrize(
    "profile_overrides",
    [
        {"coherence_threshold": "NaN"},
        {"coherence_threshold": "Infinity"},
        {"min_coincidence": 0},
        {"energy_window_seconds": 0.0},
        {"stalta_threshold": 0.0},
        {"slowness_max_s_per_m": 0.0},
        {"coherence_threshold": 0.0},
        {"trajectory_threshold": 1.01},
        {"trajectory_velocity_min_mps": 60.0, "trajectory_velocity_max_mps": 60.0},
        {"fusion_normal_score": 0.9, "fusion_strong_score": 0.8},
        {"merge_gap_seconds": -0.1},
        {"artifact_pad_seconds": -0.1},
        {"energy_bands_hz": []},
        {"energy_bands_hz": [[5.0, 10.0], [1.0, 2.0]]},
    ],
)
def test_site_create_and_patch_reject_non_runnable_profiles(
    csrf_client: tuple[TestClient, str, Settings],
    profile_overrides,
):
    client, token, settings = csrf_client
    create_payload = _site_payload(settings)
    create_payload["profile"].update(profile_overrides)
    create = client.post(
        "/api/v1/sites",
        json=create_payload,
        headers={"X-CSRF-Token": token},
    )

    valid_payload = _site_payload(
        settings,
        name="valid-for-patch",
        data_path="valid-for-patch",
    )
    valid = client.post(
        "/api/v1/sites",
        json=valid_payload,
        headers={"X-CSRF-Token": token},
    )
    patch = client.patch(
        f"/api/v1/sites/{valid.json()['id']}",
        json={"profile": profile_overrides},
        headers={"X-CSRF-Token": token},
    )

    assert create.status_code == 422
    assert patch.status_code == 422


def test_site_create_rejects_single_energy_band_without_persisting_profile(
    csrf_client: tuple[TestClient, str, Settings],
):
    client, token, settings = csrf_client
    payload = _site_payload(settings)
    payload["profile"]["energy_bands_hz"] = [[1.0, 5.0]]

    response = client.post(
        "/api/v1/sites",
        json=payload,
        headers={"X-CSRF-Token": token},
    )

    assert response.status_code == 422
    engine = create_local_engine(settings.database_url)
    try:
        sessions = session_factory(engine)
        with sessions() as session:
            assert session.scalar(select(func.count()).select_from(Site)) == 0
            assert (
                session.scalar(
                    select(func.count()).select_from(ProcessingProfile)
                )
                == 0
            )
    finally:
        engine.dispose()


def test_site_patch_rejects_single_energy_band_without_new_profile(
    csrf_client: tuple[TestClient, str, Settings],
):
    client, token, settings = csrf_client
    created = _create_site(client, token, settings).json()

    response = client.patch(
        f"/api/v1/sites/{created['id']}",
        json={"profile": {"energy_bands_hz": [[1.0, 5.0]]}},
        headers={"X-CSRF-Token": token},
    )

    assert response.status_code == 422
    unchanged = client.get(f"/api/v1/sites/{created['id']}").json()
    assert unchanged["current_profile"]["version"] == 1
    engine = create_local_engine(settings.database_url)
    try:
        sessions = session_factory(engine)
        with sessions() as session:
            assert (
                session.scalar(
                    select(func.count()).select_from(ProcessingProfile)
                )
                == 1
            )
    finally:
        engine.dispose()
