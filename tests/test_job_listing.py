from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

import pytest
from das_sentinel.persistence.models import Job, Site

from tests.test_worker_protocol import api_client  # noqa: F401


@pytest.fixture
def listing_client(api_client):  # noqa: F811
    client, _, _, source_id, profile_id = api_client
    with client.app.state.session_factory.begin() as session:
        from das_sentinel.persistence import Repositories

        repositories = Repositories(session)
        source = repositories.source_files.get(source_id)
        site_a = repositories.sites.get(source.site_id)
        site_a.data_path = r"\\archive\DAS" + "\\"
        site_b = repositories.sites.add(
            name="other-site", data_path="/archive/site-b", timezone="UTC",
            cable_type="urban",
        )
        profile_b = repositories.profiles.add(
            site_id=site_b.id, version=1, payload={}, is_current=True,
        )
        entries = [
            ("root", "succeeded", "root.h5", site_a, profile_id),
            ("run", "running", "day/run.h5", site_a, profile_id),
            ("queued-a", "queued", "day/a.h5", site_a, profile_id),
            ("queued-z", "queued", "day/z.h5", site_a, profile_id),
            ("success", "succeeded", "day/done.h5", site_a, profile_id),
            ("child", "failed", "day/child/input.h5", site_a, profile_id),
            ("sibling", "cancelled", "day-other/input.h5", site_a, profile_id),
            ("literal", "queued", "day%_/input.h5", site_a, profile_id),
            ("wildcard", "succeeded", "dayXY/input.h5", site_a, profile_id),
            ("other", "succeeded", "day/input.h5", site_b, profile_b.id),
        ]
        for job_id, job_status, relative, site, profile in entries:
            path = (
                site.data_path + relative.replace("/", "\\")
                if site.id == site_a.id else site.data_path + "/" + relative
            )
            item = repositories.source_files.add(
                site_id=site.id, path=path, fingerprint=sha256(job_id.encode()).hexdigest(),
                size_bytes=1, mtime_ns=1,
            )
            session.add(Job(
                id=job_id, source_file_id=item.id, processing_profile_id=profile,
                pipeline_version="v1", status=job_status,
                created_at=datetime(2024, 1, 1, tzinfo=UTC),
            ))
        site_a_id, site_b_id = site_a.id, site_b.id
    return client, site_a_id, site_b_id


def test_page_prioritizes_active_jobs_and_preserves_legacy_list(listing_client):
    client, _, _ = listing_client
    response = client.get("/api/v1/jobs/page", params={"limit": 2, "offset": 1})
    assert response.status_code == 200
    result = response.json()
    assert [item["id"] for item in result["items"]] == ["queued-z", "queued-a"]
    assert result["total"] == 10
    assert result["global_counts"] == {
        "queued": 3, "running": 1, "succeeded": 4, "failed": 1, "cancelled": 1,
    }
    legacy = client.get("/api/v1/jobs").json()
    assert isinstance(legacy, list)
    assert len(legacy) == 10
    assert [item["id"] for item in legacy] == [
        "child", "literal", "other", "queued-a", "queued-z", "root", "run",
        "sibling", "success", "wildcard",
    ]


def test_directory_is_exact_parent_and_keeps_global_counts(listing_client):
    client, site_id, _ = listing_client
    response = client.get("/api/v1/jobs/page", params={
        "site_id": site_id, "directory": "day", "status": "active",
    })
    assert response.status_code == 200
    result = response.json()
    assert [item["id"] for item in result["items"]] == ["run", "queued-z", "queued-a"]
    assert result["total"] == 3
    assert result["global_counts"]["cancelled"] == 1
    item = result["items"][0]
    assert item["source_filename"] == "run.h5"
    assert item["source_directory"] == "day"
    assert item["site_id"] == site_id
    assert item["site_name"] == "urban-a"


@pytest.mark.parametrize(("directory", "expected"), [
    ("", ["root"]), ("day%_", ["literal"]), ("day/child", ["child"]),
    ("missing", []),
])
def test_directory_handles_root_literal_wildcards_and_missing(listing_client, directory, expected):
    client, site_id, _ = listing_client
    response = client.get("/api/v1/jobs/page", params={"site_id": site_id, "directory": directory})
    assert response.status_code == 200
    assert [item["id"] for item in response.json()["items"]] == expected


def test_directories_group_only_jobs_across_path_styles(listing_client):
    client, site_a, site_b = listing_client
    response = client.get("/api/v1/jobs/directories")
    assert response.status_code == 200
    rows = response.json()
    assert {(row["site_id"], row["directory"], row["job_count"]) for row in rows} == {
        (site_a, "", 1), (site_a, "day", 4), (site_a, "day/child", 1),
        (site_a, "day-other", 1), (site_a, "day%_", 1), (site_a, "dayXY", 1),
        (site_b, "day", 1),
    }


@pytest.mark.parametrize("params", [
    {"status": "invalid"}, {"limit": 0}, {"limit": 101}, {"offset": -1},
    {"directory": "day"}, {"directory": "../day"}, {"directory": "day/../else"},
    {"directory": "/absolute"}, {"directory": "C:/absolute"},
    {"directory": r"\\host\share"}, {"directory": "C:relative"}, {"directory": "day\0"},
])
def test_page_rejects_invalid_filters(listing_client, params):
    client, site_id, _ = listing_client
    if "directory" in params and params != {"directory": "day"}:
        params = {**params, "site_id": site_id}
    assert client.get("/api/v1/jobs/page", params=params).status_code == 422


def test_unknown_site_and_out_of_range_page_are_empty(listing_client):
    client, _, _ = listing_client
    unknown = client.get("/api/v1/jobs/page", params={"site_id": "missing"})
    assert unknown.status_code == 200
    assert unknown.json()["total"] == 0
    assert unknown.json()["items"] == []
    beyond = client.get("/api/v1/jobs/page", params={"offset": 100})
    assert beyond.status_code == 200
    assert beyond.json()["total"] == 10
    assert beyond.json()["items"] == []


def test_page_and_directories_require_admin(listing_client):
    client, _, _ = listing_client
    client.cookies.clear()
    assert client.get("/api/v1/jobs/page").status_code == 401
    assert client.get("/api/v1/jobs/directories").status_code == 401


def test_status_filter_orders_newest_first_with_stable_ties(listing_client):
    client, _, _ = listing_client
    with client.app.state.session_factory.begin() as session:
        session.get(Job, "queued-a").created_at = datetime(2025, 1, 1, tzinfo=UTC)
    response = client.get("/api/v1/jobs/page", params={"status": "queued"})
    assert response.status_code == 200
    assert [item["id"] for item in response.json()["items"]] == ["queued-a", "queued-z", "literal"]
    assert response.json()["total"] == 3


def test_large_history_is_bounded_and_directory_counts_tasks(listing_client):
    client, site_id, _ = listing_client
    with client.app.state.session_factory.begin() as session:
        original = session.get(Job, "root")
        session.add_all([
            Job(
                id=f"extra-{number:03}", source_file_id=original.source_file_id,
                processing_profile_id=original.processing_profile_id,
                pipeline_version=f"extra-{number}", status="succeeded",
            )
            for number in range(102)
        ])
    page = client.get("/api/v1/jobs/page").json()
    assert page["total"] == 112
    assert len(page["items"]) == 50
    second = client.get("/api/v1/jobs/page", params={"offset": 50}).json()
    assert len(second["items"]) == 50
    assert {item["id"] for item in page["items"]}.isdisjoint(
        item["id"] for item in second["items"]
    )
    last = client.get("/api/v1/jobs/page", params={"offset": 100}).json()
    assert len(last["items"]) == 12
    directories = client.get("/api/v1/jobs/directories").json()
    assert next(row["job_count"] for row in directories if row["site_id"] == site_id and row["directory"] == "") == 103


def test_site_root_change_preserves_historical_jobs_and_directory_options(listing_client):
    client, site_id, _ = listing_client
    with client.app.state.session_factory.begin() as session:
        session.get(Site, site_id).data_path = r"\\archive\replacement"
    page = client.get("/api/v1/jobs/page")
    assert page.status_code == 200
    assert page.json()["total"] == 10
    assert page.json()["items"][0]["source_directory"] == "//archive/DAS/day"
    directories = client.get("/api/v1/jobs/directories")
    assert directories.status_code == 200
    assert any(row["site_id"] == site_id and row["directory"] == "//archive/DAS/day"
               and row["job_count"] == 4 for row in directories.json())


def test_historical_directory_filter_is_direct_and_combines_with_status(listing_client):
    client, site_id, _ = listing_client
    with client.app.state.session_factory.begin() as session:
        session.get(Site, site_id).data_path = r"\\archive\replacement"
    response = client.get("/api/v1/jobs/page", params={
        "site_id": site_id, "directory": r"\\archive\DAS\day", "status": "queued",
    })
    assert response.status_code == 200
    assert [row["id"] for row in response.json()["items"]] == ["queued-z", "queued-a"]
    assert response.json()["total"] == 2
    assert response.json()["global_counts"]["queued"] == 3


@pytest.mark.parametrize("wrong_scope", ["other_site", "unregistered", "current_root"])
def test_absolute_directory_filter_requires_registered_history_in_selected_site(listing_client, wrong_scope):
    client, site_a, site_b = listing_client
    with client.app.state.session_factory.begin() as session:
        session.get(Site, site_a).data_path = r"\\archive\replacement"
    site_id = site_b if wrong_scope in {"other_site", "current_root"} else site_a
    directory = {
        "other_site": "//archive/DAS/day",
        "unregistered": "//archive/missing/day",
        "current_root": "/archive/site-b/day",
    }[wrong_scope]
    assert client.get("/api/v1/jobs/page", params={
        "site_id": site_id, "directory": directory,
    }).status_code == 422


def test_historical_directory_remains_filterable_after_final_cancelled_job_deleted(listing_client):
    client, site_id, _ = listing_client
    with client.app.state.session_factory.begin() as session:
        session.get(Site, site_id).data_path = r"\\archive\replacement"
        session.delete(session.get(Job, "sibling"))
    response = client.get("/api/v1/jobs/page", params={
        "site_id": site_id, "directory": "//archive/DAS/day-other",
    })
    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["total"] == 0
