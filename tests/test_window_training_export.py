import pytest

pytest.importorskip("sqlalchemy")

from das_sentinel.persistence.models import Admin
from das_sentinel.services.event_windows import enqueue_windows
from das_sentinel.window_training import export_rows

from tests import test_event_windows as windows

context = windows.context


def reviewed_window(context):
    session, site, profile, service = context
    analysis = enqueue_windows(
        session,
        site.id,
        profile.id,
        [windows.plan()],
        catalog_matches=[
            {"catalog_id": "us-shared-quake", "phase": "P"},
            {"catalog_id": "us-shared-quake", "phase": "S"},
        ],
    )[0]
    service.claim("test")
    windows.make_result(service, analysis)
    service.result(analysis.id, "test", 1)
    session.add(Admin(id=1, password_hash="test"))
    session.flush()
    service.review(
        analysis.window_id,
        analysis_id=analysis.id,
        input_sha256=analysis.input_sha256,
        labels={"earthquake": "present", "blast": "absent", "traffic": "unknown"},
        note="",
        request_id="training-test",
        admin_id=1,
    )
    return session, site, service, analysis


def test_export_preserves_actual_catalog_identifier_and_physical_unit(context):
    session, site, service, _analysis = reviewed_window(context)
    manifest = export_rows(session, service.artifact_root, site.id, window_seconds=60)
    assert manifest["rows"][0]["catalog_ids"] == ["us-shared-quake"]
    assert manifest["rows"][0]["unit"] == "strain"


def test_export_excludes_missing_artifact_without_breaking_readiness(context):
    session, site, service, analysis = reviewed_window(context)
    path, _ = service.artifact(analysis.window_id, "waveform")
    path.unlink()
    manifest = export_rows(session, service.artifact_root, site.id, window_seconds=60)
    assert manifest["readiness"]["eligible_windows"] == 0
    assert (
        manifest["readiness"]["excluded_artifacts"][0]["window_id"]
        == analysis.window_id
    )


def test_export_normalization_keeps_original_review_binding(context, monkeypatch, tmp_path):
    import das_sentinel.window_grid as grids

    session, site, service, analysis = reviewed_window(context)
    called = []

    def normalize(rows, root):
        called.append(root)
        assert "recognition_range_m" in rows[0]
        assert rows[0]["input_sha256"] == analysis.input_sha256
        assert rows[0]["labels"] == {"earthquake": "present"}
        return [row | {"source_grid_sha256": row["grid_sha256"], "grid_sha256": "normalized"}
                for row in rows], {"channels": 834, "grid_sha256": "normalized"}

    monkeypatch.setattr(grids, "normalize_rows", normalize)
    manifest = export_rows(session, service.artifact_root, site.id, window_seconds=60,
                           normalization_root=tmp_path / "training-grid")
    assert called == [tmp_path / "training-grid"]
    assert manifest["training_grid"]["channels"] == 834
    assert manifest["readiness"]["training_grid"] == manifest["training_grid"]
    assert manifest["rows"][0]["grid_sha256"] == "normalized"
    assert manifest["rows"][0]["input_sha256"] == analysis.input_sha256


@pytest.mark.parametrize(
    "labels, expected",
    [
        ({"earthquake": "present", "blast": "absent", "traffic": "present"}, "present"),
        ({"earthquake": "absent", "blast": "present", "traffic": "present"}, "present"),
        ({"earthquake": "unknown", "blast": "present", "traffic": "unknown"}, "present"),
        ({"earthquake": "absent", "blast": "absent", "traffic": "present"}, "absent"),
        ({"earthquake": "absent", "blast": "absent", "traffic": "unknown"}, "absent"),
        ({"earthquake": "unknown", "blast": "absent", "traffic": "present"}, None),
        ({"earthquake": "unknown", "blast": "unknown", "traffic": "present"}, None),
    ],
)
def test_binary_export_maps_labels_without_changing_review(context, labels, expected):
    import json

    session, site, service, analysis = reviewed_window(context)
    service.review(
        analysis.window_id,
        analysis_id=analysis.id,
        input_sha256=analysis.input_sha256,
        labels=labels,
        note="binary export test",
        request_id="binary-export-test",
        admin_id=1,
    )
    manifest = export_rows(session, service.artifact_root, site.id, window_seconds=60)
    assert manifest["task"] == "earthquake_including_blast_vs_noise"
    assert manifest["schema_version"] == 2
    assert set(manifest["readiness"]["classes"]) == {"earthquake"}
    if expected is None:
        assert manifest["rows"] == []
        assert manifest["readiness"]["classes"]["earthquake"]["unknown"] == 1
    else:
        assert manifest["rows"][0]["labels"] == {"earthquake": expected}
        assert manifest["rows"][0]["source_labels"] == labels
        assert manifest["readiness"]["classes"]["earthquake"][expected] == 1
    from das_sentinel.persistence.window_models import WindowReview
    from sqlalchemy import select

    saved = session.scalar(select(WindowReview).where(WindowReview.request_id == "binary-export-test"))
    assert json.loads(saved.labels_json) == labels
