import hashlib
import json

import pytest
from das_sentinel.window_training import readiness, split_rows


def row(index, day, catalog=None):
    return {
        "window_id": str(index),
        "date": day,
        "catalog_ids": catalog or [],
        "read_start_utc": day + "T00:00:00+00:00",
        "read_end_utc": day + "T00:02:00+00:00",
        "grid_sha256": "grid",
        "unit": "strain",
        "window_seconds": 30,
        "labels": {"earthquake": "present" if index % 2 else "absent"},
    }


def test_joint_groups_and_unknown_readiness():
    rows = [
        row(0, "2020-01-01", ["quake"]),
        row(1, "2020-01-02", ["quake"]),
        row(2, "2020-01-02"),
        row(3, "2020-01-03"),
    ]
    split_rows(rows)
    assert rows[0]["group"] == rows[1]["group"] == rows[2]["group"]
    assert not readiness(rows, "site")["ready"]
    assert readiness([], "site")["eligible_windows"] == 0


def test_split_parses_each_rows_isolation_metadata_once(monkeypatch):
    from collections import Counter
    from datetime import datetime

    import das_sentinel.window_training as training

    parsed = Counter()

    class CountingDatetime:
        @staticmethod
        def fromisoformat(value):
            parsed[value] += 1
            return datetime.fromisoformat(value)

    class CountedList(list):
        def __init__(self, values):
            super().__init__(values)
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return super().__iter__()

    rows = [row(index, f"2020-01-0{index + 1}") for index in range(4)]
    for item in rows:
        item["catalog_ids"] = CountedList(["catalog-" + item["window_id"]])
        item["sources"] = CountedList([{"fingerprint": "source-" + item["window_id"]}])
    monkeypatch.setattr(training, "datetime", CountingDatetime)

    split_rows(rows)

    assert parsed == Counter({item[key]: 1 for item in rows for key in
                              ("read_start_utc", "read_end_utc")})
    assert [item["catalog_ids"].iterations for item in rows] == [1, 1, 1, 1]
    assert [item["sources"].iterations for item in rows] == [1, 1, 1, 1]


@pytest.mark.parametrize("reverse", [False, True])
def test_split_preserves_frozen_transitive_isolation_assignments(reverse):
    rows = [row(index, day, catalog) for index, day, catalog in [
        (0, "2020-01-01", ["catalog-a"]), (1, "2020-01-01", []),
        (2, "2020-01-02", ["catalog-a"]), (3, "2020-01-03", []),
        (4, "2020-01-04", []), (5, "2020-01-06", []), (6, "2020-01-06", []),
        (7, "2020-01-07", []), (8, "2020-01-07", []), (9, "2020-01-08", []),
        (10, "2020-01-09", []), (11, "2020-01-10", []),
    ]]
    rows[2].update(read_start_utc="2020-01-02T03:00:00+00:00", read_end_utc="2020-01-02T03:02:00+00:00")
    rows[3].update(read_start_utc="2020-01-02T03:01:00+00:00", read_end_utc="2020-01-02T03:04:00+00:00",
                   sources=[{"fingerprint": "source-a"}])
    rows[4]["sources"] = [{"fingerprint": "source-a"}]
    rows[11]["labels"] = {"earthquake": "unknown"}
    expected = {
        str(index): (group, split)
        for indices, group, split in [
            (range(5), "c6fc016e52917403523b867a7e7a61a9c367432755f17150542651de4edd8c7e", "train"),
            ([5, 6], "80162e1dc8c929edd48f944e83f3667b2e6e778f8b1de5e7132d642475e947f3", "test"),
            ([7, 8], "20a498b2629833323e1a45be9b2bb6c73b8e75ef0aeb86e73d962d0c3032fb4b", "dev"),
            ([9], "8d81fb9d8d154c5e6d6840b806b2c52e6285fb84c30152a00a09619f41b70523", "train"),
            ([10], "6e32bc5b03ab4d239b1eff1de30de5007f906dda41e8f720240bbf6481496a60", "train"),
            ([11], "71447c96d68bdc65906c5092de19f2d27634b0fb59a86baac726a961b4de4228", "train"),
        ] for index in indices
    }

    split_rows(list(reversed(rows)) if reverse else rows)

    assert {item["window_id"]: (item["group"], item["split"]) for item in rows} == expected


def test_split_balances_independent_groups_and_is_input_order_stable():
    from datetime import date, timedelta

    rows = [
        row(day * 2 + positive, (date(2020, 1, 1) + timedelta(days=day)).isoformat())
        for day in range(20)
        for positive in range(2)
    ]
    split_rows(rows)
    assert readiness(rows, "site")["ready"]
    assert {split: sum(r["split"] == split for r in rows)
            for split in ("train", "dev", "test")} == {"train": 28, "dev": 6, "test": 6}
    expected = {r["window_id"]: (r["group"], r["split"]) for r in rows}
    split_rows(list(reversed(rows)))
    assert {r["window_id"]: (r["group"], r["split"]) for r in rows} == expected


@pytest.mark.parametrize("mixed_groups", range(4))
def test_split_reserves_both_classes_when_only_three_groups_per_class(mixed_groups):
    rows = []
    day = 1
    for _ in range(mixed_groups):
        rows.extend(row(day * 2 + positive, f"2020-01-{day:02d}") for positive in range(2))
        day += 1
    for positive in range(2):
        for _ in range(3 - mixed_groups):
            rows.append(row(day * 2 + positive, f"2020-01-{day:02d}"))
            day += 1
    split_rows(rows)
    assert readiness(rows, "site")["ready"]
    expected = {r["window_id"]: (r["group"], r["split"]) for r in rows}
    split_rows(list(reversed(rows)))
    assert {r["window_id"]: (r["group"], r["split"]) for r in rows} == expected


def test_split_does_not_break_isolation_to_supply_missing_positive_group():
    rows = [row(day * 2 + positive, f"2020-01-0{day}")
            for day in range(1, 4) for positive in range(2)]
    for item in rows[:4]:
        item["sources"] = [{"fingerprint": "shared-source"}]
    split_rows(rows)
    assert len({r["group"] for r in rows[:4]}) == 1
    assert len({r["split"] for r in rows[:4]}) == 1
    assert not readiness(rows, "site")["ready"]


def test_real_training_and_inference(tmp_path):
    pytest.importorskip("torch")
    import numpy as np
    from das_sentinel.classifiers.window_model import load_bundle, prepare
    from das_sentinel.window_training import train

    rows = []
    for day in range(1, 4):
        for positive in range(2):
            item = row(day * 2 + positive, f"2020-01-0{day}")
            arrays = {
                "waveform": np.full((8, 3000), 1 + positive * 10, np.float32),
                "valid_channel_mask": np.ones(8, bool),
                "recognition_channel_mask": np.ones(8, bool),
                "region_id": np.arange(8) % 3,
            }
            path = tmp_path / f"{day}-{positive}.npz"
            np.savez(path, **arrays)
            item.update(
                waveform=str(path),
                waveform_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            rows.append(item)
    split_rows(rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 2, "task": "earthquake_including_blast_vs_noise", "rows": rows, "site_id": "synthetic"}))
    report = train(manifest, tmp_path / "model", epochs=2)
    assert report["encoder"]["test"]["classes"]["earthquake"]["known"] == 2
    assert report["encoder"]["test"]["fully_reviewed_hours"] == pytest.approx(60 / 3600)
    assert report["publication_ready"] is False
    assert report["task"] == "earthquake_including_blast_vs_noise"
    assert report["thresholds"] == [0.5]
    assert set(report["encoder"]["test"]["classes"]) == {"earthquake"}
    assert load_bundle(tmp_path / "model", "grid", unit="strain")(
        *prepare(arrays)
    ).shape == (1,)


def test_training_binds_normalized_grid_to_model_and_rejects_mismatch(tmp_path):
    pytest.importorskip("torch")
    import numpy as np
    from das_sentinel.window_training import TASK, train

    distance = np.arange(3, dtype="<f8") * 4.1
    grid = {"version": "native-channel-union-v1", "channels": 3, "spacing_m": 4.1, "start_locus": 0,
            "grid_sha256": hashlib.sha256(distance.tobytes()).hexdigest()}
    rows = []
    for day in range(1, 4):
        for positive in range(2):
            item = row(day * 2 + positive, f"2020-01-0{day}")
            path = tmp_path / f"{day}-{positive}.npz"
            np.savez(path, waveform=np.ones((3, 3000), np.float32), distance_m=distance,
                     valid_channel_mask=np.ones(3, bool), recognition_channel_mask=np.ones(3, bool),
                     region_id=np.zeros(3, np.int8), relative_seconds=np.arange(3000) / 100)
            item.update(grid_sha256=grid["grid_sha256"], grid_normalization=grid["version"], waveform=str(path),
                        waveform_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            rows.append(item)
    split_rows(rows)
    manifest = {"schema_version": 2, "task": TASK, "site_id": "synthetic", "rows": rows, "training_grid": grid}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    train(path, tmp_path / "model", epochs=1)
    metadata = json.loads((tmp_path / "model/model.json").read_text())
    assert metadata["training_grid"] == grid
    from pathlib import Path

    waveform = Path(rows[0]["waveform"])
    with np.load(waveform) as data:
        original = dict(data)
    for field in ("waveform", "valid_channel_mask", "recognition_channel_mask", "region_id"):
        np.savez(waveform, **(original | {field: original[field][:-1]}))
        rows[0]["waveform_sha256"] = hashlib.sha256(waveform.read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="grid"):
            train(path, tmp_path / ("invalid-" + field), epochs=1)
    manifest["training_grid"]["channels"] = 4
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="grid"):
        train(path, tmp_path / "invalid", epochs=1)


def test_training_weights_use_only_train_and_shuffle_reproducibly(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import numpy as np
    from das_sentinel.classifiers import window_model
    from das_sentinel.window_training import train

    rows = []
    for day in range(1, 7):
        for sample in range(day + 1):
            item = row(len(rows), f"2020-01-{day:02d}")
            item["labels"] = {"earthquake": "present" if sample == 0 else "absent"}
            path = tmp_path / f"{day}-{sample}.npz"
            np.savez(path, waveform=np.full((4, 3000), len(rows) + 1, np.float32),
                     valid_channel_mask=np.ones(4, bool), recognition_channel_mask=np.ones(4, bool),
                     region_id=np.arange(4) % 3)
            item.update(waveform=str(path), waveform_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            rows.append(item)
    split_rows(rows)
    training = [r for r in rows if r["split"] == "train"]
    positive = sum(r["labels"]["earthquake"] == "present" for r in training)
    negative = len(training) - positive
    expected_weight = negative / positive
    assert expected_weight != (len(rows) - 6) / 6
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 2, "task": "earthquake_including_blast_vs_noise",
                                    "rows": rows, "site_id": "synthetic"}))
    orders, weights = [], []
    original_forward, original_loss = window_model.WindowEncoder.forward, window_model.masked_loss

    def forward(self, planes, mask, regions):
        if torch.is_grad_enabled():
            orders.append(float(planes[0, 0, 0, 0]))
        return original_forward(self, planes, mask, regions)

    def loss(logits, targets, **kwargs):
        weights.append(kwargs.get("pos_weight"))
        return original_loss(logits, targets, **kwargs)

    monkeypatch.setattr(window_model.WindowEncoder, "forward", forward)
    monkeypatch.setattr(window_model, "masked_loss", loss)
    first = train(manifest, tmp_path / "first", epochs=2)
    assert first["training"] == {"epochs": 2, "seed": 42, "positive": positive,
                                  "negative": negative, "pos_weight": pytest.approx(expected_weight)}
    assert len(weights) == 4 * len(training)
    assert all(weight is not None and weight.item() == pytest.approx(expected_weight) for weight in weights)
    initial_order = []
    for item in training:
        with np.load(item["waveform"]) as arrays:
            initial_order.append(float(window_model.prepare(arrays)[0][0, 0, 0, 0]))
    assert orders[:len(training)] != initial_order
    assert orders[:len(training)] != orders[len(training):]
    first_orders = orders[:]
    orders.clear()
    train(manifest, tmp_path / "second", epochs=2)
    assert orders == first_orders
    first_weights = torch.load(tmp_path / "first" / "weights.pt", weights_only=True)
    second_weights = torch.load(tmp_path / "second" / "weights.pt", weights_only=True)
    assert all(torch.equal(first_weights[k], second_weights[k]) for k in first_weights)


def test_tied_predictions_do_not_gain_average_precision_from_row_order(tmp_path):
    pytest.importorskip("torch")
    import numpy as np
    from das_sentinel.window_training import train

    path = tmp_path / "same-input.npz"
    np.savez(
        path,
        waveform=np.ones((4, 3000), np.float32),
        valid_channel_mask=np.ones(4, bool),
        recognition_channel_mask=np.ones(4, bool),
        region_id=np.arange(4) % 3,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for day in range(1, 4):
        for positive in (1, 0):
            item = row(day * 2 + positive, f"2020-01-0{day}")
            item.update(waveform=str(path), waveform_sha256=digest)
            rows.append(item)
    split_rows(rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 2, "task": "earthquake_including_blast_vs_noise", "rows": rows, "site_id": "synthetic"}))
    report = train(manifest, tmp_path / "model", epochs=0)
    for network in ("encoder", "baseline"):
        for split in ("dev", "test"):
            assert (
                report[network][split]["classes"]["earthquake"]["average_precision"]
                == 0.5
            )


def test_unknown_labels_do_not_supply_fully_reviewed_false_alarm_hours(tmp_path):
    pytest.importorskip("torch")
    import numpy as np
    from das_sentinel.window_training import train

    path = tmp_path / "waveform.npz"
    np.savez(
        path,
        waveform=np.ones((4, 3000), np.float32),
        valid_channel_mask=np.ones(4, bool),
        recognition_channel_mask=np.ones(4, bool),
        region_id=np.arange(4) % 3,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    states = ["present", "absent", "unknown"]
    for day in range(1, 4):
        for index in range(3):
            item = row(day * 3 + index, f"2020-01-0{day}")
            item.update(
                waveform=str(path),
                waveform_sha256=digest,
                labels={"earthquake": states[index]},
            )
            rows.append(item)
    split_rows(rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 2, "task": "earthquake_including_blast_vs_noise", "rows": rows, "site_id": "synthetic"}))
    report = train(manifest, tmp_path / "model", epochs=0)
    for network in ("encoder", "baseline"):
        for split in ("dev", "test"):
            assert report[network][split]["fully_reviewed_hours"] == pytest.approx(60 / 3600)
            assert report[network][split]["classes"]["earthquake"]["known"] == 2


def test_training_rejects_legacy_manifest_before_loading_arrays(tmp_path):
    pytest.importorskip("torch")
    from das_sentinel.window_training import train

    manifest = tmp_path / "legacy.json"
    manifest.write_text(json.dumps({"schema_version": 1, "rows": [], "site_id": "site"}))
    with pytest.raises(ValueError, match="binary.*manifest"):
        train(manifest, tmp_path / "model")


def test_binary_readiness_needs_only_positive_and_negative_target_labels():
    rows = [row(day * 2 + positive, f"2020-01-0{day}")
            for day in range(1, 4) for positive in range(2)]
    split_rows(rows)
    state = readiness(rows, "site")
    assert state["ready"]
    assert state["classes"] == {"earthquake": {"present": 3, "absent": 3, "unknown": 0}}


def test_empty_export_and_service_failure_recovery(tmp_path, monkeypatch):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    import das_sentinel.persistence.window_models  # noqa: F401
    from das_sentinel.persistence.models import Base, Site
    from das_sentinel.services.window_training import (
        WindowTrainingError,
        WindowTrainingService,
    )
    from sqlalchemy.orm import sessionmaker

    engine = sqlalchemy.create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine)
    with factory.begin() as session:
        site = Site(
            id="site",
            name="site",
            data_path=str(tmp_path),
            timezone="UTC",
            cable_type="downhole",
        )
        session.add(site)
    service = WindowTrainingService(factory, tmp_path)
    state = service.readiness("site")
    assert state["reviewed_windows"] == state["eligible_windows"] == 0
    with pytest.raises(WindowTrainingError):
        service.create("site")
    with pytest.raises(WindowTrainingError) as error:
        service.readiness("missing")
    assert error.value.status_code == 404
    service.active = "running"
    with pytest.raises(WindowTrainingError, match="正在运行"):
        service.create("site")
    service.active = None
    folder = service.root / "interrupted"
    folder.mkdir()
    service._save(
        folder / "job.json",
        {
            "id": "interrupted",
            "site_id": "site",
            "status": "running",
            "created_at": "2020",
        },
    )
    service.close()
    recovered = WindowTrainingService(factory, tmp_path)
    assert recovered.jobs("site")[0]["status"] == "failed"
    recovered.close()
    engine.dispose()


def test_read_context_and_source_join_across_dates():
    rows = [row(0, "2020-01-01"), row(1, "2020-01-02"), row(2, "2020-01-03")]
    rows[0]["read_end_utc"] = "2020-01-02T00:01:00+00:00"
    rows[1]["sources"] = [{"fingerprint": "same"}]
    rows[2]["sources"] = [{"fingerprint": "same"}]
    split_rows(rows)
    assert len({r["group"] for r in rows}) == 1


def test_readiness_rejects_mixed_units_on_same_distance_grid():
    rows = [
        row(day * 2 + positive, f"2020-01-0{day}")
        for day in range(1, 4)
        for positive in range(2)
    ]
    split_rows(rows)
    assert readiness(rows, "site")["ready"]
    rows[-1]["unit"] = "counts"
    state = readiness(rows, "site")
    assert not state["ready"]
    assert any("单位" in reason for reason in state["reasons"])


def test_readiness_rejects_legacy_minute_even_with_complete_classes_and_groups():
    rows = [row(day * 2 + positive, f"2020-01-0{day}")
            for day in range(1, 4) for positive in range(2)]
    split_rows(rows)
    assert readiness(rows, "site")["ready"]
    rows[-1]["window_seconds"] = 60
    state = readiness(rows, "site")
    assert not state["ready"]
    assert any("30 秒" in reason for reason in state["reasons"])


def test_api_auth_csrf_and_not_ready(tmp_path):
    pytest.importorskip("sqlalchemy")
    from das_sentinel.api.auth import require_admin
    from das_sentinel.api.routes.window_training import router
    from das_sentinel.services.window_training import WindowTrainingError
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette.middleware.sessions import SessionMiddleware

    class NotReady:
        def readiness(self, site_id):
            return readiness([], site_id)

        def create(self, site_id):
            raise WindowTrainingError("整窗标签不足")

    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test-only")
    app.include_router(router, prefix="/api/v1")
    app.state.window_training_service = NotReady()
    with TestClient(app) as client:
        assert (
            client.get("/api/v1/window-training/readiness?site_id=site").status_code
            == 401
        )
        app.dependency_overrides[require_admin] = lambda: object()
        assert (
            client.get("/api/v1/window-training/readiness?site_id=site").json()[
                "eligible_windows"
            ]
            == 0
        )
        assert (
            client.post(
                "/api/v1/window-training/jobs", json={"site_id": "site"}
            ).status_code
            == 403
        )
        from das_sentinel.api.auth import require_csrf

        app.dependency_overrides[require_csrf] = lambda: object()
        response = client.post("/api/v1/window-training/jobs", json={"site_id": "site"})
        assert response.status_code == 409
        assert "整窗标签不足" in response.json()["detail"]
