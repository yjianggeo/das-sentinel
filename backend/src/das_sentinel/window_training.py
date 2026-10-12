"""Export reviewed windows in the API environment; train in an isolated Torch environment."""

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path

LABELS = ("earthquake",)
TASK = "earthquake_including_blast_vs_noise"
logger = logging.getLogger(__name__)


def binary_labels(source_labels):
    """Project review labels without treating a local vehicle label as absence."""
    if set(source_labels) == {"earthquake"}:
        return dict(source_labels)
    target_states = [source_labels[k] for k in ("earthquake", "blast")]
    if "present" in target_states:
        state = "present"
    elif all(value == "absent" for value in target_states):
        state = "absent"
    else:
        state = "unknown"
    return {"earthquake": state}


def split_rows(rows):
    # Connected components enforce joint date, catalog and native read-context isolation.
    parents = list(range(len(rows)))
    metadata = [
        (
            row["date"],
            set(row["catalog_ids"]),
            datetime.fromisoformat(row["read_start_utc"]),
            datetime.fromisoformat(row["read_end_utc"]),
            {s["fingerprint"] for s in row.get("sources", [])},
        )
        for row in rows
    ]

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, (date, catalog_ids, start, end, sources) in enumerate(metadata):
        for j in range(i):
            other_date, other_catalog_ids, other_start, other_end, other_sources = metadata[j]
            same = date == other_date or bool(catalog_ids & other_catalog_ids)
            overlap = start < other_end and other_start < end
            same_source = bool(sources & other_sources)
            if same or overlap or same_source:
                parents[root(i)] = root(j)
    members = {}
    for i in range(len(rows)):
        members.setdefault(root(i), []).append(i)
    identities = {
        g: hashlib.sha256(
            json.dumps(sorted(rows[i]["window_id"] for i in indices)).encode()
        ).hexdigest()
        for g, indices in members.items()
    }
    counts = {
        g: tuple(
            sum(rows[i]["labels"]["earthquake"] == state for i in indices)
            for state in ("present", "absent")
        )
        for g, indices in members.items()
    }
    groups = sorted(members, key=lambda g: (-sum(counts[g]), identities[g]))
    splits = ("train", "dev", "test")
    totals = [sum(count[c] for count in counts.values()) for c in range(2)]
    targets = {
        split: [total * fraction for total in totals]
        for split, fraction in zip(splits, (0.7, 0.15, 0.15))
    }
    assigned_counts = {split: [0, 0] for split in splits}
    assignments = {}

    def assign(group, split):
        assignments[group] = split
        for c in range(2):
            assigned_counts[split][c] += counts[group][c]

    # Reserve class coverage before balancing. For a binary target, mixed
    # components plus pairs of pure components cover every feasible split.
    mixed = [g for g in groups if all(counts[g])]
    positive = [g for g in groups if counts[g][0] and not counts[g][1]]
    negative = [g for g in groups if counts[g][1] and not counts[g][0]]
    for split in splits:
        if mixed:
            assign(mixed.pop(0), split)
        elif positive and negative:
            assign(positive.pop(0), split)
            assign(negative.pop(0), split)

    # Greedily minimize the change in normalized squared class-count error.
    # Whole components remain intact; exact fractions may be unattainable.
    for group in groups:
        if group in assignments:
            continue

        def cost(split, group=group):
            return sum(
                (
                    (assigned_counts[split][c] + counts[group][c] - targets[split][c]) ** 2
                    - (assigned_counts[split][c] - targets[split][c]) ** 2
                ) / max(targets[split][c], 1)
                for c in range(2)
            )

        assign(group, min(splits, key=cost))
    for i, row in enumerate(rows):
        row["group"] = identities[root(i)]
        row["split"] = assignments[root(i)]
    return rows


def readiness(rows, site_id, window_count=None, reviewed_windows=None):
    classes = {
        key: {
            state: sum(r["labels"][key] == state for r in rows)
            for state in ("present", "absent", "unknown")
        }
        for key in LABELS
    }
    reasons = []
    if any(r.get("window_seconds") != 30 for r in rows):
        reasons.append("当前模型只接受明确的 30 秒整窗，不能混入历史 60 秒样本。")
    if not rows:
        reasons.append("尚无匹配当前输入的整窗人工标签；旧 ROI 标签不能用于整窗训练。")
    if len({r["grid_sha256"] for r in rows}) != 1:
        reasons.append("首版训练需要一个经验证的相同物理网格。")
    if (
        not rows
        or any(not r.get("unit") for r in rows)
        or len({r["unit"] for r in rows}) != 1
    ):
        reasons.append("首版训练需要明确且一致的波形物理单位。")
    for split in ("train", "dev", "test"):
        subset = [r for r in rows if r.get("split") == split]
        for label in LABELS:
            if not all(
                any(r["labels"][label] == state for r in subset)
                for state in ("present", "absent")
            ):
                reasons.append(f"{split} 独立分组缺少地震（含爆破）或噪声已知样本。")
    return {
        "task": TASK,
        "site_id": site_id,
        "window_count": len(rows) if window_count is None else window_count,
        "reviewed_windows": len(rows) if reviewed_windows is None else reviewed_windows,
        "eligible_windows": len(rows),
        "classes": classes,
        "ready": not reasons,
        "reasons": reasons,
    }


def export_rows(session, artifact_root, site_id, window_seconds=30, *, normalization_root=None):
    from sqlalchemy import select

    from das_sentinel.persistence.window_models import (
        EventWindow,
        WindowAnalysis,
        WindowLabelImport,
        WindowReview,
    )
    from das_sentinel.services.event_windows import WindowError, WindowService
    from das_sentinel.services.window_label_imports import inherited_review_current

    service = WindowService(session, artifact_root)
    rows, reviewed = [], 0
    excluded = []
    windows = list(
        session.scalars(
            select(EventWindow)
            .where(EventWindow.site_id == site_id)
            .where(EventWindow.window_seconds == window_seconds)
            .order_by(EventWindow.start_utc)
        )
    )
    counts = {
        key: {"present": 0, "absent": 0, "unknown": len(windows)} for key in LABELS
    }
    for index, window in enumerate(windows, 1):
        if index % 100 == 0:
            logger.info("Exporting reviewed windows %s/%s", index, len(windows))
        analysis = (
            session.get(WindowAnalysis, window.latest_analysis_id)
            if window.latest_analysis_id
            else None
        )
        review = session.scalar(
            select(WindowReview)
            .where(WindowReview.window_id == window.id)
            .order_by(WindowReview.created_at.desc(), WindowReview.id.desc())
            .limit(1)
        )
        if (
            not analysis
            or analysis.status != "succeeded"
            or not review
            or review.analysis_id != analysis.id
            or review.input_sha256 != analysis.input_sha256
            or not service._current(analysis)
            or not inherited_review_current(session, review)
        ):
            continue
        reviewed += 1
        source_labels = json.loads(review.labels_json)
        labels = binary_labels(source_labels)
        for key in LABELS:
            if labels[key] != "unknown":
                counts[key][labels[key]] += 1
                counts[key]["unknown"] -= 1
        if not any(labels.get(k) in ("present", "absent") for k in LABELS):
            continue
        result = json.loads(analysis.result_json)
        if (
            not result.get("read_start_utc")
            or not result.get("read_end_utc")
            or not result.get("read_sources")
        ):
            continue
        try:
            path, _ = service.artifact(window.id, "waveform", analysis.id)
        except WindowError as error:
            excluded.append({"window_id": window.id, "reason": str(error)})
            continue
        catalog = json.loads(window.catalog_matches_json)
        ids = sorted(
            {
                str(c.get("catalog_id") or c.get("event_id") or c.get("id"))
                for c in catalog
                if isinstance(c, dict)
                and (c.get("catalog_id") or c.get("event_id") or c.get("id"))
            }
        )
        imported = session.get(WindowLabelImport, window.id)
        rows.append(
            {
                "window_id": window.id,
                "analysis_id": analysis.id,
                "input_sha256": analysis.input_sha256,
                "window_seconds": window.window_seconds,
                "labels": labels,
                "source_labels": source_labels,
                "label_origin": "legacy_positive_import"
                if (imported and imported.applied_review_id == review.id)
                or review.request_id.removeprefix("binary-v1:").startswith("legacy-positive-v1:")
                else "whole_window_review",
                "date": window.start_utc.date().isoformat(),
                "catalog_ids": ids,
                "read_start_utc": result["read_start_utc"],
                "read_end_utc": result["read_end_utc"],
                "sources": result["read_sources"],
                "grid_sha256": result["grid_sha256"],
                "recognition_range_m": result.get("recognition_range_m", {"start": None, "stop": None}),
                "unit": result["unit"],
                "waveform": str(path),
                "waveform_sha256": next(a["sha256"] for a in result["artifacts"] if a["kind"] == "waveform"),
            }
        )
    training_grid, normalization_error = None, None
    if normalization_root is not None and rows:
        from das_sentinel.window_grid import normalize_rows

        try:
            rows, training_grid = normalize_rows(rows, normalization_root)
        except (OSError, ValueError, KeyError) as error:
            normalization_error = f"训练距离网格校准未通过：{error}"
    split_rows(rows)
    state = readiness(rows, site_id, len(windows), reviewed)
    state["classes"] = counts
    state["excluded_artifacts"] = excluded
    state["training_grid"] = training_grid
    if normalization_error:
        state["ready"] = False
        state["reasons"].append(normalization_error)
    return {"schema_version": 2, "task": TASK, "site_id": site_id, "rows": rows,
            "readiness": state, "training_grid": training_grid}


def train(manifest_path, output, epochs=10):
    import numpy as np
    import torch

    from das_sentinel.classifiers.window_model import (
        WindowEncoder,
        masked_loss,
        prepare,
        save_bundle,
        statistics,
    )

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or manifest.get("task") != TASK:
        raise ValueError("incompatible binary training manifest; export current labels again")
    rows = manifest["rows"]
    training_grid = manifest.get("training_grid")
    if any(row.get("grid_normalization") for row in rows) and not training_grid:
        raise ValueError("normalized training grid metadata missing")
    if training_grid:
        expected_distance = (training_grid["start_locus"] + np.arange(training_grid["channels"], dtype="<f8")) * training_grid["spacing_m"]
        if (training_grid["version"] != "native-channel-union-v1"
                or hashlib.sha256(expected_distance.tobytes()).hexdigest() != training_grid["grid_sha256"]
                or any(row.get("grid_normalization") != training_grid["version"]
                       or row["grid_sha256"] != training_grid["grid_sha256"] for row in rows)):
            raise ValueError("normalized training grid contract mismatch")
    expected = split_rows([dict(row) for row in rows])
    if any(row.get("split") != check["split"] for row, check in zip(rows, expected)):
        raise ValueError("manifest split violates independent grouping")
    gate = readiness(rows, manifest["site_id"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "readiness.json").write_text(
        json.dumps(gate, ensure_ascii=False), encoding="utf-8"
    )
    if not gate["ready"]:
        raise ValueError(
            "whole-window training not ready: " + "; ".join(gate["reasons"])
        )
    torch.manual_seed(42)
    torch.set_num_threads(2)
    data = []
    for row in rows:
        if row["labels"]["earthquake"] == "unknown":
            continue
        path = Path(row["waveform"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["waveform_sha256"]:
            raise ValueError("frozen waveform hash mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            if training_grid:
                channels = training_grid["channels"]
                if (not np.array_equal(arrays["distance_m"], expected_distance)
                        or arrays["waveform"].shape != (channels, 3000)
                        or any(arrays[key].shape != (channels,) for key in
                               ("valid_channel_mask", "recognition_channel_mask", "region_id"))):
                    raise ValueError("normalized waveform grid mismatch")
            planes = prepare(arrays)
            features = statistics(arrays)
        target = torch.tensor(
            [
                {"present": 1.0, "absent": 0.0, "unknown": -1.0}[row["labels"][k]]
                for k in LABELS
            ]
        )
        data.append((row, planes, features, target))
    model, baseline = WindowEncoder(), torch.nn.Linear(6, len(LABELS))
    train_data = [item for item in data if item[0]["split"] == "train"]
    train_positive = sum(int((target == 1).sum()) for _, _, _, target in train_data)
    train_negative = sum(int((target == 0).sum()) for _, _, _, target in train_data)
    pos_weight = torch.tensor([train_negative / train_positive])
    train_features = torch.stack([features for _, _, features, _ in train_data])
    mean, scale = (
        train_features.mean(0),
        train_features.std(0, unbiased=False).clamp_min(1e-6),
    )
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(baseline.parameters()), lr=0.003
    )
    shuffle = torch.Generator().manual_seed(42)
    for _ in range(epochs):
        for index in torch.randperm(len(train_data), generator=shuffle).tolist():
            row, planes, features, target = train_data[index]
            optimizer.zero_grad()
            loss = masked_loss(model(*planes), target, pos_weight=pos_weight) + masked_loss(
                baseline((features - mean) / scale), target, pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()

    def metrics(network):
        result = {}
        for split in ("dev", "test"):
            truth, probabilities, catalog_flags = [], [], []
            started = time.perf_counter()
            with torch.no_grad():
                for row, planes, features, target in data:
                    if row["split"] == split:
                        truth.append(target.numpy())
                        catalog_flags.append(bool(row["catalog_ids"]))
                        logits = (
                            model(*planes)
                            if network == "encoder"
                            else baseline((features - mean) / scale)
                        )
                        probabilities.append(logits.sigmoid().numpy())
            elapsed = time.perf_counter() - started
            y, probability = np.stack(truth), np.stack(probabilities)
            pred, known = probability >= 0.5, y != -1
            fully_reviewed = known.all(axis=1)
            reviewed_hours = float(fully_reviewed.sum()) * 30 / 3600
            tp = ((y == 1) & pred & known).sum(0)
            fp = ((y == 0) & pred & known).sum(0)
            fn = ((y == 1) & ~pred & known).sum(0)
            precision, recall = tp / np.maximum(tp + fp, 1), tp / np.maximum(tp + fn, 1)
            f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
            ap = []
            for c in range(len(LABELS)):
                valid = y[:, c] != -1
                scores = probability[valid, c]
                order = np.argsort(-scores)
                ordered = y[valid, c][order]
                # A threshold includes every equal-scoring prediction together.
                ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(order) - 1]
                true_positives = ordered.cumsum()[ends]
                ap.append(
                    float(
                        (
                            np.diff(np.r_[0, true_positives])
                            * true_positives
                            / (ends + 1)
                        ).sum()
                        / max(ordered.sum(), 1)
                    )
                )
            result[split] = {
                "inference_seconds_per_window": elapsed / len(y),
                "catalog_associated_confirmed_missed": int(
                    ((y[:, 0] == 1) & ~pred[:, 0] & np.asarray(catalog_flags)).sum()
                ),
                "f1": float(f1[0]),
                "classes": {
                    k: {
                        "precision": float(precision[i]),
                        "recall": float(recall[i]),
                        "average_precision": ap[i],
                        "known": int(known[:, i].sum()),
                    }
                    for i, k in enumerate(LABELS)
                },
                "fully_reviewed_hours": reviewed_hours,
                "false_positive_windows_per_hour": (
                    float(((pred & (y == 0)).any(1) & fully_reviewed).sum())
                    / reviewed_hours
                    if reviewed_hours
                    else None
                ),
            }
        return result

    report = {
        "task": TASK,
        "training": {
            "epochs": epochs,
            "seed": 42,
            "positive": train_positive,
            "negative": train_negative,
            "pos_weight": train_negative / train_positive,
        },
        "encoder": metrics("encoder"),
        "baseline": metrics("baseline"),
        "thresholds": [0.5],
        "status": "offline_candidate",
        "publication_ready": False,
        "isolation": "date and overlapping native context; catalog IDs where available",
        "pretrained": False,
        "training_grid": training_grid,
    }
    save_bundle(model, output, rows[0]["grid_sha256"], report, unit=rows[0]["unit"], training_grid=training_grid)
    torch.save(
        {"weights": baseline.state_dict(), "mean": mean, "scale": scale},
        output / "baseline.pt",
    )
    (output / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--database-url", required=True)
    export.add_argument("--artifact-root", required=True)
    export.add_argument("--site-id", required=True)
    export.add_argument("--output", required=True)
    training = sub.add_parser("train")
    training.add_argument("--manifest", required=True)
    training.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.manifest, args.output)
    else:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        with Session(create_engine(args.database_url)) as session:
            manifest = export_rows(session, Path(args.artifact_root), args.site_id)
        Path(args.output).write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
