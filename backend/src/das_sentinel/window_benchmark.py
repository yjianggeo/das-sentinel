"""Offline, frozen-cohort comparison of 30-second whole-window classifiers."""

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from das_sentinel.classifiers.window_model import (
    TASK,
    WindowEncoder,
    masked_loss,
    prepare,
)
from das_sentinel.window_training import readiness, split_rows

CANDIDATES = ("tiny_pooled", "residual_pooled", "residual_waveform")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def classification_metrics(truth, probability, threshold):
    y, scores = np.asarray(truth, int), np.asarray(probability, float)
    if y.shape != scores.shape or not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("invalid binary predictions")
    pred = scores >= threshold
    tp, fp = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum())
    fn, tn = int(((y == 1) & ~pred).sum()), int(((y == 0) & ~pred).sum())
    ap = None
    if len(y) and y.sum():
        order = np.argsort(-scores)
        ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(order) - 1]
        cumulative = y[order].cumsum()[ends]
        ap = float((np.diff(np.r_[0, cumulative]) * cumulative / (ends + 1)).sum() / y.sum())
    return {"known": len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "average_precision": ap,
            "false_positive_fraction": fp / max(fp + tn, 1)}


def recall_threshold(truth, probability, target=.9):
    y, scores = np.asarray(truth), np.asarray(probability, float)
    if y.shape != scores.shape or not np.isfinite(scores).all() or not 0 < target <= 1:
        raise ValueError("invalid development threshold inputs")
    positive = np.sort(scores[y == 1])[::-1]
    if not len(positive):
        raise ValueError("development positive labels required")
    # Highest threshold meeting the requested development recall; ties stay together.
    return float(positive[math.ceil(target * len(positive)) - 1])


def validate_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or manifest.get("task") != TASK:
        raise ValueError("incompatible binary manifest")
    rows = manifest["rows"]
    if len({r["window_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate training window")
    reference = manifest.get("split_reference")
    if reference:
        reference_bytes = Path(reference["path"]).read_bytes()
        if (hashlib.sha256(reference_bytes).hexdigest() != reference["sha256"]
                or json.loads(reference_bytes).get("split_reference")):
            raise ValueError("invalid frozen split reference")
        original = validate_manifest(reference["path"])
        review_keys = {"labels", "source_labels", "label_origin", "review_id", "request_id"}
        if (original["site_id"] != manifest["site_id"]
                or original["training_grid"] != manifest["training_grid"]
                or len(rows) != len(original["rows"])
                or any({k: v for k, v in row.items() if k not in review_keys}
                       != {k: v for k, v in old.items() if k not in review_keys}
                       for row, old in zip(rows, original["rows"]))):
            raise ValueError("review update differs from frozen split reference")
    else:
        expected = split_rows([dict(row) for row in rows])
        if any((row.get("split"), row.get("group")) != (check["split"], check["group"])
               for row, check in zip(rows, expected)):
            raise ValueError("manifest split violates independent grouping")
    gate = readiness(rows, manifest["site_id"])
    if not gate["ready"]:
        raise ValueError("training not ready: " + "; ".join(gate["reasons"]))
    grid = manifest.get("training_grid")
    if not grid or grid["version"] != "native-channel-union-v1":
        raise ValueError("normalized training grid required")
    distance = (grid["start_locus"] + np.arange(grid["channels"], dtype="<f8")) * grid["spacing_m"]
    if (hashlib.sha256(distance.tobytes()).hexdigest() != grid["grid_sha256"]
            or any(r["grid_sha256"] != grid["grid_sha256"]
                   or r.get("grid_normalization") != grid["version"] for r in rows)):
        raise ValueError("normalized training grid mismatch")
    return manifest


class CachedWindows:
    """Per-window files bound resident memory without persistent mapped views."""

    def __init__(self, manifest, folder, *, waveform, pooled=True, distance_start_m=None):
        folder.mkdir()
        self.folder = folder
        self.rows = [r for r in manifest["rows"] if r["labels"]["earthquake"] != "unknown"]
        grid = manifest["training_grid"]
        count, channels = len(self.rows), grid["channels"]
        distance = (grid["start_locus"] + np.arange(channels, dtype="<f8")) * grid["spacing_m"]
        selected = distance if distance_start_m is None else distance[distance >= distance_start_m]
        self.masks = np.zeros((count, len(selected)), bool)
        self.regions = np.zeros((count, len(selected)), np.int64)
        started = time.perf_counter()
        for index, row in enumerate(self.rows):
            path = Path(row["waveform"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["waveform_sha256"]:
                raise ValueError("frozen waveform hash mismatch")
            with np.load(path, allow_pickle=False) as arrays:
                if (arrays["waveform"].shape != (channels, 3000)
                        or not np.array_equal(arrays["distance_m"], distance)
                        or not np.allclose(arrays["relative_seconds"], np.arange(3000) / 100,
                                           rtol=0, atol=1e-8)
                        or any(arrays[key].shape != (channels,) for key in
                               ("valid_channel_mask", "recognition_channel_mask", "region_id"))):
                    raise ValueError("normalized waveform grid mismatch")
                if distance_start_m is not None:
                    from das_sentinel.classifiers.window_distance import (
                        crop_distance_arrays,
                    )

                    arrays = crop_distance_arrays(arrays, distance_start_m)
                if pooled or not waveform:
                    planes, mask, regions = prepare(arrays)
                    if pooled:
                        self._save(index, "pooled", planes[0].numpy())
                if waveform:
                    from das_sentinel.classifiers.window_resnet import prepare_waveform

                    # The third plane is exactly the separately stored mask broadcast.
                    planes, mask, regions = prepare_waveform(arrays)
                    self._save(index, "waveform", planes[0, :2].numpy())
                self.masks[index], self.regions[index] = mask.numpy(), regions.numpy()
            if (index + 1) % 100 == 0 or index + 1 == count:
                print(json.dumps({"stage": "cache", "done": index + 1, "total": count,
                                  "seconds": round(time.perf_counter() - started, 1)}), flush=True)

    def _save(self, index, kind, array):
        # Closing each file releases its buffers; fsync also bounds pending writes
        # on Windows. No full-cohort view remains mapped in the training process.
        with (self.folder / f"{index}-{kind}.npy").open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())

    def inputs(self, index, candidate):
        kind = "waveform" if candidate == "residual_waveform" else "pooled"
        planes = np.load(self.folder / f"{index}-{kind}.npy", allow_pickle=False)
        if kind == "waveform":
            mask_plane = np.broadcast_to(self.masks[index, :, None], planes.shape[1:]).astype(np.float32)
            planes = np.concatenate((planes, mask_plane[None]), axis=0)
        return (torch.from_numpy(planes)[None],
                torch.from_numpy(self.masks[index]), torch.from_numpy(self.regions[index]))


def make_model(candidate):
    if candidate == "tiny_pooled":
        return WindowEncoder()
    if candidate in CANDIDATES:
        from das_sentinel.classifiers.window_resnet import ResidualWindowEncoder

        return ResidualWindowEncoder(input_kind="waveform" if candidate == "residual_waveform" else "pooled")
    raise ValueError("unknown candidate")


def load_candidate(folder):
    folder = Path(folder)
    meta = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != 1 or meta.get("task") != TASK or meta.get("window_seconds") != 30:
        raise ValueError("incompatible benchmark model")
    if hashlib.sha256((folder / "weights.pt").read_bytes()).hexdigest() != meta["weights_sha256"]:
        raise ValueError("model hash mismatch")
    model = make_model(meta["candidate"])
    model.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu", weights_only=True))
    return model.eval()


def candidate_inputs(arrays, metadata):
    """Replay an offline model with its saved distance/preprocessing contract."""
    seconds = np.asarray(arrays["relative_seconds"])
    if (seconds.shape != (3000,)
            or not np.allclose(seconds, np.arange(3000) / 100, rtol=0, atol=1e-8)):
        raise ValueError("model time grid must be 30 seconds at 100 Hz")
    grid = metadata["training_grid"]
    distance = (grid["start_locus"] + np.arange(grid["channels"], dtype="<f8")) * grid["spacing_m"]
    if not np.array_equal(arrays["distance_m"], distance):
        raise ValueError("model source distance grid mismatch")
    start = metadata.get("distance_start_m")
    if start is not None:
        from das_sentinel.classifiers.window_distance import crop_distance_arrays

        arrays = crop_distance_arrays(arrays, start)
    if metadata["candidate"] == "residual_waveform":
        from das_sentinel.classifiers.window_resnet import prepare_waveform

        return prepare_waveform(arrays)
    return prepare(arrays)


def evaluate(model, data, indices, candidate):
    model.eval()
    predictions = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index in indices:
            row = data.rows[index]
            score = model(*data.inputs(index, candidate)).sigmoid().item()
            if not math.isfinite(score):
                raise ValueError("nonfinite prediction")
            predictions.append({"window_id": row["window_id"], "split": row["split"], "group": row["group"],
                                "date": row["date"], "label_origin": row.get("label_origin", "unknown"),
                                "catalog_ids": row["catalog_ids"],
                                "truth": int(row["labels"]["earthquake"] == "present"), "probability": score})
    return predictions, time.perf_counter() - started


def summarize(predictions, threshold):
    return classification_metrics([p["truth"] for p in predictions], [p["probability"] for p in predictions], threshold)


def training_sampling_weights(rows, indices):
    """Balance classes, then independent groups, then windows within a group."""
    selected = [rows[index] for index in indices]
    if any(row.get("split") != "train" or not row.get("group")
           or row["labels"]["earthquake"] not in ("present", "absent") for row in selected):
        raise ValueError("sampling requires known train labels and independent groups")
    counts = Counter((row["labels"]["earthquake"], row["group"]) for row in selected)
    groups = Counter(label for label, _ in counts)
    if set(groups) != {"present", "absent"}:
        raise ValueError("sampling requires both train classes")
    return torch.tensor([
        .5 / (groups[row["labels"]["earthquake"]]
              * counts[(row["labels"]["earthquake"], row["group"])])
        for row in selected
    ], dtype=torch.float64)


def run_benchmark(manifest_path, output, *, epochs=10, candidates=CANDIDATES, seed=42,
                  distance_start_m=None, cache_directory=None, sampling="uniform"):
    if epochs < 1 or not candidates or len(set(candidates)) != len(candidates) or any(c not in CANDIDATES for c in candidates):
        raise ValueError("invalid benchmark configuration")
    if distance_start_m is not None and (not math.isfinite(distance_start_m) or distance_start_m < 0):
        raise ValueError("invalid distance start")
    if sampling not in ("uniform", "class_group_balanced"):
        raise ValueError("invalid training sampling")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frozen = Path(manifest_path).read_bytes()
    (output / "manifest.json").write_bytes(frozen)
    manifest = validate_manifest(output / "manifest.json")
    manifest_hash = hashlib.sha256(frozen).hexdigest()
    settings = {"epochs": epochs, "seed": seed, "optimizer": "Adam", "learning_rate": .001,
                "target_development_recall": .9, "checkpoint_selection": "development_average_precision",
                "threads": 2, "candidates": list(candidates), "manifest_sha256": manifest_hash,
                "sampling": sampling, "sampling_replacement": sampling == "class_group_balanced",
                "distance_start_m": distance_start_m,
                "distance_selection": "physical_crop_before_normalization" if distance_start_m is not None else "source_mask"}
    write_json(output / "protocol.json", settings)
    torch.set_num_threads(2)
    cache = Path(cache_directory) if cache_directory is not None else output / "cache"
    cache.parent.mkdir(parents=True, exist_ok=True)
    write_json(output / "storage.json", {"cache_directory": str(cache.resolve())})
    data = CachedWindows(manifest, cache, waveform="residual_waveform" in candidates,
                         pooled=any(candidate != "residual_waveform" for candidate in candidates),
                         distance_start_m=distance_start_m)
    indices = {split: [i for i, r in enumerate(data.rows) if r["split"] == split] for split in ("train", "dev", "test")}
    positive = sum(data.rows[i]["labels"]["earthquake"] == "present" for i in indices["train"])
    negative = len(indices["train"]) - positive
    weight = torch.tensor([1. if sampling == "class_group_balanced" else negative / positive])
    sampling_weights = (training_sampling_weights(data.rows, indices["train"])
                        if sampling == "class_group_balanced" else None)
    settings.update(steps_per_epoch=len(indices["train"]), positive_loss_weight=weight.item())
    write_json(output / "protocol.json", settings)
    report = {"task": TASK, "status": "offline_candidate", "publication_ready": False,
              "manifest_sha256": manifest_hash, "training_grid": manifest["training_grid"],
              "protocol": settings, "train_positive": positive, "train_negative": negative,
              "splits": {s: {"windows": len(ids), "positive": sum(data.rows[i]["labels"]["earthquake"] == "present" for i in ids),
                             "groups": len({data.rows[i]["group"] for i in ids}),
                             "positive_groups": len({data.rows[i]["group"] for i in ids if data.rows[i]["labels"]["earthquake"] == "present"}),
                             "label_origins": dict(Counter(data.rows[i].get("label_origin", "unknown") for i in ids))}
                         for s, ids in indices.items()},
              "individual_review_definition": "Explicit per-window reviews only; requires provenance-enriched manifest. Bulk rules and imported local labels are excluded.",
              "evaluation_scope": "Frozen historical candidate windows, not prospective continuous recordings. No operational false alarms/hour estimate.",
              "review_updated_frozen_split": bool(manifest.get("split_reference")),
              "candidates": {}}
    for candidate in candidates:
        folder = output / candidate
        folder.mkdir()
        torch.manual_seed(seed)
        model = make_model(candidate)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
        shuffle = torch.Generator().manual_seed(seed)
        best_ap, selected_epoch, history = -1., None, []
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            model.train()
            loss_sum = 0.
            selected = (torch.randperm(len(indices["train"]), generator=shuffle)
                        if sampling_weights is None else torch.multinomial(
                            sampling_weights, len(indices["train"]), replacement=True, generator=shuffle))
            sampled_classes, sampled_groups = Counter(), Counter()
            for ordinal, item in enumerate(selected.tolist(), 1):
                index = indices["train"][item]
                row = data.rows[index]
                label = row["labels"]["earthquake"]
                sampled_classes[label] += 1
                sampled_groups[label + "|" + row["group"]] += 1
                target = torch.tensor([float(data.rows[index]["labels"]["earthquake"] == "present")])
                optimizer.zero_grad(set_to_none=True)
                loss = masked_loss(model(*data.inputs(index, candidate)), target, pos_weight=weight)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                loss_sum += loss.item()
                if ordinal % 250 == 0:
                    progress = {"candidate": candidate, "epoch": epoch, "done": ordinal, "total": len(indices["train"])}
                    write_json(output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            development, _ = evaluate(model, data, indices["dev"], candidate)
            dev = summarize(development, .5)
            entry = {"epoch": epoch, "training_loss": loss_sum / len(indices["train"]),
                     "sampled_classes": dict(sampled_classes), "sampled_groups": dict(sampled_groups),
                     "sampled_unique_windows": len(set(selected.tolist())),
                     "development": dev, "seconds": time.perf_counter() - started}
            history.append(entry)
            if dev["average_precision"] > best_ap:
                best_ap, selected_epoch = dev["average_precision"], epoch
                torch.save(model.state_dict(), folder / "weights.pt")
            write_json(folder / "history.json", history)
            print(json.dumps({"candidate": candidate, **entry}), flush=True)
        model.load_state_dict(torch.load(folder / "weights.pt", weights_only=True, map_location="cpu"))
        development, _ = evaluate(model, data, indices["dev"], candidate)
        threshold = recall_threshold([p["truth"] for p in development], [p["probability"] for p in development],
                                     target=settings["target_development_recall"])
        # Test inputs are evaluated only after checkpoint and threshold are fixed.
        testing, elapsed = evaluate(model, data, indices["test"], candidate)
        result = {"selected_epoch": selected_epoch, "threshold": threshold,
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "dev": summarize(development, threshold), "test": summarize(testing, threshold),
                  "test_at_0_5": summarize(testing, .5),
                  "test_individual_reviews": summarize([p for p in testing if p["label_origin"] == "individual_review"], threshold),
                  "test_by_date": {date: summarize([p for p in testing if p["date"] == date], threshold)
                                   for date in sorted({p["date"] for p in testing})},
                  "test_seconds_per_window_including_cache_read": elapsed / len(testing)}
        metadata = {"schema_version": 1, "task": TASK, "candidate": candidate, "window_seconds": 30,
                    "sample_rate_hz": 100, "unit": manifest["rows"][0]["unit"],
                    "training_grid": manifest["training_grid"], "manifest_sha256": manifest_hash,
                    "distance_start_m": distance_start_m, "input_channels": data.masks.shape[1],
                    "weights_sha256": hashlib.sha256((folder / "weights.pt").read_bytes()).hexdigest(),
                    "status": "offline_candidate", "pretrained": False, "metrics": result,
                    "preprocess": "whole-window-log-pool-v2-30s" if candidate != "residual_waveform" else "whole-window-signed-waveform-v1-30s"}
        write_json(folder / "model.json", metadata)
        write_json(folder / "predictions.json", development + testing)
        write_json(folder / "metrics.json", result)
        report["candidates"][candidate] = result
        write_json(output / "report.json", report)
        print(json.dumps({"candidate_complete": candidate, "metrics": result}), flush=True)
    write_json(output / "progress.json", {"status": "succeeded", "candidates": list(candidates)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--distance-start-m", type=float)
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--sampling", choices=("uniform", "class_group_balanced"), default="uniform")
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=CANDIDATES)
    args = parser.parse_args()
    run_benchmark(args.manifest, args.output, epochs=args.epochs,
                  candidates=tuple(args.candidates), distance_start_m=args.distance_start_m,
                  cache_directory=args.cache_directory, sampling=args.sampling)


if __name__ == "__main__":
    main()
