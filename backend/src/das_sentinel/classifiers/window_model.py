"""Whole-window earthquake (including blast) versus noise encoder."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

LABELS = ("earthquake",)
TASK = "earthquake_including_blast_vs_noise"
PREPROCESS = "whole-window-log-pool-v2-30s"


def prepare(arrays):
    wave = np.asarray(arrays["waveform"], dtype=np.float32)
    mask = np.asarray(arrays["valid_channel_mask"], bool) & np.asarray(
        arrays["recognition_channel_mask"], bool
    )
    if (
        wave.ndim != 2
        or wave.shape[1] != 3000
        or not mask.any()
        or not np.isfinite(wave).all()
    ):
        raise ValueError("invalid whole-window waveform")
    # Keep every channel and pool all 30 seconds in 0.2-second blocks.
    safe = np.where(mask[:, None], wave, 0).reshape(wave.shape[0], 150, 20)
    signal = np.log1p(np.sqrt(np.mean(safe**2, axis=2)))
    peak = np.log1p(np.max(np.abs(safe), axis=2))
    planes = np.stack([signal, peak, np.broadcast_to(mask[:, None], signal.shape)])
    return (
        torch.from_numpy(planes.astype(np.float32))[None],
        torch.as_tensor(mask),
        torch.as_tensor(np.asarray(arrays["region_id"], dtype=np.int64)),
    )


def statistics(arrays):
    planes, mask, regions = prepare(arrays)
    values = []
    for region in range(3):
        selected = planes[0, :2, mask & (regions == region)]
        values.extend(
            [selected.mean().item(), selected.max().item()]
            if selected.numel()
            else [0.0, 0.0]
        )
    return torch.tensor(values)


class WindowEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Linear(48, len(LABELS))

    def forward(self, planes, mask, regions):
        encoded = self.encoder(planes)[0]
        pooled = []
        for region in range(3):
            selected = encoded[:, mask & (regions == region), :]
            pooled.extend(
                [selected.mean((1, 2)), selected.amax((1, 2))]
                if selected.numel()
                else [encoded.new_zeros(8), encoded.new_zeros(8)]
            )
        return self.head(torch.cat(pooled))


def masked_loss(logits, targets, *, pos_weight=None):
    known = targets != -1
    return (
        nn.functional.binary_cross_entropy_with_logits(
            logits, targets.clamp_min(0), reduction="none", pos_weight=pos_weight
        )
        * known
    ).sum() / known.sum().clamp_min(1)


def save_bundle(model, folder, grid_sha256, metrics, *, unit, training_grid=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "weights.pt")
    metadata = {
        "schema_version": 2,
        "task": TASK,
        "labels": LABELS,
        "preprocess": PREPROCESS,
        "grid_sha256": grid_sha256,
        "unit": unit,
        "window_seconds": 30,
        "weights_sha256": hashlib.sha256(
            (folder / "weights.pt").read_bytes()
        ).hexdigest(),
        "metrics": metrics,
        "status": "offline_candidate",
        "training_grid": training_grid,
    }
    (folder / "model.json").write_text(json.dumps(metadata), encoding="utf-8")


def load_bundle(folder, grid_sha256, *, unit):
    folder = Path(folder)
    meta = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    grid = meta.get("training_grid")
    if grid:
        distance = (grid["start_locus"] + np.arange(grid["channels"], dtype="<f8")) * grid["spacing_m"]
        if (grid["version"] != "native-channel-union-v1" or grid["grid_sha256"] != grid_sha256
                or hashlib.sha256(distance.tobytes()).hexdigest() != grid_sha256):
            raise ValueError("incompatible normalized model grid")
    if (
        meta["schema_version"] != 2
        or meta.get("task") != TASK
        or meta["labels"] != list(LABELS)
        or meta["preprocess"] != PREPROCESS
        or meta["grid_sha256"] != grid_sha256
        or not unit
        or meta.get("unit") != unit
        or meta.get("window_seconds") != 30
    ):
        raise ValueError("incompatible model schema/grid/unit")
    if (
        hashlib.sha256((folder / "weights.pt").read_bytes()).hexdigest()
        != meta["weights_sha256"]
    ):
        raise ValueError("model hash mismatch")
    model = WindowEncoder()
    model.load_state_dict(
        torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
    )
    return model.eval()
