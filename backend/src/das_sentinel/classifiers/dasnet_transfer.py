"""Frozen, optional DASNet-v1 features for the JJK standard waveform cohort.

This is a CPU transfer adapter, not the released marine event detector. It
retains the released three frequency views, but fixes the resized input to
6000 time pixels by 256 distance pixels and pools only the reviewed ROI.
The 100 Hz time axis is preserved so resizing cannot remove the >10 Hz view.
"""

from __future__ import annotations

import hashlib
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

CHECKPOINT_SHA256 = "f1b0e1ee00a5a4a818b4bcfb70205bcbafd0db38acfa6d04537456576ae9c590"
_SHAPE = (832, 6000)
_UNIT = "(nm/m)/s * Hz/m"


def _validate_sample(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    required = {"waveform", "distance_m", "valid_channel_mask",
                "recognition_channel_mask", "region_id"}
    if required - arrays.keys():
        raise ValueError("DASNet样本缺少必要数组")
    waveform = np.asarray(arrays["waveform"])
    if (waveform.shape != _SHAPE or waveform.dtype != np.float32
            or not np.isfinite(waveform).all()):
        raise ValueError("DASNet需要有限float32、832×6000的标准波形")
    if (metadata.get("view") != "standard" or metadata.get("sample_rate_hz") != 100
            or metadata.get("shape") != list(_SHAPE) or metadata.get("unit") != _UNIT):
        raise ValueError("DASNet需要60秒、100Hz、已核对单位的JJK应变率标准视图")
    if not np.array_equal(arrays["distance_m"], np.arange(832, dtype=np.float64) * 4):
        raise ValueError("DASNet距离网格必须为0至3324米、间距4米")
    valid = np.asarray(arrays["valid_channel_mask"])
    recognition = np.asarray(arrays["recognition_channel_mask"])
    region = np.asarray(arrays["region_id"])
    channel_indices = np.arange(832)
    expected_region = np.where(channel_indices < 75, 0,
                               np.where(channel_indices < 125, 1, 2))
    if (valid.shape != (832,) or recognition.shape != (832,) or region.shape != (832,)
            or valid.dtype != np.bool_ or recognition.dtype != np.bool_
            or not np.array_equal(region, expected_region)
            or not np.array_equal(recognition, valid & (region != 0))
            or not recognition.any()):
        raise ValueError("DASNet有效通道与非悬空识别掩码不一致")


def validate_roi(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any], event_id: str,
) -> tuple[int, int, int, int]:
    """Return one reviewed half-open (time first/stop, channel first/stop)."""
    _validate_sample(arrays, metadata)
    annotations = [annotation for annotation in metadata.get("annotations", [])
                   if annotation.get("event_id") == event_id]
    if len(annotations) != 1:
        raise ValueError("DASNet需要唯一匹配的事件ROI")
    names = ("time_start_index", "time_stop_index",
             "sample_channel_start", "sample_channel_stop")
    bounds = tuple(annotations[0].get(name) for name in names)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in bounds):
        raise ValueError("DASNet事件ROI必须使用整数索引")
    first, stop, channel_first, channel_stop = bounds
    if not (0 <= first < stop <= 6000 and 0 <= channel_first < channel_stop <= 832):
        raise ValueError("DASNet事件ROI为空或超出波形范围")
    if not arrays["recognition_channel_mask"][channel_first:channel_stop].any():
        raise ValueError("DASNet事件ROI没有有效非悬空通道")
    return bounds


def prepare_views(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any],
) -> np.ndarray:
    """Return released raw/band/high views in (3, time, channel) orientation.

    Filtering precedes resizing and uses the full 60 seconds per channel.
    Excluded channels contribute a constant zero view, regardless of signal.
    """
    _validate_sample(arrays, metadata)
    waveform = arrays["waveform"]
    recognition = arrays["recognition_channel_mask"]
    views = np.empty((3, 6000, 832), dtype=np.float32)
    filters = (None, butter(4, (2, 10), btype="bandpass", fs=100, output="sos"),
               butter(4, 10, btype="highpass", fs=100, output="sos"))
    for index, coefficients in enumerate(filters):
        view = waveform if coefficients is None else sosfiltfilt(coefficients, waveform, axis=1)
        normalized = (view - view.mean(axis=1, keepdims=True)) / (
            view.std(axis=1, keepdims=True) + 1e-6
        )
        views[index] = np.clip(normalized, -3, 3).T
    views[:, :, ~recognition] = 0
    if not np.isfinite(views).all():
        raise ValueError("DASNet预处理产生非有限值")
    return views


def pool_roi(
    features: np.ndarray,
    source_shape: tuple[int, int],
    roi: tuple[int, int, int, int],
    recognition_mask: np.ndarray,
) -> np.ndarray:
    """Pool mean/max with exact ROI-cell overlap and usable-channel weights.

    Feature maps still have the backbone's receptive-field context. A tiny
    ROI retains its intersecting cells; it is never expanded to the window.
    """
    _, height, width = features.shape
    samples, channels = source_shape
    first, stop, channel_first, channel_stop = roi
    time_edges = np.linspace(0, samples, height + 1)
    time_weights = np.maximum(0, np.minimum(time_edges[1:], stop)
                              - np.maximum(time_edges[:-1], first))
    distance_edges = np.linspace(0, channels, width + 1)
    channel_indices = np.arange(channels)
    usable = (recognition_mask & (channel_indices >= channel_first)
              & (channel_indices < channel_stop))
    overlaps = np.maximum(
        0, np.minimum(distance_edges[1:, None], channel_indices[None, :] + 1)
        - np.maximum(distance_edges[:-1, None], channel_indices[None, :]),
    )
    distance_weights = overlaps @ usable.astype(np.float64)
    weights = time_weights[:, None] * distance_weights[None, :]
    if weights.sum() <= 0:
        raise ValueError("DASNet事件ROI没有可聚合的有效特征")
    mean = (features * weights).sum(axis=(1, 2)) / weights.sum()
    maximum = features[:, weights > 0].max(axis=1)
    return np.concatenate((mean, maximum)).astype(np.float32)


class DASNetEncoder:
    """An immutable pretrained feature extractor; no local label head."""

    def __init__(self, backbone: Any, torch: Any, provenance: dict[str, Any]):
        self.backbone = backbone
        self._torch = torch
        self.provenance = provenance

    def encode_sample(
        self, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any], event_id: str,
    ) -> np.ndarray:
        roi = validate_roi(arrays, metadata, event_id)
        torch = self._torch
        views = prepare_views(arrays, metadata)
        with torch.inference_mode():
            image = torch.from_numpy(views).unsqueeze(0)
            image = torch.nn.functional.interpolate(
                image, size=(6000, 256), mode="bilinear",
                align_corners=False, antialias=True,
            )
            mean = image.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
            std = image.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
            image = (image - mean) / std
            backbone = self.backbone
            features = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(image))))
            features = backbone.layer1(features)
            parts = []
            for layer in (backbone.layer2, backbone.layer3):
                features = layer(features)
                parts.append(pool_roi(
                    features[0].numpy(), (6000, 832), roi,
                    arrays["recognition_channel_mask"],
                ))
        embedding = np.concatenate(parts).astype(np.float32)
        if embedding.shape != (3072,) or not np.isfinite(embedding).all():
            raise ValueError("DASNet编码结果不符合3072维有限向量契约")
        return embedding


def load_encoder(checkpoint: Path, *, threads: int = 4) -> DASNetEncoder:
    """Verify the official weights before loading optional CPU dependencies."""
    checkpoint = Path(checkpoint)
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError("DASNet checkpoint SHA256与官方发布不一致")
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 16:
        raise ValueError("DASNet CPU线程数必须为1至16")
    # Torch remains optional for the production API and feature-only training.
    import torch
    import torchvision
    from torchvision.models import resnet50
    from torchvision.ops.misc import FrozenBatchNorm2d

    torch.set_num_threads(threads)
    unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)
    if set(unsafe_globals) - {"argparse.Namespace"}:
        raise ValueError("DASNet checkpoint包含未允许的序列化类型")
    # The verified official file includes training arguments in a plain stdlib
    # Namespace. Keep weights_only enabled and never add arbitrary globals.
    with torch.serialization.safe_globals([Namespace]):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {name.removeprefix("backbone.body."): tensor
             for name, tensor in saved["model"].items() if name.startswith("backbone.body.")}
    backbone = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    backbone.layer1[0].conv2.stride = (2, 2)
    backbone.layer1[0].downsample[0].stride = (2, 2)
    backbone.fc = torch.nn.Identity()
    # Load layer4 too, although ROI extraction stops at layer3: every released
    # body tensor must match the architecture, without a partial-load fallback.
    backbone.load_state_dict(state, strict=True)
    backbone.requires_grad_(False)
    backbone.eval()
    provenance = {
        "adapter_version": "jjk-dasnet-roi-fulltime-v1",
        "checkpoint_url": "https://github.com/AI4EPS/models/releases/download/DASNet-v1/dasnet_v1.pth",
        "checkpoint_sha256": digest,
        "safe_load": "weights_only=True; scoped argparse.Namespace allowlist",
        "backbone": "DASNet-v1 ResNet50 FrozenBatchNorm2d; layer1 first-block stride 2",
        "loaded_backbone_tensors": len(state),
        "frequency_views_hz": ["raw", [2, 10], [10, 50]],
        "filter": "Butterworth order4, scipy sosfiltfilt, before resize",
        "normalization": "per-channel temporal mean/std + 1e-6; clip [-3,3]",
        "excluded_channels": "recognition mask false -> zero before resize",
        "input_shape_channel_time": list(_SHAPE),
        "input_unit": _UNIT,
        "input_rate_hz": 100,
        "site_regions_m": {"suspended_end": 299, "borehole_start": 499},
        "resize_shape_time_channel": [6000, 256],
        "resize": "spatial-only bilinear, align_corners=False, antialias=True; 100 Hz time axis preserved",
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "embedding": "layer2 mean/max then layer3 mean/max; 3072 float32 values",
        "roi_pool": "fractional ROI-cell overlap mean and positive-overlap max; valid non-suspended channels",
        "limitations": "ROI features retain backbone receptive-field context; resized transfer adapter is not native DASNet detector",
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cpu_threads": threads,
    }
    return DASNetEncoder(backbone, torch, provenance)
