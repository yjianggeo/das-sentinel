"""Masked self-supervised encoder for standard JJK waveform samples.

PyTorch is imported only inside the functions that need it so the regular
API environment can still import preprocessing and validation helpers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

_SHAPE = (832, 6000)
_PATCH_SHAPE = (64, 512)
_UNIT = "(nm/m)/s * Hz/m"


def _validate_standard(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any],
) -> None:
    required = {
        "waveform", "distance_m", "valid_channel_mask",
        "recognition_channel_mask",
    }
    if required - arrays.keys():
        raise ValueError("自监督样本缺少必要数组")
    waveform = np.asarray(arrays["waveform"])
    if (waveform.shape != _SHAPE or waveform.dtype != np.float32
            or not np.isfinite(waveform).all()):
        raise ValueError("自监督编码器需要有限float32、832×6000的标准波形")
    if (metadata.get("view") != "standard"
            or metadata.get("sample_rate_hz") != 100
            or metadata.get("shape") != list(_SHAPE)
            or metadata.get("unit") != _UNIT):
        raise ValueError("自监督编码器仅接受60秒、100Hz且单位已核对的标准视图")
    if not np.array_equal(
        np.asarray(arrays["distance_m"]), np.arange(832, dtype=np.float64) * 4,
    ):
        raise ValueError("自监督编码器距离网格必须为0至3324米、间距4米")
    valid = np.asarray(arrays["valid_channel_mask"])
    recognition = np.asarray(arrays["recognition_channel_mask"])
    expected = valid & (np.arange(832) >= 75) if valid.shape == (832,) else None
    if (valid.shape != (832,) or recognition.shape != (832,)
            or valid.dtype != np.bool_ or recognition.dtype != np.bool_
            or not np.array_equal(recognition, expected) or not recognition.any()):
        raise ValueError("自监督编码器有效通道与0至299米排除掩码不一致")


def prepare_standard(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any],
) -> np.ndarray:
    """Bandpass and normalize one standard waveform without temporal resizing."""
    _validate_standard(arrays, metadata)
    recognition = arrays["recognition_channel_mask"]
    output = np.zeros(_SHAPE, dtype=np.float32)
    coefficients = butter(4, (0.5, 40), btype="bandpass", fs=100, output="sos")
    filtered = sosfiltfilt(
        coefficients, arrays["waveform"][recognition], axis=1,
    )
    centered = filtered - filtered.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
    normalized = centered / (rms + 1e-6)
    output[recognition] = np.clip(normalized, -6, 6).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("自监督标准预处理产生非有限值")
    return output


def make_model(seed: int) -> Any:
    """Create the fixed compact masked autoencoder using a deterministic seed."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("模型随机种子必须为非负整数")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("自监督模型需要可选的PyTorch环境") from exc

    class MaskedAutoencoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
                torch.nn.GroupNorm(4, 16),
                torch.nn.GELU(),
                torch.nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
                torch.nn.GroupNorm(8, 32),
                torch.nn.GELU(),
                torch.nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                torch.nn.GroupNorm(8, 64),
                torch.nn.GELU(),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    64, 32, kernel_size=5, stride=2, padding=2, output_padding=1,
                ),
                torch.nn.GroupNorm(8, 32),
                torch.nn.GELU(),
                torch.nn.ConvTranspose2d(
                    32, 16, kernel_size=5, stride=2, padding=2, output_padding=1,
                ),
                torch.nn.GroupNorm(4, 16),
                torch.nn.GELU(),
                torch.nn.ConvTranspose2d(
                    16, 1, kernel_size=5, stride=2, padding=2, output_padding=1,
                ),
            )

        def encode(self, inputs: Any) -> Any:
            return self.encoder(inputs)

        def forward(self, inputs: Any) -> Any:
            return self.decoder(self.encode(inputs))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return MaskedAutoencoder()


def block_mask(valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mask exactly half the 8×64 blocks while retaining usable signal both ways."""
    valid = np.asarray(valid)
    if (valid.ndim != 4 or valid.shape[1:] != (1, *_PATCH_SHAPE)
            or valid.dtype != np.bool_):
        raise ValueError("有效掩码必须为bool[B,1,64,512]")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("block_mask需要numpy Generator")
    result = np.zeros(valid.shape, dtype=np.bool_)
    for batch_index in range(valid.shape[0]):
        effective = valid[batch_index, 0].reshape(8, 8, 8, 64).any(axis=(1, 3))
        effective_flat = np.flatnonzero(effective.ravel())
        if effective_flat.size < 2:
            raise ValueError("每个patch至少需要两个含有效像素的掩码块")
        order = rng.permutation(64)
        chosen = {int(index) for index in order[:32]}
        effective_set = {int(index) for index in effective_flat}
        hidden_effective = chosen & effective_set
        if not hidden_effective:
            remove = next(index for index in chosen if index not in effective_set)
            chosen.remove(remove)
            chosen.add(next(index for index in order[32:] if index in effective_set))
        elif hidden_effective == effective_set:
            remove = next(iter(hidden_effective))
            chosen.remove(remove)
            chosen.add(next(index for index in order[32:] if index not in effective_set))
        blocks = np.zeros((8, 8), dtype=np.bool_)
        blocks.ravel()[list(chosen)] = True
        result[batch_index, 0] = np.repeat(
            np.repeat(blocks, 8, axis=0), 64, axis=1,
        )
    return result


def masked_loss(model: Any, x: Any, valid: Any, mask: Any) -> Any:
    """Return mean reconstruction error over hidden, valid pixels only."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("自监督损失需要可选的PyTorch环境") from exc
    if (not isinstance(x, torch.Tensor) or x.shape[1:] != (1, *_PATCH_SHAPE)
            or x.dtype != torch.float32 or not torch.isfinite(x).all()):
        raise ValueError("训练信号必须为有限float32[B,1,64,512]")
    if (not isinstance(valid, torch.Tensor) or valid.shape != x.shape
            or valid.dtype != torch.bool or valid.device != x.device):
        raise ValueError("训练有效掩码必须为同设备bool且与信号同形")
    hidden = torch.as_tensor(mask, dtype=torch.bool, device=x.device)
    if hidden.shape != x.shape:
        raise ValueError("训练遮挡掩码必须与信号同形")
    score = valid & hidden
    visible = valid & ~hidden
    if not score.any() or not visible.any():
        raise ValueError("训练patch必须同时含可见和被遮挡的有效像素")
    model_input = torch.cat((torch.where(visible, x, 0), visible.to(x.dtype)), dim=1)
    reconstructed = model(model_input)
    if (reconstructed.shape != x.shape or not torch.isfinite(reconstructed).all()):
        raise ValueError("模型重建结果必须为有限float32[B,1,64,512]")
    error = reconstructed[score] - x[score]
    return torch.mean(error * error)


def _model_device(model: Any, torch: Any) -> Any:
    tensor = next(model.parameters(), None)
    if tensor is None:
        tensor = next(model.buffers(), None)
    return tensor.device if tensor is not None else torch.device("cpu")


def encode_roi(
    model: Any,
    prepared: np.ndarray,
    roi: tuple[int, int, int, int],
    recognition: np.ndarray,
    batch_size: int = 16,
) -> np.ndarray:
    """Encode all tiles intersecting a half-open ROI and return mean/max features."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("ROI编码需要可选的PyTorch环境") from exc
    prepared = np.asarray(prepared)
    recognition = np.asarray(recognition)
    if (prepared.shape != _SHAPE or prepared.dtype != np.float32
            or not np.isfinite(prepared).all()):
        raise ValueError("ROI编码需要有限float32[832,6000]标准预处理数组")
    if recognition.shape != (832,) or recognition.dtype != np.bool_:
        raise ValueError("ROI识别掩码必须为bool[832]")
    if (not isinstance(roi, tuple) or len(roi) != 4
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in roi)):
        raise ValueError("ROI必须为四个整数的半开原始坐标")
    channel_first, channel_stop, time_first, time_stop = roi
    if not (0 <= channel_first < channel_stop <= 832
            and 0 <= time_first < time_stop <= 6000):
        raise ValueError("ROI为空或超出标准波形范围")
    if not recognition[channel_first:channel_stop].any():
        raise ValueError("ROI没有有效非悬空通道")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("ROI编码batch_size必须为正整数")

    tiles: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    channel_tiles = range(channel_first // 64, (channel_stop - 1) // 64 + 1)
    time_tiles = range(time_first // 512, (time_stop - 1) // 512 + 1)
    for channel_tile in channel_tiles:
        source_channel_first = channel_tile * 64
        source_channel_stop = min(source_channel_first + 64, 832)
        channel_count = source_channel_stop - source_channel_first
        usable_channels = recognition[source_channel_first:source_channel_stop]
        for time_tile in time_tiles:
            source_time_first = time_tile * 512
            source_time_stop = min(source_time_first + 512, 6000)
            time_count = source_time_stop - source_time_first
            signal = np.zeros(_PATCH_SHAPE, dtype=np.float32)
            visible = np.zeros(_PATCH_SHAPE, dtype=np.float32)
            source = prepared[
                source_channel_first:source_channel_stop,
                source_time_first:source_time_stop,
            ]
            signal[:channel_count, :time_count] = np.where(
                usable_channels[:, None], source, np.float32(0),
            )
            visible[:channel_count, :time_count] = usable_channels[:, None]
            tiles.append(np.stack((signal, visible)))

            roi_pixels = np.zeros(_PATCH_SHAPE, dtype=np.float32)
            overlap_channel_first = max(channel_first, source_channel_first)
            overlap_channel_stop = min(channel_stop, source_channel_stop)
            overlap_time_first = max(time_first, source_time_first)
            overlap_time_stop = min(time_stop, source_time_stop)
            local_channel_first = overlap_channel_first - source_channel_first
            local_channel_stop = overlap_channel_stop - source_channel_first
            local_time_first = overlap_time_first - source_time_first
            local_time_stop = overlap_time_stop - source_time_first
            roi_usable = recognition[overlap_channel_first:overlap_channel_stop]
            roi_pixels[
                local_channel_first:local_channel_stop,
                local_time_first:local_time_stop,
            ] = roi_usable[:, None]
            weights.append(roi_pixels.reshape(8, 8, 64, 8).mean(axis=(1, 3)))

    device = _model_device(model, torch)
    feature_parts: list[np.ndarray] = []
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            for first in range(0, len(tiles), batch_size):
                inputs = torch.from_numpy(
                    np.stack(tiles[first:first + batch_size]),
                ).to(device)
                encoded = model.encode(inputs)
                if (encoded.ndim != 4 or encoded.shape[1:] != (64, 8, 64)
                        or not torch.isfinite(encoded).all()):
                    raise ValueError("编码器输出必须为有限[B,64,8,64]")
                feature_parts.append(encoded.detach().cpu().numpy())
    finally:
        if was_training:
            model.train()
    encoded_tiles = np.concatenate(feature_parts, axis=0)
    total_weight = 0.0
    weighted_sum = np.zeros(64, dtype=np.float64)
    maximum = np.full(64, -np.inf, dtype=np.float32)
    for encoded, tile_weights in zip(encoded_tiles, weights):
        weight_sum = float(tile_weights.sum())
        if weight_sum == 0:
            continue
        weighted_sum += (encoded * tile_weights[None]).sum(axis=(1, 2), dtype=np.float64)
        maximum = np.maximum(maximum, encoded[:, tile_weights > 0].max(axis=1))
        total_weight += weight_sum
    if total_weight <= 0:
        raise ValueError("ROI没有可聚合的有效特征")
    result = np.concatenate((weighted_sum / total_weight, maximum)).astype(np.float32)
    if result.shape != (128,) or not np.isfinite(result).all():
        raise ValueError("ROI编码结果必须为有限float32[128]")
    return result
