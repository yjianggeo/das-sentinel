"""Small residual encoders for 30-second DAS windows."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def prepare_waveform(arrays):
    """Prepare signed, robustly scaled 100 Hz waveforms without dropping channels."""
    try:
        wave = np.asarray(arrays["waveform"], dtype=np.float32)
        valid = np.asarray(arrays["valid_channel_mask"], dtype=bool)
        recognized = np.asarray(arrays["recognition_channel_mask"], dtype=bool)
        regions = np.asarray(arrays["region_id"], dtype=np.int64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid whole-window waveform") from exc

    if (
        wave.ndim != 2
        or wave.shape[1] != 3000
        or valid.shape != (wave.shape[0],)
        or recognized.shape != (wave.shape[0],)
        or regions.shape != (wave.shape[0],)
        or not np.isfinite(wave).all()
    ):
        raise ValueError("invalid whole-window waveform")
    mask = valid & recognized
    if not mask.any():
        raise ValueError("invalid whole-window waveform")

    # Compute both centering and scale from channels admitted by the two masks.
    # float64 intermediates avoid overflow for otherwise finite float32 inputs.
    admitted = wave[mask].astype(np.float64)
    center = np.median(admitted)
    scale = np.median(np.abs(admitted - center))
    scale = max(float(scale), float(np.finfo(np.float32).eps))
    normalized = np.where(mask[:, None], wave.astype(np.float64) - center, 0.0)
    normalized /= scale

    signed = np.arcsinh(normalized)
    amplitude = np.log1p(np.abs(normalized))
    mask_plane = np.broadcast_to(mask[:, None], wave.shape)
    planes = np.stack([signed, amplitude, mask_plane]).astype(np.float32)
    if not np.isfinite(planes).all():
        raise ValueError("invalid whole-window waveform")
    return (
        torch.from_numpy(planes)[None],
        torch.as_tensor(mask),
        torch.as_tensor(regions),
    )


class _DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        padding = (dilation, dilation)
        dilations = (dilation, dilation)
        self.norm1 = nn.GroupNorm(2, channels)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=padding,
            dilation=dilations,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(2, channels)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=padding,
            dilation=dilations,
            bias=False,
        )

    def forward(self, inputs):
        residual = F.silu(self.norm1(inputs))
        residual = self.conv1(residual)
        residual = self.conv2(F.silu(self.norm2(residual)))
        return inputs + residual


class ResidualWindowEncoder(nn.Module):
    """Encode pooled features or signed waveforms into one binary logit.

    The three residual blocks cover about 117 native channels locally. Their
    local time context is 5.81 seconds for waveforms and 6.6 seconds for pooled
    input; region-wise pooling then summarizes the full 30-second window.
    """

    _CHANNELS = 8
    _DISTANCE_KERNEL = 5
    _DISTANCE_STRIDE = 4

    def __init__(self, input_kind="pooled"):
        super().__init__()
        if input_kind not in {"pooled", "waveform"}:
            raise ValueError("input_kind must be 'pooled' or 'waveform'")
        self.input_kind = input_kind
        time_kernel = 5 if input_kind == "pooled" else 21
        time_stride = 1 if input_kind == "pooled" else 20
        self.expected_time = 150 if input_kind == "pooled" else 3000
        self.stem = nn.Conv2d(
            3,
            self._CHANNELS,
            (self._DISTANCE_KERNEL, time_kernel),
            stride=(self._DISTANCE_STRIDE, time_stride),
            padding=(self._DISTANCE_KERNEL // 2, time_kernel // 2),
            bias=False,
        )
        self.stem_norm = nn.GroupNorm(2, self._CHANNELS)
        self.blocks = nn.ModuleList(
            _DilatedResidualBlock(self._CHANNELS, dilation) for dilation in (1, 2, 4)
        )
        self.head = nn.Linear(3 * 2 * self._CHANNELS, 1)

    def forward(self, planes, mask, regions):
        self._validate_inputs(planes, mask, regions)
        mask = mask.to(device=planes.device, dtype=torch.bool)
        regions = regions.to(device=planes.device)
        active = mask & (regions >= 0) & (regions < 3)
        if not active.any():
            raise ValueError("at least one valid channel is required")

        inputs = planes * active[None, None, :, None].to(dtype=planes.dtype)
        encoded = F.silu(self.stem_norm(self.stem(inputs)))
        coverage = self._region_coverage(active, regions, dtype=encoded.dtype)
        encoded_mask = coverage.sum(0).gt(0)[None, None, :, None]
        encoded = encoded * encoded_mask
        for block in self.blocks:
            encoded = block(encoded) * encoded_mask

        pooled = []
        for region in range(3):
            weights = coverage[region]
            if weights.gt(0).any():
                mean = (encoded * weights[None, None, :, None]).sum((2, 3))
                mean = mean / (weights.sum() * encoded.shape[-1])
                selected = encoded.masked_fill(
                    ~weights.gt(0)[None, None, :, None], -torch.inf
                )
                maximum = selected.amax((2, 3))
                pooled.extend([mean[0], maximum[0]])
            else:
                zeros = encoded.new_zeros(self._CHANNELS)
                pooled.extend([zeros, zeros])
        return self.head(torch.cat(pooled))

    def _region_coverage(self, active, regions, *, dtype):
        memberships = torch.stack(
            [active & regions.eq(region) for region in range(3)]
        ).to(dtype=dtype)
        return F.avg_pool1d(
            memberships[:, None],
            kernel_size=self._DISTANCE_KERNEL,
            stride=self._DISTANCE_STRIDE,
            padding=self._DISTANCE_KERNEL // 2,
            count_include_pad=False,
        )[:, 0]

    def _validate_inputs(self, planes, mask, regions):
        if (
            not isinstance(planes, torch.Tensor)
            or planes.ndim != 4
            or planes.shape[0] != 1
            or planes.shape[1] != 3
        ):
            raise ValueError("planes must have shape (1, 3, channels, time)")
        if planes.shape[-1] != self.expected_time:
            raise ValueError(f"expected time dimension {self.expected_time}")
        if (
            not isinstance(mask, torch.Tensor)
            or not isinstance(regions, torch.Tensor)
            or mask.ndim != 1
            or regions.ndim != 1
            or mask.shape[0] != planes.shape[2]
            or regions.shape[0] != planes.shape[2]
        ):
            raise ValueError("mask and regions must match the channel dimension")
        if not torch.is_floating_point(planes) or not torch.isfinite(planes).all():
            raise ValueError("planes must contain finite floating-point values")
