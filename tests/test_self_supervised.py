from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from das_sentinel.classifiers.self_supervised import (
    block_mask,
    encode_roi,
    make_model,
    masked_loss,
    prepare_standard,
)

SHAPE = (832, 6000)
UNIT = "(nm/m)/s * Hz/m"


def _standard_sample() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    valid = np.ones(832, dtype=np.bool_)
    recognition = valid.copy()
    recognition[:75] = False
    arrays = {
        "waveform": np.zeros(SHAPE, dtype=np.float32),
        "distance_m": np.arange(832, dtype=np.float64) * 4,
        "valid_channel_mask": valid,
        "recognition_channel_mask": recognition,
    }
    metadata: dict[str, object] = {
        "view": "standard",
        "sample_rate_hz": 100,
        "shape": [832, 6000],
        "unit": UNIT,
    }
    return arrays, metadata


@pytest.mark.parametrize(
    "mutation",
    ("low_frequency", "shape", "grid", "unit", "nonfinite"),
)
def test_prepare_standard_rejects_nonstandard_inputs(mutation: str) -> None:
    arrays, metadata = _standard_sample()
    if mutation == "low_frequency":
        metadata["view"] = "low_frequency"
    elif mutation == "shape":
        arrays["waveform"] = arrays["waveform"][:, :-1]
    elif mutation == "grid":
        arrays["distance_m"][100] += 0.5
    elif mutation == "unit":
        metadata["unit"] = "strain"
    else:
        arrays["waveform"][100, 100] = np.nan

    with pytest.raises(ValueError):
        prepare_standard(arrays, metadata)


def test_prepare_standard_preserves_20hz_and_zeros_excluded_channels() -> None:
    arrays, metadata = _standard_sample()
    time = np.arange(6000, dtype=np.float32) / np.float32(100)
    signal = np.sin(2 * np.pi * np.float32(20) * time).astype(np.float32)
    arrays["waveform"][100] = signal
    arrays["waveform"][:75] = np.float32(1e20)
    poisoned = prepare_standard(arrays, metadata)

    clean_arrays = deepcopy(arrays)
    clean_arrays["waveform"][:75] = 0
    clean = prepare_standard(clean_arrays, metadata)

    assert poisoned.shape == SHAPE
    assert poisoned.dtype == np.float32
    assert np.isfinite(poisoned).all()
    np.testing.assert_array_equal(poisoned, clean)
    np.testing.assert_array_equal(poisoned[:75], 0)
    assert np.corrcoef(signal[100:-100], poisoned[100, 100:-100])[0, 1] > 0.999
    assert np.sqrt(np.mean(poisoned[100] ** 2)) == pytest.approx(1, abs=0.02)


def test_block_mask_is_seeded_exact_half_blocks_and_keeps_both_regions() -> None:
    valid = np.ones((2, 1, 64, 512), dtype=np.bool_)
    first = block_mask(valid, np.random.default_rng(20260914))
    second = block_mask(valid, np.random.default_rng(20260914))

    assert first.shape == valid.shape
    assert first.dtype == np.bool_
    np.testing.assert_array_equal(first, second)
    assert np.all(first.sum(axis=(1, 2, 3)) == 32 * 8 * 64)
    assert np.all((first & valid).any(axis=(1, 2, 3)))
    assert np.all((~first & valid).any(axis=(1, 2, 3)))
    for channel_start in range(0, 64, 8):
        for time_start in range(0, 512, 64):
            block = first[:, :, channel_start:channel_start + 8,
                          time_start:time_start + 64]
            assert np.all(block == block[:, :, :1, :1])


def test_block_mask_rejects_shape_or_one_effective_block() -> None:
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        block_mask(np.ones((1, 64, 512), dtype=np.bool_), rng)
    one_block = np.zeros((1, 1, 64, 512), dtype=np.bool_)
    one_block[:, :, :8, :64] = True
    with pytest.raises(ValueError):
        block_mask(one_block, rng)


def test_model_has_exact_shapes_and_seeded_checkpoint_roundtrip(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    first = make_model(20260914)
    second = make_model(20260914)
    inputs = torch.randn(2, 2, 64, 512)

    assert first.encode(inputs).shape == (2, 64, 8, 64)
    assert first(inputs).shape == (2, 1, 64, 512)
    conv = [module for module in first.modules()
            if isinstance(module, torch.nn.Conv2d)]
    transpose = [module for module in first.modules()
                 if isinstance(module, torch.nn.ConvTranspose2d)]
    assert [(layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride)
            for layer in conv] == [
                (2, 16, (5, 5), (2, 2)),
                (16, 32, (5, 5), (2, 2)),
                (32, 64, (5, 5), (2, 2)),
            ]
    assert len(transpose) == 3
    assert all(layer.kernel_size == (5, 5) and layer.stride == (2, 2)
               for layer in transpose)
    for left, right in zip(first.state_dict().values(), second.state_dict().values()):
        torch.testing.assert_close(left, right)

    checkpoint = tmp_path / "encoder.pt"
    torch.save(first.state_dict(), checkpoint)
    reloaded = make_model(7)
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True,
    )
    with torch.inference_mode():
        torch.testing.assert_close(first(inputs), reloaded(inputs))


def test_masked_loss_hides_input_and_scores_only_hidden_valid_pixels() -> None:
    torch = pytest.importorskip("torch")

    class Capture(torch.nn.Module):
        def forward(self, inputs):
            self.inputs = inputs.detach().clone()
            return inputs[:, :1] * 0

    x = torch.zeros((1, 1, 64, 512), dtype=torch.float32)
    valid = torch.zeros_like(x, dtype=torch.bool)
    mask = np.zeros(x.shape, dtype=np.bool_)
    x[0, 0, 0, 0] = 2
    x[0, 0, 0, 1] = 4
    x[0, 0, 0, 2] = 1000  # visible and valid: excluded from the loss
    x[0, 0, 0, 3] = 1000  # hidden but invalid: excluded from the loss
    valid[0, 0, 0, :3] = True
    mask[0, 0, 0, 0:2] = True
    mask[0, 0, 0, 3] = True
    model = Capture()

    loss = masked_loss(model, x, valid, mask)

    assert loss.ndim == 0
    assert loss.item() == pytest.approx(10)
    assert model.inputs.shape == (1, 2, 64, 512)
    assert model.inputs[0, 0, 0, 0].item() == 0
    assert model.inputs[0, 0, 0, 1].item() == 0
    assert model.inputs[0, 0, 0, 2].item() == 1000
    assert model.inputs[0, 0, 0, 3].item() == 0
    torch.testing.assert_close(
        model.inputs[:, 1:2], (valid & ~torch.as_tensor(mask)).to(torch.float32),
    )

    with pytest.raises(ValueError):
        masked_loss(model, x, valid, np.zeros(x.shape, dtype=np.bool_))


def test_encode_roi_uses_exact_weights_and_tail_padding() -> None:
    torch = pytest.importorskip("torch")

    class AverageEncoder(torch.nn.Module):
        def encode(self, inputs):
            self.inputs = inputs.detach().clone()
            pooled = torch.nn.functional.avg_pool2d(inputs[:, :1], 8, 8)
            return pooled.repeat(1, 64, 1, 1)

    prepared = np.zeros(SHAPE, dtype=np.float32)
    prepared[816:824, 5992:6000] = 2
    prepared[824:832, 5992:6000] = 6
    recognition = np.ones(832, dtype=np.bool_)
    model = AverageEncoder()

    features = encode_roi(
        model, prepared, (820, 832, 5992, 6000), recognition, batch_size=16,
    )

    assert features.shape == (128,)
    assert features.dtype == np.float32
    np.testing.assert_allclose(features[:64], np.float32(14 / 3), rtol=1e-6)
    np.testing.assert_array_equal(features[64:], np.float32(6))
    assert model.inputs.shape == (1, 2, 64, 512)
    assert model.inputs[:, 1].sum().item() == 64 * 368
    np.testing.assert_array_equal(model.inputs[0, 1, :, 368:].numpy(), 0)


def test_encode_roi_covers_every_intersecting_tile_and_masks_excluded_rows() -> None:
    torch = pytest.importorskip("torch")

    class CountingEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = 0

        def encode(self, inputs):
            self.seen += inputs.shape[0]
            pooled = torch.nn.functional.avg_pool2d(inputs[:, :1], 8, 8)
            return pooled.repeat(1, 64, 1, 1)

    recognition = np.ones(832, dtype=np.bool_)
    recognition[65:68] = False
    clean = np.ones(SHAPE, dtype=np.float32)
    poisoned = clean.copy()
    poisoned[65:68] = 1e20
    first_model = CountingEncoder()
    second_model = CountingEncoder()

    first = encode_roi(first_model, clean, (60, 68, 508, 516), recognition,
                       batch_size=2)
    second = encode_roi(second_model, poisoned, (60, 68, 508, 516), recognition,
                        batch_size=2)

    assert first_model.seen == 4
    assert second_model.seen == 4
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(second).all()


def test_encode_roi_rejects_empty_or_invalid_roi() -> None:
    pytest.importorskip("torch")
    prepared = np.zeros(SHAPE, dtype=np.float32)
    recognition = np.zeros(832, dtype=np.bool_)
    model = make_model(1)

    with pytest.raises(ValueError):
        encode_roi(model, prepared, (10, 20, 100, 200), recognition)
    with pytest.raises(ValueError):
        encode_roi(model, prepared, (20, 10, 100, 200), np.ones(832, dtype=np.bool_))
