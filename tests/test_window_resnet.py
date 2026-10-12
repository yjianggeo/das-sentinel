import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from das_sentinel.classifiers.window_model import prepare
from das_sentinel.classifiers.window_resnet import (
    ResidualWindowEncoder,
    prepare_waveform,
)


def arrays(height=8):
    time = np.linspace(0, 30, 3000, endpoint=False, dtype=np.float32)
    waveform = np.stack(
        [np.sin(2 * np.pi * (index + 1) * time / 30) for index in range(height)]
    ).astype(np.float32)
    return {
        "waveform": waveform,
        "valid_channel_mask": np.ones(height, bool),
        "recognition_channel_mask": np.ones(height, bool),
        "region_id": np.arange(height) % 3,
    }


@pytest.mark.parametrize(
    ("input_kind", "preparer"),
    [("pooled", prepare), ("waveform", prepare_waveform)],
)
def test_encoder_accepts_variable_height_and_stays_small(input_kind, preparer):
    model = ResidualWindowEncoder(input_kind=input_kind)
    assert model(*preparer(arrays(13))).shape == (1,)
    assert sum(parameter.numel() for parameter in model.parameters()) <= 50_000


def test_prepare_waveform_preserves_signed_samples_and_validates_input():
    data = arrays(5)
    planes, mask, regions = prepare_waveform(data)
    assert planes.shape == (1, 3, 5, 3000)
    assert torch.isfinite(planes).all()
    assert torch.equal(mask, torch.ones(5, dtype=torch.bool))
    assert torch.equal(regions, torch.tensor([0, 1, 2, 0, 1]))
    assert torch.equal(
        planes[0, 0],
        -prepare_waveform({**data, "waveform": -data["waveform"]})[0][0, 0],
    )

    for waveform in (
        np.ones((5, 150), np.float32),
        np.full((5, 3000), np.nan, np.float32),
    ):
        with pytest.raises(ValueError, match="waveform"):
            prepare_waveform({**data, "waveform": waveform})
    with pytest.raises(ValueError, match="waveform"):
        prepare_waveform(
            {
                **data,
                "valid_channel_mask": np.zeros(5, bool),
                "recognition_channel_mask": np.zeros(5, bool),
            }
        )


@pytest.mark.parametrize(
    ("input_kind", "preparer"),
    [("pooled", prepare), ("waveform", prepare_waveform)],
)
def test_excluded_channels_and_regions_cannot_change_prediction(input_kind, preparer):
    torch.manual_seed(7)
    data = arrays(8)
    data["recognition_channel_mask"][-1] = False
    model = ResidualWindowEncoder(input_kind=input_kind).eval()
    planes, mask, regions = preparer(data)
    expected = model(planes, mask, regions)

    planes[:, :, -1] = 1_000_000
    assert torch.equal(expected, model(planes, mask, regions))

    regions = regions.clone()
    regions[-2] = 99
    expected = model(planes, mask, regions)
    planes[:, :, -2] = -1_000_000
    assert torch.equal(expected, model(planes, mask, regions))


@pytest.mark.parametrize(
    ("input_kind", "preparer"),
    [("pooled", prepare), ("waveform", prepare_waveform)],
)
def test_encoder_has_finite_trainable_gradients(input_kind, preparer):
    torch.manual_seed(11)
    model = ResidualWindowEncoder(input_kind=input_kind)
    loss = model(*preparer(arrays(9))).square().sum()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


@pytest.mark.parametrize("input_kind", ["pooled", "waveform"])
def test_state_dict_round_trip_preserves_prediction(input_kind):
    torch.manual_seed(13)
    model = ResidualWindowEncoder(input_kind=input_kind).eval()
    inputs = prepare(arrays()) if input_kind == "pooled" else prepare_waveform(arrays())
    expected = model(*inputs)
    restored = ResidualWindowEncoder(input_kind=input_kind).eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    assert torch.equal(expected, restored(*inputs))


def test_encoder_rejects_mismatched_contracts():
    with pytest.raises(ValueError, match="input_kind"):
        ResidualWindowEncoder(input_kind="spectrogram")
    model = ResidualWindowEncoder(input_kind="pooled")
    planes, mask, regions = prepare(arrays())
    with pytest.raises(ValueError, match="time"):
        model(planes[..., :-1], mask, regions)
    with pytest.raises(ValueError, match="channel"):
        model(planes, mask[:-1], regions)
    with pytest.raises(ValueError, match="valid channel"):
        model(planes, torch.zeros_like(mask), regions)
