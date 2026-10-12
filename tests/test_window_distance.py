import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from das_sentinel.classifiers.window_distance import (
    crop_distance_arrays,
    prepare_distance_waveform,
)
from das_sentinel.classifiers.window_resnet import ResidualWindowEncoder


def arrays(height=8, samples=3000):
    distance = 503.067 + (np.arange(height, dtype=np.float64) - 123) * 4.1
    time = np.arange(samples, dtype=np.float64) / 100
    waveform = np.sin(
        np.arange(height, dtype=np.float32)[:, None] / 17
        + time.astype(np.float32)[None]
    ).astype(np.float32)
    return {
        "waveform": waveform,
        "distance_m": distance,
        "relative_seconds": time,
        "valid_channel_mask": np.ones(height, dtype=bool),
        "recognition_channel_mask": np.ones(height, dtype=bool),
        "region_id": np.arange(height, dtype=np.int8) % 3,
        "source": "frozen",
    }


def test_real_500m_crop_starts_at_503_067_and_keeps_invalid_tail():
    data = arrays(834)
    data["valid_channel_mask"][-2:] = False
    data["recognition_channel_mask"][-2:] = False
    frozen = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in data.items()
    }

    cropped = crop_distance_arrays(data, 500)

    assert cropped["waveform"].shape == (711, 3000)
    assert cropped["distance_m"][0] == pytest.approx(503.067)
    np.testing.assert_array_equal(cropped["valid_channel_mask"][-2:], [False, False])
    np.testing.assert_array_equal(
        cropped["recognition_channel_mask"][-2:], [False, False]
    )
    np.testing.assert_array_equal(cropped["region_id"], data["region_id"][123:])
    np.testing.assert_array_equal(cropped["relative_seconds"], data["relative_seconds"])
    assert cropped["source"] == "frozen"
    for key, expected in frozen.items():
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(data[key], expected)
        else:
            assert data[key] == expected


def test_npz_crop_reads_each_array_once_and_preserves_extra_keys(tmp_path, monkeypatch):
    data = arrays(8)
    data["valid_channel_mask"][5] = False
    data["recognition_channel_mask"][6] = False
    data["calibration"] = np.array([1.25, 2.5])
    path = tmp_path / "window.npz"
    np.savez(path, **data)
    reads = {}
    with np.load(path, allow_pickle=False) as saved:
        original_getitem = type(saved).__getitem__

        def getitem(self, key):
            reads[key] = reads.get(key, 0) + 1
            return original_getitem(self, key)

        monkeypatch.setattr(type(saved), "__getitem__", getitem)
        cropped = crop_distance_arrays(saved, data["distance_m"][3])

    assert reads == {key: 1 for key in data}
    np.testing.assert_array_equal(cropped["waveform"], data["waveform"][3:])
    for key in ("distance_m", "valid_channel_mask", "recognition_channel_mask", "region_id"):
        np.testing.assert_array_equal(cropped[key], data[key][3:])
    np.testing.assert_array_equal(cropped["relative_seconds"], data["relative_seconds"])
    np.testing.assert_array_equal(cropped["calibration"], [1.25, 2.5])
    assert cropped["source"].item() == "frozen"

    from das_sentinel.classifiers.window_resnet import prepare_waveform

    manual = {
        "waveform": data["waveform"][3:],
        "valid_channel_mask": np.array([True, True, False, True, True]),
        "recognition_channel_mask": np.array([True, True, True, False, True]),
        "region_id": np.array([0, 1, 2, 0, 1]),
    }
    expected, actual = prepare_waveform(manual), prepare_waveform(cropped)
    assert all(torch.equal(left, right) for left, right in zip(expected, actual))
    torch.manual_seed(29)
    model = ResidualWindowEncoder(input_kind="waveform").eval()
    with torch.inference_mode():
        assert torch.equal(model(*expected), model(*actual))


@pytest.mark.parametrize(
    "distance",
    [
        np.array([0.0, np.nan, 2.0]),
        np.array([0.0, np.inf, 2.0]),
        np.array([0.0, 2.0, 2.0]),
        np.array([0.0, 2.0, 1.0]),
    ],
)
def test_crop_rejects_nonfinite_or_nonincreasing_distance(distance):
    data = arrays(3, samples=5)
    data["distance_m"] = distance
    with pytest.raises(ValueError, match="distance_m"):
        crop_distance_arrays(data, 0)


@pytest.mark.parametrize("start", [-1, np.nan, np.inf, -np.inf])
def test_crop_rejects_invalid_start(start):
    with pytest.raises(ValueError, match="distance_start_m"):
        crop_distance_arrays(arrays(3, samples=5), start)


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("waveform", np.ones((3, 5, 1), np.float32)),
        ("distance_m", np.arange(2, dtype=float)),
        ("valid_channel_mask", np.ones(2, bool)),
        ("recognition_channel_mask", np.ones(2, bool)),
        ("region_id", np.zeros(2, np.int8)),
        ("relative_seconds", np.arange(4, dtype=float)),
    ],
)
def test_crop_rejects_dimension_mismatch(key, bad_value):
    data = arrays(3, samples=5)
    data[key] = bad_value
    with pytest.raises(ValueError, match="dimension"):
        crop_distance_arrays(data, 0)


def test_crop_requires_a_selected_recognized_valid_channel():
    data = arrays(5, samples=5)
    data["valid_channel_mask"][-2:] = False
    with pytest.raises(ValueError, match="valid recognized"):
        crop_distance_arrays(data, data["distance_m"][-2])
    with pytest.raises(ValueError, match="valid recognized"):
        crop_distance_arrays(data, data["distance_m"][-1] + 1)


def test_prepare_ignores_every_sample_before_distance_start():
    data = arrays(8)
    expected = prepare_distance_waveform(data, data["distance_m"][4])
    changed = copy.deepcopy(data)
    changed["waveform"][:4] = 1_000_000

    actual = prepare_distance_waveform(changed, data["distance_m"][4])

    assert all(torch.equal(left, right) for left, right in zip(expected, actual))
    assert actual[0].shape == (1, 3, 4, 3000)


def test_prepared_real_crop_supports_training_and_state_replay():
    torch.manual_seed(29)
    data = arrays(834)
    data["valid_channel_mask"][-2:] = False
    inputs = prepare_distance_waveform(data, 500)
    model = ResidualWindowEncoder(input_kind="waveform")

    expected = model(*inputs)
    expected.square().sum().backward()

    assert expected.shape == (1,)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    restored = ResidualWindowEncoder(input_kind="waveform")
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    assert torch.equal(expected.detach(), restored(*inputs).detach())
