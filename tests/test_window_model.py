import numpy as np
import pytest

torch = pytest.importorskip("torch")
from das_sentinel.classifiers.window_model import (
    WindowEncoder,
    load_bundle,
    masked_loss,
    prepare,
    save_bundle,
    statistics,
)


def arrays(height=8):
    return {
        "waveform": np.ones((height, 3000), np.float32),
        "valid_channel_mask": np.ones(height, bool),
        "recognition_channel_mask": np.ones(height, bool),
        "region_id": np.arange(height) % 3,
    }


def test_full_window_and_excluded_channels():
    data = arrays()
    original = statistics(data)
    data["waveform"][-1, 2500:] = 20
    assert not torch.equal(original, statistics(data))
    data["recognition_channel_mask"][-1] = False
    original = statistics(data)
    model = WindowEncoder().eval()
    original_prediction = model(*prepare(data))
    data["waveform"][-1] = 100
    assert torch.equal(original, statistics(data))
    assert torch.equal(original_prediction, model(*prepare(data)))
    assert model(*prepare(arrays(13))).shape == (1,)


def test_unknown_loss_has_no_gradient():
    logits = torch.zeros(3, requires_grad=True)
    masked_loss(logits, torch.tensor([1.0, -1.0, 0.0])).backward()
    assert logits.grad[1] == 0
    assert logits.grad[0] < 0 < logits.grad[2]


def test_positive_weight_scales_only_positive_gradient():
    targets = torch.tensor([1.0, 0.0, -1.0])
    original = torch.zeros(3, requires_grad=True)
    weighted = torch.zeros(3, requires_grad=True)
    masked_loss(original, targets).backward()
    masked_loss(weighted, targets, pos_weight=torch.tensor([5.0])).backward()
    assert weighted.grad[0] == pytest.approx(original.grad[0].item() * 5)
    assert weighted.grad[1] == original.grad[1]
    assert weighted.grad[2] == 0


def test_new_model_rejects_old_sixty_second_inputs_and_bundle(tmp_path):
    import json

    old = arrays()
    old["waveform"] = np.ones((8, 6000), np.float32)
    with pytest.raises(ValueError):
        prepare(old)
    save_bundle(WindowEncoder(), tmp_path, "grid", {}, unit="strain")
    meta = json.loads((tmp_path / "model.json").read_text())
    assert meta["window_seconds"] == 30
    meta["window_seconds"] = 60
    (tmp_path / "model.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_bundle(tmp_path, "grid", unit="strain")


def test_safe_save_load_and_grid(tmp_path):
    model = WindowEncoder().eval()
    data = prepare(arrays())
    save_bundle(model, tmp_path, "grid", {}, unit="strain")
    assert torch.equal(
        model(*data), load_bundle(tmp_path, "grid", unit="strain")(*data)
    )
    with pytest.raises(ValueError):
        load_bundle(tmp_path, "different", unit="strain")
    (tmp_path / "weights.pt").write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash"):
        load_bundle(tmp_path, "grid", unit="strain")


def test_model_loading_requires_matching_physical_unit(tmp_path):
    model = WindowEncoder().eval()
    save_bundle(model, tmp_path, "grid", {}, unit="strain")
    assert load_bundle(tmp_path, "grid", unit="strain")(*prepare(arrays())).shape == (
        1,
    )
    with pytest.raises(ValueError, match="unit"):
        load_bundle(tmp_path, "grid", unit="counts")


def test_binary_bundle_declares_task_and_rejects_other_label_contracts(tmp_path):
    import json

    save_bundle(WindowEncoder(), tmp_path, "grid", {}, unit="strain")
    path = tmp_path / "model.json"
    meta = json.loads(path.read_text())
    assert meta["schema_version"] == 2
    assert meta["task"] == "earthquake_including_blast_vs_noise"
    assert meta["labels"] == ["earthquake"]
    assert load_bundle(tmp_path, "grid", unit="strain")(*prepare(arrays())).shape == (1,)
    for change in (
        {"task": "natural_earthquake_only"},
        {"schema_version": 1, "labels": ["earthquake", "blast", "traffic"]},
    ):
        path.write_text(json.dumps({**meta, **change}))
        with pytest.raises(ValueError, match="incompatible"):
            load_bundle(tmp_path, "grid", unit="strain")
