"""Kiểm tra thêm của nhóm cho phần sửa trong train.py: lịch entropy và nạp trọng số để học chuyển giao."""
import pytest
import torch
from torch import nn

from mario_ppo.train import _ent_coef, _load_weights


def test_ent_coef_fixed_when_end_is_none():
    assert _ent_coef(0.01, None, 3_000_000, 5_000_000) == 0.01


def test_ent_coef_linear_from_start_to_end():
    assert _ent_coef(0.01, 0.0005, 0, 1000) == pytest.approx(0.01)
    assert _ent_coef(0.01, 0.0005, 500, 1000) == pytest.approx((0.01 + 0.0005) / 2)
    assert _ent_coef(0.01, 0.0005, 1000, 1000) == pytest.approx(0.0005)
    assert _ent_coef(0.01, 0.0005, 2000, 1000) == pytest.approx(0.0005)  # quá total_steps thì giữ ở end


def test_load_weights_copies_model_only(tmp_path):
    src, dst = nn.Linear(3, 2), nn.Linear(3, 2)
    path = tmp_path / "ck.pt"
    torch.save({"model": src.state_dict(), "optimizer": {}, "global_step": 3_000_000, "config": {}}, path)
    _load_weights(dst, path, "cpu")
    for k, v in src.state_dict().items():
        assert torch.equal(v, dst.state_dict()[k]), k


def test_load_weights_rejects_other_architecture(tmp_path):
    path = tmp_path / "ck.pt"
    torch.save({"model": nn.Linear(4, 2).state_dict()}, path)
    with pytest.raises(RuntimeError):
        _load_weights(nn.Linear(3, 2), path, "cpu")
