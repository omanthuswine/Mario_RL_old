"""Kiểm tra thêm của nhóm cho khóa model.arch (ngoài bộ kiểm tra công khai test_model.py)."""
import pytest
import torch
from omegaconf import OmegaConf

from mario_ppo.model import ARCHS, ActorCritic


def _obs(batch=4):
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (batch, 4, 84, 84), dtype=torch.uint8, generator=g)


@pytest.mark.parametrize("arch", ARCHS)
def test_arch_shapes_init_and_gradients(arch):
    model = ActorCritic(7, arch=arch)
    logits, value = model(_obs())
    assert logits.shape == (4, 7) and value.shape == (4,)
    assert float((torch.softmax(logits, -1) - 1 / 7).abs().max()) < 0.05
    (logits.logsumexp(-1).mean() + value.pow(2).mean()).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


@pytest.mark.parametrize("arch", ARCHS)
def test_state_dict_reloads_like_evaluate(arch):
    # evaluate.load_model dựng lại mạng từ cfg["model"] rồi nạp state_dict: phải khớp hoàn toàn.
    a, b = ActorCritic(7, **{"arch": arch}), ActorCritic(7, **{"arch": arch})
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        assert torch.equal(a(_obs())[0], b(_obs())[0])


def test_default_config_builds_model():
    cfg = OmegaConf.to_container(OmegaConf.load("configs/default.yaml"), resolve=True)
    ActorCritic(7, in_channels=cfg["env"]["frame_stack"], **cfg["model"])


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        ActorCritic(7, arch="resnet")
