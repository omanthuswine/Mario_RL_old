"""Mạng actor-critic cho ảnh Mario.

Giao diện mà train.py và evaluate.py dùng (giữ nguyên tên và chữ ký):
  ActorCritic(n_actions, in_channels=4, arch="nature")
  forward(obs)  : obs uint8 (B, C, 84, 84) -> (logits (B, n_actions) float32, value (B,) float32)
  act(obs)      : lấy mẫu hành động, trả (action long (B,), logp (B,), value (B,)) trên CPU, không gradient

Hai kiến trúc thân CNN, chọn bằng khóa cấu hình `model.arch`:
  - "nature" (mặc định): Nature-CNN của DQN (Mnih 2015), giống CleanRL ppo_atari và CnnPolicy của SB3.
    Kiến trúc chuẩn cho PPO trên ảnh game, nhẹ, chạy nhanh trên CPU.
  - "impala": IMPALA-CNN (Espeholt 2018), giống CleanRL ppo_procgen: 3 tầng [16, 32, 32] kênh, mỗi tầng
    có 2 khối dư. Cobbe 2019 (CoinRun) cho thấy nó tổng quát hóa tốt hơn Nature-CNN ở trạng thái chưa gặp,
    hợp với cách chấm (NOOP ngẫu nhiên, seed lạ). Đổi lại, khi đo trên CPU nó chạy chậm hơn Nature-CNN
    khoảng 13 lần (lượt forward 512 ảnh: 664 ms so với 52 ms), nên chỉ dùng khi có GPU.
"""
import numpy as np
import torch
from torch import nn

ARCHS = ("nature", "impala")


def _init(layer, gain=np.sqrt(2)):
    """Khởi tạo trực giao, bias bằng 0 (thường dùng cho PPO)."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def _nature_conv(in_channels):
    """3 lớp conv của Nature-CNN; với ảnh 84x84 cho ra 64 x 7 x 7 = 3136 đặc trưng."""
    return nn.Sequential(
        _init(nn.Conv2d(in_channels, 32, 8, stride=4)), nn.ReLU(),
        _init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
        _init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
        nn.Flatten(),
    )


class _ResidualBlock(nn.Module):
    """x + conv(relu(conv(relu(x)))), giữ nguyên số kênh và kích thước."""

    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv1(torch.relu(self.conv0(torch.relu(x))))


def _impala_conv(in_channels):
    """3 tầng IMPALA (conv 3x3, max-pool bước 2, 2 khối dư); 84 -> 42 -> 21 -> 11, ra 32 x 11 x 11.

    Các conv giữ khởi tạo mặc định của PyTorch như CleanRL ppo_procgen: khởi tạo trực giao gain √2
    cộng dồn qua 6 khối dư sẽ làm biên độ kích hoạt tăng nhiều lần.
    """
    layers, c = [], in_channels
    for out in (16, 32, 32):
        layers += [nn.Conv2d(c, out, 3, padding=1), nn.MaxPool2d(3, stride=2, padding=1),
                   _ResidualBlock(out), _ResidualBlock(out)]
        c = out
    return nn.Sequential(*layers, nn.Flatten(), nn.ReLU())


class ActorCritic(nn.Module):
    """Thân CNN dùng chung, hai đầu ra: logits của chính sách và giá trị trạng thái.

    Mục đích: một mạng vừa là actor (chọn hành động) vừa là critic (ước lượng giá trị cho GAE).
    Đầu chính sách khởi tạo gain 0,01 để chính sách ban đầu gần đều (khám phá tốt lúc đầu),
    đầu giá trị gain 1, các tầng ẩn gain √2 (37 Implementation Details of PPO, Huang 2022).
    """

    def __init__(self, n_actions, in_channels=4, arch="nature"):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"model.arch phải là một trong {ARCHS}, nhận được {arch!r}")
        conv = _nature_conv(in_channels) if arch == "nature" else _impala_conv(in_channels)
        hidden = 512 if arch == "nature" else 256
        with torch.no_grad():  # tính số đặc trưng sau conv thay vì ghi cứng
            n_flat = conv(torch.zeros(1, in_channels, 84, 84)).shape[1]
        self.body = nn.Sequential(conv, _init(nn.Linear(n_flat, hidden)), nn.ReLU())
        self.policy = _init(nn.Linear(hidden, n_actions), gain=0.01)
        self.value = _init(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, obs):
        h = self.body(obs.float() / 255.0)
        return self.policy(h), self.value(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        """CÓ SẴN. Lấy mẫu từ chính sách hiện tại; nhận obs ở thiết bị bất kỳ, trả tensor CPU."""
        device = next(self.parameters()).device
        logits, value = self(obs.to(device))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.cpu(), dist.log_prob(action).cpu(), value.cpu()
