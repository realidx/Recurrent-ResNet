from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def get_activation(name: str) -> Callable[[], nn.Module]:
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU
    if name in {"silu", "swish"}:
        return nn.SiLU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported activation: {name}")


class ResidualBlock(nn.Module):
    def __init__(self, width: int, activation: str = "relu") -> None:
        super().__init__()
        act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(width, width),
            act(),
            nn.Linear(width, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class ResidualStack(nn.Module):
    def __init__(self, depth: int, width: int, activation: str = "relu") -> None:
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList(
            ResidualBlock(width, activation=activation) for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
