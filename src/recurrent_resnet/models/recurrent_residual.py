from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .residual_stack import ResidualBlock


class RecurrentResidualBlock(nn.Module):
    def __init__(
        self,
        width: int,
        steps: int = 4,
        activation: str = "relu",
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.steps = steps
        self.tie_weights = tie_weights
        if tie_weights:
            self.block = ResidualBlock(width, activation=activation)
        else:
            self.blocks = nn.ModuleList(
                ResidualBlock(width, activation=activation) for _ in range(steps)
            )

    def forward(self, x: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        num_steps = steps if steps is not None else self.steps
        if self.tie_weights:
            for _ in range(num_steps):
                x = self.block(x)
            return x
        if num_steps > len(self.blocks):
            raise ValueError("steps exceeds untied depth")
        for idx in range(num_steps):
            x = self.blocks[idx](x)
        return x
