from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import variance_scaling


def _activation(use_relu: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return nn.relu if use_relu else nn.swish


def _apply_norm(norm_type: str, x: jnp.ndarray) -> jnp.ndarray:
    if norm_type == "layer_norm":
        return nn.LayerNorm()(x)
    return x


class ResidualBlock(nn.Module):
    width: int
    norm_type: str = "layer_norm"
    use_relu: int = 0
    init_identity: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        identity = x
        kernel_init = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        last_init = nn.initializers.zeros if self.init_identity else kernel_init
        activation = _activation(self.use_relu)

        x = nn.Dense(self.width, kernel_init=kernel_init, bias_init=bias_init)(x)
        x = _apply_norm(self.norm_type, x)
        x = activation(x)
        x = nn.Dense(self.width, kernel_init=kernel_init, bias_init=bias_init)(x)
        x = _apply_norm(self.norm_type, x)
        x = activation(x)
        x = nn.Dense(self.width, kernel_init=kernel_init, bias_init=bias_init)(x)
        x = _apply_norm(self.norm_type, x)
        x = activation(x)
        x = nn.Dense(self.width, kernel_init=last_init, bias_init=bias_init)(x)
        x = _apply_norm(self.norm_type, x)
        x = activation(x)
        return x + identity


class ResidualStack(nn.Module):
    width: int
    num_blocks: int
    norm_type: str = "layer_norm"
    use_relu: int = 0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for _ in range(self.num_blocks):
            x = ResidualBlock(
                width=self.width,
                norm_type=self.norm_type,
                use_relu=self.use_relu,
            )(x)
        return x


class RecurrentResidualCore(nn.Module):
    width: int
    steps: int
    tie_weights: bool = True
    norm_type: str = "layer_norm"
    use_relu: int = 0
    init_identity: bool = False
    trunc_bptt: int = 0

    def setup(self) -> None:
        if self.tie_weights:
            self.block = ResidualBlock(
                width=self.width,
                norm_type=self.norm_type,
                use_relu=self.use_relu,
                init_identity=self.init_identity,
            )
        else:
            self.blocks = [
                ResidualBlock(
                    width=self.width,
                    norm_type=self.norm_type,
                    use_relu=self.use_relu,
                    init_identity=self.init_identity,
                )
                for _ in range(self.steps)
            ]

    def __call__(self, x: jnp.ndarray, steps: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        num_steps = self.steps
        if steps is not None and isinstance(steps, int):
            num_steps = steps

        for i in range(num_steps):
            if self.tie_weights:
                block = self.block
            else:
                if i >= len(self.blocks):
                    raise ValueError("steps exceeds untied depth")
                block = self.blocks[i]
            x = block(x)
            if self.trunc_bptt > 0 and (i + 1) % self.trunc_bptt == 0:
                x = jax.lax.stop_gradient(x)
        return x
