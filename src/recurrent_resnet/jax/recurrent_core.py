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
    pre_norm: bool = False
    alpha: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        identity = x
        kernel_init = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        last_init = nn.initializers.zeros if self.init_identity else kernel_init
        activation = _activation(self.use_relu)

        if self.pre_norm:
            x = _apply_norm(self.norm_type, x)
            x = nn.Dense(self.width, kernel_init=kernel_init, bias_init=bias_init)(x)
            x = activation(x)
            x = nn.Dense(self.width, kernel_init=kernel_init, bias_init=bias_init)(x)
            x = activation(x)
            x = nn.Dense(self.width, kernel_init=kernel_init, bias_init=bias_init)(x)
            x = activation(x)
            x = nn.Dense(self.width, kernel_init=last_init, bias_init=bias_init)(x)
            return identity + self.alpha * x

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
        return identity + self.alpha * x


class StepConditionedBlock(nn.Module):
    width: int
    norm_type: str = "layer_norm"
    use_relu: int = 0
    init_identity: bool = False
    alpha: float = 1.0
    variant: str = "film"

    @nn.compact
    def __call__(self, x: jnp.ndarray, step_embed: jnp.ndarray) -> jnp.ndarray:
        if self.variant not in {"film", "gated"}:
            raise ValueError(f"Unsupported variant: {self.variant}")

        kernel_init = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        last_init = nn.initializers.zeros if self.init_identity else kernel_init
        activation = _activation(self.use_relu)

        u = _apply_norm(self.norm_type, x)
        gamma_beta = nn.Dense(
            2 * self.width,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )(step_embed)
        gamma, beta = jnp.split(gamma_beta, 2, axis=-1)
        u = (1.0 + gamma) * u + beta

        delta = nn.Dense(self.width, kernel_init=kernel_init, bias_init=bias_init)(u)
        delta = activation(delta)
        delta = nn.Dense(self.width, kernel_init=kernel_init, bias_init=bias_init)(delta)
        delta = activation(delta)
        delta = nn.Dense(self.width, kernel_init=kernel_init, bias_init=bias_init)(delta)
        delta = activation(delta)
        delta = nn.Dense(self.width, kernel_init=last_init, bias_init=bias_init)(delta)

        if self.variant == "gated":
            gate_in = jnp.concatenate([u, step_embed], axis=-1)
            gate = nn.Dense(
                self.width,
                kernel_init=kernel_init,
                bias_init=bias_init,
            )(gate_in)
            delta = delta * nn.sigmoid(gate)

        return x + self.alpha * delta


class ResidualStack(nn.Module):
    width: int
    num_blocks: int
    norm_type: str = "layer_norm"
    use_relu: int = 0
    pre_norm: bool = False
    alpha: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for _ in range(self.num_blocks):
            x = ResidualBlock(
                width=self.width,
                norm_type=self.norm_type,
                use_relu=self.use_relu,
                pre_norm=self.pre_norm,
                alpha=self.alpha,
            )(x)
        return x


class RecurrentResidualCore(nn.Module):
    width: int
    steps: int
    tie_weights: bool = True
    stages: int = 1
    variant: str = "plain"
    step_embed_dim: int = 0
    norm_type: str = "layer_norm"
    use_relu: int = 0
    init_identity: bool = False
    trunc_bptt: int = 0
    pre_norm: bool = False
    alpha_mode: str = "fixed"
    alpha_init: float = 1.0

    def setup(self) -> None:
        if self.variant not in {"plain", "film", "gated"}:
            raise ValueError(f"Unknown variant: {self.variant}")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")

        self._alpha = self._init_alpha()
        self._num_stages = self._resolve_stages()
        if self.steps % self._num_stages != 0:
            raise ValueError("steps must be divisible by stages")
        self._steps_per_stage = self.steps // self._num_stages

        if self.variant in {"film", "gated"}:
            embed_dim = self.step_embed_dim if self.step_embed_dim > 0 else self.width
            self.step_embed = nn.Embed(self.steps, embed_dim)
            block_cls = StepConditionedBlock
            block_kwargs = dict(
                width=self.width,
                norm_type=self.norm_type,
                use_relu=self.use_relu,
                init_identity=self.init_identity,
                alpha=self._alpha,
                variant=self.variant,
            )
        else:
            self.step_embed = None
            block_cls = ResidualBlock
            block_kwargs = dict(
                width=self.width,
                norm_type=self.norm_type,
                use_relu=self.use_relu,
                init_identity=self.init_identity,
                pre_norm=self.pre_norm,
                alpha=self._alpha,
            )

        if self._num_stages == 1:
            self.block = block_cls(**block_kwargs)
        else:
            self.blocks = [block_cls(**block_kwargs) for _ in range(self._num_stages)]

    def __call__(self, x: jnp.ndarray, steps: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        num_steps = self.steps
        if steps is not None and isinstance(steps, int):
            num_steps = steps
        if num_steps > self.steps:
            raise ValueError("steps exceeds configured depth")

        for i in range(num_steps):
            if self._num_stages == 1:
                block = self.block
            else:
                stage_idx = min(i // self._steps_per_stage, self._num_stages - 1)
                block = self.blocks[stage_idx]

            if self.variant in {"film", "gated"}:
                step_embed = self._lookup_step_embed(i, x)
                x = block(x, step_embed)
            else:
                x = block(x)
            if self.trunc_bptt > 0 and (i + 1) % self.trunc_bptt == 0:
                x = jax.lax.stop_gradient(x)
        return x

    def _resolve_stages(self) -> int:
        if self.stages < 1:
            raise ValueError("stages must be >= 1")
        if self.tie_weights and self.stages == 1:
            return 1
        if self.tie_weights and self.stages > 1:
            return self.stages
        if not self.tie_weights and self.stages == 1:
            return self.steps
        return min(self.stages, self.steps)

    def _init_alpha(self) -> jnp.ndarray | float:
        if self.alpha_mode == "scaled":
            return 1.0 / float(self.steps)
        if self.alpha_mode == "learned":
            init_value = self.alpha_init / float(self.steps)
            return self.param(
                "alpha",
                lambda key: jnp.asarray(init_value, dtype=jnp.float32),
            )
        return float(self.alpha_init)

    def _lookup_step_embed(self, step_idx: int, x: jnp.ndarray) -> jnp.ndarray:
        if self.step_embed is None:
            raise ValueError("step_embed is not initialized for this variant")
        embed = self.step_embed(jnp.asarray(step_idx))
        if x.ndim == 1:
            return embed
        return jnp.broadcast_to(embed, (x.shape[0], embed.shape[-1]))
