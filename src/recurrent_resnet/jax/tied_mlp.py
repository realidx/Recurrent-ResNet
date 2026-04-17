"""Weight-tied recurrent SwiGLU MLP with additive step embeddings and LayerScale.

Architecture:
    x = input_projection(input)
    for k in range(K):
        u = LayerNorm(x) + e_k             # pre-LN, then additive step embedding
        a = W_a @ u                        # SwiGLU gate projection
        g = W_g @ u                        # SwiGLU value projection
        h = a ⊙ SiLU(g)                    # SwiGLU activation
        Δ = W_o @ h                        # Output projection
        x = x + α * Δ                      # LayerScale residual
    y = output_projection(x)
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import variance_scaling


class SwiGLUBlock(nn.Module):
    """Single recurrent SwiGLU block with additive step conditioning and LayerScale.

    Attributes:
        width: Hidden dimension of the block.
        ffn_mult: Multiplier for FFN intermediate dimension (default 8/3 for SwiGLU).
        layerscale_init: Initial value for LayerScale parameter.
    """
    width: int
    ffn_mult: float = 8 / 3
    layerscale_init: float = 1e-2

    @nn.compact
    def __call__(self, x: jnp.ndarray, step_embed: jnp.ndarray) -> jnp.ndarray:
        """Apply the block with additive step conditioning.

        Args:
            x: Input tensor of shape (..., width).
            step_embed: Additive step embedding of shape (..., width).

        Returns:
            Output tensor of shape (..., width).
        """
        ffn_dim = int(self.width * self.ffn_mult)

        kernel_init = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # Pre-LN followed by additive step embedding.
        u = nn.LayerNorm()(x)
        u = u + step_embed

        # SwiGLU: h = a ⊙ SiLU(g)
        a = nn.Dense(ffn_dim, kernel_init=kernel_init, bias_init=bias_init, name="W_a")(u)
        g = nn.Dense(ffn_dim, kernel_init=kernel_init, bias_init=bias_init, name="W_g")(u)
        h = a * nn.silu(g)

        # Output projection
        delta = nn.Dense(self.width, kernel_init=kernel_init, bias_init=bias_init, name="W_o")(h)

        # LayerScale: learnable scalar initialized to small value
        alpha = self.param(
            "layerscale",
            lambda key: jnp.asarray(self.layerscale_init, dtype=jnp.float32),
        )

        return x + alpha * delta


class TiedMLP(nn.Module):
    """Tied MLP with one repeated SwiGLU block applied for K iterations.

    Attributes:
        width: Hidden dimension.
        num_iters: Number of recurrent refinement iterations (K).
        ffn_mult: Multiplier for FFN intermediate dimension.
        layerscale_init: Initial value for LayerScale parameter.
        trunc_bptt: If > 0, detach gradients every this many steps.
    """
    width: int
    num_iters: int = 4
    ffn_mult: float = 8 / 3
    layerscale_init: float = 1e-2
    trunc_bptt: int = 0

    def setup(self) -> None:
        """Initialize the repeated block and step embeddings."""
        self.total_steps = self.num_iters

        self.step_embed = nn.Embed(
            num_embeddings=self.total_steps,
            features=self.width,
            embedding_init=nn.initializers.zeros,
        )

        self.block = SwiGLUBlock(
            width=self.width,
            ffn_mult=self.ffn_mult,
            layerscale_init=self.layerscale_init,
            name="block",
        )

    def __call__(self, x: jnp.ndarray, steps: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Apply the tied MLP.

        Args:
            x: Input tensor of shape (..., width).
            steps: Optional override for number of recurrent steps (must be <= total_steps).
                   Can be an int or a scalar jnp.ndarray.

        Returns:
            Output tensor of shape (..., width).
        """
        num_steps = self.total_steps
        if steps is not None:
            # Handle both int and jnp.ndarray
            if isinstance(steps, int):
                num_steps = steps
            else:
                num_steps = int(steps)
            if num_steps > self.total_steps:
                raise ValueError(f"steps ({num_steps}) exceeds total_steps ({self.total_steps})")

        for step_idx in range(num_steps):
            step_embed = self._lookup_step_embed(step_idx, x)
            x = self.block(x, step_embed)

            # Truncated BPTT
            if self.trunc_bptt > 0 and (step_idx + 1) % self.trunc_bptt == 0:
                x = jax.lax.stop_gradient(x)

        return x

    def _lookup_step_embed(self, step_idx: int, x: jnp.ndarray) -> jnp.ndarray:
        embed = self.step_embed(jnp.asarray(step_idx))
        if x.ndim == 1:
            return embed
        return jnp.broadcast_to(embed, (x.shape[0], embed.shape[-1]))


class TiedMLPEncoder(nn.Module):
    """Encoder using TiedMLP as the core, matching the interface of existing encoders.

    Structure:
        1. Input projection: Dense
        2. TiedMLP core
        3. Output projection: Dense

    Attributes:
        width: Hidden dimension.
        output_dim: Output dimension (e.g., 64 for critic encoders).
        num_iters: Number of iterations in TiedMLP.
        ffn_mult: FFN expansion multiplier.
        layerscale_init: LayerScale initialization value.
        trunc_bptt: Truncated BPTT window (0 = disabled).
    """
    width: int
    output_dim: int = 64
    num_iters: int = 4
    ffn_mult: float = 8 / 3
    layerscale_init: float = 1e-2
    trunc_bptt: int = 0

    @nn.compact
    def __call__(self, x: jnp.ndarray, steps: Optional[int] = None) -> jnp.ndarray:
        """Encode input through the TiedMLP.

        Args:
            x: Input tensor.
            steps: Optional override for number of recurrent steps.

        Returns:
            Encoded output tensor.
        """
        kernel_init = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # Input projection
        x = nn.Dense(self.width, kernel_init=kernel_init, bias_init=bias_init)(x)

        # TiedMLP core
        x = TiedMLP(
            width=self.width,
            num_iters=self.num_iters,
            ffn_mult=self.ffn_mult,
            layerscale_init=self.layerscale_init,
            trunc_bptt=self.trunc_bptt,
        )(x, steps=steps)

        # Output projection
        x = nn.Dense(self.output_dim, kernel_init=kernel_init, bias_init=bias_init)(x)

        return x
