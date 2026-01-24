"""Tied MLP architecture with SwiGLU, FiLM conditioning, and LayerScale.

This module implements a weight-tied MLP architecture where L distinct blocks
are applied cyclically for K iterations, giving L*K total block applications.

Architecture per block:
    u = (1 + γ_k) ⊙ LayerNorm(x) + β_k    # FiLM conditioning on step k
    a = W_a @ u                            # SwiGLU gate projection
    g = W_g @ u                            # SwiGLU value projection
    h = a ⊙ SiLU(g)                        # SwiGLU activation
    Δ = W_o @ h                            # Output projection
    x = x + α * Δ                          # LayerScale residual
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import variance_scaling


class SwiGLUBlock(nn.Module):
    """Single SwiGLU block with FiLM conditioning and LayerScale.

    Attributes:
        width: Hidden dimension of the block.
        ffn_mult: Multiplier for FFN intermediate dimension (default 8/3 for SwiGLU).
        layerscale_init: Initial value for LayerScale parameter.
    """
    width: int
    ffn_mult: float = 8 / 3
    layerscale_init: float = 1e-2

    @nn.compact
    def __call__(self, x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
        """Apply the block with FiLM conditioning.

        Args:
            x: Input tensor of shape (..., width).
            gamma: FiLM scale parameter of shape (..., width).
            beta: FiLM shift parameter of shape (..., width).

        Returns:
            Output tensor of shape (..., width).
        """
        ffn_dim = int(self.width * self.ffn_mult)

        kernel_init = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # FiLM conditioning: u = (1 + γ) ⊙ LayerNorm(x) + β
        u = nn.LayerNorm()(x)
        u = (1.0 + gamma) * u + beta

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
    """Tied MLP with L distinct blocks applied cyclically for K iterations.

    The total number of block applications is L * K (or num_blocks * num_iters).
    For comparison with untied baselines:
        - Loop-1: 1 block × 4 iters = 4 block applications
        - Loop-2: 2 blocks × 2 iters = 4 block applications
        - Loop-4: 4 blocks × 1 iter = 4 block applications (equivalent to untied)

    Attributes:
        width: Hidden dimension.
        num_blocks: Number of distinct blocks (L).
        num_iters: Number of iterations to apply the blocks (K).
        ffn_mult: Multiplier for FFN intermediate dimension.
        layerscale_init: Initial value for LayerScale parameter.
        step_embed_dim: Dimension of step embeddings (defaults to width if 0).
        trunc_bptt: If > 0, detach gradients every this many steps.
    """
    width: int
    num_blocks: int = 1
    num_iters: int = 4
    ffn_mult: float = 8 / 3
    layerscale_init: float = 1e-2
    step_embed_dim: int = 0
    trunc_bptt: int = 0

    def setup(self) -> None:
        """Initialize blocks and step embeddings."""
        self.total_steps = self.num_blocks * self.num_iters

        # Step embedding dimension
        embed_dim = self.step_embed_dim if self.step_embed_dim > 0 else self.width

        # Learned step embeddings for FiLM: outputs [gamma, beta] for each step
        # We embed each step index to produce gamma and beta
        self.step_embed = nn.Embed(
            num_embeddings=self.total_steps,
            features=embed_dim,
        )

        # Project step embedding to gamma and beta
        self.film_proj = nn.Dense(
            2 * self.width,
            kernel_init=nn.initializers.zeros,  # Initialize near identity
            bias_init=nn.initializers.zeros,
        )

        # Create L distinct blocks
        self.blocks = [
            SwiGLUBlock(
                width=self.width,
                ffn_mult=self.ffn_mult,
                layerscale_init=self.layerscale_init,
                name=f"block_{i}",
            )
            for i in range(self.num_blocks)
        ]

    def __call__(self, x: jnp.ndarray, steps: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Apply the tied MLP.

        Args:
            x: Input tensor of shape (..., width).
            steps: Optional override for number of total steps (must be <= total_steps).
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
            # Get block index (cyclic)
            block_idx = step_idx % self.num_blocks
            block = self.blocks[block_idx]

            # Get FiLM parameters for this step
            step_embed = self.step_embed(jnp.asarray(step_idx))

            # Broadcast for batched inputs
            if x.ndim > 1:
                step_embed = jnp.broadcast_to(step_embed, (x.shape[0], step_embed.shape[-1]))

            # Project to gamma and beta
            film_params = self.film_proj(step_embed)
            gamma, beta = jnp.split(film_params, 2, axis=-1)

            # Apply block
            x = block(x, gamma, beta)

            # Truncated BPTT
            if self.trunc_bptt > 0 and (step_idx + 1) % self.trunc_bptt == 0:
                x = jax.lax.stop_gradient(x)

        return x


class TiedMLPEncoder(nn.Module):
    """Encoder using TiedMLP as the core, matching the interface of existing encoders.

    Structure:
        1. Input projection: Dense -> LayerNorm -> Swish
        2. TiedMLP core
        3. Output projection: Dense

    Attributes:
        width: Hidden dimension.
        output_dim: Output dimension (e.g., 64 for critic encoders).
        num_blocks: Number of distinct blocks in TiedMLP.
        num_iters: Number of iterations in TiedMLP.
        ffn_mult: FFN expansion multiplier.
        layerscale_init: LayerScale initialization value.
        step_embed_dim: Step embedding dimension.
        trunc_bptt: Truncated BPTT window (0 = disabled).
    """
    width: int
    output_dim: int = 64
    num_blocks: int = 1
    num_iters: int = 4
    ffn_mult: float = 8 / 3
    layerscale_init: float = 1e-2
    step_embed_dim: int = 0
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
        x = nn.LayerNorm()(x)
        x = nn.swish(x)

        # TiedMLP core
        x = TiedMLP(
            width=self.width,
            num_blocks=self.num_blocks,
            num_iters=self.num_iters,
            ffn_mult=self.ffn_mult,
            layerscale_init=self.layerscale_init,
            step_embed_dim=self.step_embed_dim,
            trunc_bptt=self.trunc_bptt,
        )(x, steps=steps)

        # Output projection
        x = nn.Dense(self.output_dim, kernel_init=kernel_init, bias_init=bias_init)(x)

        return x
