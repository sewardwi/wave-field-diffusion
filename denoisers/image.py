"""
WaveFieldDenoiser — diffusion backbone for 2D images.

Architecture (DiT-inspired):
    Noisy image
      → patchify (P×P → token per patch)
      → linear patch embedding + 2D sin/cos positional embedding
      → timestep embedding (sinusoidal → MLP)
      → N × WaveFieldBlock
      → FinalLayer → unpatchify → predicted noise ε (or v)

Shared building blocks (WaveFieldBlock, TimestepEmbedder, FinalLayer,
positional embeddings) live in wave_field/blocks.py and are also used by
the audio denoiser.

Conditioning modes:
  'physics' — timestep modulates wave kernels α/ω/φ + FFN FiLM + residual gates
  'adaln'   — DiT AdaLN-Zero on attention and FFN branches

Spatial variant: use_2d_kernel=True swaps in 2D radial wave kernels.
Self-conditioning (Chen 2022): use_self_cond=True doubles the input channels.
"""

import torch
import torch.nn as nn

from wave_field.blocks import (
    TimestepEmbedder,
    WaveFieldBlock,
    FinalLayer,
    pos_embed_2d_sincos,
)


class WaveFieldDenoiser(nn.Module):
    """
    Wave Field diffusion backbone for images.

    Args:
        image_size:    int or (H, W) — must be divisible by patch_size
        in_channels:   image channels (1=MNIST, 3=CIFAR)
        patch_size:    patch side length
        dim, depth, num_heads, timestep_dim, conditioning: model hyperparams
        use_2d_kernel: use 2D radial wave kernels (vs 1D over flattened patches)
        use_self_cond: enable self-conditioning extra input channel
    """

    def __init__(
        self,
        image_size,
        in_channels: int,
        patch_size: int = 4,
        dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        timestep_dim: int = 128,
        conditioning: str = "physics",
        use_2d_kernel: bool = False,
        use_self_cond: bool = False,
    ):
        super().__init__()

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        H, W = image_size
        assert H % patch_size == 0 and W % patch_size == 0, (
            f"image_size {image_size} must be divisible by patch_size {patch_size}"
        )

        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.dim = dim
        self.conditioning = conditioning
        self.use_self_cond = use_self_cond

        self.ph = H // patch_size
        self.pw = W // patch_size
        self.num_patches = self.ph * self.pw
        self.patch_dim = in_channels * patch_size * patch_size

        embed_in = self.patch_dim * (2 if use_self_cond else 1)
        self.patch_embed = nn.Linear(embed_in, dim)
        self.register_buffer(
            "pos_embed",
            pos_embed_2d_sincos(self.ph, self.pw, dim),
            persistent=False,
        )

        self.time_embed = TimestepEmbedder(timestep_dim)

        self.blocks = nn.ModuleList([
            WaveFieldBlock(
                dim=dim, num_heads=num_heads, seq_len=self.num_patches,
                timestep_dim=timestep_dim, conditioning=conditioning,
                use_2d_kernel=use_2d_kernel,
                height=self.ph, width=self.pw,
            )
            for _ in range(depth)
        ])

        self.final = FinalLayer(dim, self.patch_dim, timestep_dim, conditioning)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)

    def patchify(self, x: torch.Tensor, channels: int | None = None) -> torch.Tensor:
        """
        (B, C, H, W) → (B, N, C*P*P).  `channels` overrides the assumed channel
        count (used for self-conditioned input where C = 2 * in_channels).
        """
        B, C, H, W = x.shape
        P = self.patch_size
        flat_dim = (channels if channels is not None else self.in_channels) * P * P
        x = x.reshape(B, C, self.ph, P, self.pw, P)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, self.num_patches, flat_dim)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, patch_dim) → (B, C, H, W)."""
        B = x.shape[0]
        P = self.patch_size
        C = self.in_channels
        x = x.view(B, self.ph, self.pw, C, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, self.ph * P, self.pw * P)
        return x

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor,
                self_cond: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x_noisy:   (B, C, H, W) noisy image in [-1, 1]
            t:         (B,) integer timesteps
            self_cond: (B, C, H, W) optional previous-step x_0 estimate
        Returns:
            (B, C, H, W) — predicted noise ε (or v with v-pred)
        """
        if self.use_self_cond:
            if self_cond is None:
                self_cond = torch.zeros_like(x_noisy)
            x_in = torch.cat([x_noisy, self_cond], dim=1)
            patches = self.patchify(x_in, channels=2 * self.in_channels)
        else:
            patches = self.patchify(x_noisy)

        h = self.patch_embed(patches) + self.pos_embed
        t_emb = self.time_embed(t)
        for block in self.blocks:
            h = block(h, t_emb)
        out = self.final(h, t_emb)
        return self.unpatchify(out)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
