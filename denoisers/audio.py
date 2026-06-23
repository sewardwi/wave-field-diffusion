"""
WaveFieldAudioDenoiser — 1D variant of the diffusion backbone for raw waveforms.

Architecture:
    Raw waveform (B, 1, L_samples)
      → 1D patchify (Conv1d stride=patch_size) → (B, N, dim)
      → 1D sinusoidal positional embedding
      → timestep embedding
      → N × WaveFieldBlock (with 1D wave kernels)
      → FinalLayer → unpatchify → predicted noise (B, 1, L_samples)

This is the regime where wave field's O(n log n) advantage over softmax
attention's O(n²) actually pays off: at sequence length 1024+ tokens,
standard attention becomes expensive while wave field stays cheap.

Reuses every shared block from wave_field/blocks.py — only patch/unpatch
and the 1D positional embedding differ from the image case.
"""

import torch
import torch.nn as nn

from wave_field.blocks import (
    TimestepEmbedder,
    LabelEmbedder,
    WaveFieldBlock,
    FinalLayer,
    timestep_sinusoidal,
)


def pos_embed_1d_sincos(seq_len: int, dim: int) -> torch.Tensor:
    """1D sinusoidal positional embedding. Returns (1, seq_len, dim)."""
    pe = timestep_sinusoidal(torch.arange(seq_len), dim)   # (seq_len, dim)
    return pe.unsqueeze(0)


class WaveFieldAudioDenoiser(nn.Module):
    """
    Wave Field diffusion backbone for raw audio waveforms.

    Args:
        sequence_length: total samples per clip (e.g., 16384 for ~1 s at 16 kHz)
        in_channels:     audio channels (1 = mono)
        patch_size:      samples per token (16 → 1024 tokens at length 16384)
        dim, depth, num_heads, timestep_dim, conditioning, use_self_cond:
            same semantics as the image WaveFieldDenoiser
    """

    def __init__(
        self,
        sequence_length: int,
        in_channels: int = 1,
        patch_size: int = 16,
        dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        timestep_dim: int = 128,
        conditioning: str = "physics",
        use_self_cond: bool = False,
        dynamic_filter: bool = False,
        gating: str = "pointwise",
        num_classes: int | None = None,
        class_dropout_prob: float = 0.1,
    ):
        super().__init__()
        assert sequence_length % patch_size == 0, (
            f"sequence_length {sequence_length} must be divisible by patch_size {patch_size}"
        )

        self.sequence_length = sequence_length
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_patches = sequence_length // patch_size
        self.patch_dim = in_channels * patch_size
        self.dim = dim
        self.conditioning = conditioning
        self.use_self_cond = use_self_cond
        self.num_classes = num_classes

        # 1D patch embedding via strided Conv1d (doubled channels if self-cond)
        embed_in_ch = in_channels * (2 if use_self_cond else 1)
        self.patch_embed = nn.Conv1d(
            in_channels=embed_in_ch, out_channels=dim,
            kernel_size=patch_size, stride=patch_size,
        )

        self.register_buffer(
            "pos_embed",
            pos_embed_1d_sincos(self.num_patches, dim),
            persistent=False,
        )

        self.time_embed = TimestepEmbedder(timestep_dim)
        if num_classes is not None:
            self.label_embed = LabelEmbedder(num_classes, timestep_dim, class_dropout_prob)

        # Audio is genuinely 1D — use 1D wave kernels
        self.blocks = nn.ModuleList([
            WaveFieldBlock(
                dim=dim, num_heads=num_heads, seq_len=self.num_patches,
                timestep_dim=timestep_dim, conditioning=conditioning,
                use_2d_kernel=False,
                dynamic_filter=dynamic_filter, gating=gating,
            )
            for _ in range(depth)
        ])

        self.final = FinalLayer(dim, self.patch_dim, timestep_dim, conditioning)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, patch_dim) → (B, in_channels, sequence_length)."""
        B, N, _ = x.shape
        P = self.patch_size
        C = self.in_channels
        x = x.view(B, N, C, P).permute(0, 2, 1, 3).contiguous()    # (B, C, N, P)
        return x.view(B, C, N * P)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor,
                self_cond: torch.Tensor | None = None,
                y: torch.Tensor | None = None,
                force_drop_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x_noisy:   (B, in_channels, sequence_length) noisy waveform in [-1, 1]
            t:         (B,) integer timesteps
            self_cond: (B, in_channels, sequence_length) optional x_0 estimate
            y:         (B,) integer class labels (if num_classes set)
            force_drop_ids: (B,) bool — force labels to null (CFG uncond pass)
        Returns:
            (B, in_channels, sequence_length) — predicted noise (or v if v-pred)
        """
        if self.use_self_cond:
            if self_cond is None:
                self_cond = torch.zeros_like(x_noisy)
            x_in = torch.cat([x_noisy, self_cond], dim=1)
        else:
            x_in = x_noisy

        # Conv1d patch embedding: (B, C, L) → (B, dim, N)
        h = self.patch_embed(x_in).transpose(1, 2)        # (B, N, dim)
        h = h + self.pos_embed

        t_emb = self.time_embed(t)
        if self.num_classes is not None and y is not None:
            t_emb = t_emb + self.label_embed(y, self.training, force_drop_ids)
        for block in self.blocks:
            h = block(h, t_emb)

        out = self.final(h, t_emb)
        return self.unpatchify(out)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
