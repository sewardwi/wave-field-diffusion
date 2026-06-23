"""
Shared building blocks for wave field diffusion denoisers.

These are *task-agnostic* — used by both image (denoisers/image.py) and audio
(denoisers/audio.py) denoisers.  Anything specific to a data shape (patchify,
positional embedding for that shape, final unpatchify) belongs in the
corresponding denoiser module, not here.

Contents:
  - timestep_sinusoidal:       Ho-2020-style sinusoidal timestep encoding
  - pos_embed_2d_sincos:       2D factorized sin/cos positional embedding
  - TimestepEmbedder:          sin encoding → 2-layer MLP
  - modulate:                  AdaLN scale-shift helper
  - WaveFieldBlock:            one transformer block with wave field attention
  - FinalLayer:                final norm + linear prediction head
"""

import math
import torch
import torch.nn as nn

from .attention import WaveFieldAttention, WaveFieldAttention2D


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------

def timestep_sinusoidal(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Sinusoidal timestep embedding (Ho et al. 2020).
    t: (B,) integer or float timesteps
    Returns: (B, dim)
    """
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def pos_embed_2d_sincos(grid_h: int, grid_w: int, dim: int) -> torch.Tensor:
    """
    Factorized 2D sin/cos positional embedding (ViT-style).
    Encodes y-coordinate in first half of dim, x-coordinate in second half.
    Returns: (1, grid_h * grid_w, dim)
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sin/cos pos embed"
    half = dim // 2
    y_emb = timestep_sinusoidal(torch.arange(grid_h), half)   # (gh, half)
    x_emb = timestep_sinusoidal(torch.arange(grid_w), half)   # (gw, half)
    y_grid = y_emb[:, None, :].expand(grid_h, grid_w, half)
    x_grid = x_emb[None, :, :].expand(grid_h, grid_w, half)
    pos = torch.cat([y_grid, x_grid], dim=-1)                 # (gh, gw, dim)
    return pos.reshape(1, grid_h * grid_w, dim)


class TimestepEmbedder(nn.Module):
    """Sinusoidal encoding → 2-layer MLP → timestep_dim embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_sinusoidal(t, self.dim))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift. shift/scale are (B, dim)."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into a vector, with label dropout for classifier-free
    guidance (Ho & Salimans 2022; DiT, Peebles & Xie 2023).

    An extra embedding row at index `num_classes` is the learned "null" / un-
    conditional token. During training, labels are randomly replaced by null
    with probability `dropout_prob` so the same network learns both the
    conditional and unconditional scores needed for guidance at sampling time.
    """

    def __init__(self, num_classes: int, dim: int, dropout_prob: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embedding_table = nn.Embedding(num_classes + 1, dim)

    def token_drop(self, labels: torch.Tensor, force_drop_ids: torch.Tensor | None = None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids.bool()
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels: torch.Tensor, train: bool,
                force_drop_ids: torch.Tensor | None = None) -> torch.Tensor:
        if (train and self.dropout_prob > 0) or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


# ---------------------------------------------------------------------------
# Wave Field Block
# ---------------------------------------------------------------------------

class WaveFieldBlock(nn.Module):
    """
    One transformer block with Wave Field Attention.

    conditioning='physics':
        - WaveFieldAttention internally modulates kernels via t_emb
        - FFN gets a FiLM scale+shift from t_emb
        - Both branches have time-conditioned residual gates
    conditioning='adaln':
        - DiT AdaLN-Zero: 6 modulation values per block (shift/scale/gate
          for attention branch + FFN branch)
        - Attention does not receive t_emb internally
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        seq_len: int,
        timestep_dim: int,
        conditioning: str = "physics",
        use_2d_kernel: bool = False,
        height: int | None = None,
        width: int | None = None,
        dynamic_filter: bool = False,
        gating: str = "pointwise",
        aniso_kernel: bool = False,
    ):
        super().__init__()
        self.conditioning = conditioning

        # LayerNorm — no learned affine when using AdaLN (affine handled externally)
        use_affine = conditioning != "adaln"
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=use_affine)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=use_affine)

        # Wave Field Attention (1D or 2D)
        wfa_ts_dim = timestep_dim if conditioning == "physics" else None
        if use_2d_kernel:
            assert height is not None and width is not None
            self.attn = WaveFieldAttention2D(
                dim=dim, num_heads=num_heads,
                height=height, width=width,
                timestep_dim=wfa_ts_dim, conditioning=conditioning,
                dynamic_filter=dynamic_filter, gating=gating,
                aniso_kernel=aniso_kernel,
            )
        else:
            self.attn = WaveFieldAttention(
                dim=dim, num_heads=num_heads, seq_len=seq_len,
                timestep_dim=wfa_ts_dim, conditioning=conditioning,
                dynamic_filter=dynamic_filter, gating=gating,
            )

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

        if conditioning == "adaln":
            # 6 mod values: (shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn)
            self.adaln_mod = nn.Sequential(
                nn.SiLU(),
                nn.Linear(timestep_dim, 6 * dim),
            )
            nn.init.zeros_(self.adaln_mod[-1].weight)
            nn.init.zeros_(self.adaln_mod[-1].bias)
        else:
            # Physics mode: FFN FiLM + time-conditioned residual gates
            self.ffn_film = nn.Sequential(
                nn.SiLU(),
                nn.Linear(timestep_dim, 2 * dim),
            )
            nn.init.zeros_(self.ffn_film[-1].weight)
            nn.init.zeros_(self.ffn_film[-1].bias)
            self.res_gates = nn.Sequential(
                nn.SiLU(),
                nn.Linear(timestep_dim, 2 * dim),
            )
            nn.init.zeros_(self.res_gates[-1].weight)
            nn.init.zeros_(self.res_gates[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.conditioning == "adaln":
            mods = self.adaln_mod(t_emb)
            shift1, scale1, gate1, shift2, scale2, gate2 = mods.chunk(6, dim=-1)
            x = x + gate1.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift1, scale1))
            x = x + gate2.unsqueeze(1) * self.ffn(modulate(self.norm2(x), shift2, scale2))
        else:
            res = self.res_gates(t_emb)
            g_attn, g_ffn = res.chunk(2, dim=-1)
            g_attn = (1.0 + g_attn).unsqueeze(1)
            g_ffn = (1.0 + g_ffn).unsqueeze(1)
            x = x + g_attn * self.attn(self.norm1(x), t_emb)
            film = self.ffn_film(t_emb)
            scale_ffn, shift_ffn = film.chunk(2, dim=-1)
            x = x + g_ffn * self.ffn(modulate(self.norm2(x), shift_ffn, scale_ffn))
        return x


# ---------------------------------------------------------------------------
# Final prediction layer
# ---------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """Last norm + linear that produces patch-level predictions."""

    def __init__(self, dim: int, out_dim: int, timestep_dim: int, conditioning: str):
        super().__init__()
        self.conditioning = conditioning
        use_affine = conditioning != "adaln"
        self.norm = nn.LayerNorm(dim, elementwise_affine=use_affine)
        self.linear = nn.Linear(dim, out_dim)

        if conditioning == "adaln":
            self.adaln_mod = nn.Sequential(
                nn.SiLU(),
                nn.Linear(timestep_dim, 2 * dim),
            )
            nn.init.zeros_(self.adaln_mod[-1].weight)
            nn.init.zeros_(self.adaln_mod[-1].bias)

        # Zero-init linear → starts predicting zero (stable for diffusion)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.conditioning == "adaln":
            shift, scale = self.adaln_mod(t_emb).chunk(2, dim=-1)
            x = modulate(self.norm(x), shift, scale)
        else:
            x = self.norm(x)
        return self.linear(x)
