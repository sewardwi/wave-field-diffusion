"""
WaveFieldDenoiser: full diffusion backbone using Wave Field Attention.

Architecture (DiT-inspired):
    Noisy image
      → patchify (P×P → token per patch)
      → linear patch embedding + learned positional embedding
      → timestep embedding (sinusoidal → MLP)
      → N × WaveFieldBlock
      → FinalLayer (AdaLN-style norm + linear)
      → unpatchify → predicted noise ε

Supports two timestep conditioning modes:
  'physics'  (Option B, plan §Phase 1):
      Timestep embedding directly modulates α, ω, φ of wave kernels.
      FFN gets a lightweight FiLM scale/shift from the timestep.
  'adaln'    (Option A, plan §Phase 1):
      DiT-style AdaLN-Zero: 6 modulation values per block (shift/scale/gate
      for attention branch + FFN branch).  Attention uses no internal kernel
      conditioning.

Spatial variant (use_2d_kernel=True, for Phase 2):
    WaveFieldAttention2D is used instead of 1D, operating on the H×W patch
    grid with radially-symmetric 2D kernels.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import WaveFieldAttention, WaveFieldAttention2D


# ---------------------------------------------------------------------------
# Helpers
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
    # Broadcast to (gh, gw, half) each
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
        emb = timestep_sinusoidal(t, self.dim)
        return self.mlp(emb)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift. shift/scale are (B, dim)."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Wave Field Block
# ---------------------------------------------------------------------------

class WaveFieldBlock(nn.Module):
    """
    One transformer block with Wave Field Attention.

    conditioning='physics':
        - WaveFieldAttention internally modulates kernels via t_emb
        - FFN gets a FiLM scale+shift from t_emb (lightweight, no gating)
    conditioning='adaln':
        - DiT AdaLN-Zero: modulation values from t_emb for attn + FFN branches
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
                dim=dim,
                num_heads=num_heads,
                height=height,
                width=width,
                timestep_dim=wfa_ts_dim,
                conditioning=conditioning,
            )
        else:
            self.attn = WaveFieldAttention(
                dim=dim,
                num_heads=num_heads,
                seq_len=seq_len,
                timestep_dim=wfa_ts_dim,
                conditioning=conditioning,
            )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

        if conditioning == "adaln":
            # 6 modulation values: (shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn)
            # Zero-init → identity residuals at start (DiT AdaLN-Zero)
            self.adaln_mod = nn.Sequential(
                nn.SiLU(),
                nn.Linear(timestep_dim, 6 * dim),
            )
            nn.init.zeros_(self.adaln_mod[-1].weight)
            nn.init.zeros_(self.adaln_mod[-1].bias)
        else:
            # Physics mode: FiLM for FFN + time-conditioned residual gates for
            # both attention and FFN branches.  Lets the block attenuate or
            # amplify each branch as a function of the diffusion timestep.
            self.ffn_film = nn.Sequential(
                nn.SiLU(),
                nn.Linear(timestep_dim, 2 * dim),
            )
            nn.init.zeros_(self.ffn_film[-1].weight)
            nn.init.zeros_(self.ffn_film[-1].bias)
            # Two scalar gates (per channel) — one per residual branch
            self.res_gates = nn.Sequential(
                nn.SiLU(),
                nn.Linear(timestep_dim, 2 * dim),
            )
            nn.init.zeros_(self.res_gates[-1].weight)
            nn.init.zeros_(self.res_gates[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.conditioning == "adaln":
            mods = self.adaln_mod(t_emb)                         # (B, 6*dim)
            shift1, scale1, gate1, shift2, scale2, gate2 = mods.chunk(6, dim=-1)
            # Attention branch
            x = x + gate1.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift1, scale1))
            # FFN branch
            x = x + gate2.unsqueeze(1) * self.ffn(modulate(self.norm2(x), shift2, scale2))
        else:
            # Time-conditioned residual gates: (1 + g) per branch, identity-init
            res = self.res_gates(t_emb)                          # (B, 2*dim)
            g_attn, g_ffn = res.chunk(2, dim=-1)
            g_attn = (1.0 + g_attn).unsqueeze(1)                 # (B, 1, dim)
            g_ffn  = (1.0 + g_ffn).unsqueeze(1)
            # Attention with physics-modulated kernels
            x = x + g_attn * self.attn(self.norm1(x), t_emb)
            # FFN with FiLM from timestep
            film = self.ffn_film(t_emb)                          # (B, 2*dim)
            scale_ffn, shift_ffn = film.chunk(2, dim=-1)
            x = x + g_ffn * self.ffn(modulate(self.norm2(x), shift_ffn, scale_ffn))

        return x


# ---------------------------------------------------------------------------
# Final prediction layer
# ---------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """Last norm + linear that predicts noise patches."""

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

        # Zero-init linear to start from zero prediction
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.conditioning == "adaln":
            shift, scale = self.adaln_mod(t_emb).chunk(2, dim=-1)
            x = modulate(self.norm(x), shift, scale)
        else:
            x = self.norm(x)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Full denoiser
# ---------------------------------------------------------------------------

class WaveFieldDenoiser(nn.Module):
    """
    Wave Field Diffusion Model — full denoising backbone.

    Takes (noisy_image, timestep) and predicts the added noise ε.

    Args:
        image_size:    int or (H, W) — must be divisible by patch_size
        in_channels:   image channels (1=MNIST, 3=CIFAR)
        patch_size:    patch side length (default 4)
        dim:           model dimension
        depth:         number of WaveFieldBlocks
        num_heads:     attention heads
        timestep_dim:  timestep embedding dimension
        conditioning:  'physics' or 'adaln'
        use_2d_kernel: use 2D radial wave kernels (Phase 2)
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

        # Patch geometry
        self.ph = H // patch_size   # number of patches vertically
        self.pw = W // patch_size   # number of patches horizontally
        self.num_patches = self.ph * self.pw
        self.patch_dim = in_channels * patch_size * patch_size

        # Patch embedding: flat patch → model dim
        self.patch_embed = nn.Linear(self.patch_dim, dim)
        # 2D sin/cos positional embedding — gives spatial layout for free
        # (vs. learned 1D embedding that has to discover row/column structure)
        self.register_buffer(
            "pos_embed",
            pos_embed_2d_sincos(self.ph, self.pw, dim),
            persistent=False,
        )

        # Timestep embedder
        self.time_embed = TimestepEmbedder(timestep_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            WaveFieldBlock(
                dim=dim,
                num_heads=num_heads,
                seq_len=self.num_patches,
                timestep_dim=timestep_dim,
                conditioning=conditioning,
                use_2d_kernel=use_2d_kernel,
                height=self.ph,
                width=self.pw,
            )
            for _ in range(depth)
        ])

        # Final prediction
        self.final = FinalLayer(dim, self.patch_dim, timestep_dim, conditioning)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)

    # ------------------------------------------------------------------
    # Patch utilities
    # ------------------------------------------------------------------

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, patch_dim) where N = ph * pw."""
        B, C, H, W = x.shape
        P = self.patch_size
        # Rearrange into patches
        x = x.reshape(B, C, self.ph, P, self.pw, P)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()   # (B, ph, pw, C, P, P)
        x = x.view(B, self.num_patches, self.patch_dim)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, patch_dim) → (B, C, H, W)."""
        B = x.shape[0]
        P = self.patch_size
        C = self.in_channels
        x = x.view(B, self.ph, self.pw, C, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()   # (B, C, ph, P, pw, P)
        x = x.view(B, C, self.ph * P, self.pw * P)
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_noisy: (B, C, H, W) noisy image in [-1, 1]
            t:       (B,) integer diffusion timesteps in [0, T-1]
        Returns:
            eps_pred: (B, C, H, W) predicted noise
        """
        # Embed patches
        h = self.patch_embed(self.patchify(x_noisy)) + self.pos_embed  # (B, N, dim)

        # Embed timestep
        t_emb = self.time_embed(t)   # (B, timestep_dim)

        # Apply blocks
        for block in self.blocks:
            h = block(h, t_emb)

        # Predict noise
        out = self.final(h, t_emb)           # (B, N, patch_dim)
        return self.unpatchify(out)          # (B, C, H, W)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
