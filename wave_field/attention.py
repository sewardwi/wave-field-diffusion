"""
Wave Field Attention modules.

Core mechanism from Wave Field LLM, adapted for diffusion:
  - 1D non-causal kernel:  k(t) = exp(-α|t|) · cos(ω·t + φ)
  - 2D radial kernel:      k(r) = exp(-α·r) · cos(ω·r + φ),  r = sqrt(x²+y²)
  - Applied via FFT convolution: O(n log n)
  - Gated by content-dependent sigmoid(Q)
  - Per-head learnable α (damping), ω (frequency), φ (phase)

Two timestep conditioning modes (selectable via `conditioning` arg):
  - 'physics'  (Option B): timestep embedding directly modulates α, ω, φ
                           Early timesteps → broad smooth kernels (low ω, high α)
                           Late timesteps  → sharp oscillatory kernels (high ω, low α)
  - 'adaln'    (Option A): conditioning handled externally by WaveFieldBlock (AdaLN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# 1-D Wave Field Attention
# ---------------------------------------------------------------------------

class WaveFieldAttention(nn.Module):
    """
    Non-causal 1D Wave Field Attention.

    Projects x → Q, K, V (K unused in pure wave-field path).
    FFT-convolves V with learned per-head damped wave kernel.
    Gates result with sigmoid(gate_proj(Q)).

    Args:
        dim:           model dimension
        num_heads:     number of attention heads
        seq_len:       sequence length (needed to build kernel grid)
        timestep_dim:  if provided and conditioning=='physics', modulate kernels with t_emb
        conditioning:  'physics' | 'adaln' | None
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        seq_len: int,
        timestep_dim: int | None = None,
        conditioning: str = "physics",
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seq_len = seq_len
        self.conditioning = conditioning

        # Projections — no bias on QKV (standard in modern transformers)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        # Per-head base wave parameters
        # log_alpha: log damping coefficient (exp → positive)
        self.log_alpha = nn.Parameter(torch.zeros(num_heads))
        # omega: angular frequency — init spread across heads for natural specialization
        self.omega = nn.Parameter(torch.linspace(0.5, 4.0, num_heads))
        # phi: phase offset
        self.phi = nn.Parameter(torch.zeros(num_heads))

        # Content-dependent gating: sigmoid(gate_q(q) ⊙ gate_k(k))
        # Multiplicative Q⊙K interaction gives content-aware gating without
        # full O(n²) attention matrix.
        self.gate_q = nn.Linear(self.head_dim, self.head_dim)
        self.gate_k = nn.Linear(self.head_dim, self.head_dim)

        # Frequency-domain content modulation: scales V's spectrum based on
        # a content summary, giving content-conditional convolution.
        # 1 + tanh(...) keeps the modulation around 1 (identity at init).
        self.freq_mod = nn.Linear(dim, num_heads)

        # Physics conditioning: small MLP maps t_emb → (Δlog_α, Δω, Δφ) per head
        self.use_ts_cond = conditioning == "physics" and timestep_dim is not None
        if self.use_ts_cond:
            self.ts_to_params = nn.Sequential(
                nn.Linear(timestep_dim, timestep_dim),
                nn.SiLU(),
                nn.Linear(timestep_dim, 3 * num_heads),
            )
            # Zero-init last layer so perturbation starts at 0
            nn.init.zeros_(self.ts_to_params[-1].weight)
            nn.init.zeros_(self.ts_to_params[-1].bias)

    def _build_kernel(self, t_emb=None):
        """
        Construct the per-head wave kernel in circular (FFT-compatible) layout.

        Circular layout: indices [0, 1, ..., L//2, -(L//2-1), ..., -1]
        corresponds to lags [0, +1, ..., +L//2, -(L//2-1), ..., -1].

        Returns:
            kernel: (B, H, L) if batched else (H, L)
            batched: bool
        """
        L = self.seq_len
        device = self.log_alpha.device

        alpha = torch.exp(self.log_alpha)   # (H,)  always positive
        omega = self.omega                   # (H,)
        phi = self.phi                       # (H,)

        batched = self.use_ts_cond and t_emb is not None
        if batched:
            B = t_emb.shape[0]
            params = self.ts_to_params(t_emb)                  # (B, 3H)
            d_log_alpha, d_omega, d_phi = params.view(B, 3, self.num_heads).unbind(dim=1)
            # Multiply alpha by exp(delta) so modulation is multiplicative and stays positive
            alpha = alpha.unsqueeze(0) * torch.exp(d_log_alpha)  # (B, H)
            omega = omega.unsqueeze(0) + d_omega                  # (B, H)
            phi   = phi.unsqueeze(0)   + d_phi                    # (B, H)

        # Circular time grid: lag t[n] for n = 0..L-1
        t = torch.arange(L, dtype=torch.float32, device=device)
        t = torch.where(t <= L // 2, t, t - L)   # (L,)  values in [-(L//2-1), L//2]
        t_abs = t.abs()

        if batched:
            # (B, H, 1) × (1, 1, L) → (B, H, L)
            a = alpha[:, :, None]
            w = omega[:, :, None]
            p = phi[:, :, None]
            t_     = t[None, None, :]
            t_abs_ = t_abs[None, None, :]
        else:
            # (H, 1) × (1, L) → (H, L)
            a = alpha[:, None]
            w = omega[:, None]
            p = phi[:, None]
            t_     = t[None, :]
            t_abs_ = t_abs[None, :]

        kernel = torch.exp(-a * t_abs_) * torch.cos(w * t_ + p)
        # L1-normalize each kernel — keeps output magnitude stable across heads
        # and timesteps regardless of (α, ω). "Energy conservation" under the kernel.
        kernel_norm = kernel.abs().sum(dim=-1, keepdim=True).clamp(min=1e-6)
        kernel = kernel / kernel_norm
        return kernel, batched

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:     (B, L, D) input sequence
            t_emb: (B, timestep_dim) optional timestep embedding
        Returns:
            out:   (B, L, D)
        """
        B, L, D = x.shape
        H = self.num_heads
        Dh = self.head_dim

        qkv = self.qkv(x)                               # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)                  # each (B, L, D)

        # Reshape to multi-head: (B, H, L, Dh)
        q = q.view(B, L, H, Dh).permute(0, 2, 1, 3)
        k = k.view(B, L, H, Dh).permute(0, 2, 1, 3)
        v = v.view(B, L, H, Dh).permute(0, 2, 1, 3)

        # Build kernel and FFT-convolve V along sequence dimension
        kernel, batched = self._build_kernel(t_emb)

        # FFT convolution runs in fp32: torch.fft has no bf16/fp16 kernels, so
        # under autocast `v`/`kernel` arrive as half precision and rfft raises.
        # Cast the FFT operands up and restore the surrounding dtype afterwards.
        in_dtype = v.dtype
        V_fft = torch.fft.rfft(v.float(), n=L, dim=2)  # (B, H, L//2+1, Dh)

        # Frequency-domain content modulation: scale V's spectrum per-head
        # using a global content summary. Identity-init via 1 + tanh(...).
        content_summary = x.mean(dim=1)                              # (B, D)
        freq_scale = (1.0 + torch.tanh(self.freq_mod(content_summary))).float()  # (B, H)
        V_fft = V_fft * freq_scale[:, :, None, None]                 # broadcast

        kernel = kernel.float()
        if batched:
            k_fft = torch.fft.rfft(kernel, n=L, dim=2) # (B, H, L//2+1)
            out_fft = V_fft * k_fft.unsqueeze(-1)       # (B, H, L//2+1, Dh)
        else:
            k_fft = torch.fft.rfft(kernel, n=L, dim=1) # (H, L//2+1)
            out_fft = V_fft * k_fft[None, :, :, None]  # (B, H, L//2+1, Dh)

        out = torch.fft.irfft(out_fft, n=L, dim=2).to(in_dtype)   # (B, H, L, Dh)

        # Q⊙K content-dependent gating (no full attention matrix)
        gate = torch.sigmoid(self.gate_q(q) * self.gate_k(k))   # (B, H, L, Dh)
        out = out * gate

        # Merge heads and project
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# 2-D Wave Field Attention (Phase 2 — CIFAR-10 spatial kernels)
# ---------------------------------------------------------------------------

class WaveFieldAttention2D(nn.Module):
    """
    2D Wave Field Attention for spatial image generation (Phase 2).

    Radially-symmetric kernel:
        k(x, y) = exp(-α · sqrt(x²+y²)) · cos(ω · sqrt(x²+y²) + φ)

    Operates on a flattened (H*W) spatial sequence while maintaining 2D
    spatial structure for the FFT convolution.

    Args:
        dim:        model dimension
        num_heads:  number of attention heads
        height:     spatial height (in patches)
        width:      spatial width (in patches)
        timestep_dim, conditioning: same as WaveFieldAttention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        height: int,
        width: int,
        timestep_dim: int | None = None,
        conditioning: str = "physics",
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.height = height
        self.width = width
        self.conditioning = conditioning

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        # Init log_alpha much smaller than 1D: max radial distance on an H×W grid
        # is sqrt((H-1)²+(W-1)²). With α=exp(0)=1 most of the kernel is already
        # ~zero at that range, causing collapse. Start at α≈0.1 (log_alpha=-2.3).
        self.log_alpha = nn.Parameter(torch.full((num_heads,), -2.3))
        self.omega = nn.Parameter(torch.linspace(0.5, 4.0, num_heads))
        self.phi = nn.Parameter(torch.zeros(num_heads))

        self.gate_q = nn.Linear(self.head_dim, self.head_dim)
        self.gate_k = nn.Linear(self.head_dim, self.head_dim)

        # Frequency-domain content modulation
        self.freq_mod = nn.Linear(dim, num_heads)

        self.use_ts_cond = conditioning == "physics" and timestep_dim is not None
        if self.use_ts_cond:
            self.ts_to_params = nn.Sequential(
                nn.Linear(timestep_dim, timestep_dim),
                nn.SiLU(),
                nn.Linear(timestep_dim, 3 * num_heads),
            )
            nn.init.zeros_(self.ts_to_params[-1].weight)
            nn.init.zeros_(self.ts_to_params[-1].bias)

    def _build_kernel_2d(self, t_emb=None):
        """
        Build 2D radially-symmetric wave kernel in circular (FFT-compatible) layout.

        Returns:
            kernel: (B, H, Gy, Gx) if batched else (H, Gy, Gx)
            batched: bool
        """
        Gy, Gx = self.height, self.width
        device = self.log_alpha.device

        alpha = torch.exp(self.log_alpha)
        omega = self.omega
        phi = self.phi

        batched = self.use_ts_cond and t_emb is not None
        if batched:
            B = t_emb.shape[0]
            params = self.ts_to_params(t_emb).view(B, 3, self.num_heads)
            d_log_alpha, d_omega, d_phi = params.unbind(dim=1)
            alpha = alpha.unsqueeze(0) * torch.exp(d_log_alpha)   # (B, H)
            omega = omega.unsqueeze(0) + d_omega
            phi   = phi.unsqueeze(0)   + d_phi

        # 2D circular grids for y and x
        ty = torch.arange(Gy, dtype=torch.float32, device=device)
        tx = torch.arange(Gx, dtype=torch.float32, device=device)
        ty = torch.where(ty <= Gy // 2, ty, ty - Gy)
        tx = torch.where(tx <= Gx // 2, tx, tx - Gx)
        ty_grid, tx_grid = torch.meshgrid(ty, tx, indexing="ij")   # (Gy, Gx)
        r = torch.sqrt(ty_grid ** 2 + tx_grid ** 2)                # (Gy, Gx)

        if batched:
            # (B, H, 1, 1) × (1, 1, Gy, Gx)
            a = alpha[:, :, None, None]
            w = omega[:, :, None, None]
            p = phi[:, :, None, None]
            r_ = r[None, None, :, :]
        else:
            a = alpha[:, None, None]
            w = omega[:, None, None]
            p = phi[:, None, None]
            r_ = r[None, :, :]

        kernel = torch.exp(-a * r_) * torch.cos(w * r_ + p)
        # L1-normalize the 2D kernel for stable output magnitude
        kernel_norm = kernel.abs().sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        kernel = kernel / kernel_norm
        return kernel, batched

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:     (B, Gy*Gx, D) flattened spatial sequence
            t_emb: (B, timestep_dim) optional
        Returns:
            out:   (B, Gy*Gx, D)
        """
        B, L, D = x.shape
        Gy, Gx = self.height, self.width
        assert L == Gy * Gx
        H = self.num_heads
        Dh = self.head_dim

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, H, L, Dh)
        q = q.view(B, L, H, Dh).permute(0, 2, 1, 3)
        k = k.view(B, L, H, Dh).permute(0, 2, 1, 3)
        v = v.view(B, L, H, Dh).permute(0, 2, 1, 3)

        # Reshape v to 2D spatial, then permute for rfft2: (B, H, Dh, Gy, Gx)
        v_2d = v.view(B, H, Gy, Gx, Dh).permute(0, 1, 4, 2, 3)

        # FFT convolution runs in fp32: torch.fft has no bf16/fp16 kernels, so
        # under autocast `v`/`kernel` arrive as half precision and rfft2 raises.
        in_dtype = v.dtype
        V_fft = torch.fft.rfft2(v_2d.float(), s=(Gy, Gx))    # (B, H, Dh, Gy, Gx//2+1)

        # Frequency-domain content modulation
        content_summary = x.mean(dim=1)                              # (B, D)
        freq_scale = (1.0 + torch.tanh(self.freq_mod(content_summary))).float()  # (B, H)
        V_fft = V_fft * freq_scale[:, :, None, None, None]

        kernel, batched = self._build_kernel_2d(t_emb)
        k_fft = torch.fft.rfft2(kernel.float(), s=(Gy, Gx))
        if batched:
            # kernel: (B, H, Gy, Gx) → rfft2 → (B, H, Gy, Gx//2+1)
            # Broadcast over Dh: (B, H, 1, Gy, Gx//2+1)
            out_fft = V_fft * k_fft[:, :, None, :, :]
        else:
            # kernel: (H, Gy, Gx) → rfft2 → (H, Gy, Gx//2+1)
            out_fft = V_fft * k_fft[None, :, None, :, :]

        out_2d = torch.fft.irfft2(out_fft, s=(Gy, Gx)).to(in_dtype)   # (B, H, Dh, Gy, Gx)

        # Back to (B, H, Gy, Gx, Dh) for gating
        out_2d = out_2d.permute(0, 1, 3, 4, 2)

        # Q⊙K content gating (2D-reshaped)
        q_2d = q.view(B, H, Gy, Gx, Dh)
        k_2d = k.view(B, H, Gy, Gx, Dh)
        gate = torch.sigmoid(self.gate_q(q_2d) * self.gate_k(k_2d))
        out_2d = out_2d * gate

        # Back to sequence: (B, L, D)
        out = out_2d.view(B, H, L, Dh).permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out)
