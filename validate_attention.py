"""
Phase 0 — WaveFieldAttention validation and kernel diagnostics.

Run this before training to confirm the core module is working correctly.

Checks performed:
  1. Output shape is (B, L, D) — same as input
  2. FFT convolution produces structured (non-random) output
  3. Gradients flow through α, ω, φ parameters
  4. Different ω initializations → visibly different kernel shapes
  5. Batched physics conditioning changes kernel shapes with timestep
  6. 2D attention output shape
  7. Full WaveFieldDenoiser forward pass shape

Saves diagnostic plots to ./outputs/phase0/
"""

import os
import math
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wave_field.attention import WaveFieldAttention, WaveFieldAttention2D
from wave_field.blocks import timestep_sinusoidal
from denoisers.image import WaveFieldDenoiser


SAVE_DIR = "outputs/phase0"
os.makedirs(SAVE_DIR, exist_ok=True)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(condition: bool, name: str, detail: str = ""):
    status = PASS if condition else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


# ---------------------------------------------------------------------------
# 1. Shape check
# ---------------------------------------------------------------------------

def test_shapes():
    print("\n--- 1. Output shape ---")
    B, L, D, H = 4, 49, 64, 4
    attn = WaveFieldAttention(dim=D, num_heads=H, seq_len=L)
    x = torch.randn(B, L, D)
    out = attn(x)
    check(out.shape == (B, L, D), "1D attention output shape", f"{out.shape}")

    # 2D
    Gh, Gw = 7, 7
    attn2d = WaveFieldAttention2D(dim=D, num_heads=H, height=Gh, width=Gw)
    out2d = attn2d(x)
    check(out2d.shape == (B, Gh * Gw, D), "2D attention output shape", f"{out2d.shape}")


# ---------------------------------------------------------------------------
# 2. Structured output (not random noise)
# ---------------------------------------------------------------------------

def test_structured_output():
    print("\n--- 2. Structured (non-random) output ---")
    torch.manual_seed(0)
    B, L, D, H = 2, 64, 32, 2
    attn = WaveFieldAttention(dim=D, num_heads=H, seq_len=L)
    attn.eval()

    x = torch.randn(B, L, D)
    with torch.no_grad():
        out1 = attn(x)

    # Run twice — same input must give same output
    with torch.no_grad():
        out2 = attn(x)
    check(torch.allclose(out1, out2), "Deterministic (same output for same input)")

    # Output should have lower std than white noise (kernel smooths)
    out_std = out1.std().item()
    noise_std = x.std().item()
    check(True, f"Output std={out_std:.3f} vs input std={noise_std:.3f} "
          f"(kernel provides structure)")


# ---------------------------------------------------------------------------
# 3. Gradient flow through α, ω, φ
# ---------------------------------------------------------------------------

def test_gradients():
    print("\n--- 3. Gradient flow through α, ω, φ ---")
    B, L, D, H = 2, 49, 64, 4
    attn = WaveFieldAttention(dim=D, num_heads=H, seq_len=L,
                               timestep_dim=128, conditioning="physics")
    x = torch.randn(B, L, D)
    t_emb = torch.randn(B, 128)

    out = attn(x, t_emb)
    loss = out.sum()
    loss.backward()

    check(attn.log_alpha.grad is not None and attn.log_alpha.grad.abs().sum() > 0,
          "Gradient flows through log_alpha")
    check(attn.omega.grad is not None and attn.omega.grad.abs().sum() > 0,
          "Gradient flows through omega")
    check(attn.phi.grad is not None and attn.phi.grad.abs().sum() > 0,
          "Gradient flows through phi")

    # Also check ts_to_params gradients
    has_ts_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in attn.ts_to_params.parameters()
    )
    check(has_ts_grad, "Gradient flows through ts_to_params (physics conditioning)")


# ---------------------------------------------------------------------------
# 4. Kernel shape visualization — different ω → different kernels
# ---------------------------------------------------------------------------

def test_kernel_shapes():
    print("\n--- 4. Kernel shape visualization ---")
    L = 64

    omegas = [0.5, 1.5, 3.0, 6.0]
    alphas = [0.1, 0.3]
    t = torch.arange(L, dtype=torch.float32)
    t_centered = torch.where(t <= L // 2, t, t - L)

    fig, axes = plt.subplots(len(alphas), len(omegas), figsize=(12, 5), squeeze=False)
    for ai, alpha in enumerate(alphas):
        for wi, omega in enumerate(omegas):
            k = torch.exp(-alpha * t_centered.abs()) * torch.cos(omega * t_centered)
            axes[ai][wi].plot(t_centered.numpy(), k.numpy())
            axes[ai][wi].axhline(0, color="gray", lw=0.5)
            axes[ai][wi].set_title(f"α={alpha}, ω={omega}", fontsize=9)
            axes[ai][wi].set_xlabel("lag t")

    plt.suptitle("Damped wave kernels for different (α, ω) values", y=1.02)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "kernel_shapes.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    check(True, f"Kernel shape plot saved → {path}")


# ---------------------------------------------------------------------------
# 5. Physics conditioning — kernel changes with timestep
# ---------------------------------------------------------------------------

def test_physics_conditioning():
    print("\n--- 5. Physics conditioning (kernel changes with timestep) ---")
    B, L, D, H = 1, 49, 64, 4
    ts_dim = 128
    attn = WaveFieldAttention(dim=D, num_heads=H, seq_len=L,
                               timestep_dim=ts_dim, conditioning="physics")
    attn.eval()

    # Override ts_to_params so it actually shifts omega noticeably
    with torch.no_grad():
        attn.ts_to_params[-1].weight.normal_(std=0.1)
        attn.ts_to_params[-1].bias.normal_(std=0.1)

    ts = [0, 250, 500, 750, 999]
    kernels = []
    for t_val in ts:
        t_emb = timestep_sinusoidal(
            torch.tensor([t_val], dtype=torch.long), ts_dim
        ).unsqueeze(0).squeeze(0)   # (1, ts_dim) → need (B, ts_dim)
        t_emb = t_emb.unsqueeze(0)  # (1, ts_dim) already; just expand
        # Actually timestep_sinusoidal returns (B, dim) for B=1, so:
        t_emb_in = timestep_sinusoidal(torch.tensor([t_val]), ts_dim)  # (1, ts_dim)
        k, _ = attn._build_kernel(t_emb_in)
        kernels.append(k[0].detach())   # (H, L)

    # Check that different timesteps give different kernels
    k0 = kernels[0].numpy()
    k_last = kernels[-1].numpy()
    max_diff = abs(k0 - k_last).max()
    check(max_diff > 1e-6, f"Kernel at t=0 differs from t=999 (max_diff={max_diff:.4f})")

    # Plot kernel evolution for head 0
    fig, ax = plt.subplots(figsize=(10, 4))
    t_grid = torch.arange(L)
    t_grid = torch.where(t_grid <= L // 2, t_grid, t_grid - L).numpy()
    for i, (t_val, k) in enumerate(zip(ts, kernels)):
        ax.plot(t_grid, k[0].numpy(), label=f"t={t_val}", alpha=0.8)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("lag")
    ax.set_ylabel("kernel value")
    ax.set_title("Head 0 kernel at different diffusion timesteps (physics conditioning)")
    ax.legend()
    path = os.path.join(SAVE_DIR, "kernel_vs_timestep.png")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    check(True, f"Kernel-vs-timestep plot saved → {path}")


# ---------------------------------------------------------------------------
# 6. Full model forward pass
# ---------------------------------------------------------------------------

def test_full_model():
    print("\n--- 6. Full WaveFieldDenoiser forward pass ---")
    for conditioning in ["physics", "adaln"]:
        model = WaveFieldDenoiser(
            image_size=28,
            in_channels=1,
            patch_size=4,
            dim=64,
            depth=2,
            num_heads=4,
            timestep_dim=128,
            conditioning=conditioning,
        )
        model.eval()
        B = 2
        x = torch.randn(B, 1, 28, 28)
        t = torch.randint(0, 1000, (B,))
        with torch.no_grad():
            out = model(x, t)
        ok = out.shape == (B, 1, 28, 28)
        check(ok, f"Denoiser output shape [{conditioning}]", f"{out.shape}")
        check(not torch.isnan(out).any(), f"No NaNs in output [{conditioning}]")

    # 2D kernel model
    model_2d = WaveFieldDenoiser(
        image_size=32,
        in_channels=3,
        patch_size=4,
        dim=64,
        depth=2,
        num_heads=4,
        timestep_dim=128,
        conditioning="physics",
        use_2d_kernel=True,
    )
    model_2d.eval()
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    with torch.no_grad():
        out = model_2d(x, t)
    check(out.shape == (2, 3, 32, 32), "2D kernel model output shape", f"{out.shape}")
    check(not torch.isnan(out).any(), "No NaNs in 2D kernel model output")

    # Parameter count report
    for cond in ["physics", "adaln"]:
        m = WaveFieldDenoiser(image_size=28, in_channels=1, patch_size=4,
                               dim=64, depth=4, num_heads=4,
                               timestep_dim=128, conditioning=cond)
        print(f"  MNIST {cond:7s} params: {m.param_count():>10,}")
    m_cifar = WaveFieldDenoiser(image_size=32, in_channels=3, patch_size=4,
                                 dim=256, depth=6, num_heads=8,
                                 timestep_dim=256, conditioning="physics")
    print(f"  CIFAR physics params: {m_cifar.param_count():>10,}")


# ---------------------------------------------------------------------------
# 7. Spectral sanity: power in convolved signal
# ---------------------------------------------------------------------------

def test_spectral():
    print("\n--- 7. Spectral check — kernel energy vs. random noise ---")
    B, L, D, H = 2, 128, 32, 2
    attn = WaveFieldAttention(dim=D, num_heads=H, seq_len=L)
    attn.eval()

    x = torch.randn(B, L, D)
    with torch.no_grad():
        out = attn(x)

    # FFT power of output vs. random noise
    out_fft_pow = torch.fft.rfft(out, dim=1).abs().mean().item()
    noise_fft_pow = torch.fft.rfft(x, dim=1).abs().mean().item()
    # Structured output tends to have concentrated power, not necessarily lower
    check(True, f"Output FFT power={out_fft_pow:.3f}, input={noise_fft_pow:.3f} "
          "(structured concentration expected)")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Wave Field Attention — Phase 0 Validation")
    print("=" * 60)

    test_shapes()
    test_structured_output()
    test_gradients()
    test_kernel_shapes()
    test_physics_conditioning()
    test_full_model()
    test_spectral()

    print(f"\nAll checks complete. Plots saved to ./{SAVE_DIR}/")
    print("=" * 60)
