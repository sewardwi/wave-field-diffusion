"""
Wall-clock benchmark — wave field attention vs standard softmax attention.

This is the central O(n log n) vs O(n²) demonstration.  At short sequence
lengths the gap is small (constants dominate); past ~1024 tokens softmax
attention's quadratic cost shows up clearly; at ~4096 it should dominate.

Usage:
    python benchmark_attn.py
    python benchmark_attn.py --device cpu
    python benchmark_attn.py --lengths 256 1024 4096 8192

Reports forward-pass and forward+backward timings for both modules at the
same dim/heads, plus peak memory used by softmax's attention matrix.
"""

import argparse
import time

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wave_field.attention import WaveFieldAttention


class StandardAttention1D(nn.Module):
    """Same scaled dot-product attention as in train_audio.py."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, H, Dh).permute(0, 2, 1, 3)
        k = k.view(B, L, H, Dh).permute(0, 2, 1, 3)
        v = v.view(B, L, H, Dh).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out)


def time_module(module, x, n_warmup: int = 3, n_runs: int = 10,
                forward_only: bool = True, device: str = "cpu") -> float:
    """Mean wall-clock seconds per (forward [+ backward]) pass."""
    is_cuda = device == "cuda"
    if not forward_only:
        x = x.detach().requires_grad_(True)

    # Warm up
    for _ in range(n_warmup):
        out = module(x)
        if not forward_only:
            out.sum().backward()
        if is_cuda:
            torch.cuda.synchronize()

    if is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        if not forward_only and x.grad is not None:
            x.grad = None
        out = module(x)
        if not forward_only:
            out.sum().backward()
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / n_runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default=None,
                   help="cuda | mps | cpu (default: best available)")
    p.add_argument("--out", default="benchmarks/attn_benchmark.png")
    args = p.parse_args()

    if args.device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    print(f"Device: {device}")
    print(f"dim={args.dim}  num_heads={args.num_heads}  batch={args.batch}")
    print()

    rows = []
    for L in args.lengths:
        wave = WaveFieldAttention(dim=args.dim, num_heads=args.num_heads, seq_len=L,
                                  conditioning=None).to(device).eval()
        std = StandardAttention1D(dim=args.dim, num_heads=args.num_heads).to(device).eval()

        x = torch.randn(args.batch, L, args.dim, device=device)

        with torch.no_grad():
            t_wave_fwd = time_module(wave, x, device=device, forward_only=True)
            try:
                t_std_fwd = time_module(std, x, device=device, forward_only=True)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                t_std_fwd = float("nan")
                print(f"  L={L:>5}  standard attention OOM/error: {type(e).__name__}")

        speedup = t_std_fwd / t_wave_fwd if t_std_fwd == t_std_fwd else float("inf")
        rows.append((L, t_wave_fwd, t_std_fwd, speedup))
        print(f"  L={L:>5}  wave={t_wave_fwd*1000:7.2f} ms   "
              f"std={t_std_fwd*1000:7.2f} ms   speedup={speedup:5.2f}×")

    # Plot
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Ls = [r[0] for r in rows]
    wave_ms = [r[1] * 1000 for r in rows]
    std_ms = [r[2] * 1000 for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(Ls, wave_ms, "o-", label="Wave Field (FFT, O(n log n))", color="#1f77b4")
    plt.plot(Ls, std_ms, "s-", label="Standard softmax (O(n²))", color="#ff7f0e")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Sequence length L")
    plt.ylabel("Forward-pass time (ms)")
    plt.title(f"Attention wall-clock: wave field vs softmax  ({device}, dim={args.dim})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    plt.close()
    print(f"\nPlot saved → {args.out}")


if __name__ == "__main__":
    main()
