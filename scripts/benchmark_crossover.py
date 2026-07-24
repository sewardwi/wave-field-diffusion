"""
Crossover benchmark: wave-field attention vs softmax over a sequence-length sweep.

The efficiency claim only means something against the softmax people actually
use. This times the attention OPERATOR in isolation at increasing sequence
lengths and compares three implementations at matched dim/heads/batch:

  - softmax (naive)   : materialized (q@kᵀ).softmax()@v — O(L²) memory, the
                        strawman the earlier benchmarks used (OOMs early).
  - softmax (flash)   : F.scaled_dot_product_attention — FlashAttention /
                        memory-efficient kernel, O(L) memory. THE fair baseline.
  - wave              : WaveFieldAttention (FFT conv) — O(L log L). Optionally
                        also the upgraded operator (--upgraded).

For each length it reports forward-only ms, forward+backward ms, and peak GPU
memory, handling OOM gracefully (naive softmax will drop out long before the
others). Prints the crossover length where wave overtakes flash-softmax on
each axis, writes JSON, and (if matplotlib) a plot.

No training, no dataset — pure timing on random weights, so it runs in minutes.

Usage:
    python scripts/benchmark_crossover.py                                   # default sweep
    python scripts/benchmark_crossover.py --lengths 1024 4096 16384 65536
    python scripts/benchmark_crossover.py --batch_size 4 --dim 256 --heads 8
    python scripts/benchmark_crossover.py --no-naive                        # skip the strawman
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wave_field.attention import WaveFieldAttention


class SoftmaxAttention(nn.Module):
    """Matched-interface softmax attention; `flash=True` uses the SDPA kernel."""

    def __init__(self, dim, num_heads, flash: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.flash = flash
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, H, Dh).permute(0, 2, 1, 3)
        k = k.view(B, L, H, Dh).permute(0, 2, 1, 3)
        v = v.view(B, L, H, Dh).permute(0, 2, 1, 3)
        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
            out = attn @ v
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out)


def build(kind, dim, heads, seq_len):
    if kind == "softmax_naive":
        return SoftmaxAttention(dim, heads, flash=False)
    if kind == "softmax_flash":
        return SoftmaxAttention(dim, heads, flash=True)
    if kind == "wave_base":
        return WaveFieldAttention(dim=dim, num_heads=heads, seq_len=seq_len)
    if kind == "wave_upgraded":
        return WaveFieldAttention(dim=dim, num_heads=heads, seq_len=seq_len,
                                  dynamic_filter=True, gating="hyena")
    raise ValueError(kind)


def time_one(kind, dim, heads, L, batch, device, warmup, iters, amp):
    """Return dict(fwd_ms, step_ms, peak_mb) or {'error': ...} on OOM."""
    def amp_ctx():
        if amp and device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False)
        return torch.autocast(device_type="cpu", enabled=False)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    try:
        torch.manual_seed(0)
        model = build(kind, dim, heads, L).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        x = torch.randn(batch, L, dim, device=device)

        def step():
            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                out = model(x)
                loss = out.float().pow(2).mean()
            loss.backward()
            opt.step()

        def fwd():
            with torch.no_grad(), amp_ctx():
                model(x)

        for _ in range(warmup):
            step()
        sync()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        step(); sync()
        peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else None

        step_ts, fwd_ts = [], []
        for _ in range(iters):
            sync(); t0 = time.perf_counter(); step(); sync()
            step_ts.append((time.perf_counter() - t0) * 1e3)
        for _ in range(iters):
            sync(); t0 = time.perf_counter(); fwd(); sync()
            fwd_ts.append((time.perf_counter() - t0) * 1e3)

        return {
            "fwd_ms": round(statistics.median(fwd_ts), 3),
            "step_ms": round(statistics.median(step_ts), 3),
            "peak_mb": round(peak_mb, 1) if peak_mb is not None else None,
            "error": None,
        }
    except RuntimeError as e:
        msg = str(e).split("\n")[0]
        return {"fwd_ms": None, "step_ms": None, "peak_mb": None, "error": msg}
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()


def crossover(lengths, series, ref_key, cmp_key, field):
    """First length where series[cmp_key] beats series[ref_key] on `field` (lower=better)."""
    prev = None
    for L in lengths:
        a = series[ref_key].get(L, {}).get(field)
        b = series[cmp_key].get(L, {}).get(field)
        if a is None or b is None:
            continue
        wins = b < a
        if wins and prev is False:
            return L
        if wins and prev is None:
            return L  # wave already ahead at the shortest measured length
        prev = wins
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[1024, 2048, 4096, 8192, 16384, 32768, 65536])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--no-naive", dest="naive", action="store_false",
                   help="skip the naive-softmax strawman (it OOMs on long L anyway)")
    p.add_argument("--upgraded", action="store_true",
                   help="also benchmark the dynamic-filter+hyena wave operator")
    p.add_argument("--out_dir", default="outputs/crossover")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    amp = device.type == "cuda"
    os.makedirs(args.out_dir, exist_ok=True)

    kinds = []
    if args.naive:
        kinds.append("softmax_naive")
    kinds += ["softmax_flash", "wave_base"]
    if args.upgraded:
        kinds.append("wave_upgraded")

    print(f"Device: {device}  amp={amp}  batch={args.batch_size}  dim={args.dim}  heads={args.heads}")
    if device.type != "cuda":
        print("  NOTE: not CUDA — memory unmeasured; FFT constants can distort short-L timings. "
              "Treat as a smoke test; the real crossover is a CUDA measurement.")

    series = {k: {} for k in kinds}
    for L in args.lengths:
        print(f"\n-- L={L} ({L} tokens) --")
        for k in kinds:
            r = time_one(k, args.dim, args.heads, L, args.batch_size, device,
                         args.warmup, args.iters, amp)
            series[k][L] = r
            if r["error"]:
                print(f"  {k:<16} OOM/err: {r['error'][:60]}")
            else:
                pk = f"{r['peak_mb']:.0f}MB" if r["peak_mb"] is not None else "mem n/a"
                print(f"  {k:<16} fwd {r['fwd_ms']:>8.2f}ms  step {r['step_ms']:>8.2f}ms  {pk}")

    # Crossover points: wave_base vs softmax_flash (the fair comparison)
    cross = {}
    if "softmax_flash" in series and "wave_base" in series:
        cross["wave_base_beats_flash_step_ms_at_L"] = crossover(
            args.lengths, series, "softmax_flash", "wave_base", "step_ms")
        cross["wave_base_beats_flash_peak_mb_at_L"] = crossover(
            args.lengths, series, "softmax_flash", "wave_base", "peak_mb")

    out = {
        "config": {
            "device": (torch.cuda.get_device_name(0) if device.type == "cuda" else device.type),
            "torch": torch.__version__, "batch_size": args.batch_size,
            "dim": args.dim, "heads": args.heads, "amp_bf16": amp,
            "lengths": args.lengths, "warmup": args.warmup, "iters": args.iters,
        },
        "series": {k: {str(L): v for L, v in d.items()} for k, d in series.items()},
        "crossover": cross,
    }
    out_path = os.path.join(args.out_dir, "crossover.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Summary
    print("\n=== Crossover summary (wave_base vs FlashAttention softmax) ===")
    cs = cross.get("wave_base_beats_flash_step_ms_at_L")
    cm = cross.get("wave_base_beats_flash_peak_mb_at_L")
    print(f"  wave faster per step from:      {cs if cs else 'not within swept range'} tokens")
    print(f"  wave lower peak memory from:    {cm if cm else 'not within swept range'} tokens")
    if series.get("wave_base") and series.get("softmax_flash"):
        Lmax = max(args.lengths)
        wb = series["wave_base"].get(Lmax, {})
        sf = series["softmax_flash"].get(Lmax, {})
        if wb.get("step_ms") and sf.get("step_ms"):
            print(f"  at L={Lmax}: wave {sf['step_ms']/wb['step_ms']:.2f}x faster/step", end="")
            if wb.get("peak_mb") and sf.get("peak_mb"):
                print(f", {sf['peak_mb']/wb['peak_mb']:.2f}x less memory", end="")
            print(f" vs flash-softmax")

    # Optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
        styles = {"softmax_naive": ("naive softmax O(L²)", "s--", "#d62728"),
                  "softmax_flash": ("FlashAttention softmax", "o-", "#ff7f0e"),
                  "wave_base": ("wave field O(L log L)", "^-", "#1f77b4"),
                  "wave_upgraded": ("wave (dyn+hyena)", "v-", "#2ca02c")}
        for k in kinds:
            lab, mk, col = styles[k]
            Ls = [L for L in args.lengths if series[k].get(L, {}).get("step_ms")]
            a1.plot(Ls, [series[k][L]["step_ms"] for L in Ls], mk, label=lab, color=col)
            Lm = [L for L in args.lengths if series[k].get(L, {}).get("peak_mb")]
            if Lm:
                a2.plot(Lm, [series[k][L]["peak_mb"] for L in Lm], mk, label=lab, color=col)
        for ax in (a1, a2):
            ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("sequence length (tokens)")
            ax.legend(); ax.grid(True, alpha=0.3)
        a1.set_ylabel("forward+backward ms"); a1.set_title("Step time")
        a2.set_ylabel("peak GPU memory (MB)"); a2.set_title("Peak memory")
        fig.suptitle(f"Wave vs softmax scaling ({out['config']['device']}, "
                     f"dim={args.dim}, heads={args.heads}, bs={args.batch_size})")
        fig.tight_layout()
        plot_path = os.path.join(args.out_dir, "crossover.png")
        fig.savefig(plot_path, dpi=110, bbox_inches="tight")
        print(f"\nPlot  → {plot_path}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")

    print(f"Wrote → {out_path}")


if __name__ == "__main__":
    main()
