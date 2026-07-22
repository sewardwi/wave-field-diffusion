"""
Wall-clock + peak-memory benchmark for the real SC09 denoisers at L=1024 tokens.

The O(n log n) claim has so far rested on benchmark_attn.py, which times the
attention *module* in isolation on synthetic tensors. This times the actual
WaveFieldAudioDenoiser end-to-end — the same model class training uses — at the
production audio config (16384 samples / patch 16 = 1024 tokens, dim 128,
depth 6), doing a real forward + backward step. It isolates the architecture
cost cleanly (deterministic single forward, no self-cond stochasticity) so the
softmax O(L²) vs wave O(L log L) difference shows up as measured ms and MB, not
projection.

Three operators, matched everywhere except the attention:
  - standard        : softmax attention  (O(L²) attn matrix)
  - wave (base)      : fixed damped-wave FFT conv
  - wave (upgraded)  : + dynamic spectral filter + hyena gating

Writes <out_dir>/audio_bench.json and prints a table. GPU-only for the memory
numbers; on CPU it still reports timings with memory=null (for a local smoke).

Usage:
    python scripts/benchmark_audio_models.py --out_dir outputs/audio_bench
    python scripts/benchmark_audio_models.py --batch_size 64 --warmup 10 --iters 30
"""

import argparse
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.sc09 import TARGET_LEN  # 16384
import train_audio


# The three operators to compare. conditioning is held at "physics" throughout
# (it barely touches attention cost); only the attention path varies.
CONFIGS = [
    {"label": "standard (softmax)",   "attn": "standard", "dynamic_filter": False, "gating": "pointwise"},
    {"label": "wave (base)",          "attn": "wave",     "dynamic_filter": False, "gating": "pointwise"},
    {"label": "wave (dyn+hyena)",     "attn": "wave",     "dynamic_filter": True,  "gating": "hyena"},
]


def build(cfg, device):
    args = types.SimpleNamespace(
        patch_size=16, dim=128, depth=6, num_heads=4, timestep_dim=128,
        conditioning="physics", self_cond=True,
        dynamic_filter=cfg["dynamic_filter"], gating=cfg["gating"],
        num_classes=None, class_dropout=0.1,
    )
    model = train_audio.make_audio_model(args, use_standard_attn=(cfg["attn"] == "standard"))
    return model.to(device)


def bench_one(cfg, device, batch_size, warmup, iters, amp):
    """Return dict with fwd_ms, step_ms (fwd+bwd), peak_mb, params — or an error."""
    torch.manual_seed(0)
    model = build(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    params = sum(p.numel() for p in model.parameters())

    def amp_ctx():
        if amp and device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False)
        return torch.autocast(device_type="cpu", enabled=False)

    x = torch.randn(batch_size, 1, TARGET_LEN, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)

    def one_step():
        opt.zero_grad(set_to_none=True)
        with amp_ctx():
            pred = model(x, t)          # self_cond=None -> zeros; single clean forward
            loss = pred.pow(2).mean()
        loss.backward()
        opt.step()

    def one_fwd():
        with torch.no_grad(), amp_ctx():
            model(x, t)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Warmup (kernel autotune, allocator caching)
    for _ in range(warmup):
        one_step()
    sync()

    # Peak memory over a fresh training step
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    one_step(); sync()
    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else None

    # Timed forward+backward steps
    step_ts = []
    for _ in range(iters):
        sync(); t0 = time.perf_counter()
        one_step()
        sync(); step_ts.append((time.perf_counter() - t0) * 1e3)

    # Timed forward-only (inference/sampling cost)
    fwd_ts = []
    for _ in range(iters):
        sync(); t0 = time.perf_counter()
        one_fwd()
        sync(); fwd_ts.append((time.perf_counter() - t0) * 1e3)

    return {
        "params": params,
        "fwd_ms_median": round(statistics.median(fwd_ts), 3),
        "step_ms_median": round(statistics.median(step_ts), 3),
        "step_ms_mean": round(statistics.mean(step_ts), 3),
        "peak_mb": round(peak_mb, 1) if peak_mb is not None else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="outputs/audio_bench")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    header = {
        "device": (torch.cuda.get_device_name(0) if device.type == "cuda" else device.type),
        "torch": torch.__version__,
        "seq_len_samples": TARGET_LEN,
        "tokens": TARGET_LEN // 16,
        "batch_size": args.batch_size,
        "amp_bf16": bool(args.amp and device.type == "cuda"),
        "warmup": args.warmup, "iters": args.iters,
        "note": "single clean fwd/step (self_cond=zeros); isolates the attention operator.",
    }
    print(f"Device: {header['device']}  torch {header['torch']}  "
          f"tokens={header['tokens']}  bs={args.batch_size}  amp={header['amp_bf16']}")
    if device.type != "cuda":
        print("  NOTE: not CUDA — memory is unmeasured and at tiny batch the FFT "
              "constants can make wave look slower. The O(L²) crossover shows on "
              "CUDA at real batch size; treat CPU/MPS output as a smoke test only.")

    results = []
    for cfg in CONFIGS:
        print(f"  benchmarking {cfg['label']} …", flush=True)
        try:
            r = bench_one(cfg, device, args.batch_size, args.warmup, args.iters, args.amp)
            r["error"] = None
        except RuntimeError as e:
            msg = str(e).split("\n")[0]
            print(f"    FAILED: {msg}")
            r = {"error": msg, "params": None, "fwd_ms_median": None,
                 "step_ms_median": None, "step_ms_mean": None, "peak_mb": None}
        r["label"] = cfg["label"]
        results.append(r)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = {"config": header, "results": results}
    out_path = Path(args.out_dir) / "audio_bench.json"
    out_path.write_text(json.dumps(out, indent=2))

    # Table + relative-to-softmax factors
    print(f"\n=== Audio model benchmark @ {header['tokens']} tokens, bs {args.batch_size} ===")
    print(f"  {'operator':<20} {'params':>10} {'fwd ms':>9} {'step ms':>9} {'peak MB':>9}")
    print(f"  {'-'*20} {'-'*10} {'-'*9} {'-'*9} {'-'*9}")
    base = next((r for r in results if r["label"].startswith("standard") and r["step_ms_median"]), None)
    for r in results:
        if r["error"]:
            print(f"  {r['label']:<20} {'—':>10} {'OOM/err':>9} {'':>9} {'':>9}")
            continue
        pk = f"{r['peak_mb']:.0f}" if r["peak_mb"] is not None else "n/a"
        print(f"  {r['label']:<20} {r['params']:>10,} {r['fwd_ms_median']:>9.2f} "
              f"{r['step_ms_median']:>9.2f} {pk:>9}")
    if base:
        print(f"\n  vs softmax (step time / peak mem):")
        for r in results:
            if r["error"] or r["label"].startswith("standard"):
                continue
            spd = base["step_ms_median"] / r["step_ms_median"]
            speed = f"{spd:.2f}x faster" if spd >= 1 else f"{1/spd:.2f}x slower"
            mem = (base["peak_mb"] / r["peak_mb"]) if (r["peak_mb"] and base["peak_mb"]) else None
            mems = f"{mem:.2f}x less memory" if mem else "mem n/a (CPU/MPS)"
            print(f"    {r['label']:<20} {speed} step, {mems}")
    print(f"\nWrote → {out_path}")


if __name__ == "__main__":
    main()
