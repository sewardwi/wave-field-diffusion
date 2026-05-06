"""
Cross-run comparison for CIFAR-10 ablations.

Auto-discovers any directory under ../outputs/ whose name starts with "cifar_"
and that contains a final checkpoint, then generates side-by-side diagnostics.

Run from the comparison/ directory:
    python compare_cifar.py
    python compare_cifar.py --runs cifar_wave_physics_2d_sc cifar_standard_adaln_sc

Outputs (saved into this folder):
  - cifar_sample_comparison.png   — generated samples, one column per run
  - cifar_loss_comparison.png     — training curves overlaid
  - cifar_per_timestep_loss.png   — per-timestep MSE (low-t = fine detail)
  - cifar_kernels2d.png           — 2D wave kernels per (run × block × t)
                                     (only for wave-attention runs)

Each plot is sized to scale with the number of runs included.
"""

import argparse
import glob
import os
import sys

# Add parent so wave_field/denoisers/train_cifar are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from denoisers.image import WaveFieldDenoiser
from wave_field.diffusion import DDPMDiffusion
from train_cifar import StandardAttention

OUT_DIR = os.path.dirname(__file__)
OUTPUTS = os.path.join(os.path.dirname(__file__), "..", "outputs")


# ---------------------------------------------------------------------------
# Discovery & loading
# ---------------------------------------------------------------------------

def discover_runs(filter_names=None):
    """Find all CIFAR run directories with at least one checkpoint."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(OUTPUTS, "cifar*"))):
        name = os.path.basename(path)
        if filter_names and name not in filter_names:
            continue
        ckpts = sorted(glob.glob(os.path.join(path, "checkpoint_epoch*.pt")))
        if ckpts:
            runs[name] = ckpts[-1]   # use latest checkpoint
    return runs


def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]

    model = WaveFieldDenoiser(
        image_size=32, in_channels=3,
        patch_size=a.get("patch_size", 4),
        dim=a["dim"], depth=a["depth"],
        num_heads=a["num_heads"], timestep_dim=a["timestep_dim"],
        conditioning=a["conditioning"],
        use_2d_kernel=(a.get("kernel", "1d") == "2d"),
        use_self_cond=a.get("self_cond", False),
    )

    if a.get("attn", "wave") == "standard":
        for block in model.blocks:
            block.attn = StandardAttention(dim=a["dim"], num_heads=a["num_heads"])

    state = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, ckpt["losses"], a, sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def per_timestep_loss(model, diffusion, device, n_samples: int = 128):
    x = torch.randn(n_samples, 3, 32, 32, device=device)
    ts, losses = [], []
    with torch.no_grad():
        for t_val in range(0, 1000, 20):
            t = torch.full((n_samples,), t_val, dtype=torch.long, device=device)
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise)
            eps_pred = diffusion._model_eps(model, x_t, t)
            losses.append((eps_pred - noise).pow(2).mean().item())
            ts.append(t_val)
    return ts, losses


def render_2d_kernels(model, diffusion, device, num_ts: int = 4):
    """Returns a list-of-lists of (kernel_avg_over_heads, t_value) per block."""
    rows = []
    if not model.blocks:
        return rows
    ts = torch.linspace(0, diffusion.T - 1, num_ts, dtype=torch.long, device=device)
    t_embs = model.time_embed(ts)

    for block in model.blocks:
        attn = block.attn
        block_imgs = []
        if not hasattr(attn, "_build_kernel_2d"):
            block_imgs = None
        else:
            for ti in range(num_ts):
                with torch.no_grad():
                    kernel, batched = attn._build_kernel_2d(t_embs[ti:ti+1])
                    k = kernel[0] if batched else kernel
                    block_imgs.append((k.mean(0).cpu().numpy(), ts[ti].item()))
        rows.append(block_imgs)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=None,
                   help="Optional subset of run directory names (under outputs/)")
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--n_samples", type=int, default=64)
    args = p.parse_args()

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    runs = discover_runs(filter_names=args.runs)
    if not runs:
        print(f"No CIFAR runs found in {OUTPUTS}/cifar*  with a checkpoint_epoch*.pt")
        return

    print(f"Discovered runs:")
    for name, ckpt in runs.items():
        print(f"  {name}  ←  {os.path.basename(ckpt)}")

    diffusion = DDPMDiffusion(1000, schedule="cosine", parameterization="v")
    diffusion.to(device)

    models, all_losses, all_args, params, ts_losses = {}, {}, {}, {}, {}
    for label, ckpt in runs.items():
        print(f"\nLoading {label} …")
        model, losses, a, n = load_model(ckpt, device)
        models[label]      = model
        all_losses[label]  = losses
        all_args[label]    = a
        params[label]      = n
        ts_losses[label]   = per_timestep_loss(model, diffusion, device)
        print(f"  params={n:,}  final_loss={losses[-1]:.5f}")

    # Use a stable color palette in run-discovery order
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
               "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    colors = {label: palette[i % len(palette)] for i, label in enumerate(models)}

    n_runs = len(models)

    # -----------------------------------------------------------------------
    # 1. Sample grid (one column per run) — generated with shared random seed
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, n_runs, figsize=(7 * n_runs, 8.5), squeeze=False)
    for ax, (label, model) in zip(axes[0], models.items()):
        torch.manual_seed(42)
        with torch.no_grad():
            samples = diffusion.ddim_sample(model, (args.n_samples, 3, 32, 32), device,
                                            num_steps=args.ddim_steps, eta=0.0, progress=False)
        imgs = (samples.clamp(-1, 1) + 1) / 2
        grid = torchvision.utils.make_grid(imgs, nrow=8, padding=2)
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
        a = all_args[label]
        title = (f"{label}\n"
                 f"attn={a.get('attn','?')} cond={a.get('conditioning','?')} "
                 f"kernel={a.get('kernel','?')} sc={a.get('self_cond', False)}\n"
                 f"{params[label]:,} params  |  final loss {all_losses[label][-1]:.4f}")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.suptitle("CIFAR-10 sample comparison", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "cifar_sample_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {path}")

    # -----------------------------------------------------------------------
    # 2. Training loss overlay
    # -----------------------------------------------------------------------
    plt.figure(figsize=(9, 4))
    for label, losses in all_losses.items():
        plt.plot(losses, label=f"{label}  (final {losses[-1]:.4f})",
                 color=colors[label])
    plt.xlabel("Epoch"); plt.ylabel("Loss (min-SNR weighted)")
    plt.title("CIFAR-10 training loss")
    plt.legend(fontsize=8); plt.tight_layout()
    path = os.path.join(OUT_DIR, "cifar_loss_comparison.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved → {path}")

    # -----------------------------------------------------------------------
    # 3. Per-timestep loss — the key calibration diagnostic
    # -----------------------------------------------------------------------
    plt.figure(figsize=(9, 4))
    for label, (ts, ls) in ts_losses.items():
        plt.plot(ts, ls, label=label, color=colors[label])
    plt.xlabel("Diffusion timestep t (0 = clean, 999 = pure noise)")
    plt.ylabel("MSE loss")
    plt.title("Per-timestep noise prediction loss (CIFAR-10)\n"
              "Low-t = fine detail recovery (drives sample quality)")
    plt.legend(fontsize=8); plt.tight_layout()
    path = os.path.join(OUT_DIR, "cifar_per_timestep_loss.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved → {path}")

    # -----------------------------------------------------------------------
    # 4. 2D kernel grid — shows spatial structure across runs
    # -----------------------------------------------------------------------
    wave_runs = [label for label, m in models.items()
                 if any(hasattr(b.attn, "_build_kernel_2d") for b in m.blocks)]
    if wave_runs:
        num_ts = 4
        max_blocks = max(len(models[label].blocks) for label in wave_runs)

        fig, axes = plt.subplots(
            len(wave_runs) * num_ts, max_blocks,
            figsize=(2 * max_blocks, 2 * num_ts * len(wave_runs)),
            squeeze=False,
        )
        # Hide everything by default
        for r in range(axes.shape[0]):
            for c in range(axes.shape[1]):
                axes[r][c].axis("off")

        for ri, label in enumerate(wave_runs):
            blocks = render_2d_kernels(models[label], diffusion, device, num_ts=num_ts)
            for bi, block_imgs in enumerate(blocks):
                if block_imgs is None:
                    continue
                for ti, (k_avg, t_val) in enumerate(block_imgs):
                    ax = axes[ri * num_ts + ti][bi]
                    vmax = max(abs(k_avg.min()), abs(k_avg.max()), 1e-6)
                    ax.imshow(k_avg, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                    if ti == 0 and bi == 0:
                        ax.set_ylabel(label, fontsize=8)
                    if ti == 0:
                        ax.set_title(f"Block {bi}", fontsize=8)
                    if bi == 0:
                        ax.text(-0.4, 0.5, f"t={t_val}", transform=ax.transAxes,
                                fontsize=7, ha="right", va="center")
                    ax.set_xticks([]); ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(True)

        plt.suptitle("2D wave kernels across runs (avg over heads)", y=1.0)
        plt.tight_layout()
        path = os.path.join(OUT_DIR, "cifar_kernels2d.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Saved → {path}")

    # -----------------------------------------------------------------------
    # 5. Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"{'Run':<40} {'Params':>10} {'Final loss':>12} {'Low-t MSE (t<100)':>18}")
    print("-" * 90)
    for label in models:
        ts, ls = ts_losses[label]
        low_t_mse = sum(l for t, l in zip(ts, ls) if t < 100) / max(1, sum(1 for t in ts if t < 100))
        print(f"{label:<40} {params[label]:>10,} {all_losses[label][-1]:>12.5f} {low_t_mse:>18.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
