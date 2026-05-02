"""
Three-way comparison: wave field kernels vs standard attention, across conditioning modes.

Run from the comparison/ directory:
    python compare_models.py

Checkpoints are resolved relative to the parent outputs/ directory.
Outputs overwrite any existing comparison plots in this folder.

The three runs compared:
    A  mnist_v6_physics       — Wave field kernels  + Physics gates/FiLM
    B  mnist_standard_physics — Softmax attention   + Physics gates/FiLM  (same cond as A)
    C  mnist_standard         — Softmax attention   + AdaLN-Zero           (original baseline)

A vs B isolates the attention mechanism (everything else identical).
B vs C isolates the conditioning mechanism (same attention, different conditioning).
"""

import os
import sys

# Add parent directory so wave_field and train_mnist are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wave_field.model import WaveFieldDenoiser
from wave_field.diffusion import DDPMDiffusion
from train_mnist import StandardAttention

OUT_DIR = os.path.dirname(__file__)   # save into comparison/ itself
OUTPUTS = os.path.join(os.path.dirname(__file__), "..", "outputs")

RUNS = {
    "A: Wave + Physics":      os.path.join(OUTPUTS, "mnist_v6_physics",       "checkpoint_epoch0300.pt"),
    "B: Standard + Physics":  os.path.join(OUTPUTS, "mnist_standard_physics", "checkpoint_epoch0300.pt"),
    "C: Standard + AdaLN":    os.path.join(OUTPUTS, "mnist_standard",         "checkpoint_epoch0300.pt"),
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]

    model = WaveFieldDenoiser(
        image_size=28, in_channels=1,
        patch_size=a.get("patch_size", 4),
        dim=a["dim"], depth=a["depth"],
        num_heads=a["num_heads"], timestep_dim=a["timestep_dim"],
        conditioning=a["conditioning"],
        use_2d_kernel=(a.get("kernel", "1d") == "2d"),
    )

    # Swap in standard attention if checkpoint was trained with --attn standard
    if a.get("attn", "wave") == "standard":
        for block in model.blocks:
            block.attn = StandardAttention(dim=a["dim"], num_heads=a["num_heads"])

    state = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    return model, ckpt["losses"], n_params


# ---------------------------------------------------------------------------
# Per-timestep loss
# ---------------------------------------------------------------------------

def per_timestep_loss(model, diffusion, device, n_samples=256):
    x = torch.randn(n_samples, 1, 28, 28, device=device)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    diffusion = DDPMDiffusion(1000, schedule="cosine", parameterization="v")
    diffusion.to(device)

    models, all_losses, params, ts_losses = {}, {}, {}, {}

    for label, ckpt_path in RUNS.items():
        if not os.path.exists(ckpt_path):
            print(f"  SKIP {label} — checkpoint not found: {ckpt_path}")
            continue
        print(f"  Loading {label} …")
        model, losses, n = load_model(ckpt_path, device)
        models[label]    = model
        all_losses[label] = losses
        params[label]    = n
        ts_losses[label] = per_timestep_loss(model, diffusion, device)
        print(f"    params={n:,}  final_loss={losses[-1]:.5f}")

    if not models:
        print("No checkpoints found — run training first.")
        return

    n_runs = len(models)
    colors = {"A: Wave + Physics": "#1f77b4",
              "B: Standard + Physics": "#ff7f0e",
              "C: Standard + AdaLN": "#2ca02c"}

    # -----------------------------------------------------------------------
    # 1. Sample grid (one column per run)
    # -----------------------------------------------------------------------
    N = 64
    nrow = 8
    fig, axes = plt.subplots(1, n_runs, figsize=(8 * n_runs, 9))
    if n_runs == 1:
        axes = [axes]

    for ax, (label, model) in zip(axes, models.items()):
        torch.manual_seed(42)
        with torch.no_grad():
            samples = diffusion.ddim_sample(model, (N, 1, 28, 28), device,
                                             num_steps=50, progress=False)
        imgs = (samples.clamp(-1, 1) + 1) / 2
        grid = torchvision.utils.make_grid(imgs, nrow=nrow)
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{label}\n{params[label]:,} params  |  loss {all_losses[label][-1]:.4f}",
                     fontsize=10)
        ax.axis("off")

    plt.suptitle("Sample quality — wave field kernels vs standard attention", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "sample_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {path}")

    # -----------------------------------------------------------------------
    # 2. Training loss curves
    # -----------------------------------------------------------------------
    plt.figure(figsize=(9, 4))
    for label, losses in all_losses.items():
        plt.plot(losses, label=f"{label}  (final {losses[-1]:.4f})",
                 color=colors.get(label))
    plt.xlabel("Epoch"); plt.ylabel("Loss (min-SNR v-pred)")
    plt.title("Training loss curves")
    plt.legend(fontsize=9); plt.tight_layout()
    path = os.path.join(OUT_DIR, "loss_comparison.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved → {path}")

    # -----------------------------------------------------------------------
    # 3. Per-timestep loss (the key diagnostic)
    # -----------------------------------------------------------------------
    plt.figure(figsize=(9, 4))
    for label, (ts, ls) in ts_losses.items():
        plt.plot(ts, ls, label=label, color=colors.get(label))
    plt.xlabel("Diffusion timestep t (0 = clean, 999 = pure noise)")
    plt.ylabel("MSE loss")
    plt.title("Per-timestep noise prediction loss\n"
              "Low t = fine detail recovery (critical for sample quality)")
    plt.legend(fontsize=9); plt.tight_layout()
    path = os.path.join(OUT_DIR, "per_timestep_loss.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved → {path}")

    # -----------------------------------------------------------------------
    # 4. Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{'Run':<28} {'Params':>10} {'Final loss':>12} {'Low-t MSE (t<100)':>18}")
    print("-" * 60)
    for label in models:
        ts, ls = ts_losses[label]
        low_t_mse = sum(l for t, l in zip(ts, ls) if t < 100) / max(1, sum(1 for t in ts if t < 100))
        print(f"{label:<28} {params[label]:>10,} {all_losses[label][-1]:>12.5f} {low_t_mse:>18.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
