"""
Phase 1 — MNIST training script.

Trains WaveFieldDenoiser on MNIST (28×28, grayscale) with standard DDPM.

Usage:
    python train_mnist.py                              # physics conditioning
    python train_mnist.py --conditioning adaln         # AdaLN conditioning
    python train_mnist.py --conditioning physics --epochs 200 --dim 128 --depth 6

Key diagnostics logged:
  - Training loss per epoch
  - Sample grids every --sample_every epochs
  - Wave kernel parameter snapshots (α, ω, φ per head) vs. timestep
"""

import argparse
import os
import math
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wave_field.model import WaveFieldDenoiser
from wave_field.diffusion import DDPMDiffusion, EMA


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Train Wave Field Denoiser on MNIST")
    p.add_argument("--conditioning", default="physics", choices=["physics", "adaln"],
                   help="Timestep conditioning mode (default: physics)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dim", type=int, default=64, help="Model dimension")
    p.add_argument("--depth", type=int, default=4, help="Number of Wave Field Blocks")
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--timestep_dim", type=int, default=128)
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--kernel", default="2d", choices=["1d", "2d"],
                   help="1D sequence kernels or 2D spatial kernels (default: 2d)")
    p.add_argument("--patch_size", type=int, default=4,
                   help="Patch size — 28 must be divisible (4→49 tokens, 7→16 tokens)")
    p.add_argument("--save_dir", default="outputs/mnist",
                   help="Directory for checkpoints and sample images")
    p.add_argument("--sample_every", type=int, default=10,
                   help="Generate samples every N epochs")
    p.add_argument("--ddim_steps", type=int, default=50,
                   help="DDIM steps for fast sampling (set to 0 for full DDPM)")
    p.add_argument("--parameterization", default="v", choices=["eps", "v"],
                   help="v-prediction (recommended for cosine) or ε-prediction")
    p.add_argument("--ema_decay", type=float, default=0.9999,
                   help="EMA decay; sample from EMA weights for clean outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def save_sample_grid(samples: torch.Tensor, path: str, nrow: int = 8):
    """Save a grid of samples. samples in [-1, 1]."""
    imgs = (samples.clamp(-1, 1) + 1) / 2   # → [0, 1]
    grid = torchvision.utils.make_grid(imgs, nrow=nrow, padding=2)
    plt.figure(figsize=(nrow, math.ceil(imgs.shape[0] / nrow)))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_kernel_params(model: WaveFieldDenoiser, diffusion: DDPMDiffusion,
                       path: str, num_ts_samples: int = 20):
    """
    Plot learned wave kernel params (α, ω, φ) per head as a function of
    conditioning timestep.  Only meaningful for physics conditioning.
    """
    device = next(model.parameters()).device
    model.eval()

    ts = torch.linspace(0, diffusion.T - 1, num_ts_samples, dtype=torch.long, device=device)
    t_embs = model.time_embed(ts)   # (num_ts, timestep_dim)

    all_alphas, all_omegas, all_phis = [], [], []
    for block in model.blocks:
        attn = block.attn
        if not hasattr(attn, "use_ts_cond") or not attn.use_ts_cond:
            return   # no physics conditioning, skip

        with torch.no_grad():
            params = attn.ts_to_params(t_embs)                    # (num_ts, 3H)
            B, H = num_ts_samples, attn.num_heads
            d_log_alpha, d_omega, d_phi = params.view(B, 3, H).unbind(dim=1)
            alpha_base = torch.exp(attn.log_alpha)
            alpha = alpha_base.unsqueeze(0) * torch.exp(d_log_alpha)
            omega = attn.omega.unsqueeze(0) + d_omega
            phi   = attn.phi.unsqueeze(0)   + d_phi

        all_alphas.append(alpha.cpu().numpy())
        all_omegas.append(omega.cpu().numpy())
        all_phis.append(phi.cpu().numpy())

    ts_np = ts.cpu().numpy()
    n_blocks = len(all_alphas)
    fig, axes = plt.subplots(n_blocks, 3, figsize=(12, 3 * n_blocks), squeeze=False)

    for i in range(n_blocks):
        axes[i][0].set_title(f"Block {i}: α (damping)")
        axes[i][1].set_title(f"Block {i}: ω (frequency)")
        axes[i][2].set_title(f"Block {i}: φ (phase)")
        for h in range(all_alphas[i].shape[1]):
            axes[i][0].plot(ts_np, all_alphas[i][:, h], label=f"head {h}")
            axes[i][1].plot(ts_np, all_omegas[i][:, h], label=f"head {h}")
            axes[i][2].plot(ts_np, all_phis[i][:, h], label=f"head {h}")
        for j in range(3):
            axes[i][j].set_xlabel("timestep t")
            axes[i][j].legend(fontsize=6)

    plt.suptitle("Wave kernel parameters vs diffusion timestep", y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_training_curve(losses: list, path: str):
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("MNIST Training Curve")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])  # → [-1, 1]
    train_ds = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=min(4, os.cpu_count()), pin_memory=(device.type != "cpu"),
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    use_2d = (args.kernel == "2d")
    model = WaveFieldDenoiser(
        image_size=28,
        in_channels=1,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        timestep_dim=args.timestep_dim,
        conditioning=args.conditioning,
        use_2d_kernel=use_2d,
    ).to(device)

    print(f"Parameters: {model.param_count():,}")
    print(f"Patches: {model.num_patches}  (patch_size={args.patch_size})  kernel={'2D' if use_2d else '1D'}")

    # Save config
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ------------------------------------------------------------------
    # Diffusion
    # ------------------------------------------------------------------
    diffusion = DDPMDiffusion(
        num_timesteps=args.num_timesteps,
        schedule="cosine",
        parameterization=args.parameterization,
    )
    diffusion.to(device)

    ema = EMA(model, decay=args.ema_decay)

    # ------------------------------------------------------------------
    # Optimizer + schedule
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{args.epochs}", leave=False)
        for x, _ in pbar:
            x = x.to(device)
            B = x.shape[0]

            # Uniform timestep sample
            t = torch.randint(0, args.num_timesteps, (B,), device=device)

            loss = diffusion.p_losses(model, x, t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch:3d} | loss = {avg_loss:.5f} | lr = {scheduler.get_last_lr()[0]:.2e}")

        # ------------------------------------------------------------------
        # Periodic evaluation
        # ------------------------------------------------------------------
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            ema.ema_model.eval()
            with torch.no_grad():
                sample_shape = (64, 1, 28, 28)
                if args.ddim_steps > 0:
                    samples = diffusion.ddim_sample(
                        ema.ema_model, sample_shape, device,
                        num_steps=args.ddim_steps, eta=0.0, progress=False
                    )
                else:
                    samples = diffusion.sample(
                        ema.ema_model, sample_shape, device, progress=False
                    )

            save_sample_grid(
                samples,
                os.path.join(args.save_dir, f"samples_epoch{epoch:04d}.png"),
            )

            if args.conditioning == "physics":
                plot_kernel_params(
                    model, diffusion,
                    os.path.join(args.save_dir, f"kernels_epoch{epoch:04d}.png"),
                )

            # Checkpoint (saves both raw and EMA weights)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "losses": losses,
                    "args": vars(args),
                },
                os.path.join(args.save_dir, f"checkpoint_epoch{epoch:04d}.pt"),
            )

    # ------------------------------------------------------------------
    # Final outputs
    # ------------------------------------------------------------------
    plot_training_curve(losses, os.path.join(args.save_dir, "training_curve.png"))
    print(f"\nDone. Outputs saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
