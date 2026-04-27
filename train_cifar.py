"""
Phase 2 — CIFAR-10 training script.

Extends Phase 1 to CIFAR-10 (32×32, RGB) with two configurable approaches:

  Approach A (--kernel 1d):
      Same 1D sequence-of-patches architecture as MNIST Phase 1, scaled up.
      64 patches of dim 48 → embed to 256.

  Approach B (--kernel 2d):
      2D spatial wave kernels via WaveFieldAttention2D, operating on the
      8×8 patch grid with radially-symmetric damped wave kernels.

Additional ablation flag:
  --attn standard   →  swap in standard scaled-dot-product attention (baseline)
  --attn wave       →  Wave Field Attention (default)

Usage:
    python train_cifar.py                               # physics + 1D kernels
    python train_cifar.py --kernel 2d                   # physics + 2D kernels
    python train_cifar.py --conditioning adaln          # AdaLN baseline
    python train_cifar.py --conditioning adaln --attn standard  # standard DiT baseline
"""

import argparse
import os
import math
import json
import torch
import torch.nn as nn
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
from wave_field.diffusion import DDPMDiffusion


# ---------------------------------------------------------------------------
# Standard attention baseline (for ablation)
# ---------------------------------------------------------------------------

class StandardAttention(nn.Module):
    """Scaled dot-product attention baseline — same interface as WaveFieldAttention."""

    def __init__(self, dim, num_heads, seq_len=None, timestep_dim=None, conditioning=None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, t_emb=None):
        B, L, D = x.shape
        H = self.num_heads
        Dh = self.head_dim

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, H, Dh).permute(0, 2, 1, 3)
        k = k.view(B, L, H, Dh).permute(0, 2, 1, 3)
        v = v.view(B, L, H, Dh).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale    # (B, H, L, L)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out)


class WaveFieldDenoiserWithStandardAttn(WaveFieldDenoiser):
    """
    WaveFieldDenoiser with standard attention substituted in for ablation.
    Overrides the attn module in each block after construction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for block in self.blocks:
            block.attn = StandardAttention(
                dim=self.dim,
                num_heads=block.attn.num_heads,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Train Wave Field Denoiser on CIFAR-10")
    p.add_argument("--conditioning", default="physics", choices=["physics", "adaln"])
    p.add_argument("--kernel", default="1d", choices=["1d", "2d"],
                   help="1D sequence kernels (Approach A) or 2D spatial kernels (Approach B)")
    p.add_argument("--attn", default="wave", choices=["wave", "standard"],
                   help="Attention type: wave field or standard (ablation)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--timestep_dim", type=int, default=256)
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--save_dir", default="outputs/cifar10")
    p.add_argument("--sample_every", type=int, default=20)
    p.add_argument("--ddim_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_sample_grid(samples: torch.Tensor, path: str, nrow: int = 8):
    imgs = (samples.clamp(-1, 1) + 1) / 2
    grid = torchvision.utils.make_grid(imgs, nrow=nrow, padding=2)
    plt.figure(figsize=(nrow, math.ceil(imgs.shape[0] / nrow)))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def visualize_2d_kernels(model: WaveFieldDenoiser, path: str, num_ts: int = 5):
    """
    For 2D kernel models: render the learned 2D wave kernels at several
    diffusion timesteps.  Shows whether they resemble bandpass / Gabor filters.
    """
    device = next(model.parameters()).device
    ts = torch.linspace(0, 999, num_ts, dtype=torch.long, device=device)
    t_embs = model.time_embed(ts)

    n_blocks = len(model.blocks)
    fig, axes = plt.subplots(num_ts, n_blocks, figsize=(3 * n_blocks, 3 * num_ts),
                              squeeze=False)

    for bi, block in enumerate(model.blocks):
        attn = block.attn
        if not hasattr(attn, "_build_kernel_2d"):
            axes[0][bi].set_title(f"Block {bi}: N/A (1D)")
            continue

        for ti in range(num_ts):
            with torch.no_grad():
                t_emb_i = t_embs[ti:ti+1]
                kernel, batched = attn._build_kernel_2d(t_emb_i)
                # kernel: (1, num_heads, H, W) or (num_heads, H, W)
                k = kernel[0] if batched else kernel   # (num_heads, H, W)
                k_avg = k.mean(0).cpu().numpy()        # average over heads

            ax = axes[ti][bi]
            im = ax.imshow(k_avg, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_title(f"Block {bi}, t={ts[ti].item()}", fontsize=8)
            ax.axis("off")

    plt.suptitle("2D wave kernels at different timesteps (avg over heads)")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_training_curve(losses: list, path: str, title: str = "Training Curve"):
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    run_name = f"cifar_{args.conditioning}_{args.kernel}_{args.attn}"
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}  |  Run: {run_name}")

    # ------------------------------------------------------------------
    # Dataset — standard CIFAR-10 augmentation
    # ------------------------------------------------------------------
    transform_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
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

    if args.attn == "standard":
        # Baseline: DiT-style with standard attention (no wave kernels)
        ModelClass = WaveFieldDenoiserWithStandardAttn
    else:
        ModelClass = WaveFieldDenoiser

    model = ModelClass(
        image_size=32,
        in_channels=3,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        timestep_dim=args.timestep_dim,
        conditioning=args.conditioning,
        use_2d_kernel=use_2d,
    ).to(device)

    print(f"Parameters: {model.param_count():,}")
    print(f"Patches: {model.num_patches}  patch_size={args.patch_size}  "
          f"kernel={'2D' if use_2d else '1D'}")

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ------------------------------------------------------------------
    # Diffusion + optimizer
    # ------------------------------------------------------------------
    diffusion = DDPMDiffusion(num_timesteps=args.num_timesteps)
    diffusion.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Warm-up 5 epochs + cosine decay
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

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
            t = torch.randint(0, args.num_timesteps, (x.shape[0],), device=device)

            loss = diffusion.p_losses(model, x, t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch:3d} | loss = {avg_loss:.5f} | lr = {scheduler.get_last_lr()[0]:.2e}")

        # ------------------------------------------------------------------
        # Periodic diagnostics
        # ------------------------------------------------------------------
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                shape = (64, 3, 32, 32)
                if args.ddim_steps > 0:
                    samples = diffusion.ddim_sample(
                        model, shape, device,
                        num_steps=args.ddim_steps, eta=0.0, progress=False
                    )
                else:
                    samples = diffusion.sample(model, shape, device, progress=False)

            save_sample_grid(
                samples,
                os.path.join(save_dir, f"samples_epoch{epoch:04d}.png"),
            )

            if use_2d and args.attn == "wave":
                visualize_2d_kernels(
                    model,
                    os.path.join(save_dir, f"kernels2d_epoch{epoch:04d}.png"),
                )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "losses": losses,
                    "args": vars(args),
                },
                os.path.join(save_dir, f"checkpoint_epoch{epoch:04d}.pt"),
            )

    plot_training_curve(
        losses,
        os.path.join(save_dir, "training_curve.png"),
        title=f"CIFAR-10 — {run_name}",
    )
    print(f"\nDone. Outputs in: {save_dir}")


if __name__ == "__main__":
    main()
