"""
Phase 2 — CIFAR-10 training script.

Brings the CIFAR backbone up to the same training setup as Phase 1 MNIST:
cosine noise schedule, v-prediction, EMA weights for sampling, min-SNR loss
weighting (γ=5), self-conditioning (Chen 2022), and DDIM-50 sampling.

Architecture variants (configurable):
  --kernel 1d | 2d           1D sequence kernels vs. 2D radial spatial kernels
  --attn   wave | standard   Wave field attention vs. standard softmax (baseline)
  --conditioning physics | adaln

The natural ablation set (matches what comparison/compare_cifar.py expects):
    A  --attn wave     --conditioning physics  --kernel 2d   (the headline run)
    B  --attn standard --conditioning physics                (isolates attention)
    C  --attn standard --conditioning adaln                  (DiT-like baseline)
    D  --attn wave     --conditioning adaln    --kernel 2d   (isolates conditioning)

Usage:
    python train_cifar.py --attn wave     --conditioning physics --kernel 2d \\
        --save_dir outputs/cifar_v2_wave_physics_2d
    python train_cifar.py --attn standard --conditioning physics \\
        --save_dir outputs/cifar_v2_standard_physics
    python train_cifar.py --attn standard --conditioning adaln \\
        --save_dir outputs/cifar_v2_standard_adaln
    python train_cifar.py --attn wave     --conditioning adaln    --kernel 2d \\
        --save_dir outputs/cifar_v2_wave_adaln_2d
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

from denoisers.image import WaveFieldDenoiser
from wave_field.diffusion import DDPMDiffusion, EMA


# ---------------------------------------------------------------------------
# Standard attention baseline (drop-in swap, identical interface to wave field)
# ---------------------------------------------------------------------------

class StandardAttention(nn.Module):
    """Scaled dot-product attention — same forward signature as WaveFieldAttention."""

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, t_emb=None):  # noqa: ARG002
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, H, Dh).permute(0, 2, 1, 3)
        k = k.view(B, L, H, Dh).permute(0, 2, 1, 3)
        v = v.view(B, L, H, Dh).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out)


def make_standard_model(image_size, in_channels, patch_size, dim, depth,
                         num_heads, timestep_dim, conditioning, use_self_cond=False,
                         num_classes=None, class_dropout_prob=0.1):
    """WaveFieldDenoiser with attention swapped out for standard softmax attention."""
    model = WaveFieldDenoiser(
        image_size=image_size, in_channels=in_channels, patch_size=patch_size,
        dim=dim, depth=depth, num_heads=num_heads, timestep_dim=timestep_dim,
        conditioning=conditioning, use_2d_kernel=False,
        use_self_cond=use_self_cond,
        num_classes=num_classes, class_dropout_prob=class_dropout_prob,
    )
    for block in model.blocks:
        block.attn = StandardAttention(dim=dim, num_heads=num_heads)
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Train Wave Field Denoiser on CIFAR-10")
    p.add_argument("--conditioning", default="physics", choices=["physics", "adaln"])
    p.add_argument("--kernel", default="2d", choices=["1d", "2d"],
                   help="1D sequence kernels (Approach A) or 2D spatial kernels (Approach B)")
    p.add_argument("--dynamic_filter", action=argparse.BooleanOptionalAction, default=False,
                   help="Data-dependent spectral filter on the wave kernel "
                        "(ΔK̂(x) added to the static kernel spectrum). Wave attn only.")
    p.add_argument("--gating", default="pointwise", choices=["pointwise", "hyena"],
                   help="Wave-attn gate: 'pointwise' (sigmoid q⊙k) or 'hyena' "
                        "(q ⊙ conv(k ⊙ v), position-dependent routing). Wave attn only.")
    p.add_argument("--aniso_kernel", action=argparse.BooleanOptionalAction, default=False,
                   help="Anisotropic oriented (Gabor-like) 2D kernel instead of "
                        "isotropic radial — adds orientation selectivity. 2D wave only.")
    p.add_argument("--attn", default="wave", choices=["wave", "standard"],
                   help="Attention type — wave field or standard softmax (baseline)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--timestep_dim", type=int, default=256)
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--save_dir", default="outputs/cifar")
    p.add_argument("--sample_every", type=int, default=20)
    p.add_argument("--ddim_steps", type=int, default=50,
                   help="DDIM steps for fast sampling (set to 0 for full DDPM)")
    p.add_argument("--parameterization", default="v", choices=["eps", "v"])
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--self_cond", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_classes", type=int, default=None,
                   help="Enable class-conditional generation + CFG with this many "
                        "classes (CIFAR-10 → 10). Omit for unconditional.")
    p.add_argument("--class_dropout", type=float, default=0.1,
                   help="Label-dropout prob for classifier-free-guidance training.")
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="CFG scale used for eval-grid sampling (1.0 = no guidance).")
    p.add_argument("--sampler", default="ddim", choices=["ddim", "dpmpp", "ddpm"],
                   help="Sampler for periodic eval grids (dpmpp = DPM-Solver++(2M)).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def do_sample(diffusion, model, shape, device, args, y=None):
    """Dispatch the chosen sampler with optional class labels + CFG."""
    gs = args.guidance_scale
    if args.sampler == "dpmpp":
        return diffusion.dpmpp_2m_sample(model, shape, device,
                                         num_steps=max(args.ddim_steps, 1),
                                         progress=False, y=y, guidance_scale=gs)
    if args.sampler == "ddpm" or args.ddim_steps == 0:
        return diffusion.sample(model, shape, device, progress=False,
                                y=y, guidance_scale=gs)
    return diffusion.ddim_sample(model, shape, device, num_steps=args.ddim_steps,
                                 eta=0.0, progress=False, y=y, guidance_scale=gs)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def save_sample_grid(samples: torch.Tensor, path: str, nrow: int = 8, title: str = ""):
    """Save a grid of CIFAR samples (RGB). samples in [-1, 1]."""
    imgs = (samples.clamp(-1, 1) + 1) / 2
    grid = torchvision.utils.make_grid(imgs, nrow=nrow, padding=2)
    plt.figure(figsize=(nrow, math.ceil(imgs.shape[0] / nrow) + (0.5 if title else 0)))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    if title:
        plt.title(title, fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_kernel_params(model, diffusion, path, num_ts_samples: int = 20):
    """
    Plot per-head wave kernel (α, ω, φ) vs diffusion timestep — the headline
    physics-conditioning diagnostic.  Only meaningful for physics conditioning
    on a wave-field attention model.
    """
    device = next(model.parameters()).device
    model.eval()

    ts = torch.linspace(0, diffusion.T - 1, num_ts_samples, dtype=torch.long, device=device)
    t_embs = model.time_embed(ts)

    rows = []
    for block in model.blocks:
        attn = block.attn
        if not hasattr(attn, "use_ts_cond") or not attn.use_ts_cond:
            return  # not physics-conditioned wave attention → skip
        if getattr(attn, "aniso_kernel", False):
            return  # anisotropic kernel has 5 directional params/head → different viz
        with torch.no_grad():
            params = attn.ts_to_params(t_embs)
            B_, H = num_ts_samples, attn.num_heads
            d_log_alpha, d_omega, d_phi = params.view(B_, 3, H).unbind(dim=1)
            alpha = torch.exp(attn.log_alpha).unsqueeze(0) * torch.exp(d_log_alpha)
            omega = attn.omega.unsqueeze(0) + d_omega
            phi = attn.phi.unsqueeze(0) + d_phi
        rows.append((alpha.cpu().numpy(), omega.cpu().numpy(), phi.cpu().numpy()))

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n), squeeze=False)
    ts_np = ts.cpu().numpy()
    for i, (a, w, p) in enumerate(rows):
        axes[i][0].set_title(f"Block {i}: α (damping)")
        axes[i][1].set_title(f"Block {i}: ω (frequency)")
        axes[i][2].set_title(f"Block {i}: φ (phase)")
        for h in range(a.shape[1]):
            axes[i][0].plot(ts_np, a[:, h], label=f"h{h}")
            axes[i][1].plot(ts_np, w[:, h], label=f"h{h}")
            axes[i][2].plot(ts_np, p[:, h], label=f"h{h}")
        for j in range(3):
            axes[i][j].set_xlabel("timestep t")
            axes[i][j].legend(fontsize=6)
    plt.suptitle("Wave kernel parameters vs diffusion timestep (CIFAR)", y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def visualize_2d_kernels(model, path, num_ts: int = 5):
    """
    Render learned 2D radial wave kernels at several timesteps, per block.
    Verifies the kernels haven't collapsed and shows their spatial structure.
    """
    device = next(model.parameters()).device
    ts = torch.linspace(0, 999, num_ts, dtype=torch.long, device=device)
    t_embs = model.time_embed(ts)

    n_blocks = len(model.blocks)
    fig, axes = plt.subplots(num_ts, n_blocks, figsize=(2.2 * n_blocks, 2.2 * num_ts),
                              squeeze=False)
    for bi, block in enumerate(model.blocks):
        attn = block.attn
        if not hasattr(attn, "_build_kernel_2d"):
            axes[0][bi].set_title(f"Block {bi}: N/A (1D)", fontsize=8)
            for ti in range(num_ts):
                axes[ti][bi].axis("off")
            continue
        for ti in range(num_ts):
            with torch.no_grad():
                kernel, batched = attn._build_kernel_2d(t_embs[ti:ti+1])
                k = kernel[0] if batched else kernel
                k_avg = k.mean(0).cpu().numpy()
            ax = axes[ti][bi]
            vmax = max(abs(k_avg.min()), abs(k_avg.max()), 1e-6)
            ax.imshow(k_avg, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"Blk {bi}, t={ts[ti].item()}", fontsize=7)
            ax.axis("off")
    plt.suptitle("2D wave kernels at different timesteps (avg over heads)", y=1.0)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_training_curve(losses: list, path: str, title: str = "Training Curve"):
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Epoch"); plt.ylabel("Loss (min-SNR weighted)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    # If user didn't customize save_dir, build a descriptive default
    if args.save_dir == "outputs/cifar":
        run_name = f"cifar_{args.attn}_{args.conditioning}"
        if args.attn == "wave":
            run_name += f"_{args.kernel}"
            if args.dynamic_filter:
                run_name += "_dyn"
            if args.gating == "hyena":
                run_name += "_hyena"
            if args.aniso_kernel:
                run_name += "_aniso"
        if args.self_cond:
            run_name += "_sc"
        if args.num_classes is not None:
            run_name += "_cond"
        args.save_dir = os.path.join("outputs", run_name)
    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}  |  save_dir: {args.save_dir}")

    # ------------------------------------------------------------------
    # Dataset — CIFAR-10 with horizontal flip augmentation
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
        model = make_standard_model(
            image_size=32, in_channels=3, patch_size=args.patch_size,
            dim=args.dim, depth=args.depth, num_heads=args.num_heads,
            timestep_dim=args.timestep_dim, conditioning=args.conditioning,
            use_self_cond=args.self_cond,
            num_classes=args.num_classes, class_dropout_prob=args.class_dropout,
        ).to(device)
    else:
        model = WaveFieldDenoiser(
            image_size=32, in_channels=3, patch_size=args.patch_size,
            dim=args.dim, depth=args.depth, num_heads=args.num_heads,
            timestep_dim=args.timestep_dim, conditioning=args.conditioning,
            use_2d_kernel=use_2d, use_self_cond=args.self_cond,
            dynamic_filter=args.dynamic_filter, gating=args.gating,
            aniso_kernel=args.aniso_kernel,
            num_classes=args.num_classes, class_dropout_prob=args.class_dropout,
        ).to(device)

    print(f"Parameters: {model.param_count():,}")
    print(f"Patches: {model.num_patches}  patch_size={args.patch_size}  "
          f"attn={args.attn}  cond={args.conditioning}  "
          f"kernel={'2D' if (args.attn == 'wave' and use_2d) else '1D'}  "
          f"self_cond={args.self_cond}")

    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ------------------------------------------------------------------
    # Diffusion + EMA + optimizer
    # ------------------------------------------------------------------
    diffusion = DDPMDiffusion(
        num_timesteps=args.num_timesteps,
        schedule="cosine",
        parameterization=args.parameterization,
    )
    diffusion.to(device)
    ema = EMA(model, decay=args.ema_decay)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Warm-up 5 epochs + cosine decay (kept from old script — useful for CIFAR)
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
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
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device) if args.num_classes is not None else None
            t = torch.randint(0, args.num_timesteps, (x.shape[0],), device=device)
            loss = diffusion.p_losses(model, x, t, y=y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg = epoch_loss / len(loader)
        losses.append(avg)
        print(f"Epoch {epoch:3d} | loss = {avg:.5f} | lr = {scheduler.get_last_lr()[0]:.2e}")

        # ------------------------------------------------------------------
        # Periodic evaluation — sample, kernel diagnostics, checkpoint
        # ------------------------------------------------------------------
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            ema.ema_model.eval()
            with torch.no_grad():
                shape = (64, 3, 32, 32)
                # Conditional: a per-class grid (each row a class); CFG via guidance_scale
                y_s = (torch.arange(64, device=device) % args.num_classes
                       if args.num_classes is not None else None)
                samples = do_sample(diffusion, ema.ema_model, shape, device, args, y_s)

            sample_title = (f"epoch {epoch} | attn={args.attn} cond={args.conditioning} "
                            f"{'sc' if args.self_cond else ''} | loss {avg:.4f}")
            save_sample_grid(
                samples,
                os.path.join(args.save_dir, f"samples_epoch{epoch:04d}.png"),
                title=sample_title,
            )

            # 2D kernel viz when using wave + 2d
            if args.attn == "wave" and use_2d:
                visualize_2d_kernels(
                    ema.ema_model,
                    os.path.join(args.save_dir, f"kernels2d_epoch{epoch:04d}.png"),
                )
            # Per-head α/ω/φ vs t plot when using physics conditioning + wave
            if args.attn == "wave" and args.conditioning == "physics":
                plot_kernel_params(
                    ema.ema_model, diffusion,
                    os.path.join(args.save_dir, f"kernel_params_epoch{epoch:04d}.png"),
                )

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
                "args": vars(args),
            }, os.path.join(args.save_dir, f"checkpoint_epoch{epoch:04d}.pt"))

    plot_training_curve(
        losses,
        os.path.join(args.save_dir, "training_curve.png"),
        title=f"CIFAR-10 — {os.path.basename(args.save_dir)}",
    )
    print(f"\nDone. Outputs in: {args.save_dir}")


if __name__ == "__main__":
    main()
