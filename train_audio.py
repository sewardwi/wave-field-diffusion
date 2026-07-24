"""
Phase B (stretch goal) — SC09 spoken-digit audio diffusion.

Trains WaveFieldAudioDenoiser on SC09 (digit subset of Speech Commands) with
the same training setup as MNIST (cosine schedule, v-prediction, EMA, min-SNR,
self-conditioning).  Default configuration: 1 s clips at 16 kHz = 16,384
samples → 1024 tokens at patch_size=16.  This is the regime where wave field's
O(n log n) FFT advantage over O(n²) softmax attention actually shows up.

Usage:
    python train_audio.py                            # wave field, physics conditioning
    python train_audio.py --attn standard            # standard attention baseline
    python train_audio.py --conditioning adaln       # AdaLN baseline

Diagnostics logged:
  - Training loss per epoch
  - Sample audio (.wav files) every --sample_every epochs
  - Spectrogram and waveform plots of generated samples
  - Wave kernel parameter snapshots vs timestep (when conditioning=physics)
"""

import argparse
import os
import math
import json
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from denoisers.audio import WaveFieldAudioDenoiser
from datasets.sc09 import SC09, TARGET_SR, TARGET_LEN
from wave_field.diffusion import DDPMDiffusion, EMA
from metrics.classifier import SC09Classifier
from metrics.frechet import extract_features, frechet_distance, classifier_metrics


# ---------------------------------------------------------------------------
# Standard 1D attention baseline (drop-in swap)
# ---------------------------------------------------------------------------

class StandardAttention1D(nn.Module):
    """Scaled dot-product attention with the same interface as WaveFieldAttention."""

    def __init__(self, dim, num_heads):
        super().__init__()
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
        # scaled_dot_product_attention routes to FlashAttention / memory-
        # efficient kernels (O(L) memory, no materialized L×L matrix) — the
        # correct modern softmax baseline. Default scale = head_dim**-0.5.
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out)


def make_audio_model(args, *, use_standard_attn: bool):
    """Build either wave-field or standard-attention 1D denoiser."""
    model = WaveFieldAudioDenoiser(
        sequence_length=TARGET_LEN,
        in_channels=1,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        timestep_dim=args.timestep_dim,
        conditioning=args.conditioning,
        use_self_cond=args.self_cond,
        dynamic_filter=args.dynamic_filter, gating=args.gating,
        num_classes=args.num_classes, class_dropout_prob=args.class_dropout,
    )
    if use_standard_attn:
        for block in model.blocks:
            block.attn = StandardAttention1D(dim=args.dim, num_heads=args.num_heads)
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Train wave field audio diffusion on SC09")
    p.add_argument("--attn", default="wave", choices=["wave", "standard"])
    p.add_argument("--dynamic_filter", action=argparse.BooleanOptionalAction, default=False,
                   help="Data-dependent spectral filter on the wave kernel. Wave attn only.")
    p.add_argument("--gating", default="pointwise", choices=["pointwise", "hyena"],
                   help="Wave-attn gate: 'pointwise' (sigmoid q⊙k) or 'hyena' "
                        "(q ⊙ conv(k ⊙ v), position-dependent routing). Wave attn only.")
    p.add_argument("--conditioning", default="physics", choices=["physics", "adaln"])
    p.add_argument("--num_classes", type=int, default=None,
                   help="Enable class-conditional generation + CFG (SC09 digits → 10). "
                        "Directly targets the wave+physics mode-collapse via the "
                        "CFG diversity/quality knob. Omit for unconditional.")
    p.add_argument("--class_dropout", type=float, default=0.1,
                   help="Label-dropout prob for classifier-free-guidance training.")
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="CFG scale for sampling (1.0 = no guidance).")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--timestep_dim", type=int, default=128)
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--patch_size", type=int, default=16,
                   help="Samples per token. patch_size=16 → 1024 tokens at 16 kHz × 1 s")
    p.add_argument("--save_dir", default="outputs/sc09")
    p.add_argument("--sample_every", type=int, default=10)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0,
                   help="DDIM stochasticity for sampling. 0=deterministic, 1=full "
                        "DDPM variance. >0 injects noise each step, which can break "
                        "a self-conditioning collapse attractor (see diagnostic).")
    p.add_argument("--parameterization", default="v", choices=["eps", "v"])
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--self_cond", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n_samples", type=int, default=8,
                   help="Number of audio samples to save as .wav at each evaluation")
    p.add_argument("--n_metric_samples", type=int, default=1000,
                   help="Number of samples generated for FSD/accuracy at each eval. "
                        "Set to 0 to disable in-training metrics.")
    p.add_argument("--n_real_features", type=int, default=2000,
                   help="Real clips used to compute reference FSD statistics.")
    p.add_argument("--classifier_weights", default="metrics/weights/sc09_classifier.pt",
                   help="Path to trained SC09 classifier (feature extractor for FSD).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="bf16 autocast for training + sampling (CUDA only; ignored "
                        "on CPU/MPS where it has no benefit or support).")
    p.add_argument("--fast", action="store_true",
                   help="Throughput preset for big GPUs: batch_size=256, lr=4e-4. "
                        "Fewer, larger gradient steps — speeds wall-clock at the "
                        "cost of fewer updates per epoch (consider more epochs).")
    args = p.parse_args()
    if args.fast:
        args.batch_size = 256
        args.lr = 4e-4
    return args


# ---------------------------------------------------------------------------
# Visualization & sample saving
# ---------------------------------------------------------------------------

def save_audio_grid(samples: torch.Tensor, out_dir: str, prefix: str):
    """Save individual .wav files plus a combined waveform/spectrogram plot."""
    os.makedirs(out_dir, exist_ok=True)
    samples = samples.clamp(-1, 1).cpu()
    n = samples.shape[0]

    # Save individual wav files. soundfile writes .wav natively; torchaudio.save
    # now routes through torchcodec (not installed) and would crash here.
    for i in range(n):
        path = os.path.join(out_dir, f"{prefix}_{i:02d}.wav")
        sf.write(path, samples[i, 0].numpy(), TARGET_SR)   # (L,) mono, float in [-1, 1]

    # Waveform + log-mel spectrogram grid
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5), squeeze=False)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=1024, hop_length=256, n_mels=64
    )

    for i in range(n):
        wav = samples[i, 0]
        axes[0][i].plot(wav.numpy(), linewidth=0.4)
        axes[0][i].set_ylim(-1.05, 1.05)
        axes[0][i].set_xticks([]); axes[0][i].set_yticks([])
        axes[0][i].set_title(f"Sample {i}", fontsize=8)

        mel_spec = mel(wav.unsqueeze(0))                          # (1, 64, T)
        log_mel = torch.log(mel_spec + 1e-6).squeeze(0).numpy()
        axes[1][i].imshow(log_mel, origin="lower", aspect="auto", cmap="magma")
        axes[1][i].set_xticks([]); axes[1][i].set_yticks([])

    axes[0][0].set_ylabel("waveform", fontsize=8)
    axes[1][0].set_ylabel("log-mel", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_grid.png"), dpi=100, bbox_inches="tight")
    plt.close()


def plot_kernel_params_audio(model, diffusion, path, num_ts_samples=20):
    """Plot per-head wave kernel (α, ω, φ) vs diffusion timestep."""
    device = next(model.parameters()).device
    model.eval()
    ts = torch.linspace(0, diffusion.T - 1, num_ts_samples, dtype=torch.long, device=device)
    t_embs = model.time_embed(ts)

    rows = []
    for block in model.blocks:
        attn = block.attn
        if not hasattr(attn, "use_ts_cond") or not attn.use_ts_cond:
            return  # not physics-conditioned → nothing to plot
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
    plt.suptitle("Wave kernel parameters vs diffusion timestep (audio)", y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_training_curve(losses, path):
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("SC09 audio diffusion training")
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()


# ---------------------------------------------------------------------------
# Metric helpers (FSD + classifier readouts + mel-distribution comparison)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batched(diffusion, model, n_total, batch_size, device, ddim_steps, eta=0.0,
                     num_classes=None, guidance_scale=1.0):
    """
    Generate n_total audio clips in chunks of batch_size. Returns CPU tensor.
    If num_classes is set, draws uniform random digit labels per chunk (matching
    SC09's ~uniform class prior) and applies classifier-free guidance.
    """
    chunks, done = [], 0
    while done < n_total:
        b = min(batch_size, n_total - done)
        shape = (b, 1, TARGET_LEN)
        y = (torch.randint(0, num_classes, (b,), device=device)
             if num_classes is not None else None)
        if ddim_steps > 0:
            x = diffusion.ddim_sample(model, shape, device, num_steps=ddim_steps,
                                       eta=eta, progress=False, y=y,
                                       guidance_scale=guidance_scale)
        else:
            x = diffusion.sample(model, shape, device, progress=False, y=y,
                                 guidance_scale=guidance_scale)
        chunks.append(x.cpu()); done += b
    return torch.cat(chunks, dim=0)


def save_mel_distribution_plot(real: torch.Tensor, gen: torch.Tensor, path: str):
    """Side-by-side mean log-mel spectrogram: real vs generated vs difference."""
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=1024, hop_length=256, n_mels=64,
    )

    def mean_log_mel(x: torch.Tensor) -> np.ndarray:
        specs = mel(x.cpu())                              # (B, 1, n_mels, T)
        return torch.log(specs.clamp(min=1e-6)).squeeze(1).mean(dim=0).numpy()

    real_lm = mean_log_mel(real)
    gen_lm  = mean_log_mel(gen.clamp(-1, 1))
    diff_lm = gen_lm - real_lm

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, m, title, cmap in zip(
        axes,
        [real_lm, gen_lm, diff_lm],
        ["Real (mean log-mel)", "Generated (mean log-mel)", "Generated − Real"],
        ["magma", "magma", "RdBu_r"],
    ):
        im = ax.imshow(m, origin="lower", aspect="auto", cmap=cmap)
        ax.set_title(title); ax.set_xlabel("time"); ax.set_ylabel("mel")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=100, bbox_inches="tight"); plt.close()


def plot_metric_curves(history: list, path: str):
    """Three-panel: FSD / confident-accuracy / class-entropy over epochs."""
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fsd    = [h["frechet_sc09_distance"] for h in history]
    acc    = [h["confident_accuracy"] for h in history]
    ent    = [h["class_entropy"] for h in history]
    ent_u  = history[0].get("class_entropy_uniform", math.log(10))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, fsd, "o-"); axes[0].set_title("Frechet SC09 Distance"); axes[0].set_xlabel("epoch")
    axes[1].plot(epochs, acc, "o-"); axes[1].set_title("Confident accuracy (top-1 ≥ 0.5)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylim(0, 1)
    axes[2].plot(epochs, ent, "o-"); axes[2].axhline(ent_u, ls="--", color="gray", label=f"uniform={ent_u:.2f}")
    axes[2].set_title("Predicted-class entropy"); axes[2].set_xlabel("epoch"); axes[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=100, bbox_inches="tight"); plt.close()


def load_classifier_and_real_features(args, train_ds, device):
    """Load SC09 classifier and extract reference real-data features. None if disabled."""
    if args.n_metric_samples <= 0:
        return None, None
    path = args.classifier_weights
    if not os.path.exists(path):
        print(f"[metrics disabled] classifier weights not found at {path}. "
              "Run: python -m metrics.train_classifier --task sc09")
        return None, None

    classifier = SC09Classifier().to(device)
    classifier.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    classifier.eval()
    print(f"Loaded SC09 classifier for FSD: {path}")

    # Cache features from a fixed real-data subsample
    n_real = min(args.n_real_features, len(train_ds))
    idx = torch.randperm(len(train_ds))[:n_real]
    real = torch.stack([train_ds[i][0] for i in idx.tolist()], dim=0)
    print(f"  extracting features for {n_real} real clips…")
    feats_real = extract_features(classifier, real, batch_size=128, device=device)
    return classifier, (feats_real, real)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    # bf16 autocast pays off on CUDA tensor cores; on CPU/MPS it's a no-op
    # (no benefit / patchy bf16 support), so gate it to keep those paths fp32.
    # cache_enabled=False: on older torch (pod images ship 2.4) the autocast
    # weight cache retains detached casts created under no_grad, which zeroed
    # gradients for cast-cached weights on self-cond batches and crashed
    # standard+adaln outright. p_losses no longer runs its first pass under
    # no_grad, but the cache buys little for a model this small — keep it off.
    use_amp = args.amp and device.type == "cuda"
    amp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                      cache_enabled=False)) \
        if use_amp else nullcontext
    print(f"AMP (bf16 autocast): {use_amp}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print("Loading SC09 (digit subset of Speech Commands)…")
    train_ds = SC09(root="./data", subset="training")
    print(f"  → {len(train_ds)} training clips")
    # Dataset is cached in RAM, so __getitem__ is a trivial index. Worker
    # processes would each copy the ~2GB cache and add IPC overhead for no
    # gain, so load in the main process (num_workers=0).
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    use_std = (args.attn == "standard")
    model = make_audio_model(args, use_standard_attn=use_std).to(device)
    print(f"Parameters: {model.param_count():,}")
    print(f"Tokens: {model.num_patches}  attn={args.attn}  cond={args.conditioning}  "
          f"self_cond={args.self_cond}")

    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    diffusion = DDPMDiffusion(
        num_timesteps=args.num_timesteps,
        schedule="cosine",
        parameterization=args.parameterization,
    )
    diffusion.to(device)
    ema = EMA(model, decay=args.ema_decay)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ------------------------------------------------------------------
    # Sample-quality metrics: classifier + cached real-data features
    # ------------------------------------------------------------------
    classifier, real_pack = load_classifier_and_real_features(args, train_ds, device)
    metric_history: list = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{args.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device)                                     # (B, 1, L)
            y = y.to(device) if args.num_classes is not None else None
            t = torch.randint(0, args.num_timesteps, (x.shape[0],), device=device)
            with amp_ctx():
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
        # Periodic evaluation
        # ------------------------------------------------------------------
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            ema.ema_model.eval()

            # (a) small batch saved as listenable .wav grid (visual / audible debug)
            with torch.no_grad(), amp_ctx():
                shape = (args.n_samples, 1, TARGET_LEN)
                y_listen = (torch.arange(args.n_samples, device=device) % args.num_classes
                            if args.num_classes is not None else None)
                if args.ddim_steps > 0:
                    listen_samples = diffusion.ddim_sample(
                        ema.ema_model, shape, device,
                        num_steps=args.ddim_steps, eta=args.eta, progress=False,
                        y=y_listen, guidance_scale=args.guidance_scale,
                    )
                else:
                    listen_samples = diffusion.sample(ema.ema_model, shape, device,
                                                      progress=False, y=y_listen,
                                                      guidance_scale=args.guidance_scale)
            sample_dir = os.path.join(args.save_dir, f"samples_epoch{epoch:04d}")
            save_audio_grid(listen_samples, sample_dir, prefix="gen")

            # (b) larger sample population for FSD / classifier metrics
            if classifier is not None:
                feats_real, real_audio = real_pack
                print(f"  generating {args.n_metric_samples} samples for FSD…")
                with amp_ctx():
                    metric_samples = generate_batched(
                        diffusion, ema.ema_model,
                        n_total=args.n_metric_samples,
                        batch_size=max(8, args.batch_size),
                        device=device, ddim_steps=args.ddim_steps, eta=args.eta,
                        num_classes=args.num_classes,
                        guidance_scale=args.guidance_scale,
                    )
                feats_gen = extract_features(classifier, metric_samples.clamp(-1, 1),
                                              batch_size=128, device=device)
                fsd = frechet_distance(feats_real, feats_gen)
                cm = classifier_metrics(classifier, metric_samples.clamp(-1, 1),
                                         batch_size=128, device=device)
                row = {"epoch": epoch, "frechet_sc09_distance": fsd, **cm}
                metric_history.append(row)
                print(f"  FSD={fsd:.3f}  confident_acc={cm['confident_accuracy']:.3f}  "
                      f"entropy={cm['class_entropy']:.3f}/{cm['class_entropy_uniform']:.3f}")

                # Mel-distribution comparison plot uses a fixed real subsample
                # vs the freshly generated batch (smaller = faster to plot).
                save_mel_distribution_plot(
                    real_audio[: min(256, real_audio.shape[0])],
                    metric_samples[: min(256, metric_samples.shape[0])],
                    os.path.join(args.save_dir, f"mel_dist_epoch{epoch:04d}.png"),
                )
                plot_metric_curves(metric_history, os.path.join(args.save_dir, "metric_curves.png"))
                with open(os.path.join(args.save_dir, "metric_history.json"), "w") as f:
                    json.dump(metric_history, f, indent=2)

            if args.conditioning == "physics" and args.attn == "wave":
                plot_kernel_params_audio(
                    ema.ema_model, diffusion,
                    os.path.join(args.save_dir, f"kernels_epoch{epoch:04d}.png"),
                )

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
                "args": vars(args),
            }, os.path.join(args.save_dir, f"checkpoint_epoch{epoch:04d}.pt"))

    plot_training_curve(losses, os.path.join(args.save_dir, "training_curve.png"))
    print(f"\nDone. Outputs in: {args.save_dir}")
    if metric_history:
        final = metric_history[-1]
        print(f"Final metrics: FSD={final['frechet_sc09_distance']:.3f}  "
              f"confident_acc={final['confident_accuracy']:.3f}  "
              f"entropy={final['class_entropy']:.3f}")
        print("For 10k-sample final eval, run: "
              f"python -m metrics.evaluate {args.save_dir} --n_samples 10000")


if __name__ == "__main__":
    main()
