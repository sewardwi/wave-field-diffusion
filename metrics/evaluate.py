"""
One-shot sample-quality evaluation for a saved diffusion checkpoint.

Loads the EMA weights from a checkpoint directory, generates N samples via
DDIM, and computes the appropriate sample-quality metric for that modality:

  - CIFAR:  FID via clean-fid against CIFAR-10 train set
  - MNIST:  Frechet MNIST Distance via metrics/weights/mnist_classifier.pt
            + classifier accuracy/entropy on samples
  - SC09:   Frechet SC09 Distance via metrics/weights/sc09_classifier.pt
            + classifier accuracy/entropy + mel-spectrogram dist plot

Usage:
    python -m metrics.evaluate outputs/cifar_wave_physics_2d_sc
    python -m metrics.evaluate outputs/mnist_v7_selfcond --n_samples 5000
    python -m metrics.evaluate outputs/sc09_wave_physics --modality sc09

Output:
    <checkpoint_dir>/metrics.json
    <checkpoint_dir>/eval_samples.png         (image grid OR mel-spec dist plot)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wave_field.diffusion import DDPMDiffusion
from denoisers.image import WaveFieldDenoiser
from denoisers.audio import WaveFieldAudioDenoiser
from metrics.classifier import MNISTClassifier, SC09Classifier
from metrics.frechet import extract_features, frechet_distance, classifier_metrics


WEIGHTS_DIR = Path(__file__).parent / "weights"


# ---------------------------------------------------------------------------
# Standard-attention swap (mirrors train_{mnist,cifar,audio}.py)
# ---------------------------------------------------------------------------

class StandardAttention(nn.Module):
    """Drop-in for WaveFieldAttention(.forward(x, t_emb=None))."""

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
        # scaled_dot_product_attention routes to FlashAttention / memory-
        # efficient kernels (O(L) memory, no materialized L×L matrix) — the
        # correct modern softmax baseline. Default scale = head_dim**-0.5.
        # Identical math to the materialized path, so old softmax checkpoints
        # load and evaluate unchanged.
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Modality detection + model reconstruction
# ---------------------------------------------------------------------------

def detect_modality(ckpt_dir: Path, override: str | None) -> str:
    if override is not None:
        return override
    name = ckpt_dir.name.lower()
    for m in ("cifar", "mnist", "sc09"):
        if m in name:
            return m
    raise SystemExit(
        f"Could not detect modality from '{ckpt_dir.name}'. Pass --modality {{mnist,cifar,sc09}}"
    )


def build_image_model(cfg: dict, image_size: int, in_channels: int):
    is_wave = cfg.get("attn", "wave") == "wave"
    model = WaveFieldDenoiser(
        image_size=image_size,
        in_channels=in_channels,
        patch_size=cfg["patch_size"],
        dim=cfg["dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        timestep_dim=cfg["timestep_dim"],
        conditioning=cfg["conditioning"],
        use_2d_kernel=(cfg.get("kernel") == "2d" and is_wave),
        use_self_cond=cfg.get("self_cond", False),
        # New wave-operator flags must match the trained architecture or the EMA
        # weights won't load (silently → garbage eval). Read straight from config.
        dynamic_filter=cfg.get("dynamic_filter", False) and is_wave,
        gating=cfg.get("gating", "pointwise") if is_wave else "pointwise",
        aniso_kernel=cfg.get("aniso_kernel", False) and is_wave,
        num_classes=cfg.get("num_classes", None),
        class_dropout_prob=cfg.get("class_dropout", 0.1),
    )
    if not is_wave:
        for block in model.blocks:
            block.attn = StandardAttention(dim=cfg["dim"], num_heads=cfg["num_heads"])
    return model


def build_audio_model(cfg: dict):
    from datasets.sc09 import TARGET_LEN
    is_wave = cfg.get("attn", "wave") == "wave"
    model = WaveFieldAudioDenoiser(
        sequence_length=TARGET_LEN,
        in_channels=1,
        patch_size=cfg["patch_size"],
        dim=cfg["dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        timestep_dim=cfg["timestep_dim"],
        conditioning=cfg["conditioning"],
        use_self_cond=cfg.get("self_cond", False),
        dynamic_filter=cfg.get("dynamic_filter", False) and is_wave,
        gating=cfg.get("gating", "pointwise") if is_wave else "pointwise",
        num_classes=cfg.get("num_classes", None),
        class_dropout_prob=cfg.get("class_dropout", 0.1),
    )
    if not is_wave:
        for block in model.blocks:
            block.attn = StandardAttention(dim=cfg["dim"], num_heads=cfg["num_heads"])
    return model


def find_latest_checkpoint(ckpt_dir: Path) -> Path:
    ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))
    if not ckpts:
        raise SystemExit(f"No checkpoint_epoch*.pt found in {ckpt_dir}")
    return ckpts[-1]


# ---------------------------------------------------------------------------
# Sample generation (batched, memory-safe)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(
    model: nn.Module,
    diffusion: DDPMDiffusion,
    sample_shape: tuple,
    n_total: int,
    batch_size: int,
    device: torch.device,
    ddim_steps: int,
    eta: float = 0.0,
    num_classes: int | None = None,
    guidance_scale: float = 1.0,
    sampler: str = "ddim",
) -> torch.Tensor:
    """
    Generate n_total samples in batches of batch_size. Returns CPU tensor.
    For conditional models, draws uniform random class labels per batch (matching
    the ~uniform class prior of CIFAR/MNIST/SC09) and applies CFG.
    """
    # bf16 autocast on CUDA tensor cores ~halves sampling wall-clock; no-op
    # elsewhere. FFT ops stay fp32 under autocast, so the wave path is safe.
    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if device.type == "cuda" else nullcontext())
    chunks = []
    n_done = 0
    pbar = tqdm(total=n_total, desc=f"Generating {n_total} samples")
    with amp_ctx:
        while n_done < n_total:
            b = min(batch_size, n_total - n_done)
            shape = (b,) + tuple(sample_shape)
            y = (torch.randint(0, num_classes, (b,), device=device)
                 if num_classes is not None else None)
            if sampler == "dpmpp":
                x = diffusion.dpmpp_2m_sample(model, shape, device, num_steps=ddim_steps,
                                              progress=False, y=y, guidance_scale=guidance_scale)
            else:
                x = diffusion.ddim_sample(model, shape, device, num_steps=ddim_steps,
                                          eta=eta, progress=False, y=y,
                                          guidance_scale=guidance_scale)
            chunks.append(x.detach().cpu())
            n_done += b
            pbar.update(b)
    pbar.close()
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# Modality-specific real-data + metric routines
# ---------------------------------------------------------------------------

def real_mnist_tensor(n: int) -> torch.Tensor:
    """Return n real MNIST images in [-1, 1], shape (n, 1, 28, 28)."""
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    idx = torch.randperm(len(ds))[:n]
    return torch.stack([ds[i][0] for i in idx], dim=0)


def real_sc09_tensor(n: int) -> torch.Tensor:
    """Return n real SC09 clips, shape (n, 1, 16384) in [-1, 1]."""
    from datasets.sc09 import SC09
    # We only need a random subset, so skip the full RAM cache (which would
    # decode all ~31k clips up front just to read n of them).
    ds = SC09(root="./data", subset="training", cache=False)
    idx = torch.randperm(len(ds))[:n]
    return torch.stack([ds[i][0] for i in idx], dim=0)


def eval_mnist(samples: torch.Tensor, device: torch.device, n_real: int) -> dict:
    cls = MNISTClassifier().to(device)
    cls.load_state_dict(torch.load(WEIGHTS_DIR / "mnist_classifier.pt",
                                   map_location=device, weights_only=True))
    real = real_mnist_tensor(n_real)
    feats_real = extract_features(cls, real, batch_size=512, device=device)
    feats_gen  = extract_features(cls, samples.clamp(-1, 1), batch_size=512, device=device)
    fmd = frechet_distance(feats_real, feats_gen)
    cm = classifier_metrics(cls, samples.clamp(-1, 1), batch_size=512, device=device)
    cm["frechet_mnist_distance"] = fmd
    return cm


def eval_sc09(samples: torch.Tensor, device: torch.device, n_real: int,
              out_dir: Path) -> dict:
    cls = SC09Classifier().to(device)
    cls.load_state_dict(torch.load(WEIGHTS_DIR / "sc09_classifier.pt",
                                   map_location=device, weights_only=True))
    real = real_sc09_tensor(n_real)
    feats_real = extract_features(cls, real, batch_size=128, device=device)
    feats_gen  = extract_features(cls, samples.clamp(-1, 1), batch_size=128, device=device)
    fsd = frechet_distance(feats_real, feats_gen)
    cm = classifier_metrics(cls, samples.clamp(-1, 1), batch_size=128, device=device)
    cm["frechet_sc09_distance"] = fsd

    # Mean log-mel spectrogram comparison (real vs generated)
    _save_mel_distribution_plot(real, samples, out_dir / "mel_distribution.png")
    return cm


def _save_mel_distribution_plot(real: torch.Tensor, gen: torch.Tensor, path: Path):
    """Side-by-side: mean log-mel spectrogram for real vs generated batches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torchaudio

    from datasets.sc09 import TARGET_SR
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=1024, hop_length=256, n_mels=64,
    )

    def mean_log_mel(x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            specs = mel(x.cpu())                            # (B, 1, n_mels, T)
            log = torch.log(specs.clamp(min=1e-6)).squeeze(1).mean(dim=0).numpy()
        return log

    real_lm = mean_log_mel(real)
    gen_lm  = mean_log_mel(gen.clamp(-1, 1))
    diff_lm = gen_lm - real_lm

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, m, title in zip(axes, [real_lm, gen_lm, diff_lm],
                            ["Real (mean log-mel)", "Generated (mean log-mel)",
                             "Generated − Real"]):
        im = ax.imshow(m, origin="lower", aspect="auto",
                       cmap="magma" if "Real" in title or "Generated " in title else "RdBu_r")
        ax.set_title(title); ax.set_xlabel("time"); ax.set_ylabel("mel")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


CIFAR_REAL_PNG_DIR = Path("./data/cifar10_real_pngs")


def prepare_cifar_real_pngs(n: int = 50_000) -> Path:
    """
    Dump n real CIFAR-10 train images to a shared cache dir as PNGs.
    One-time setup so all CIFAR eval calls reuse the same real-image directory
    for FID computation. Bypasses clean-fid's broken reference-stats download.
    """
    CIFAR_REAL_PNG_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(CIFAR_REAL_PNG_DIR.glob("*.png"))
    if len(existing) >= n:
        return CIFAR_REAL_PNG_DIR

    print(f"  preparing {n} real CIFAR-10 PNGs in {CIFAR_REAL_PNG_DIR} …")
    import torchvision
    from PIL import Image
    ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    # ds[i] returns (PIL.Image, label). Save as-is — no resize, no normalization.
    for i in range(len(existing), n):
        img, _ = ds[i]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img.save(CIFAR_REAL_PNG_DIR / f"{i:05d}.png")
    return CIFAR_REAL_PNG_DIR


def eval_cifar(samples: torch.Tensor | None, out_dir: Path, n_real: int,
                fid_device: str = "cpu", reuse_samples: bool = False) -> dict:
    """
    CIFAR FID via clean-fid in dir-vs-dir mode against a local copy of CIFAR-10
    train images.  Avoids clean-fid's hardcoded reference-stats URL which is
    currently 404ing on the CMU host.

    fid_device:    clean-fid's Inception runs here ('cpu' is safe on Mac/MPS;
                   'cuda' on CUDA machines). Inception on MPS is unsupported.
    reuse_samples: if True, skip PNG writing and use whatever's already in
                   <out_dir>/fid_gen_tmp/.  Used to recover from a previous
                   FID-only failure without regenerating samples.
    """
    from cleanfid import fid
    import torchvision.utils as vutils

    gen_dir = out_dir / "fid_gen_tmp"
    gen_dir.mkdir(exist_ok=True)

    if reuse_samples:
        existing = sorted(gen_dir.glob("*.png"))
        if not existing:
            raise SystemExit(f"--reuse_samples set but no PNGs found in {gen_dir}")
        print(f"  reusing {len(existing)} existing PNG samples from {gen_dir}")
        n_generated = len(existing)
    else:
        if samples is None:
            raise SystemExit("samples is None and reuse_samples=False")
        # Existing PNGs from a prior run would skew FID — clear them.
        for f in gen_dir.glob("*.png"):
            f.unlink()
        imgs = ((samples.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
        for i, img in enumerate(imgs):
            vutils.save_image(img.float() / 255, gen_dir / f"{i:05d}.png")
        n_generated = int(samples.shape[0])

    real_dir = prepare_cifar_real_pngs(n=max(n_real, n_generated))

    print(f"  computing FID on {fid_device} (Inception V3, dir-vs-dir)…")
    score = fid.compute_fid(
        str(real_dir), str(gen_dir),
        mode="clean", batch_size=64, device=torch.device(fid_device),
        num_workers=0, use_dataparallel=False,
    )
    return {
        "fid_clean_cifar10_train": float(score),
        "n_generated": n_generated,
        "n_real": int(len(sorted(real_dir.glob("*.png")))),
        "fid_device": fid_device,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint_dir")
    p.add_argument("--modality", default=None, choices=["mnist", "cifar", "sc09"],
                   help="Default: auto-detect from directory name.")
    p.add_argument("--n_samples", type=int, default=10000,
                   help="Number of samples to generate (default 10000 for final eval).")
    p.add_argument("--n_real", type=int, default=10000,
                   help="Number of real samples used for Frechet baseline (MNIST/SC09).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=None,
                   help="CFG scale for conditional checkpoints. Default: read from "
                        "config.json if trained conditional, else 1.0 (no guidance).")
    p.add_argument("--sampler", default=None, choices=["ddim", "dpmpp"],
                   help="Override sampler. Default: read from config.json (else ddim).")
    p.add_argument("--eta", type=float, default=0.0,
                   help="DDIM stochasticity. 0=deterministic, 1=full DDPM variance. "
                        ">0 injects noise per step — use to A/B whether a "
                        "self-conditioning collapse is a sampling attractor.")
    p.add_argument("--fid_device", default=None,
                   help="Device for clean-fid Inception ('cpu' on Mac, 'cuda' otherwise). "
                        "Default: auto.")
    p.add_argument("--reuse_samples", action="store_true",
                   help="CIFAR only — skip sample generation and reuse PNGs in "
                        "<checkpoint_dir>/fid_gen_tmp/. Useful for retrying after "
                        "a FID-step failure.")
    p.add_argument("--out_name", default="metrics.json",
                   help="Output filename inside checkpoint_dir. Use e.g. "
                        "metrics_g2.0.json for guidance sweeps so the canonical "
                        "metrics.json isn't overwritten.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Inception on MPS is not supported by clean-fid; default to CPU on Mac.
    if args.fid_device is None:
        args.fid_device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_dir():
        raise SystemExit(f"Not a directory: {ckpt_dir}")
    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing config.json in {ckpt_dir}")
    cfg = json.loads(config_path.read_text())

    modality = detect_modality(ckpt_dir, args.modality)
    print(f"Modality: {modality}")
    print(f"Config:   {config_path}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device:   {device}")

    # ------------------------------------------------------------------
    # Build model + load EMA weights
    # ------------------------------------------------------------------
    if modality == "mnist":
        model = build_image_model(cfg, image_size=28, in_channels=1)
        sample_shape = (1, 28, 28)
    elif modality == "cifar":
        model = build_image_model(cfg, image_size=32, in_channels=3)
        sample_shape = (3, 32, 32)
    else:
        model = build_audio_model(cfg)
        sample_shape = (1, 16384)
    model = model.to(device)

    ckpt_path = find_latest_checkpoint(ckpt_dir)
    print(f"Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("ema_state_dict") or ckpt.get("model_state_dict")
    if state is None:
        raise SystemExit("Checkpoint missing both ema_state_dict and model_state_dict")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  missing keys:    {len(missing)}\n  unexpected keys: {len(unexpected)}")
    model.eval()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Generate samples
    # ------------------------------------------------------------------
    diffusion = DDPMDiffusion(
        num_timesteps=cfg["num_timesteps"],
        schedule="cosine",
        parameterization=cfg.get("parameterization", "v"),
    )
    diffusion.to(device)

    # Conditional eval: sample with uniform random labels + CFG so FID is
    # meaningful. num_classes/guidance/sampler default to what was trained.
    num_classes = cfg.get("num_classes", None)
    guidance_scale = (args.guidance_scale if args.guidance_scale is not None
                      else cfg.get("guidance_scale", 1.0 if num_classes is None else 1.5))
    sampler = args.sampler or cfg.get("sampler", "ddim")

    if args.reuse_samples and modality == "cifar":
        print("Skipping sample generation (--reuse_samples).")
        samples = None
    else:
        if num_classes is not None:
            print(f"Conditional eval: {num_classes} classes, "
                  f"guidance_scale={guidance_scale}, sampler={sampler}")
        samples = generate_samples(
            model, diffusion, sample_shape,
            n_total=args.n_samples, batch_size=args.batch_size,
            device=device, ddim_steps=args.ddim_steps, eta=args.eta,
            num_classes=num_classes, guidance_scale=guidance_scale, sampler=sampler,
        )
        print(f"Samples: {tuple(samples.shape)}  range=[{samples.min():.3f}, {samples.max():.3f}]")

    # ------------------------------------------------------------------
    # Modality-specific metrics
    # ------------------------------------------------------------------
    if modality == "mnist":
        metrics_out = eval_mnist(samples, device, n_real=min(args.n_real, 60000))
        metric_key = "frechet_mnist_distance"
    elif modality == "cifar":
        metrics_out = eval_cifar(samples, ckpt_dir, n_real=args.n_real,
                                  fid_device=args.fid_device,
                                  reuse_samples=args.reuse_samples)
        metric_key = "fid_clean_cifar10_train"
    else:
        metrics_out = eval_sc09(samples, device, n_real=min(args.n_real, 30000),
                                out_dir=ckpt_dir)
        metric_key = "frechet_sc09_distance"

    metrics_out.update({
        "modality": modality,
        "checkpoint": str(ckpt_path.name),
        "n_samples": int(args.n_samples),
        "ddim_steps": int(args.ddim_steps),
        "eta": float(args.eta),
        "sampler": sampler,
        "guidance_scale": float(guidance_scale),
        "num_classes": num_classes,
    })

    out_path = ckpt_dir / args.out_name
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("\n=== Metrics ===")
    print(f"  {metric_key}: {metrics_out[metric_key]:.4f}")
    if "confident_accuracy" in metrics_out:
        print(f"  confident_accuracy: {metrics_out['confident_accuracy']:.3f}")
        print(f"  class_entropy:      {metrics_out['class_entropy']:.3f}  "
              f"(uniform = {metrics_out['class_entropy_uniform']:.3f})")
    print(f"\nWrote → {out_path}")


if __name__ == "__main__":
    main()
