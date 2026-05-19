"""
Frechet distance between real and generated sample distributions.

The Frechet distance between two multivariate Gaussians fit to feature
embeddings is the standard population-level "sample quality" metric:
    d² = ||μ_r - μ_g||² + tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^½)

This module provides three pieces:
  - extract_features:    run a classifier over a tensor batch → (N, D) features
  - frechet_distance:    closed-form FD between two feature batches
  - classifier_metrics:  classification accuracy + entropy on generated samples
                         (uses the classifier's classification head, not embeddings)

CIFAR uses Inception via clean-fid externally; this module covers MNIST and SC09.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    classifier: nn.Module,
    x: torch.Tensor,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Run `classifier.embed(x)` in batches and return concatenated features.

    Args:
        classifier:  has .embed() returning (B, D)
        x:           (N, ...) input tensor, on cpu or device
        batch_size:  batched inference for memory
        device:      where to run the classifier (default: classifier's device)
    Returns:
        (N, D) features on CPU
    """
    classifier.eval()
    if device is None:
        device = next(classifier.parameters()).device

    feats = []
    for i in range(0, x.shape[0], batch_size):
        batch = x[i : i + batch_size].to(device)
        f = classifier.embed(batch).detach().cpu()
        feats.append(f)
    return torch.cat(feats, dim=0)


# ---------------------------------------------------------------------------
# Frechet distance
# ---------------------------------------------------------------------------

def frechet_distance(
    feats_real: torch.Tensor | np.ndarray,
    feats_gen:  torch.Tensor | np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Closed-form Frechet distance between Gaussians fit to two feature batches.

    Args:
        feats_real: (N_r, D) features from real samples
        feats_gen:  (N_g, D) features from generated samples
        eps:        regularizer added to covariance diagonal if sqrtm hits a
                    non-finite result (numerical guard, follows clean-fid)
    Returns:
        Scalar Frechet distance.
    """
    if isinstance(feats_real, torch.Tensor):
        feats_real = feats_real.detach().cpu().numpy()
    if isinstance(feats_gen, torch.Tensor):
        feats_gen = feats_gen.detach().cpu().numpy()
    feats_real = feats_real.astype(np.float64)
    feats_gen  = feats_gen.astype(np.float64)

    mu_r, sig_r = feats_real.mean(axis=0), np.cov(feats_real, rowvar=False)
    mu_g, sig_g = feats_gen.mean(axis=0),  np.cov(feats_gen,  rowvar=False)

    diff = mu_r - mu_g
    # Matrix square root of Σ_r Σ_g.
    covmean, _ = linalg.sqrtm(sig_r @ sig_g, disp=False)
    if not np.isfinite(covmean).all():
        # Regularize and retry.
        offset = np.eye(sig_r.shape[0]) * eps
        covmean = linalg.sqrtm((sig_r + offset) @ (sig_g + offset))

    # Numerical sqrtm can yield small imaginary components; discard them.
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            # If the imaginary part is meaningful, something is wrong upstream.
            raise RuntimeError(
                "Frechet sqrtm returned non-trivial imaginary part — check features."
            )
        covmean = covmean.real

    return float(diff @ diff + np.trace(sig_r) + np.trace(sig_g) - 2 * np.trace(covmean))


# ---------------------------------------------------------------------------
# Classification-based readouts
# ---------------------------------------------------------------------------

@torch.no_grad()
def classifier_metrics(
    classifier: nn.Module,
    samples: torch.Tensor,
    batch_size: int = 256,
    confidence_threshold: float = 0.5,
    device: torch.device | None = None,
) -> dict:
    """
    Run the classifier head on generated samples and report:
      - confident_accuracy:  fraction of samples whose top-1 softmax probability
                             exceeds `confidence_threshold`.  Approximates
                             "what fraction of generated outputs look like a
                             real digit class at all?"
      - mean_top1_confidence: mean of top-1 softmax probabilities
      - class_entropy:       Shannon entropy (nats) of the predicted-class
                             histogram.  log(num_classes) = uniform = good;
                             low entropy = mode collapse.
      - class_counts:        list of per-class predicted counts.

    Note: confident_accuracy is NOT compared against ground-truth labels (we
    have none for generated samples). It's a "looks-like-a-digit-at-all" score.
    """
    classifier.eval()
    if device is None:
        device = next(classifier.parameters()).device

    confs = []
    preds = []
    for i in range(0, samples.shape[0], batch_size):
        batch = samples[i : i + batch_size].to(device)
        probs = F.softmax(classifier(batch), dim=-1)
        top_conf, top_pred = probs.max(dim=-1)
        confs.append(top_conf.cpu())
        preds.append(top_pred.cpu())
    confs = torch.cat(confs)
    preds = torch.cat(preds)

    num_classes = classifier.NUM_CLASSES
    counts = torch.bincount(preds, minlength=num_classes).float()
    probs = counts / counts.sum().clamp(min=1)
    entropy = -(probs * (probs.clamp(min=1e-12)).log()).sum().item()

    return {
        "confident_accuracy": (confs >= confidence_threshold).float().mean().item(),
        "mean_top1_confidence": confs.mean().item(),
        "class_entropy": entropy,
        "class_entropy_uniform": float(np.log(num_classes)),  # reference upper bound
        "class_counts": counts.int().tolist(),
    }
