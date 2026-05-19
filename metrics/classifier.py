"""
Tiny classifiers used as feature extractors for Frechet-distance metrics.

These are NOT diffusion models — they are dataset-specific embedders that
play the same role Inception V3 plays for natural-image FID.  We train each
once on the real dataset, then use the penultimate-layer features to compute
Frechet distance between real and generated sample distributions.

  MNISTClassifier  →  2D CNN, ~50K params, embed_dim=64
  SC09Classifier   →  1D CNN, ~250K params, embed_dim=128

Both expose:
  forward(x)          → logits (B, num_classes)
  embed(x)            → penultimate features (B, embed_dim)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MNIST: small 2D CNN
# ---------------------------------------------------------------------------

class MNISTClassifier(nn.Module):
    """Tiny CNN for 28x28 grayscale digits. Used as MNIST embedder for FMD."""

    EMBED_DIM = 64
    NUM_CLASSES = 10

    def __init__(self):
        super().__init__()
        # 28→14→7 spatial via two stride-2 convs
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                                         # (B, 64, 1, 1)
            nn.Flatten(),                                                    # (B, 64)
        )
        self.classifier = nn.Linear(self.EMBED_DIM, self.NUM_CLASSES)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 28, 28) in [-1, 1] or [0, 1]. Returns (B, 64)."""
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))


# ---------------------------------------------------------------------------
# SC09: small 1D CNN
# ---------------------------------------------------------------------------

class SC09Classifier(nn.Module):
    """
    Tiny 1D CNN for 16384-sample audio clips at 16 kHz.

    Strided-conv frontend reduces 16384 → 1024 → 256 → 64 → 16 timesteps.
    Penultimate features (after global pool) used as audio embedding for FSD.
    """

    EMBED_DIM = 128
    NUM_CLASSES = 10

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 16384 → 1024
            nn.Conv1d(1, 32, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            # 1024 → 256
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            # 256 → 64
            nn.Conv1d(64, 96, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(96), nn.ReLU(inplace=True),
            # 64 → 16
            nn.Conv1d(96, 128, kernel_size=4, stride=4),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),                       # (B, 128, 1)
            nn.Flatten(),                                  # (B, 128)
        )
        self.classifier = nn.Linear(self.EMBED_DIM, self.NUM_CLASSES)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 16384) in [-1, 1]. Returns (B, 128)."""
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
