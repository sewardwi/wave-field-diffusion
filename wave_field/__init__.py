"""
wave_field — core architecture for wave field diffusion.

This package contains *only* the reusable, task-agnostic architecture:
the wave field attention mechanism, the shared transformer building blocks,
and the diffusion process / EMA infrastructure.

Task-specific denoisers (image, audio) live in `denoisers/`.
Dataset loaders live in `datasets/`.
"""

from .attention import WaveFieldAttention, WaveFieldAttention2D
from .blocks import (
    WaveFieldBlock,
    TimestepEmbedder,
    FinalLayer,
    timestep_sinusoidal,
    pos_embed_2d_sincos,
    modulate,
)
from .diffusion import DDPMDiffusion, EMA

__all__ = [
    "WaveFieldAttention",
    "WaveFieldAttention2D",
    "WaveFieldBlock",
    "TimestepEmbedder",
    "FinalLayer",
    "timestep_sinusoidal",
    "pos_embed_2d_sincos",
    "modulate",
    "DDPMDiffusion",
    "EMA",
]
