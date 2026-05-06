"""
denoisers — task-specific diffusion backbones built on the wave_field core.

Each denoiser uses the shared `WaveFieldBlock` machinery from `wave_field`,
adding only the data-shape-specific parts: patch embedding, positional
embedding for that shape, and unpatchify.
"""

from .image import WaveFieldDenoiser
from .audio import WaveFieldAudioDenoiser

__all__ = ["WaveFieldDenoiser", "WaveFieldAudioDenoiser"]
