"""
SC09 dataset loader — the digit subset of Google Speech Commands.

Filters Speech Commands v2 to the 10 digit classes ("zero" through "nine"),
resamples to 16 kHz, pads/truncates each clip to exactly 16,384 samples
(~1.024 s), and normalizes amplitude to [-1, 1].

Each item is returned as (waveform: (1, 16384), label: int 0..9).
"""

import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


DIGIT_WORDS = ["zero", "one", "two", "three", "four",
               "five", "six", "seven", "eight", "nine"]
WORD_TO_LABEL = {w: i for i, w in enumerate(DIGIT_WORDS)}

TARGET_SR = 16_000
TARGET_LEN = 16_384      # ~1.024 s at 16 kHz; power of 2 friendly for FFT


class SC09(Dataset):
    """
    Args:
        root:    download directory (will fetch ~2GB SpeechCommands if missing)
        subset:  'training' | 'validation' | 'testing'  (passed to torchaudio)
        device:  optional pre-load to GPU memory if dataset fits
    """

    def __init__(self, root: str = "./data", subset: str = "training"):
        super().__init__()
        os.makedirs(root, exist_ok=True)
        # torchaudio's SPEECHCOMMANDS auto-downloads; we filter to digit classes
        full = torchaudio.datasets.SPEECHCOMMANDS(
            root=root, download=True, subset=subset
        )
        # Filter to digit classes only — keep absolute paths for fast __getitem__.
        # torchaudio's _walker format varies by version: in some it's archive-
        # relative, in others it's cwd-relative or absolute. We only rely on the
        # parent directory name (the spoken word) to filter.
        self.items = []
        walker = getattr(full, "_walker", None)
        if walker is None:
            # Fallback: walk the filesystem directly.
            archive = Path(getattr(full, "_path", Path(root) / "SpeechCommands" / "speech_commands_v0.02"))
            walker = [str(p) for word in DIGIT_WORDS for p in (archive / word).glob("*.wav")]

        for rel in walker:
            p = Path(rel)
            word = p.parent.name
            if word not in WORD_TO_LABEL:
                continue
            abs_path = str(p if p.is_absolute() else p.resolve())
            self.items.append((abs_path, WORD_TO_LABEL[word]))

        self._resamplers = {}   # cache resamplers per source sample rate

    def _get_resampler(self, sr: int) -> torchaudio.transforms.Resample:
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=TARGET_SR
            )
        return self._resamplers[sr]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        # soundfile reads .wav natively (torchaudio.load now needs torchcodec)
        data, sr = sf.read(path, dtype="float32", always_2d=True)   # (samples, channels)
        wav = torch.from_numpy(np.ascontiguousarray(data.T))         # (channels, samples)

        # Mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample to 16 kHz if needed
        if sr != TARGET_SR:
            wav = self._get_resampler(sr)(wav)

        # Pad or truncate to TARGET_LEN
        L = wav.shape[-1]
        if L < TARGET_LEN:
            wav = torch.nn.functional.pad(wav, (0, TARGET_LEN - L))
        elif L > TARGET_LEN:
            wav = wav[:, :TARGET_LEN]

        # Normalize peak amplitude into [-1, 1]
        peak = wav.abs().max().clamp(min=1e-8)
        wav = wav / peak

        return wav, label
