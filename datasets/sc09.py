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

    def __init__(self, root: str = "./data", subset: str = "training",
                 cache: bool = True):
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

        # Optionally decode the whole (small, ~2GB) dataset into a RAM tensor
        # once, so __getitem__ becomes a cheap index instead of a per-epoch
        # disk read + resample. This is the single biggest training speedup on
        # a fast GPU, which otherwise sits idle waiting on CPU audio decoding.
        self._cache = None
        self._labels = None
        if cache:
            self._build_cache()

    def _get_resampler(self, sr: int) -> torchaudio.transforms.Resample:
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=TARGET_SR
            )
        return self._resamplers[sr]

    def _decode(self, path: str) -> torch.Tensor:
        """Read one .wav → mono, 16 kHz, exactly TARGET_LEN samples, peak-normed."""
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
        return wav / peak

    def _build_cache(self) -> None:
        n = len(self.items)
        print(f"  caching {n} clips into RAM "
              f"(~{n * TARGET_LEN * 4 / 1e9:.1f} GB)…", flush=True)
        buf = torch.empty(n, 1, TARGET_LEN, dtype=torch.float32)
        labels = torch.empty(n, dtype=torch.long)
        for i, (path, label) in enumerate(self.items):
            buf[i] = self._decode(path)
            labels[i] = label
        self._cache = buf
        self._labels = labels

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        if self._cache is not None:
            return self._cache[idx], int(self._labels[idx])
        path, label = self.items[idx]
        return self._decode(path), label
