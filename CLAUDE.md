# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Exploratory ML research: replacing softmax self-attention in a diffusion denoiser with a wave-equation-inspired mechanism — per-head damped oscillation kernels `k(r) = exp(-α·r)·cos(ω·r + φ)` applied via FFT convolution (O(n log n)). The central hypothesis is that conditioning the kernel physics (α, ω, φ) on the diffusion timestep is a better inductive bias than generic AdaLN conditioning. See README.md for results so far (headline: per-timestep MSE and Frechet metrics disagree, differently on MNIST vs CIFAR). The SC09 audio leg (1024 tokens, where O(n log n) matters) has infrastructure in place but runs are pending.

## Environment & commands

Python 3.11 venv at `.venv/` — use `.venv/bin/python` (deps in `requirements.txt`; PyTorch + clean-fid + torchaudio). No test suite, linter, or build system; the correctness gate is:

```bash
python validate_attention.py       # Phase-0 checks: shapes, gradient flow through α/ω/φ,
                                   # kernel diagnostics → outputs/phase0/
```

Training (one script per dataset; heavy runs are done on rented RunPod GPUs, not locally):

```bash
python train_mnist.py --attn wave --conditioning physics --save_dir outputs/my_run
python train_cifar.py ...          # same flag surface; dim=256 depth=6 defaults
python train_audio.py ...          # SC09 waveforms, 16384 samples / patch 16 → 1024 tokens
```

Key shared flags: `--attn {wave,standard}`, `--conditioning {physics,adaln}`, `--kernel {1d,2d}`, `--dynamic_filter`, `--gating {pointwise,hyena}`, `--aniso_kernel`, `--self_cond` (default on), `--num_classes 10` + `--guidance_scale` for class-conditional CFG, `--sampler {ddim,dpmpp,ddpm}`.

Evaluation (Frechet metrics; modality auto-detected from the checkpoint dir name, override with `--modality`):

```bash
python -m metrics.evaluate outputs/my_run --n_samples 10000
python -m metrics.train_classifier --task {mnist,sc09}   # feature extractors; weights committed in metrics/weights/
python benchmark_attn.py                                  # wave vs softmax wall-clock scaling
cd comparison && python compare_cifar.py                  # cross-run sample/loss/kernel diagnostics
```

Full ablation sweeps (designed for unattended RunPod boxes):

```bash
bash scripts/run_image_ablation.sh    # cumulative CIFAR feature ablation (+optional MNIST)
bash scripts/run_sc09_ablation.sh     # 4-config SC09 sweep
# Env knobs: SMOKE=1 (2-epoch dry run), FAST=1 (big-GPU batch), EPOCHS=N,
# AUTOPUSH=1 (git-push small artifacts after each run), SHUTDOWN=1 (self-terminate pod)
```

## Architecture

- **`wave_field/`** — reusable, task-agnostic core. `attention.py`: `WaveFieldAttention` (1D) and `WaveFieldAttention2D` (radial kernel over the patch grid), including the optional dynamic spectral filter (content-adaptive filter expressed as coefficients over a Hann frequency basis — `hann_freq_basis`), hyena-style gating, and anisotropic kernels. `blocks.py`: `WaveFieldBlock` (transformer block with time-conditioned residual gates), timestep/label embedders, AdaLN modulation. `diffusion.py`: `DDPMDiffusion` (cosine schedule, v-prediction, min-SNR-γ=5 weighting, DDIM/DPM++/DDPM samplers) and `EMA` (decay 0.9999 — sampling always uses EMA weights).
- **`denoisers/`** — task-specific DiT-style backbones that patchify input, stack `WaveFieldBlock`s, and unpatchify (`image.py`, `audio.py`).
- **`datasets/sc09.py`** — Speech Commands digit subset, 16 kHz, exactly 16384 samples/clip.
- **`metrics/`** — sample-quality eval. `evaluate.py` rebuilds a model from a run dir's `config.json` + latest checkpoint, generates samples, computes clean-FID (CIFAR) or classifier-based Frechet distance (MNIST FMD / SC09 FSD), writes `metrics.json` into the run dir.
- **`comparison/`** — analysis scripts + written-up findings (`comparison/README.md` has the metric-disagreement discussion); **`results/`** — committed aggregate tables and sample grids.

### The two conditioning modes (the core experiment)

`physics`: a small MLP maps the timestep embedding to per-head perturbations of (α, ω, φ), so kernels reshape across the reverse process (broad/smooth at high noise → sharp/oscillatory at low noise). `adaln`: kernels are static; conditioning happens via standard AdaLN in the block. Every ablation crosses `--attn` × `--conditioning`, and matched parameter counts between arms matter when comparing.

### Run-directory convention

Each training run writes to `outputs/<name>/` (gitignored): `config.json`, checkpoints (`*.pt`), `training.log`, sample grids, and eventually `metrics.json`. The ablation scripts treat an existing `metrics.json` as "run complete" and skip it — delete it to force a re-run. Run names encode the config (e.g. `cifar_wave_dyn_hyena_sc` = wave attn + dynamic filter + hyena gating + self-conditioning). Only small artifacts (json/log/png) get pushed from pods; checkpoints stay on the pod.

## Gotchas

- Never run a model forward under `torch.no_grad()` inside an autocast region that a later grad-enabled forward shares (the self-conditioning pass in `p_losses` is the canonical case). On older torch (pod images ship ~2.4) the autocast weight cache retains the detached casts, silently zeroing gradients for those weights — and crashing `loss.backward()` in configs with no fp32 parameter path (standard attn + adaln). `p_losses` records grad on its first pass and detaches instead; `train_audio.py` additionally sets `cache_enabled=False`.

- `docs/API_KEYS.md` and `docs/INTERVIEW_NOTES.md` are gitignored on purpose — never commit them.
- Kernels are L1-normalized so different (α, ω) give consistent magnitudes; keep this invariant if touching kernel construction.
- The `--attn standard` softmax baseline (`StandardAttention`) is duplicated in each training script *and* in `metrics/evaluate.py` (used to rebuild checkpoints for eval) — changing it means changing all copies.
