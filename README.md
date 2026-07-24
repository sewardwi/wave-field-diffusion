# Wave Field Diffusion

An exploratory research project that replaces standard self-attention in a diffusion model's denoising backbone with a wave-equation-inspired mechanism: damped oscillation kernels applied via FFT convolution. The thesis is that if the forward diffusion process is dissipative (heat-equation-like, progressively destroying signal), then the reverse generative process should naturally be parameterized by wave dynamics — propagation operators that explicitly undo dissipation rather than the generic content-routing of attention.

## The core mechanism

Each attention head learns three scalar parameters: a damping coefficient α, an angular frequency ω, and a phase φ. From these, the head constructs a 2D radial kernel

```
k(r) = exp(-α · r) · cos(ω · r + φ),     r = √(x² + y²)
```

defined over the patch grid. The value tensor V is then convolved with this kernel via FFT — `irfft2(rfft2(V) · rfft2(k))` — giving an O(n log n) replacement for the O(n²) softmax attention matrix. A content-dependent gate `sigmoid(W_q(Q) ⊙ W_k(K))` lets the output respond to local content without reintroducing quadratic cost, and a frequency-domain modulation step rescales V's spectrum based on a global content summary so the convolution becomes content-conditional.

The interesting property is what happens when α and ω are conditioned on the diffusion timestep. At high noise levels, the model wants broad smooth kernels (high α, low ω) to capture global structure; at low noise levels, it wants sharp oscillatory kernels (low α, high ω) to recover fine detail. A small MLP maps the timestep embedding to per-head perturbations of (α, ω, φ), so the wave kernels physically reshape themselves as the reverse diffusion process unfolds. This is the central inductive bias the project tests: that the *physics* of the architecture should track the noise schedule.

## Training setup

The model is trained with v-prediction on a cosine noise schedule, which keeps the loss balanced across timesteps where ε-prediction would degenerate. Min-SNR loss weighting (γ=5) caps the contribution of extreme-SNR timesteps, focusing capacity on the mid-noise range where the digit shape actually emerges. Sampling is done from an EMA copy of the weights (decay 0.9999) using DDIM with 50 steps; the EMA averages out optimizer noise that would otherwise leave visible texture in the samples.

Each transformer block carries time-conditioned residual gates so each branch can attenuate or amplify as a function of the timestep, and the wave kernels are L1-normalized so different (α, ω) values produce contributions at consistent magnitudes — a discrete analog of energy conservation under a dissipative kernel.

## What this is testing

Three questions, in order of how cleanly they can be answered:

1. **Can a wave-field-only backbone learn the reverse diffusion process at all?** Yes — the kernel diagnostics show the right qualitative behavior, with α curving down from high values at high t to low values at low t, and heads specializing to different frequencies.
2. **Does the physics-motivated timestep conditioning beat generic AdaLN at the same parameter count?** Mixed result. Physics conditioning has lower per-timestep MSE but a *higher* FID — see Results.
3. **Does the FFT-based architecture become competitive at scale, where O(n log n) actually matters?** Partly, and only for long sequences. Against a real FlashAttention baseline (not the naive softmax earlier numbers used), wave is a **speed** win only beyond ~8k tokens — where it grows to ~10× faster than FlashAttention at 65k — and is **never** a memory win (FlashAttention uses ~1.5× less throughout). At SC09's 1024 tokens it is actually slower and heavier than FlashAttention. Separately, at matched budget it beats softmax on sample *quality* (FSD ~16 vs ~26). See "Efficiency vs FlashAttention" below for the honest crossover.

## Results

The headline finding is **per-timestep MSE and Frechet sample-quality disagree, and they disagree differently on MNIST vs CIFAR-10**.

**CIFAR-10** (10,000 samples vs 50,000 CIFAR-10 train, clean-FID, Inception V3):

|     | Attention            | Conditioning | FID ↓     | Low-t MSE |
|-----|----------------------|:------------:|----------:|----------:|
| A   | wave (2D radial FFT) | physics      | 87.93     | 2.70      |
| B   | softmax              | physics      | 63.74     | **2.60**  |
| C   | wave (2D radial FFT) | AdaLN        | 81.95     | 3.99      |
| D   | softmax              | AdaLN        | **56.35** | 4.85      |

Standard attention beats wave-field attention on FID at every conditioning level. AdaLN beats physics conditioning on FID — the opposite of what per-timestep MSE says.

**MNIST** (10,000 samples vs 10,000 MNIST train, classifier-based Frechet MNIST Distance):

|     | Attention            | Conditioning | self-cond | FMD ↓    | Low-t MSE |
|-----|----------------------|:------------:|:---------:|---------:|----------:|
| A   | wave (2D radial FFT) | physics      | yes       | **2.19** | 3.53      |
| B   | softmax              | physics      | no        | 4.18     | 3.85      |
| C   | softmax              | AdaLN        | no        | 4.80     | **5.92**  |

A's FMD advantage is partly the self-conditioning confound. Within the matched no-self-cond cohort (B vs C), physics conditioning beats AdaLN on FMD by 15 % — opposite direction from CIFAR.

Full analysis, the metric-disagreement section, and the SC09 placeholder are in [comparison/README.md](comparison/README.md). Aggregated numbers + provenance in [results/](results/), with sample grids per configuration in [results/samples/](results/samples/).

### Wave-operator upgrades flip the CIFAR result (2026-07)

The base wave kernel is content-independent — a fixed filter per (head, timestep). Three opt-in upgrades make the operator content-adaptive while staying O(n log n): a data-dependent spectral filter (`--dynamic_filter`, Orchid/AFFNet-style ΔK̂(x) in a smooth Hann frequency basis), Hyena order-2 gating (`--gating hyena`, pre/post data-controlled gates around the convolution), and anisotropic oriented kernels (`--aniso_kernel`, Gabor-like directional selectivity). A cumulative ablation (200 epochs, 10k-sample clean-FID):

| CIFAR-10 run (all self-cond)      | FID ↓     |
|-----------------------------------|----------:|
| softmax + AdaLN (baseline)        | 57.40     |
| wave + physics (base operator)    | 85.86     |
| + dynamic filter                  | 58.79     |
| + hyena gating                    | 56.80     |
| + anisotropic kernels             | **55.63** |

The dynamic filter alone recovers 27 FID points, and the full stack **beats the matched softmax baseline**. The content-independence of the base kernel — not the wave parameterization itself — was the bottleneck.

### SC09 audio — first clean run (2026-07-18)

The earlier SC09 numbers were invalidated by a training bug: the self-conditioning pass ran under `torch.no_grad()` inside the bf16 autocast region, and on the training pod's torch 2.4 the autocast weight cache retained the detached casts — silently zeroing gradients for most weights on ~half of all batches, and crashing the softmax+AdaLN config outright (its every parameter path runs through a cast-cached op). The rerun uses fixed code ([wave_field/diffusion.py](wave_field/diffusion.py)). Two caveats: all runs used the FAST preset (batch 256 at the same epoch count → ~4× fewer optimizer steps than intended), and the wave runs used the **base** operator — the upgrades that flipped CIFAR were not enabled on audio yet.

|     | Attention | Conditioning | FSD ↓  | class entropy ↑ (uniform 2.30) | status |
|-----|-----------|:------------:|-------:|:---:|--------|
| A   | wave      | physics      | 43.6   | 0.83 | complete (10k eval) |
| B   | softmax   | physics      | 30.0\* | 1.53\* | trained 100 epochs; final 10k eval lost to OOM (\*epoch-100 in-training eval, 1k samples) |
| C   | wave      | AdaLN        | 66.5   | 0.72 | complete (10k eval) |
| D   | softmax   | AdaLN        | —      | —    | OOM in epoch 1 |

Three findings. **(1)** Softmax+physics decisively beats the base wave operator on audio and keeps near-uniform class coverage while both wave runs collapse onto two digits ("two"/"six") — so the mode collapse is a property of the content-independent kernel, not the training bug; the same diagnosis that motivated the dynamic filter on images. **(2)** These softmax runs used *naive* materialized attention, which OOM'd the 24 GB GPU at batch 256 (killing run D and run B's final eval) while the wave runs fit. This is a limitation of naive softmax, **not** a wave memory advantage — FlashAttention removes the wall and in fact uses *less* memory than wave (see "Efficiency vs FlashAttention"). **(3)** The obvious next experiment — the upgraded wave operator (`--dynamic_filter --gating hyena`) on audio — is the run below. Full details and caveats in [results/sc09_fsd_table.json](results/sc09_fsd_table.json).

### SC09 audio — upgraded operator + class-conditioning (2026-07-22)

The upgrade that flipped CIFAR (`--dynamic_filter --gating hyena`), now run on audio, at batch 64 (10k-sample FSD). Both a plain 2×2 (physics/AdaLN) and a class-conditional + CFG (w=2) version:

|     | Conditioning | class-cond + CFG | FSD ↓    | class entropy ↑ (uniform 2.30) |
|-----|:------------:|:----------------:|---------:|:---:|
| wave + physics (base, 07-18) | physics | no | 43.6 | 0.83 |
| upgraded | physics | no | 14.9 | 1.57 |
| upgraded | AdaLN   | no | 19.0 | 1.92 |
| upgraded | physics | yes | 8.5 | 2.18 |
| upgraded | AdaLN   | yes | **8.0** | **2.22** |

The upgrade reproduces the image result on audio: FSD drops **3–3.5×** (physics 43.6→14.9, AdaLN 66.5→19.0), and it **fixes the mode collapse on its own** — every digit is now generated (entropy 1.57–1.92, up from ~0.8) without any class-conditioning. That confirms the collapse was a property of the content-independent base kernel, not an inherent limit of wave attention. The upgraded unconditional model (14.9) also clears the softmax baseline from the previous run (30.0), the same flip seen on CIFAR.

Class-conditioning + CFG then adds a further ~2× (to FSD ~8) and drives class balance to near-uniform (entropy ~2.22). One nuance: physics conditioning's edge is specific to the unconditional regime (14.9 vs 19.0 for AdaLN); once explicit labels + CFG are added the two conditioning schemes converge and AdaLN is marginally ahead (8.0 vs 8.5).

> **Correction (from the matched run below).** This section originally read the FSD drop (43.6→14.9, 66.5→19.0) as evidence the operator *upgrade* fixed audio quality and the mode collapse — with the caveat that the base runs used batch 256. The matched batch-64 rerun shows that caveat was the whole story: **the improvement was training budget, not the operator.** At batch 64, the *base* operator already reaches 16.9/16.1 and covers all ten digits, and the dynamic-filter/hyena upgrade adds little (physics 16.9→14.9) or *hurts* (AdaLN 16.1→19.0). See below.

### SC09 audio — matched batch-64 head-to-head (2026-07-22)

The clean comparison the earlier runs couldn't support: softmax, base wave, and upgraded wave, all unconditional, **all at batch 64 / 100 epochs / lr 1e-4 / 10k-sample FSD** — no budget confound. (Efficiency is covered separately in the next section, against a fair baseline.)

**Quality** (FSD ↓; class entropy in parentheses, uniform = 2.30):

| operator (bs 64, unconditional) | physics | AdaLN |
|---------------------------------|--------:|------:|
| softmax                         | 25.4 (1.19) | 27.4 (0.98) |
| wave (base)                     | 16.9 (1.82) | **16.1** (1.36) |
| wave (dynamic filter + hyena)   | **14.9** (1.57) | 19.0 (1.92) |

Two quality conclusions:

1. **At matched budget, the wave operator beats softmax on audio quality** — decisively (FSD ~16 vs ~26) and with better class coverage. The earlier run's apparent "softmax wins on quality" was purely the budget confound; with equal optimizer steps the ordering flips, matching the image result.

2. **The dynamic-filter/hyena upgrade that won on CIFAR does *not* transfer to audio.** At matched budget it gives a small gain for physics (16.9→14.9) and a regression for AdaLN (16.1→19.0). The 3–3.5× "improvement from the upgrade" reported on 2026-07-18/22 was training budget (batch 64 vs 256), not the operator — the mode-collapse fix too: the base operator at batch 64 already covers all ten digits (entropy 1.36–1.82, up from ~0.8 at batch 256). The upgrade's benefit appears to be modality-specific.

The efficiency picture is in the next section — and it is **not** the "~4× cheaper" this section originally claimed. That number was measured against naive materialized softmax; against a real FlashAttention baseline it does not survive.

### Efficiency vs FlashAttention — the honest crossover (2026-07-24)

The efficiency claims above and in earlier sections compared wave against **naive materialized softmax** (`(qkᵀ).softmax()·v`, which builds the full L×L matrix). Nobody trains attention that way anymore — the standard is `F.scaled_dot_product_attention` (FlashAttention / memory-efficient kernels, O(L) memory). Re-measured against *that* baseline, the story changes substantially.

**Full model at 1024 tokens** (RTX 4090, bf16, bs 64, real `WaveFieldAudioDenoiser`, fwd+bwd step) — the naive-vs-fair difference, and why the old number was wrong:

| softmax baseline | step ms | peak GPU MB |
|------------------|--------:|------------:|
| naive (materialized) — *old strawman* | 159.8 | 14,387 |
| FlashAttention (SDPA) — *fair*        | **36.5** | **2,448** |
| wave (base)                           | 58.9 | 3,394 |

Against the fair baseline, **at 1024 tokens FlashAttention is ~1.6× faster and uses ~1.4× less memory than wave** — the opposite of the retracted claim. The strawman had inflated softmax by 4.4× (time) and 5.9× (memory); that inflation *was* the entire apparent advantage.

**Operator-level sweep, 1k→64k tokens** (bs 8, dim 256, 8 heads; naive softmax OOMs past 4k):

| tokens | flash softmax | wave (base) | flash softmax | wave (base) |
|-------:|--------------:|------------:|--------------:|------------:|
|        | **step ms**   | **step ms** | **peak MB**   | **peak MB** |
| 1,024  | 1.6           | 7.1         | 94            | 132         |
| 4,096  | 4.2           | 6.4         | 321           | 472         |
| 8,192  | 15.0          | **9.3**     | 624           | 926         |
| 16,384 | 55.9          | 18.8        | 1,230         | 1,834       |
| 65,536 | 881.7         | **84.4**    | 4,867         | 7,279       |

Two clean findings:

- **Speed: wave overtakes FlashAttention at ~8,192 tokens.** Below that, flash is faster (4.5× at 1k); above it, wave's O(L log L) pulls away from flash's O(L²) — **~10× faster at 65k tokens**. This is the genuine, honestly-measured asymptotic advantage.
- **Memory: FlashAttention wins at every length.** Wave uses ~1.4–1.5× *more* memory throughout, because its FFT path materializes complex spectra while flash tiles and materializes nothing. **The memory advantage claimed earlier does not exist against a real baseline.**

So the efficiency thesis, stated honestly: **the wave operator is a speed win only for long sequences (≳8k tokens), where it can be many times faster than FlashAttention, and it is never a memory win.** SC09 at 1024 tokens sits *below* the crossover — on that task wave buys better quality at higher compute and memory cost, not lower. The efficiency payoff belongs to genuinely long-context settings, which is where future work should aim. Scaling curves: [outputs/crossover/crossover.png](outputs/crossover/crossover.png).

### CFG guidance sweep, CIFAR-10 (2026-07-18)

Both class-conditional models were retrained (the originals' checkpoints died with their pod) and evaluated at six guidance scales on the same checkpoint. Caveat: the retrains used batch 256 at 200 epochs — **half the optimizer steps** of the table above — so these FIDs are only comparable within this table, not to the July numbers.

| guidance w | wave full stack | softmax + AdaLN |
|-----------:|----------------:|----------------:|
| 1.0        | 83.7            | 108.4           |
| 1.25       | 80.0            | 102.5           |
| 1.5        | 76.5            | 96.3            |
| 1.75       | 73.2            | 90.4            |
| 2.0        | 70.4            | 85.1            |
| 3.0        | **63.9**        | **69.5**        |

FID falls monotonically through w=3 with no minimum in range — the earlier one-point conclusion that "CFG at 1.5 makes things worse" was an artifact; the optimum for these models sits at w≥3. Two more observations: the wave stack dominates softmax at *every* guidance scale, and it degrades far more gracefully under the halved training budget (at w=1.0: 83.7 vs 108.4, against 55.6 vs 57.4 for the fully-trained unconditional models) — evidence that the physics-constrained operator is markedly more sample-efficient. The EMA checkpoints are committed (`outputs/cifar_*_cond/checkpoint_epoch0200_ema.pt`), so extending the sweep past w=3 requires no retraining.

### Open items

1. **The efficiency payoff is a long-context story now, so test a genuinely long-context task** (≳8k-token audio/video/high-res image) where wave is faster than FlashAttention — 1024-token SC09 sits below the crossover and shows no efficiency win. This is where the thesis has to prove itself.
2. **Reduce wave's memory constant.** It uses ~1.5× more memory than FlashAttention because the FFT path materializes complex spectra; closing that gap (in-place FFT, fp16 spectra, chunking) is what would turn the speed-only win into a speed-and-memory win.
3. **Benchmark against the real competition** — Mamba/S4, Hyena — not just softmax. Those are the sub-quadratic baselines a reviewer or buyer will expect; beating softmax is table stakes.
4. **Why does the dynamic-filter/hyena upgrade help images but not audio?** It regressed AdaLN on SC09 at matched budget — content-adaptive filter overfitting the short 64-token image grid, or a 1D-vs-2D kernel interaction?
5. Scale the matched audio comparison up (longer training / larger models) to see whether the wave quality lead over softmax holds or widens out of the undertrained regime.
6. Lower priority: audio CFG guidance sweep (only w=2 tested; CIFAR optimum w≥3); CIFAR guidance sweep on fully-trained batch-128 conditional models past w=3.

## Sources
Using knowledge and inspiration gathered from:

1. https://arxiv.org/abs/2503.13615
2. https://discuss.huggingface.co/t/wave-field-llm-o-n-log-n-attention-via-wave-equation-dynamics-within-5-of-standard-transformer/173625
3. https://github.com/badaramoni/wave-field-llm
4. https://wavefieldlab.com/