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
3. **Does the FFT-based architecture become competitive at scale, where O(n log n) actually matters?** Wall-clock crossover is real (`benchmarks/attn_benchmark.png` — wave field is ~7× faster than softmax at L=2048), and the SC09 audio leg (1024 tokens) now provides a memory data point too: at batch 256, softmax attention's ~4 GB per-layer attention matrix OOM'd a 24 GB GPU while the wave-field runs fit easily. On sample *quality* at that length, however, softmax still wins — see the SC09 section below.

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

Three findings. **(1)** Softmax+physics decisively beats the base wave operator on audio and keeps near-uniform class coverage while both wave runs collapse onto two digits ("two"/"six") — so the mode collapse is a property of the content-independent kernel, not the training bug; the same diagnosis that motivated the dynamic filter on images. **(2)** The O(L²) memory wall is concrete: at batch 256 and L=1024 tokens the softmax attention matrix is ~4 GB per layer, which OOM'd the 24 GB GPU (killing run D and run B's final eval) while the wave runs fit with room to spare. **(3)** The obvious next experiment — the upgraded wave operator (`--dynamic_filter --gating hyena`) on audio — has not yet been run. Full details and caveats in [results/sc09_fsd_table.json](results/sc09_fsd_table.json).

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

1. SC09 with the upgraded wave operator (`--dynamic_filter --gating hyena`) — the single most informative missing run.
2. Class-conditional SC09 + CFG to attack the mode collapse directly.
3. Rerun the guidance sweep on fully-trained (batch-128) conditional models, extending past w=3.
4. Training-recipe scaling (longer runs, patch size 2, wider/deeper) — at matched budget the architecture question is answered on images; absolute FID is now recipe-limited.

## Sources
Using knowledge and inspiration gathered from:

1. https://arxiv.org/abs/2503.13615
2. https://discuss.huggingface.co/t/wave-field-llm-o-n-log-n-attention-via-wave-equation-dynamics-within-5-of-standard-transformer/173625
3. https://github.com/badaramoni/wave-field-llm
4. https://wavefieldlab.com/