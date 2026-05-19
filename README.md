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
3. **Does the FFT-based architecture become competitive at scale, where O(n log n) actually matters?** Wall-clock crossover is real (`benchmarks/attn_benchmark.png` — wave field is ~7× faster than softmax at L=2048), but the only experiment that lives in the regime where the asymptotic gain matters is the SC09 audio leg, which is implemented but not yet run.

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

The SC09 audio leg is the missing experiment — it's the only setting in this project where the sequence length (1024 tokens) puts wave field's `O(n log n)` advantage in the regime that matters ([benchmarks/attn_benchmark.png](benchmarks/attn_benchmark.png) shows ~7× speedup at L=2048). Infrastructure is in place ([train_audio.py](train_audio.py), [metrics/evaluate.py](metrics/evaluate.py)); FSD slots are reserved in [results/sc09_fsd_table.json](results/sc09_fsd_table.json) with `fsd: null` pending the four ablation runs.

## Sources
Using knowledge and inspiration gathered from:

1. https://arxiv.org/abs/2503.13615
2. https://discuss.huggingface.co/t/wave-field-llm-o-n-log-n-attention-via-wave-equation-dynamics-within-5-of-standard-transformer/173625
3. https://github.com/badaramoni/wave-field-llm
4. https://wavefieldlab.com/