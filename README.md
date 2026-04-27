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
2. **Does the physics-motivated timestep conditioning beat generic AdaLN at the same parameter count?** Currently a near-tie on MNIST.
3. **Does the FFT-based architecture become competitive at scale, where O(n log n) actually matters?** This is the open question CIFAR-10 and longer sequences are meant to address.


## Sources
Using knowledge and inspiration gathered from:

1. https://arxiv.org/abs/2503.13615
2. https://discuss.huggingface.co/t/wave-field-llm-o-n-log-n-attention-via-wave-equation-dynamics-within-5-of-standard-transformer/173625
3. https://github.com/badaramoni/wave-field-llm
4. https://wavefieldlab.com/