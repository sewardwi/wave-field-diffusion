# Wave Field Diffusion: Exploratory Project Plan

## Thesis

A diffusion model's denoising backbone can be parameterized with wave-equation dynamics (damped oscillation kernels applied via FFT convolution) instead of standard self-attention, yielding an architecture where the reverse generative process explicitly invokes wave propagation to undo a dissipative forward process. This could offer O(n log n) scaling for high-resolution generation and physically interpretable multi-scale denoising.

---

## Phase 0 — Foundations & Implementation Scaffold (2–3 weeks)

### Read & annotate
- **Wave Field LLM**: Study the README and docs carefully (you have the full architectural spec from the GitHub page). Key details: the damped kernel formula, the FFT convolution pipeline, bilinear scatter/gather, and head specialization behavior. You won't have the source code, but the mechanism is simple enough to reimplement (see "Build the core module" below).
- **García-Pintos et al.**: Focus on Sections II–III (Hamiltonian construction for trajectory reversal, feedback protocol). Extract the core mathematical insight: given a stochastic forward process, one can construct an explicit generator for the reverse. Note how this relates to the score function in diffusion theory.
- **Diffusion model fundamentals**: Revisit DDPM (Ho et al. 2020), score-based SDEs (Song et al. 2021), and the Diffusion Transformer (DiT, Peebles & Xie 2023). Pay special attention to how the denoising network architecture interacts with the noise schedule.
- **Adjacent work**: Skim Hyena (Poli et al. 2023) and S4/Mamba for context on sub-quadratic sequence models applied beyond language. Check if anyone has tried convolution-kernel-based backbones for diffusion (FNO-based generators, spectral diffusion, etc.).

### Build the core module from scratch

You don't have access to the Wave Field LLM source, but the README provides the full spec. Implement a standalone `WaveFieldAttention` module in PyTorch:

```python
# Pseudocode — the entire core mechanism

class WaveFieldAttention(nn.Module):
    # Per-head learnable parameters: α (damping), ω (frequency), φ (phase)
    # 1. Project input to Q, K, V
    # 2. Build kernel: k[t] = exp(-α|t|) * cos(ω*t + φ) on a grid t ∈ [-L, L]
    # 3. FFT-convolve V with k:
    #      V_fft = torch.fft.rfft(V)
    #      k_fft = torch.fft.rfft(k, n=V.shape[-1])
    #      out = torch.fft.irfft(V_fft * k_fft)
    # 4. Gate output with content-dependent gating (sigmoid(linear(Q)))
    # 5. Multi-head concat + output projection
```

**Simplification vs. the original**: Skip the bilinear scatter/gather onto a continuous field. For diffusion on a fixed patch grid, you can convolve directly on the sequence — the continuous field abstraction is an optimization for variable-length sequences that you don't need. This cuts out the hardest-to-reverse-engineer part of their code.

**Validation step**: Before plugging this into a diffusion model, sanity-check the module in isolation. Feed in random sequences, verify the FFT convolution is working (output should be smooth/structured, not random), verify gradients flow through α, ω, φ, and verify that different initializations of ω produce visibly different kernel shapes.

### Deliverable
A 2–3 page lit review / working note that maps the mathematical correspondences:
| Diffusion concept | Wave field analog | García-Pintos analog |
|---|---|---|
| Forward SDE (heat equation) | Dissipation on the field | Arrow of time (measurement) |
| Score function ∇ log p_t | Wave kernel convolution | Feedback Hamiltonian |
| Noise schedule β(t) | Damping coefficient α | Monitoring strength |
| Multi-scale denoising | Head frequency specialization | — |
| Classifier-free guidance | Gated field coupling | Feedback steering |

---

## Phase 1 — Minimal Proof of Concept (2–3 weeks)

### Goal
Get a wave-field denoising network to generate *anything recognizable* on a toy dataset. Don't chase quality — chase signal that the architecture can learn the reverse process at all.

### Dataset
MNIST or Fashion-MNIST (28×28, grayscale). Simple enough that architecture issues won't be masked by data complexity.

### Architecture: WaveFieldDenoiser

**Key design decisions:**

1. **Flatten the image to a 1D sequence.** Treat the 28×28 image as a length-784 sequence (or use patches: 4×4 patches → length-49 sequence of dim-16 tokens). Patching is probably better to start.

2. **Remove causality.** Wave Field LLM uses causal (one-sided) kernels. For diffusion denoising, use the *full* (non-causal) damped oscillation:
   ```
   k(t) = exp(-α|t|) · cos(ω·t + φ)    for t ∈ ℝ
   ```
   This is still O(n log n) via FFT — just don't zero out the negative side.

3. **Condition on timestep.** Standard approach: embed the diffusion timestep t via sinusoidal encoding, then use it to modulate the wave parameters. Two options to explore:
   - **Option A (simple):** AdaLN — timestep embedding modulates LayerNorm scale/shift (as in DiT).
   - **Option B (physics-motivated):** Timestep *directly modulates* α, ω, φ of the wave kernels. Early timesteps (high noise) → low ω, high α (broad, smooth kernels for global structure). Late timesteps (low noise) → high ω, low α (sharp, oscillatory kernels for fine detail). This is the more interesting option and the whole point of the project.

4. **Architecture skeleton:**
   ```
   Noisy patches + timestep embedding
       │
   [Patch Embedding (linear projection)]
       │
   [Wave Field Block × N]
       │── LayerNorm
       │── Non-Causal Wave Field Attention (FFT convolution)
       │     │── QKV projection
       │     │── Scatter to continuous field
       │     │── FFT convolve with damped wave kernel
       │     │── (Optional) cross-head field coupling
       │     │── Gather from field
       │── LayerNorm + FFN (GELU)
       │── (Every K layers) Field Interference
       │
   [LayerNorm → Linear → Predict noise ε]
   ```

5. **Hyperparameters to start:**
   - Patches: 4×4 → 49 tokens, dim 64
   - 4 wave field layers, 4 heads
   - Field size: 256
   - ~500K–1M parameters (keep it small)

### Training
- Standard DDPM training: sample t ~ Uniform, add noise, predict ε
- Linear noise schedule, 1000 timesteps
- Adam, lr 1e-4, batch size 128
- Train for ~50K steps (should be enough for MNIST)

### Evaluation & diagnostics
- **FID on MNIST** (just to have a number, not the point yet)
- **Visual inspection** of generated samples at 10K, 25K, 50K steps
- **Physics diagnostics** (borrowed from Wave Field LLM):
  - Plot learned (α, ω, φ) per head as a function of conditioning timestep t
  - Does frequency specialization emerge? Do heads separate into low-ω (global) and high-ω (local)?
  - Energy flow through layers: is conservation approximately maintained?
- **Baseline**: Train a simple DiT-style model with the same param count and standard attention. This is your comparison.

### Deliverable
A notebook with: training curves, sample grids, head specialization plots, and a brief write-up of what worked vs. didn't.

---

## Phase 2 — Extend to 2D and CIFAR-10 (2–3 weeks)

### Goal
Move to a real (if small) image dataset and a proper 2D wave field.

### Key change: 2D wave kernels

The 1D kernel `k(t) = exp(-α|t|)·cos(ωt + φ)` generalizes to 2D:
```
k(x, y) = exp(-α·√(x² + y²)) · cos(ω·√(x² + y²) + φ)
```
This is a radially symmetric damped wave — isotropic by default. Can be made anisotropic with separate (αx, αy, ωx, ωy) if needed.

2D FFT convolution is still O(n log n) where n = H×W.

### Architecture modifications
- Input: 32×32×3 CIFAR-10 images
- Patch size 4×4 → 8×8 = 64 patches of dim 48 (or operate directly on the 2D pixel field)
- Two approaches to try:
  - **Approach A**: Keep 1D sequence of patches, apply 1D wave kernels (simpler, same as Phase 1 but scaled up)
  - **Approach B**: Operate on 2D field directly with 2D FFT wave convolution (more principled, more engineering)
- Scale up: 6–8 layers, 8 heads, dim 128–256, ~5–10M params

### Experiments
1. Compare Option A vs Option B timestep conditioning (AdaLN vs physics-modulated kernels)
2. Ablate: wave field attention vs. standard attention vs. linear attention, same param count
3. If 2D works: visualize the learned 2D wave kernels at different timesteps. Do they look like bandpass filters? Do they resemble Gabor-like wavelets?

### Deliverable
- FID scores on CIFAR-10 (unconditional generation)
- Kernel visualization gallery
- Clear answer to: does timestep-conditioned wave parameterization outperform generic AdaLN conditioning?

---

## Phase 3 — Analysis & Write-up (1–2 weeks)

### Questions to answer
1. **Does wave field attention learn meaningful frequency decomposition for denoising?** The headline finding would be if heads clearly specialize: low-frequency heads dominate at high-noise timesteps, high-frequency heads dominate at low-noise timesteps.
2. **Is there a quality/efficiency tradeoff worth reporting?** Even if FID isn't SOTA, if wave field gets within X% at significantly lower compute (especially for high resolution), that's a meaningful result.
3. **Does the physics-motivated timestep conditioning (modulating α, ω, φ directly) outperform generic conditioning?** This tests whether the inductive bias from wave physics actually helps.
4. **Connection to the García-Pintos framework**: Can you characterize the learned reverse process as approximately implementing a "feedback Hamiltonian" that reverses the forward diffusion? This is the theory-paper angle.

### Write-up structure
If results are promising, target a workshop paper (4–6 pages) or a solid blog post / arXiv preprint:
- Motivation: PDE duality (heat eq forward, wave eq reverse)
- Architecture: non-causal wave field denoising blocks
- Experiments: MNIST → CIFAR-10, ablations, head specialization analysis
- Discussion: connections to quantum arrow of time, limitations, future work

---

## Stretch Goals (if things go well)

- **Long-sequence diffusion**: Apply to 1D audio waveform generation (e.g., SC09 spoken digits) where the O(n log n) scaling actually matters vs. O(n²) attention on long sequences.
- **Schrödinger bridge formulation**: Replace the standard DDPM forward/reverse with a Schrödinger bridge, and see if wave kernels are a natural parameterization for the drift.
- **Hybrid architecture**: Wave field attention for global structure (low-frequency denoising) + standard local attention for fine detail. Like a frequency-domain U-Net.
- **Quantum generative model**: If you want to go full circle to the García-Pintos paper — implement a small quantum circuit diffusion model (using Stim or Qiskit) where the reverse process is parameterized by a variational Hamiltonian. Very speculative, very cool, very publishable if it works.

---

## Tools & Resources

- **Codebase starting point**: Implement `WaveFieldAttention` from scratch using the README spec (see Phase 0). Cleaner than adapting their LLM code anyway since you need non-causal, timestep-conditioned, potentially 2D variants.
- **Diffusion framework**: Use a lightweight implementation (lucidrains/denoising-diffusion-pytorch or roll your own — you've done enough PyTorch)
- **Compute**: Should be doable on a single consumer GPU (MNIST/CIFAR-10 at these scales). The whole point of wave fields is efficiency.
- **Key libraries**: PyTorch, torchvision, torch.fft, matplotlib (for kernel visualizations), clean-fid (for FID computation)

---

## Timeline Summary

| Phase | Duration | Milestone |
|---|---|---|
| 0 — Literature + core module | 2–3 weeks | Working note + validated WaveFieldAttention module |
| 1 — MNIST PoC | 2–3 weeks | Wave field denoiser generates recognizable digits |
| 2 — CIFAR-10 + 2D | 2–3 weeks | FID scores, kernel visualizations, ablations |
| 3 — Analysis & paper | 1–2 weeks | Write-up with clear answers to key questions |
| **Total** | **~7–11 weeks** | |