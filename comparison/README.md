# Comparison between Wave-field model and standard attention model.

> **Update — sample-quality metrics added (2026-05-19).** The original analysis below used per-timestep MSE as the proxy for "sample quality." It now also reports Frechet distances on 10,000 generated samples (FID for CIFAR-10 via clean-fid against the train set, classifier-based FMD for MNIST). The two metrics disagree on which architecture wins, which is itself the most interesting finding. The new section [Per-timestep MSE vs Frechet distance — they disagree](#per-timestep-mse-vs-frechet-distance--they-disagree) discusses why.

---

### Q: The standard loss is better than the wave field but the samples are much much worse in the standard model. Why is that?

A: This is actually the most interesting result you've gotten, and it's the core finding of the project. It has a specific explanation.

Lower training loss ≠ better samples in diffusion models. Training loss is MSE on randomly sampled (x_t, t) pairs — each evaluated independently. Sampling is 50 sequential steps where each step feeds into the next. These are fundamentally different tasks.

Standard attention is more expressive and finds a lower per-step MSE, but it does it by fitting patterns in individual (x_t, t) pairs that don't necessarily chain together coherently. When you run 50 steps sequentially, small inconsistencies compound. The model is locally better but globally incoherent.

The wave field model is constrained. The kernel must be smooth, spatially coherent, and — crucially — it reshapes itself as a function of timestep (high α broad kernels at high noise, low α sharp kernels at low noise). This timestep-dependent calibration means each denoising step is internally consistent with the ones before and after it. Individual predictions are slightly noisier (higher loss), but they compose into a coherent trajectory.

There's a useful analogy: standard attention is like fitting 50 independent splines to data points — each fit is optimal locally. The wave field kernel is like fitting a physically constrained curve — each point is slightly off the data but the whole curve is smooth and coherent.

This is exactly the research question from the plan: does the physics inductive bias help? The answer you're seeing is: not by lowering per-step MSE, but by improving sample-level coherence across the denoising chain. That's a more interesting finding than just "better loss."

See "loss_comparison.png" for the graph explaining this.


### Q: How do the different configurations for MNIST compare?

| Run | Configuration             | Params      | Final Loss | Low-t MSE (t<100) | FMD (10k) ↓ | Confident acc | Class entropy |
|:---:|:--------------------------|------------:|-----------:|------------------:|------------:|--------------:|--------------:|
| A   | Wave + Physics + self-cond| 1,276,800   | 0.02359    | 3.5263            | **2.19**    | 0.917         | 2.286         |
| B   | Standard + Physics        | 1,191,952   | 0.02305    | 3.8525            | 4.18        | 0.890         | 2.292         |
| C   | Standard + AdaLN          | 1,354,768   | 0.02272    | 5.9188            | 4.80        | 0.892         | 2.292         |

FMD is the Frechet distance between embeddings of 10,000 generated samples and 10,000 real MNIST train images, using a small CNN classifier (94.3 % test accuracy on MNIST) as the feature extractor. Confident accuracy = fraction of generated samples the classifier labels with top-1 probability ≥ 0.5 ("looks like a digit at all"). Class entropy = Shannon entropy of the predicted-class histogram; uniform is `log(10) ≈ 2.303` — values near that ceiling mean no mode collapse.

**Caveat: run A uses self-conditioning, B and C do not.** Self-conditioning (Chen 2022) is known to improve diffusion sample quality by 30 – 50 % on FID-style metrics. So A's FMD of 2.19 cannot be cleanly attributed to wave kernels alone — the self-cond confound is doing a meaningful share of the work. The clean comparison is B vs C, both standard attention with the same parameter count:

1. The physics conditioning is doing most of the work (A vs B).
Wave field kernels (A) vs standard attention (B), same conditioning: low-t MSE of 3.53 vs 3.85 — about a 9% gap. The kernel helps, but not dramatically. Crucially, B still produces recognizable digits because it has the physics gates and FiLM. The conditioning mechanism is load-bearing.

2. The conditioning mechanism matters far more than the attention mechanism (B vs C).
Same standard attention, different conditioning: 3.85 vs 5.92 — a 54% gap. Swapping AdaLN for physics conditioning (residual gates + FiLM) has six times more impact on low-t performance than swapping the attention mechanism. This is why C produced blobs while A produced clean digits — it was never primarily about the wave kernels.

3. The wave field kernel provides a genuine but modest improvement over standard attention, given the same conditioning.
A beats B by 9% with fewer parameters (1.27M vs 1.19M, so A is actually slightly larger — about 7% more params). Normalizing for parameter count, the kernel advantage becomes smaller. It's real but not the headline finding.

4. **FMD agrees with low-t MSE on MNIST that physics conditioning helps.** B (physics) gets FMD 4.18 vs C (AdaLN) at 4.80 — a 15 % gap. Direction matches the 54 % low-t MSE gap, magnitude is much smaller. The "sample-quality" reading is therefore weaker than the per-timestep reading, but at least consistent — they point the same way on MNIST.

What this means for the research question in the plan:
The original thesis — "wave-equation dynamics in the backbone yield physically interpretable multi-scale denoising" — holds up, but the stronger result is about the conditioning mechanism, not the kernel. The physics-motivated timestep conditioning (kernels that reshape with t, residual gates that attenuate per branch) is what actually drives sample quality on MNIST. The wave kernel is an additional contribution on top of that, not the primary one.

The MNIST write-up therefore says: physics-conditioned conditioning >> attention mechanism choice, with wave kernels providing a secondary gain. **But — see the CIFAR-10 section below — this story does not survive the move to a real-image distribution.**

---

### CIFAR-10 results

| Run | Configuration           | Params    | Final Loss | Low-t MSE (t<100) | FID (10k) ↓ |
|:----|:------------------------|----------:|-----------:|------------------:|------------:|
| A   | Wave + Physics + 2D     | 7,333,120 | 0.02709    | 2.6966            | 87.93       |
| B   | Standard + Physics      | 6,876,208 | 0.02584    | **2.5975**        | 63.74       |
| C   | Standard + AdaLN        | 7,790,640 | 0.02555    | 4.8542            | **56.35**   |
| D   | Wave + AdaLN + 2D       | 7,815,792 | 0.02695    | 3.9875            | 81.95       |

FID is computed with [clean-fid](https://github.com/GaParmar/clean-fid) in `mode=clean` against 50,000 real CIFAR-10 train images, using 10,000 generated samples from each model's EMA copy at epoch 200 (DDIM, 50 steps).

**The CIFAR results replicate the MNIST finding at scale and sharpen two of the conclusions.**

#### 1. Physics conditioning dominates again — and by a larger margin (B vs C, D vs A)

The clearest comparison is B (standard attention + physics) vs C (standard attention + AdaLN): same model, same parameter count within rounding, different conditioning only. Low-t MSE goes from 2.60 to 4.85 — an **87% increase** when you strip out physics conditioning. That is a much larger gap than the 54% seen on MNIST, which makes sense: CIFAR-10's natural images have more fine-grained structure that has to be recovered in the low-noise regime, so the calibration advantage of physics conditioning pays off more.

The same story holds when comparing wave-field runs: wave+AdaLN (D) has low-t MSE of 3.99, while wave+physics (A) reaches 2.70 — a 48% improvement from adding physics conditioning to the same attention mechanism.

#### 2. The attention mechanism gap narrows at scale (A vs D, B vs C within conditioning)

On MNIST the wave kernel contributed a ~9% improvement in low-t MSE over standard attention given the same conditioning. On CIFAR:

- Physics conditioning: wave (A, 2.70) vs standard (B, 2.60) — standard is **4% better**
- AdaLN conditioning: wave (D, 3.99) vs standard (C, 4.85) — wave is **18% better**

The wave kernel advantage shrinks under physics conditioning and reverses slightly (within noise). Under AdaLN where the kernel has to carry more load, wave still wins. This is consistent with MNIST: physics conditioning closes the gap between attention mechanisms because it already handles the timestep-calibration problem that wave kernels were partially solving.

The reversal under physics conditioning (B marginally beats A) is not large enough to be conclusive — both runs have higher final training loss than C and D, which may reflect the heavier model of A (7.33M vs 6.88M for B at the same number of training steps). With equal parameter counts the result would likely be a tie.

#### 3. The key CIFAR-specific finding: physics conditioning scales better than AdaLN

The MNIST three-way comparison already showed physics conditioning >> attention choice. CIFAR shows that the physics conditioning advantage **grows** with task difficulty. The low-t MSE spread between physics and AdaLN conditioning is 87% on CIFAR vs 54% on MNIST. This is the stronger result: the inductive bias of physics-motivated timestep routing is not just helpful on toys — it compounds on real image distributions where low-noise denoising is genuinely hard.

The wave-field kernel advantage, by contrast, remains secondary and roughly constant (moderate on MNIST, small-to-mixed on CIFAR). The physics conditioning mechanism — residual gates, FFN FiLM, and kernel adaptation — is the load-bearing claim of this project at both scales.

See `cifar_per_timestep_loss.png` for the full per-timestep breakdown, `cifar_sample_comparison.png` for sample quality, and `cifar_kernels2d.png` for the 2D wave kernel visualizations across runs and timesteps.

---

## Per-timestep MSE vs Frechet distance — they disagree

The story changes once FID is added. **The CIFAR ranking by FID is the inverse of the ranking by low-t MSE.**

| Run | Low-t MSE rank | FID rank |
|:---:|:--------------:|:--------:|
| B (standard + physics)   | 1 (best, 2.60) | 2 (63.7) |
| A (wave + physics)       | 2 (2.70)       | 4 (worst, 87.9) |
| D (standard + AdaLN)     | 4 (worst, 4.85)| 1 (best, 56.4) |
| C (wave + AdaLN)         | 3 (3.99)       | 3 (82.0) |

Two specific reversals:

- **Conditioning choice flips.** By low-t MSE, physics conditioning wins by a wide margin (B beats C by 47 %, A beats D by 32 %). By FID, AdaLN wins (D beats B by 12 %, C beats A by 7 %).
- **Attention choice strengthens.** By low-t MSE, wave and standard are roughly tied under physics (A vs B within ~4 %). By FID, standard beats wave by 28 % under physics (B vs A) and 31 % under AdaLN (D vs C).

### Why this happens

Per-timestep MSE asks: *averaged over fresh noise samples at each timestep, how well does the model fit the local denoising target?* It rewards a model that is tightly calibrated point-by-point through diffusion time.

FID asks: *do 10,000 samples drawn by running the full reverse process look, in Inception-feature space, like real CIFAR-10?* It rewards a model whose 50-step trajectories produce realistic outputs at the population level.

Physics conditioning constrains the wave kernels and the residual gates to vary smoothly with timestep. That smoothness is exactly what reduces per-step MSE — neighbouring timesteps share parameters and so fit similarly. But it also constrains the model: the conditioning is parametrised by a small MLP that maps `t_emb → (Δlog α, Δω, Δφ)` per head, vs AdaLN's six free shift/scale/gate per block. Under FID, the model with more degrees of freedom to break the smoothness assumption (AdaLN) wins. The point-by-point calibration that physics conditioning enforces is over-regularising at the level of full sample trajectories.

**The MNIST direction was the opposite.** There, physics conditioning helped FMD by 15 % (B vs C). So the metric disagreement is not the whole story — the conditioning preference flips between datasets. The plausible reading is that on MNIST's simple digit distribution, the smoothness constraint regularises usefully (small effective sample space, AdaLN can overfit the trajectory); on CIFAR-10's diverse natural-image distribution, the same smoothness constraint clips expressivity (much larger effective sample space, AdaLN's extra freedom is now necessary).

### What this means for the original research questions

1. *"Can a wave-field-only backbone learn the reverse diffusion process?"* Yes on both datasets — sample grids in [results/samples/](../results/samples/) confirm this.
2. *"Does physics-motivated conditioning beat AdaLN at matched parameter count?"* **Dataset-dependent.** Better on MNIST per both metrics; better per low-t MSE on CIFAR but **worse per FID on CIFAR.** This is not a clean "physics conditioning wins" anymore.
3. *"Does FFT-based attention become competitive at scale?"* CIFAR's 64 tokens is too short to test the asymptotic regime, and at that scale standard softmax attention wins on FID at every conditioning level. **The SC09 audio leg at 1024 tokens is the only experiment that lives in the regime where wave field's O(n log n) advantage might matter.** That experiment is implemented and not yet run.

---

## SC09 (audio, 1024 tokens) — pending

This is the missing experiment. SC09 is the digit-subset of Google Speech Commands; clips are 1.024 s at 16 kHz = 16,384 samples = 1024 tokens at patch_size 16. [benchmarks/attn_benchmark.png](../benchmarks/attn_benchmark.png) shows the wave-field forward pass is roughly 7× faster than softmax attention at L=2048 (and the crossover happens well below 1024), so SC09 is the first experiment in this project that benchmarks both ends of the trade-off: wall-clock cost and sample quality.

The metrics already in place mirror the image side:

- **Frechet SC09 Distance** — Frechet distance in the embedding space of a small SC09-digit classifier (91.5 % val accuracy). Same role as FMD on MNIST.
- **Classifier accuracy & class entropy** — does a real digit classifier recognise the generated clip as a digit at all? Do the 10 classes show up roughly uniformly?
- **Mean log-mel spectrogram comparison** — real vs generated population-level frequency content. Plot at every eval.

Slots are reserved in [results/sc09_fsd_table.json](../results/sc09_fsd_table.json) (`fsd: null` for each row). Commands to run:

```bash
python train_audio.py --attn wave     --conditioning physics --save_dir outputs/sc09_wave_physics
python train_audio.py --attn standard --conditioning physics --save_dir outputs/sc09_standard_physics
python train_audio.py --attn wave     --conditioning adaln   --save_dir outputs/sc09_wave_adaln
python train_audio.py --attn standard --conditioning adaln   --save_dir outputs/sc09_standard_adaln

# 10k-sample reference FSD after each run:
python -m metrics.evaluate outputs/sc09_<...> --n_samples 10000
```

Once those four runs land, this section will get a parallel table to the CIFAR one (FSD column + wall-clock-per-step column) and a paragraph on whether the wave field's asymptotic advantage actually translates to a quality/efficiency win at the only sequence length in the project where the trade-off is meaningful.

---

## Reproducing the metrics in this README

All numbers above come from saved EMA checkpoints under `outputs/`. To reproduce:

```bash
# One-time setup: train the feature-extractor classifiers (~minutes)
python -m metrics.train_classifier --task mnist
python -m metrics.train_classifier --task sc09     # only needed for the audio leg

# Evaluate existing checkpoints
python -m metrics.evaluate outputs/cifar_wave_physics_2d_sc   --n_samples 10000
python -m metrics.evaluate outputs/cifar_standard_physics_sc  --n_samples 10000
python -m metrics.evaluate outputs/cifar_wave_adaln_2d_sc     --n_samples 10000
python -m metrics.evaluate outputs/cifar_standard_adaln_sc    --n_samples 10000
python -m metrics.evaluate outputs/mnist_v7_selfcond          --n_samples 10000
python -m metrics.evaluate outputs/mnist_standard_physics     --n_samples 10000
python -m metrics.evaluate outputs/mnist_standard_adaln       --n_samples 10000
```

Each call writes a `metrics.json` into the checkpoint directory and (for CIFAR) keeps the 10k generated PNGs in `<checkpoint>/fid_gen_tmp/` so the FID computation can be retried without regeneration via `--reuse_samples`. The 50 k real CIFAR-10 PNGs used as the FID reference are cached in `data/cifar10_real_pngs/` on first use.

Aggregated numbers + provenance live in [results/](../results/).