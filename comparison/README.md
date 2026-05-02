# Comparison between Wave-field model and standard attention model.

### Q: The standard loss is better than the wave field but the samples are much much worse in the standard model. Why is that?

A: This is actually the most interesting result you've gotten, and it's the core finding of the project. It has a specific explanation.

Lower training loss ≠ better samples in diffusion models. Training loss is MSE on randomly sampled (x_t, t) pairs — each evaluated independently. Sampling is 50 sequential steps where each step feeds into the next. These are fundamentally different tasks.

Standard attention is more expressive and finds a lower per-step MSE, but it does it by fitting patterns in individual (x_t, t) pairs that don't necessarily chain together coherently. When you run 50 steps sequentially, small inconsistencies compound. The model is locally better but globally incoherent.

The wave field model is constrained. The kernel must be smooth, spatially coherent, and — crucially — it reshapes itself as a function of timestep (high α broad kernels at high noise, low α sharp kernels at low noise). This timestep-dependent calibration means each denoising step is internally consistent with the ones before and after it. Individual predictions are slightly noisier (higher loss), but they compose into a coherent trajectory.

There's a useful analogy: standard attention is like fitting 50 independent splines to data points — each fit is optimal locally. The wave field kernel is like fitting a physically constrained curve — each point is slightly off the data but the whole curve is smooth and coherent.

This is exactly the research question from the plan: does the physics inductive bias help? The answer you're seeing is: not by lowering per-step MSE, but by improving sample-level coherence across the denoising chain. That's a more interesting finding than just "better loss."

See "loss_comparison.png" for the graph explaining this.


### Q: How do the different configurations compare?

| Run | Configuration       | Params      | Final Loss | Low-t MSE (t<100) |
|:---:|:--------------------|------------:|-----------:|------------------:|
| A   | Wave + Physics      | 1,274,752   | 0.02359    | 3.5263            |
| B   | Standard + Physics  | 1,191,952   | 0.02305    | 3.8525            |
| C   | Standard + AdaLN    | 1,354,768   | 0.02272    | 5.9188            |

1. The physics conditioning is doing most of the work (A vs B).
Wave field kernels (A) vs standard attention (B), same conditioning: low-t MSE of 3.53 vs 3.85 — about a 9% gap. The kernel helps, but not dramatically. Crucially, B still produces recognizable digits because it has the physics gates and FiLM. The conditioning mechanism is load-bearing.

2. The conditioning mechanism matters far more than the attention mechanism (B vs C).
Same standard attention, different conditioning: 3.85 vs 5.92 — a 54% gap. Swapping AdaLN for physics conditioning (residual gates + FiLM) has six times more impact on low-t performance than swapping the attention mechanism. This is why C produced blobs while A produced clean digits — it was never primarily about the wave kernels.

3. The wave field kernel provides a genuine but modest improvement over standard attention, given the same conditioning.
A beats B by 9% with fewer parameters (1.27M vs 1.19M, so A is actually slightly larger — about 7% more params). Normalizing for parameter count, the kernel advantage becomes smaller. It's real but not the headline finding.

What this means for the research question in the plan:
The original thesis — "wave-equation dynamics in the backbone yield physically interpretable multi-scale denoising" — holds up, but the stronger result is about the conditioning mechanism, not the kernel. The physics-motivated timestep conditioning (kernels that reshape with t, residual gates that attenuate per branch) is what actually drives sample quality. The wave kernel is an additional contribution on top of that, not the primary one.

The honest write-up says: physics-conditioned conditioning >> attention mechanism choice, with wave kernels providing a secondary gain. That's still publishable and arguably more interesting — it says the inductive bias about how timestep information flows through the architecture matters more than whether mixing is done by softmax or FFT convolution.