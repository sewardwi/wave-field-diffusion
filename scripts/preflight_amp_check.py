"""
Preflight: verify the AMP self-conditioning gradient path on the actual device.

This directly exercises the bug that killed sc09_standard_adaln (no_grad
self-cond pass inside an autocast region + autocast weight cache on older
torch => zero grads / disconnected graph). It builds the exact worst-case
config (standard attention + adaln: no fp32 parameter path), forces the
self-cond branch, and asserts every parameter receives a gradient under the
same autocast settings train_audio.py uses.

Exit 0 = safe to train.  Exit 1 = do NOT burn a night of GPU time.

Set WFD_PREFLIGHT_DEVICE=cpu to run the logic without CUDA (local testing).
"""

import os
import sys
import types

import torch

# Runnable as `python scripts/preflight_amp_check.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    dev = os.environ.get("WFD_PREFLIGHT_DEVICE", "cuda")
    if dev == "cuda":
        if not torch.cuda.is_available():
            print("PREFLIGHT FAIL: no CUDA GPU visible.")
            return 1
        p = torch.cuda.get_device_properties(0)
        print(f"CUDA {torch.version.cuda}  {p.name}  {p.total_memory / 1e9:.1f} GB  "
              f"torch {torch.__version__}")

    import train_audio
    from wave_field.diffusion import DDPMDiffusion

    args = types.SimpleNamespace(
        patch_size=16, dim=32, depth=2, num_heads=4, timestep_dim=32,
        conditioning="adaln", self_cond=True, dynamic_filter=False,
        gating="pointwise", num_classes=None, class_dropout=0.1,
    )
    torch.manual_seed(0)
    model = train_audio.make_audio_model(args, use_standard_attn=True).to(dev)
    # Perturb away from zero-init so every weight contributes to the loss —
    # at init the zero-init output head makes upstream grads legitimately zero.
    with torch.no_grad():
        for q in model.parameters():
            q.add_(0.02 * torch.randn_like(q))

    diffusion = DDPMDiffusion(num_timesteps=100, schedule="cosine",
                              parameterization="v")
    diffusion.to(torch.device(dev))
    x = torch.randn(4, 1, 16384, device=dev)
    t = torch.randint(0, 100, (4,), device=dev)

    # Mirror train_audio.py's amp_ctx exactly.
    amp = torch.autocast(device_type=dev, dtype=torch.bfloat16,
                         cache_enabled=False)
    with amp:
        loss = diffusion.p_losses(model, x, t, self_cond_prob=1.0)

    if not loss.requires_grad:
        print("PREFLIGHT FAIL: loss has no grad_fn — the AMP self-cond bug is "
              "present on this device/torch version.")
        return 1
    loss.backward()

    zero = [n for n, q in model.named_parameters()
            if q.grad is None or q.grad.abs().sum() == 0]
    if zero:
        print(f"PREFLIGHT FAIL: {len(zero)} params received zero gradient on a "
              f"self-cond batch, e.g. {zero[:4]}")
        return 1

    n = sum(1 for _ in model.parameters())
    print(f"Preflight OK: AMP self-cond backward on {dev} — all {n} params "
          f"received gradients.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
