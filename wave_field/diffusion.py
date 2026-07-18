"""
DDPM diffusion process (Ho et al. 2020).

Forward: q(x_t | x_0) = N(sqrt(ᾱ_t) x_0, (1 − ᾱ_t) I)
Loss:    E[||ε − ε_θ(x_t, t)||²]
Sample:  DDPM (full T steps) or DDIM (accelerated, configurable steps)
"""

import math
import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm


class EMA:
    """
    Exponential moving average of model weights.
    Sample from EMA weights instead of training weights — produces noticeably
    cleaner DDPM samples (Karras 2022, Ho 2020). decay=0.9999 standard.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


class DDPMDiffusion:
    """
    Precomputes the complete noise schedule and all derived tensors.

    schedule='linear': original DDPM (Ho et al. 2020)
    schedule='cosine': improved schedule (Nichol & Dhariwal 2021)

    parameterization='eps': model predicts noise (original DDPM)
    parameterization='v':   model predicts v = α·ε − σ·x_0 (Salimans & Ho 2022)
                             — recommended for cosine schedule, balanced loss
                             across all timesteps including extreme SNR.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "cosine",
        parameterization: str = "v",
    ):
        self.T = num_timesteps
        self.parameterization = parameterization

        if schedule == "cosine":
            # Nichol & Dhariwal 2021: alpha_bar_t = cos²((t/T + s)/(1+s) * π/2) / cos²(s/(1+s) * π/2)
            s = 0.008
            steps = num_timesteps + 1
            t = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
            alpha_bar_cos = torch.cos(((t / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alpha_bar_cos = alpha_bar_cos / alpha_bar_cos[0]
            betas = 1.0 - (alpha_bar_cos[1:] / alpha_bar_cos[:-1])
            betas = betas.clamp(max=0.999)
        else:
            # --- linear β schedule ---
            betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        # Cast to float32 for model compatibility
        self.betas = betas.float()
        self.alphas = alphas.float()
        self.alpha_bar = alpha_bar.float()
        self.alpha_bar_prev = alpha_bar_prev.float()

        self.sqrt_alpha_bar = torch.sqrt(alpha_bar).float()
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar).float()

        # For x_0 prediction from ε
        self.sqrt_recip_alpha_bar = torch.sqrt(1.0 / alpha_bar).float()
        self.sqrt_recipm1_alpha_bar = torch.sqrt(1.0 / alpha_bar - 1.0).float()

        # Posterior q(x_{t-1} | x_t, x_0)
        post_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.posterior_variance = post_var.float()
        self.posterior_log_var_clipped = torch.log(post_var.clamp(min=1e-20)).float()
        self.posterior_mean_coef1 = (betas * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)).float()
        self.posterior_mean_coef2 = ((1.0 - alpha_bar_prev) * torch.sqrt(alphas) / (1.0 - alpha_bar)).float()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, broadcast_shape) -> torch.Tensor:
        """
        Gather schedule values at timesteps t and broadcast to broadcast_shape.
        arr: (T,)  t: (B,)  → out: (B, 1, 1, 1, ...)
        """
        out = arr.to(t.device)[t]
        while out.dim() < len(broadcast_shape):
            out = out.unsqueeze(-1)
        return out

    def to(self, device):
        """Move all schedule tensors to device."""
        for attr in vars(self):
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))
        return self

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample x_t from q(x_t | x_0).
        x_t = sqrt(ᾱ_t) x_0 + sqrt(1 − ᾱ_t) ε
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        s_ab = self._extract(self.sqrt_alpha_bar, t, x_start.shape)
        s_omab = self._extract(self.sqrt_one_minus_alpha_bar, t, x_start.shape)
        return s_ab * x_start + s_omab * noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def p_losses(
        self,
        model: torch.nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
        min_snr_gamma: float | None = 5.0,
        self_cond_prob: float = 0.5,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Training loss with optional Min-SNR weighting (Hang et al. 2023) and
        self-conditioning (Chen et al. 2022).

        Self-conditioning: if model.use_self_cond is True, with probability
        `self_cond_prob` we run the model once, recover an x_0 estimate,
        detach it, and feed it back as `self_cond` on a second forward pass.
        The other (1−p) of the time we train with zeros so the model handles
        both cases at inference.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)

        # Self-conditioning: optional first pass to get x_0 estimate
        use_sc = getattr(model, "use_self_cond", False)
        self_cond = None
        if use_sc and self_cond_prob > 0 and torch.rand(1).item() < self_cond_prob:
            # This pass must NOT run under torch.no_grad(): p_losses is called
            # inside the trainer's autocast region, and autocast caches weight
            # casts per region. A no-grad pass populates that cache with
            # *detached* casts which the grad pass below then reuses — every
            # cast-cached weight silently gets zero gradient, and configs with
            # no fp32 parameter path (standard attn + adaln) crash on
            # backward. Recording grad here is cheap; the graph is dropped by
            # detach() below and never backprop'd.
            first_pred = self._call_model(model, x_noisy, t, y=y)
            if self.parameterization == "v":
                s_ab = self._extract(self.sqrt_alpha_bar, t, x_start.shape)
                s_omab = self._extract(self.sqrt_one_minus_alpha_bar, t, x_start.shape)
                # x_0 = α·x_t − σ·v
                self_cond = s_ab * x_noisy - s_omab * first_pred
            else:
                s_recip = self._extract(self.sqrt_recip_alpha_bar, t, x_start.shape)
                s_recipm1 = self._extract(self.sqrt_recipm1_alpha_bar, t, x_start.shape)
                self_cond = s_recip * x_noisy - s_recipm1 * first_pred
            self_cond = self_cond.detach()
            del first_pred  # free the first-pass graph before the main pass

        # Main forward pass (with self_cond if computed, else implicit zeros)
        pred = self._call_model(model, x_noisy, t, self_cond, y=y)

        if self.parameterization == "v":
            s_ab = self._extract(self.sqrt_alpha_bar, t, x_start.shape)
            s_omab = self._extract(self.sqrt_one_minus_alpha_bar, t, x_start.shape)
            target = s_ab * noise - s_omab * x_start
        else:
            target = noise

        # Per-element MSE → per-sample mean → per-sample SNR weighting
        sq = (pred - target) ** 2
        per_sample = sq.flatten(1).mean(dim=1)   # (B,)

        if min_snr_gamma is not None:
            ab = self._extract(self.alpha_bar, t, (t.shape[0],))   # (B,)
            snr = ab / (1.0 - ab).clamp(min=1e-8)
            if self.parameterization == "v":
                w = snr.clamp(max=min_snr_gamma) / (snr + 1.0)
            else:
                w = snr.clamp(max=min_snr_gamma) / snr.clamp(min=1e-8)
            return (w * per_sample).mean()
        return per_sample.mean()

    @staticmethod
    def _call_model(model, x_t, t, self_cond=None, y=None, force_drop_ids=None):
        """Forward pass, threading self_cond / class labels if the model uses them."""
        kwargs = {}
        if getattr(model, "use_self_cond", False):
            kwargs["self_cond"] = self_cond
        if getattr(model, "num_classes", None) is not None and y is not None:
            kwargs["y"] = y
            if force_drop_ids is not None:
                kwargs["force_drop_ids"] = force_drop_ids
        return model(x_t, t, **kwargs)

    def _model_eps(self, model: torch.nn.Module, x_t: torch.Tensor, t: torch.Tensor,
                   self_cond: torch.Tensor | None = None,
                   y: torch.Tensor | None = None,
                   force_drop_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Run the model and return ε prediction regardless of parameterization."""
        pred = self._call_model(model, x_t, t, self_cond, y, force_drop_ids)
        if self.parameterization == "v":
            # v = α·ε − σ·x_0, and x_t = α·x_0 + σ·ε
            # → ε = α·v + σ·x_t  (using α² + σ² = 1)
            s_ab = self._extract(self.sqrt_alpha_bar, t, x_t.shape)
            s_omab = self._extract(self.sqrt_one_minus_alpha_bar, t, x_t.shape)
            return s_ab * pred + s_omab * x_t
        return pred

    def _guided_eps(self, model: torch.nn.Module, x_t: torch.Tensor, t: torch.Tensor,
                    self_cond: torch.Tensor | None = None,
                    y: torch.Tensor | None = None,
                    guidance_scale: float = 1.0) -> torch.Tensor:
        """
        ε prediction with optional classifier-free guidance (Ho & Salimans 2022).

        guidance_scale == 1 or no labels → plain conditional/unconditional ε.
        Otherwise:  ε = ε_uncond + w · (ε_cond − ε_uncond),  where ε_uncond uses
        the learned null token (force_drop_ids=1).
        """
        if y is None or guidance_scale == 1.0 or getattr(model, "num_classes", None) is None:
            return self._model_eps(model, x_t, t, self_cond, y)
        eps_cond = self._model_eps(model, x_t, t, self_cond, y)
        drop = torch.ones(x_t.shape[0], device=x_t.device)
        eps_uncond = self._model_eps(model, x_t, t, self_cond, y, force_drop_ids=drop)
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    # ------------------------------------------------------------------
    # Reverse process helpers
    # ------------------------------------------------------------------

    def _predict_xstart(
        self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor
    ) -> torch.Tensor:
        """Recover x_0 estimate from predicted noise."""
        c1 = self._extract(self.sqrt_recip_alpha_bar, t, x_t.shape)
        c2 = self._extract(self.sqrt_recipm1_alpha_bar, t, x_t.shape)
        return (c1 * x_t - c2 * eps_pred).clamp(-1.0, 1.0)

    def _q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ):
        """Mean, variance, and log-variance of q(x_{t-1} | x_t, x_0)."""
        c1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        c2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = c1 * x_start + c2 * x_t
        var = self._extract(self.posterior_variance, t, x_t.shape)
        log_var = self._extract(self.posterior_log_var_clipped, t, x_t.shape)
        return mean, var, log_var

    # ------------------------------------------------------------------
    # Sampling — DDPM (full)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample_step(
        self, model: torch.nn.Module, x: torch.Tensor, t_idx: int,
        self_cond: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> tuple:
        """
        Single DDPM reverse step: sample x_{t-1} from p_θ(x_{t-1} | x_t).
        Returns (x_next, x_start) so callers can thread x_start as self_cond.
        """
        B = x.shape[0]
        t = torch.full((B,), t_idx, dtype=torch.long, device=x.device)
        eps_pred = self._guided_eps(model, x, t, self_cond, y, guidance_scale)
        x_start = self._predict_xstart(x, t, eps_pred)
        mean, _, log_var = self._q_posterior(x_start, x, t)
        noise = torch.randn_like(x) if t_idx > 0 else torch.zeros_like(x)
        return mean + (0.5 * log_var).exp() * noise, x_start

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: tuple,
        device: torch.device | str,
        progress: bool = True,
        y: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Full DDPM reverse diffusion from pure noise → image."""
        x = torch.randn(shape, device=device)
        x_start = None   # threaded as self_cond when use_self_cond is on
        steps = list(reversed(range(self.T)))
        if progress:
            steps = tqdm(steps, desc="DDPM sampling", total=self.T)
        for t_idx in steps:
            x, x_start = self.p_sample_step(model, x, t_idx, self_cond=x_start,
                                            y=y, guidance_scale=guidance_scale)
        return x

    # ------------------------------------------------------------------
    # Sampling — DDIM (accelerated)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        model: torch.nn.Module,
        shape: tuple,
        device: torch.device | str,
        num_steps: int = 50,
        eta: float = 0.0,
        progress: bool = True,
        y: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        DDIM sampling (Song et al. 2021) — much faster than full DDPM.

        eta=0: deterministic (DDIM)
        eta=1: matches DDPM variance
        y / guidance_scale: optional class labels + classifier-free guidance.
        """
        T = self.T
        step_ratio = T // num_steps
        # Timestep subsequence (evenly spaced, descending)
        ts = list(reversed(range(0, T, step_ratio)))

        x = torch.randn(shape, device=device)
        x_start_prev = None   # threaded as self_cond
        if progress:
            ts_iter = tqdm(ts, desc=f"DDIM sampling ({num_steps} steps)")
        else:
            ts_iter = ts

        for i, t_curr in enumerate(ts_iter):
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1

            B = x.shape[0]
            t_tensor = torch.full((B,), t_curr, dtype=torch.long, device=device)
            eps_pred = self._guided_eps(model, x, t_tensor, x_start_prev, y, guidance_scale)

            ab_t = self._extract(self.alpha_bar, t_tensor, x.shape)

            if t_prev >= 0:
                t_prev_tensor = torch.full((B,), t_prev, dtype=torch.long, device=device)
                ab_prev = self._extract(self.alpha_bar, t_prev_tensor, x.shape)
            else:
                ab_prev = torch.ones_like(ab_t)

            # Predicted x_0 — no clamp here; clamping intermediates compounds errors
            # over many small steps. Final output is clamped after the loop.
            x_start = (x - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
            x_start_prev = x_start   # feed forward to next step's self-conditioning

            # DDIM variance
            sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).sqrt()

            # Direction pointing to x_t
            dir_xt_coef = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt()

            noise = torch.randn_like(x) if (t_prev >= 0 and eta > 0) else torch.zeros_like(x)
            x = ab_prev.sqrt() * x_start + dir_xt_coef * eps_pred + sigma * noise

        return x

    # ------------------------------------------------------------------
    # Sampling — DPM-Solver++(2M)  (Lu et al. 2022)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def dpmpp_2m_sample(
        self,
        model: torch.nn.Module,
        shape: tuple,
        device: torch.device | str,
        num_steps: int = 50,
        progress: bool = True,
        y: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        DPM-Solver++(2M) — 2nd-order multistep deterministic solver in data
        (x_0) prediction form. Higher sample quality than DDIM at the same (or
        fewer) steps, with no retraining. Supports class labels + CFG.

        Works in the half-log-SNR variable λ = log(α/σ); each step uses the
        current and previous x_0 predictions for a 2nd-order update.
        """
        T = self.T
        step_ratio = T // num_steps
        ts = list(reversed(range(0, T, step_ratio)))   # descending current timesteps

        x = torch.randn(shape, device=device)
        x0_prev = None          # previous x_0 estimate (for 2nd-order term)
        x_start_prev = None     # threaded as self_cond
        h_last = None
        idxs = range(len(ts))
        if progress:
            idxs = tqdm(idxs, desc=f"DPM++(2M) sampling ({num_steps} steps)")

        for i in idxs:
            B = x.shape[0]
            t_tensor = torch.full((B,), ts[i], dtype=torch.long, device=device)
            eps = self._guided_eps(model, x, t_tensor, x_start_prev, y, guidance_scale)

            ab_t = self._extract(self.alpha_bar, t_tensor, x.shape)
            alpha_t, sigma_t = ab_t.sqrt(), (1.0 - ab_t).sqrt()
            x0 = (x - sigma_t * eps) / alpha_t          # unclamped x_0 prediction
            x_start_prev = x0

            last = i + 1 == len(ts)
            if not last:
                t_next = torch.full((B,), ts[i + 1], dtype=torch.long, device=device)
                ab_n = self._extract(self.alpha_bar, t_next, x.shape)
            else:
                ab_n = torch.ones_like(ab_t)
            alpha_n, sigma_n = ab_n.sqrt(), (1.0 - ab_n).sqrt()

            lam_t = torch.log(alpha_t) - torch.log(sigma_t.clamp(min=1e-8))
            lam_n = torch.log(alpha_n) - torch.log(sigma_n.clamp(min=1e-8))
            h = lam_n - lam_t

            if x0_prev is None or last:
                D = x0                                   # 1st-order (also on final step)
            else:
                r = h_last / h
                D = (1.0 + 1.0 / (2.0 * r)) * x0 - (1.0 / (2.0 * r)) * x0_prev

            x = (sigma_n / sigma_t) * x - alpha_n * (torch.exp(-h) - 1.0) * D
            x0_prev, h_last = x0, h

        return x.clamp(-1.0, 1.0)
