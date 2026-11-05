#!/usr/bin/env python
"""
Tests for the DDPM training-objective options: cosine schedule, v-prediction, Min-SNR.

Each of these is easy to implement in a way that trains without error and produces
subtly wrong samples -- a v-trained model decoded as eps yields plausible-looking noise,
not a crash. These check the maths rather than that the code runs.
"""

import numpy as np
import torch

from src.training.diffusion import (get_beta_schedule, noise_estimation_loss,
                                    generalized_steps, compute_alpha)

print("=" * 60)
print("Diffusion Objective Tests")
print("=" * 60)

T = 1000

# ---------------------------------------------------------------- [1] cosine schedule
cos = get_beta_schedule("cosine", beta_start=1e-4, beta_end=2e-2, num_diffusion_timesteps=T)
lin = get_beta_schedule("linear", beta_start=1e-4, beta_end=2e-2, num_diffusion_timesteps=T)
assert cos.shape == (T,), cos.shape
assert (cos >= 0).all() and (cos <= 0.999).all(), (cos.min(), cos.max())

ab_cos = np.cumprod(1 - cos)
ab_lin = np.cumprod(1 - lin)
# The whole point of cosine: signal survives much longer through the schedule.
mid = T // 2
assert ab_cos[mid] > ab_lin[mid], (ab_cos[mid], ab_lin[mid])
assert ab_cos[-1] < 0.01, ab_cos[-1]          # still fully destroyed by the end
print(f"[1] cosine keeps signal longer: alpha_bar at midpoint "
      f"{ab_cos[mid]:.4f} (cosine) vs {ab_lin[mid]:.4f} (linear); ends at {ab_cos[-1]:.2e}")

# ---------------------------------------------------------------- [2] v-prediction identity
# v = sqrt(a)*eps - sqrt(1-a)*x0, and the sampler must invert it back to the same x0/eps.
torch.manual_seed(0)
B, H, W = 4, 16, 16
betas = torch.from_numpy(lin).float()
t_idx = torch.tensor([10, 300, 700, 990])
a = compute_alpha(betas, t_idx)
x0 = torch.randn(B, 1, H, W)
eps = torch.randn(B, 1, H, W)
xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
v = a.sqrt() * eps - (1 - a).sqrt() * x0

x0_rec = a.sqrt() * xt - (1 - a).sqrt() * v
eps_rec = (1 - a).sqrt() * xt + a.sqrt() * v
assert torch.allclose(x0_rec, x0, atol=1e-4), (x0_rec - x0).abs().max().item()
assert torch.allclose(eps_rec, eps, atol=1e-4), (eps_rec - eps).abs().max().item()
print(f"[2] v-prediction inverts exactly: max |x0 err| {(x0_rec-x0).abs().max():.2e}, "
      f"max |eps err| {(eps_rec-eps).abs().max():.2e}")


# ---------------------------------------------------------------- [3] loss targets differ
class _Echo(torch.nn.Module):
    """Returns a fixed tensor -- isolates the loss maths from any model behaviour."""
    def __init__(self, out):
        super().__init__()
        self.out = out

    def forward(self, x, t):
        return self.out


x0_pair = torch.cat([torch.randn(B, 1, H, W), x0], dim=1)   # [cond, clean]
zero = torch.zeros(B, 1, H, W)

l_eps = noise_estimation_loss(_Echo(zero), x0_pair, t_idx, eps, betas, prediction_type="eps")
l_v = noise_estimation_loss(_Echo(zero), x0_pair, t_idx, eps, betas, prediction_type="v")
assert torch.isfinite(l_eps) and torch.isfinite(l_v)
assert not torch.isclose(l_eps, l_v), (l_eps.item(), l_v.item())
print(f"[3] eps and v regress different targets: loss {l_eps:.1f} vs {l_v:.1f}")

# a perfect eps-prediction must score ~0 under the eps objective
l_perfect = noise_estimation_loss(_Echo(eps), x0_pair, t_idx, eps, betas, prediction_type="eps")
assert l_perfect.item() < 1e-6, l_perfect.item()
print(f"[3] perfect eps prediction scores {l_perfect.item():.2e}")

# ---------------------------------------------------------------- [4] Min-SNR reweighting
# It must down-weight LOW-noise (high-SNR) timesteps relative to unweighted training --
# that is the entire purpose. Compare one very-low-noise step against a very-noisy one.
lo_noise = torch.tensor([5, 5, 5, 5])       # early t -> high SNR
hi_noise = torch.tensor([995, 995, 995, 995])


def _ratio(t_sel):
    unw = noise_estimation_loss(_Echo(zero), x0_pair, t_sel, eps, betas,
                                prediction_type="eps", min_snr_gamma=0.0)
    wtd = noise_estimation_loss(_Echo(zero), x0_pair, t_sel, eps, betas,
                                prediction_type="eps", min_snr_gamma=5.0)
    return (wtd / unw).item()


r_lo, r_hi = _ratio(lo_noise), _ratio(hi_noise)
assert r_lo < r_hi, (r_lo, r_hi)
assert r_lo < 0.5, r_lo          # high-SNR steps must be substantially suppressed
assert 0.9 < r_hi <= 1.0 + 1e-6  # noisy steps essentially untouched
print(f"[4] Min-SNR down-weights the easy end: high-SNR step scaled x{r_lo:.3f}, "
      f"low-SNR step x{r_hi:.3f}")

# ---------------------------------------------------------------- [5] sampler honours type
# A v-trained model decoded as eps must NOT silently produce the same thing.
seq = list(range(0, T, T // 10))
cond = torch.randn(B, 1, H, W)
xin = torch.randn(B, 1, H, W)
model = _Echo(torch.randn(B, 1, H, W))
out_eps, _ = generalized_steps(xin, cond, seq, model, betas, eta=0.0, prediction_type="eps")
out_v, _ = generalized_steps(xin, cond, seq, model, betas, eta=0.0, prediction_type="v")
assert not torch.allclose(out_eps[-1], out_v[-1], atol=1e-3), \
    "sampler ignored prediction_type -- a v-trained model would be decoded as eps"
print("[5] sampler decodes eps and v differently, as it must")

print("\n" + "=" * 60)
print("All diffusion-objective tests PASSED")
print("=" * 60)
