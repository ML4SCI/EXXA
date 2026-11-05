#!/usr/bin/env python
"""
Test EMAHelper's bias-corrected warmup: the bug that produced PSNR 14dB /
SSIM 0.55 on DDPM eval was an EMA snapshot still stuck near its random init
after ~1300 short-run steps (mu=0.999 has a ~1000-step time-constant).
"""

import torch
import torch.nn as nn

from src.training.diffusion import EMAHelper

print("=" * 60)
print("EMAHelper Warmup Test")
print("=" * 60)

torch.manual_seed(0)
model = nn.Linear(4, 4)
target = torch.randn(4, 4)
with torch.no_grad():
    model.weight.copy_(target)

ema = EMAHelper(mu=0.999)
ema.register(model)

# Move the "trained" weights away from init, then EMA-update for a short run
# (mirrors ~1300 steps: far too few for mu=0.999 to leave its starting point
# without warmup).
with torch.no_grad():
    model.weight.add_(1.0)  # simulate training moving weights
for _ in range(50):
    ema.update(model)

shadow = ema.shadow["weight"]
dist_to_init = (shadow - target).abs().mean().item()
dist_to_trained = (shadow - model.weight).abs().mean().item()
print(f"[1] after 50 short-run updates: dist_to_init={dist_to_init:.4f} "
      f"dist_to_trained={dist_to_trained:.4f}")
assert dist_to_trained < dist_to_init, (
    "EMA shadow should have moved closer to the trained weights than to init "
    "within 50 steps -- warmup regressed to the pre-fix frozen-near-init bug"
)

# state_dict/load_state_dict round-trip must preserve num_updates so a resumed
# run doesn't reset the warmup ramp.
ema2 = EMAHelper(mu=0.999)
ema2.register(nn.Linear(4, 4))
ema2.load_state_dict(ema.state_dict())
assert ema2.num_updates == ema.num_updates == 50
print(f"[2] state_dict round-trip preserved num_updates={ema2.num_updates}")

# Old-format checkpoints (bare shadow dict, no num_updates) must still load.
ema3 = EMAHelper(mu=0.999)
ema3.register(nn.Linear(4, 4))
ema3.load_state_dict({"weight": target})
assert ema3.num_updates == 0
print("[3] backward-compat load of bare shadow dict OK")

print("=" * 60)
print("All EMA warmup tests passed!")
print("=" * 60)
