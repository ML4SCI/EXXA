"""Candidate D: loss_amp is ~0 when the prediction's wiggle has clean's amplitude, ~0.5 when it keeps half, ~1 when it doubles it, and ignores a
smooth (rotation-like) M1 gradient. Needs torch (run at the start of the Kaggle training kernel). Run: PYTHONPATH=. python3 tests/test_loss_amp.py"""
import os, sys
import torch
sys.path.insert(0, "experiments")
src = open("experiments/train_unified.py").read()
i = src.index("def loss_amp"); j = src.index("def loss_fn")
ns = {"torch": torch, "Fn": torch.nn.functional}; exec(src[i:j], ns)
loss_amp = ns["loss_amp"]

n, band, H = 2, 10, 96
nb = 2 * band + 1
yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing="ij")
v = (torch.arange(nb, dtype=torch.float32) - band).view(nb, 1, 1)


def cube(amp, grad=0.03):
    vc = grad * (xx - H / 2) + amp * torch.sin(2 * torch.pi * yy / 12.0)       # rotation gradient + wiggle (channels)
    one = torch.exp(-0.5 * ((v - vc[None]) / 2.0) ** 2)                        # (nb, H, W)
    return torch.cat([one] * n)                                                 # n crops


clean = cube(1.0)
zero = torch.zeros(clean.shape[0], 1, H, H)
for amp, want in ((1.0, 0.0), (0.5, 0.5), (2.0, 1.0)):
    got = float(loss_amp(zero, cube(amp), clean, n, band))
    assert abs(got - want) < 0.12, (amp, got, want)
flat = float(loss_amp(zero, cube(1.0, grad=0.05), clean, n, band))     # same wiggle, different smooth gradient: not penalised
assert flat < 0.05, flat
print("loss_amp ok: amplitude 1 / 0.5 / 2 ->", [round(float(loss_amp(zero, cube(a), clean, n, band)), 3) for a in (1.0, 0.5, 2.0)])
