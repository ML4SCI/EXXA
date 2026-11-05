"""loss_m1 returns the line-centre shift in channels, ignores pixels without a line, and is zero for a perfect prediction.
Run: PYTHONPATH=. python3 tests/test_loss_m1.py"""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import train_unified as tu

band, n, h = 10, 2, 8
v = torch.arange(2 * band + 1, dtype=torch.float32) - band
line = lambda c0: torch.exp(-0.5 * ((v - c0) / 2.0) ** 2).view(1, -1, 1, 1).expand(n, -1, h, h)
clean = line(0.0).clone()
clean[:, :, :, : h // 2] = 0.0                                   # half the pixels carry no line: masked out
flat = lambda a: a.reshape(n * (2 * band + 1), h, h)
for shift in (0.0, 1.0, -2.5):
    pred = line(shift).clone(); pred[:, :, :, : h // 2] = 0.0
    corr = (flat(pred) - flat(clean)).unsqueeze(1)                  # base = clean, so pred = base + corr
    l, frac = tu.loss_m1(corr, flat(clean), flat(clean), n, band)
    assert abs(l.item() - abs(shift)) < 0.05, (shift, l.item())
    assert abs(frac - 0.5) < 1e-6, frac
print("loss_m1 ok")
