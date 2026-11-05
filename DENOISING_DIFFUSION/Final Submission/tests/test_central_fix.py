"""Candidate A: central_flux_fix brings a 35 percent over-bright compact centre back to the truth and leaves the line shape (M1) and the outer disc unchanged.
Light (96 px, no model). Run: PYTHONPATH=. python3 tests/test_central_fix.py"""
import numpy as np
import importlib.util, os, sys
sys.path.insert(0, "experiments")
spec = importlib.util.spec_from_file_location("tu_cf", "experiments/train_unified.py")
src = open("experiments/train_unified.py").read()
i = src.index("def central_flux_fix"); j = src.index("_NATIVE = {}")
ns = {"np": np}; exec(src[i:j], ns)
cf = ns["central_flux_fix"]
n, C, beam = 96, 40, 4.0
yy, xx = np.indices((n, n)); r = np.hypot(yy - 48, xx - 48)
v = np.linspace(-2, 2, C)
vc = 0.02 * (xx - 48)
truth = (np.exp(-r / 8.0)[None] * np.exp(-0.5 * ((v[:, None, None] - vc[None]) / 0.3) ** 2)).astype(np.float32)
boost = 1 + 0.35 * np.exp(-0.5 * (r / beam) ** 2)
pred = (truth * boost[None]).astype(np.float32)
dirty = truth + np.random.default_rng(0).normal(0, 0.002, truth.shape).astype(np.float32)
fixed, info = cf(pred, dirty, beam)
c = r <= beam
before, after = pred[:, c].sum() / truth[:, c].sum(), fixed[:, c].sum() / truth[:, c].sum()
outer = r > 5 * beam
assert abs(after - 1) < 0.06 and before > 1.2, (before, after)
assert np.allclose(fixed[:, outer], pred[:, outer])
m1 = lambda x: (x * v[:, None, None]).sum(0) / np.maximum(x.sum(0), 1e-30)
assert np.nanmax(np.abs(m1(fixed) - m1(pred))[c]) < 1e-5
print(f"central_flux_fix ok: centre flux / truth {before:.3f} -> {after:.3f}; M1 unchanged; outer disc unchanged; {info}")
