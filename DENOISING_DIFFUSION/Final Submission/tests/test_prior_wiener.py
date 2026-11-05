"""prior_wiener: with the true signal as the prior it is the oracle filter, so it must beat the noisy cube by a wide margin; with the noisy cube itself as
the prior it must not amplify anything. --train-beam-px: a redraw lands on the grid where the beam spans the asked number of px.
Run: PYTHONPATH=. python3 tests/test_prior_wiener.py"""
import os, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import train_unified as tu

rng = np.random.default_rng(0)
C, n = 40, 64
clean = ndimage.gaussian_filter(rng.standard_normal((C, n, n)), (2, 4, 4)).astype(np.float32)
clean /= clean.std()
d = clean + 1.0 * rng.standard_normal(clean.shape).astype(np.float32)
mse = lambda a: float(((a - clean) ** 2).mean())
w_or = tu.prior_wiener(d, clean)
w_self = tu.prior_wiener(d, d)
assert mse(w_or) < 0.3 * mse(d), (mse(w_or), mse(d))
assert mse(w_self) <= mse(d) + 1e-6, (mse(w_self), mse(d))
print(f"prior_wiener ok: mse noisy {mse(d):.3f}, oracle prior {mse(w_or):.3f}, self prior {mse(w_self):.3f}")

tu.REAL.update(snr=(10.0, 10.0), chan_a=0.0, beam_max=1.0, beam_px=(10.0, 10.0))
cube = tu.TrainCube("synthetic", "line", np.abs(clean), np.abs(clean) + 0.1 * rng.standard_normal(clean.shape).astype(np.float32), 0.1, fwhm_px=5.0)
cube.redraw(rng, [256, 600])
assert cube.S == 128, cube.S                               # 64 px * 10 / 5 = 128: the beam now spans 10 px
print(f"train-beam-px ok: native {n} px, beam 5 px -> redraw at {cube.S} px")
