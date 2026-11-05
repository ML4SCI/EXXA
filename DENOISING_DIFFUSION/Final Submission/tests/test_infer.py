"""infer(): off or a narrow beam -> the plain model; a wide beam -> run on a coarser grid, output back at full size.
Run: PYTHONPATH=. python3 tests/test_infer.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import train_unified as tu

seen = []
def fake(model, arm, d, w, dv, bs=8):
    seen.append(d.shape[-1]); return (w + 0.5 * (d - w)).astype(np.float32)
tu.denoise = fake
rng = np.random.default_rng(0)
d, w = rng.standard_normal((4, 240, 240)).astype(np.float32), np.zeros((4, 240, 240), np.float32)
tu.INFER_BEAM = 0.0
assert np.allclose(tu.infer(None, "wiener", d, w, 0.1, 20.0), 0.5 * d) and seen[-1] == 240
tu.INFER_BEAM = 10.0
out = tu.infer(None, "wiener", d, w, 0.1, 11.0)              # within 1.2 x: untouched
assert seen[-1] == 240 and np.allclose(out, 0.5 * d)
for mode in ("corr", "full"):
    tu.INFER_MODE = mode
    out = tu.infer(None, "wiener", d, w, 0.1, 20.0)          # beam 20 px -> grid where it spans 10 px: 120
    assert seen[-1] == 120 and out.shape == d.shape, (seen[-1], out.shape)
    assert out.std() < 0.5 * d.std()                         # coarse grid averaged the white noise down
tu.INFER_BEAM = 10.0; tu.INFER_MODE = "full"
out = tu.infer(None, "wiener", d, w, 0.1, 20.0, ai=False)      # deconvolution regime: the cube as given
assert seen[-1] == 240 and np.allclose(out, 0.5 * d)
print("infer ok")
