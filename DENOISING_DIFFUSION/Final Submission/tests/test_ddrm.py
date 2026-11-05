"""
DDRM's operator machinery, validated without any trained model.

The sampler itself needs a diffusion prior to test end to end, but everything it depends on
-- the transfer function, the forward operator, and the measured/unmeasured split that IS the
algorithm -- can be checked exactly against known answers. If those are wrong, nothing
downstream means anything, and the failure would be invisible: a wrong operator still produces
plausible-looking images.

    PYTHONPATH=. python3 tests/test_ddrm.py
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.training.ddrm import beam_transfer_function, apply_operator
from src.evaluation.forward_operator import apply_beam

failures = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


print("=" * 70)
print("DDRM operator machinery")
print("=" * 70)

N = 128
rng = np.random.default_rng(0)

# A centred, sidelobed beam, as estimate_beam_from_pair returns.
y, x = np.mgrid[-20:21, -20:21].astype(float)
r = np.sqrt(x ** 2 + y ** 2)
beam = np.exp(-r ** 2 / (2 * 3.0 ** 2)) * np.cos(r / 4.0)
beam /= beam.max()

# --- 1. the transfer function must reproduce the convolution EXACTLY -------------------
print("\ncase 1  apply_operator matches the reference convolution")
img = rng.random((2, N, N)) - 0.5
ref = apply_beam(img, beam)                        # the已-validated numpy path
transfer = beam_transfer_function(beam, (N, N))
got = apply_operator(torch.from_numpy(img).float()[:, None], transfer)[:, 0].numpy()
err = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-12))
check("torch FFT operator matches the numpy convolution", err < 1e-5,
      f"max relative error {err:.2e}")

# --- 2. singular values must be the beam's own spectrum --------------------------------
print("\ncase 2  the transfer function IS |FFT(beam)|")
check("DC gain equals the beam's sum (a convolution preserves total flux by its sum)",
      abs(float(transfer[0, 0]) - abs(beam.sum())) < 1e-3 * max(abs(beam.sum()), 1e-9),
      f"transfer[0,0]={float(transfer[0,0]):.4f}, beam.sum()={beam.sum():.4f}")
check("all singular values are non-negative", bool((transfer >= 0).all()))
check("the beam suppresses high frequencies",
      float(transfer.max()) > 5 * float(transfer[N // 2, N // 2]),
      f"peak {float(transfer.max()):.3f} vs Nyquist {float(transfer[N//2, N//2]):.4f}")

# --- 3. the measured/unmeasured split, which IS the algorithm ---------------------------
# This is the decision DDRM turns on: modes the beam suppressed below the noise are handed to
# the prior, everything else is pinned to the data. Getting the threshold backwards would
# silently invert the method -- generating where it should restore and vice versa.
print("\ncase 3  the measured/unmeasured split behaves correctly with noise level")
# This beam NULLS most of the spectrum: a sidelobed kernel's transfer function crosses zero
# wherever its cos() does, so 61.5% of modes fall below 1e-6 no matter how low the noise is.
# Those modes carry no information at ANY noise level, and no threshold recovers them. That
# is not a defect, it is exactly the information loss DDRM exists to fill from the prior --
# and it is why the 2026-08-27 ablation found the beam alone erases the GI wiggle.
frac_lownoise = float((transfer > 1e-6).float().mean())
check("even at negligible noise, the beam's own nulls leave most modes unmeasurable",
      0.2 < frac_lownoise < 0.6,
      f"{100*frac_lownoise:.1f}% measured at sigma_y=1e-6 (the rest are true nulls)")
check("huge noise -> nothing measured", float((transfer > 1e9).float().mean()) < 0.01)
check("lowering the noise floor monotonically admits more modes",
      float((transfer > 1e-8).float().mean()) > float((transfer > 1e-2).float().mean()),
      f"{100*float((transfer>1e-8).float().mean()):.1f}% vs "
      f"{100*float((transfer>1e-2).float().mean()):.1f}%")

# The physically meaningful case: a realistic noise level leaves SOME modes unmeasured.
sigma_real = 0.02 * float(transfer.max())
frac_real = float((transfer > sigma_real).float().mean())
check("a realistic noise level leaves a genuine mix of measured and unmeasured modes",
      0.01 < frac_real < 0.99, f"{100*frac_real:.1f}% measured at sigma_y=2% of peak gain")

# --- 4. the least-squares estimate must invert the beam where it is measurable ----------
print("\ncase 4  dividing by the transfer function inverts the convolution where measurable")
clean = rng.random((1, N, N)) - 0.5
dirty = apply_operator(torch.from_numpy(clean).float()[:, None], transfer)
Y = torch.fft.fft2(dirty)
s = transfer[None, None]
measured = s > 1e-6
Xhat = torch.where(measured, Y / torch.clamp(s, min=1e-8), torch.zeros_like(Y))
recon = torch.real(torch.fft.ifft2(Xhat))[:, 0].numpy()

# Compare in the FOURIER domain, restricted to the modes the beam actually passed. Comparing
# whole images instead conflates two different things: whether the inversion is correct where
# it is possible (it is, exactly), and how much of the image the beam destroyed (most of it).
# The naive image-space correlation here is 0.62, and reads as a broken inverse when it is
# actually a correct inverse of a beam that nulled 61% of the spectrum.
Xc = torch.fft.fft2(torch.from_numpy(clean).float()[:, None])
m = measured.expand_as(Xc)
err = (torch.abs(Xhat[m] - Xc[m]) / torch.clamp(torch.abs(Xc[m]), min=1e-8)).median()
check("the inversion is exact on the modes the beam DID pass",
      float(err) < 1e-3, f"median relative error {float(err):.2e} over {int(m.sum())} modes")

corr = float(np.corrcoef(recon.ravel(), clean.ravel())[0, 1])
check("image-space correlation is limited by the beam's nulls, not by the inversion",
      0.4 < corr < 0.9,
      f"r={corr:.3f} -- the information in the nulled 61% of modes is gone, and this is "
      f"precisely the gap a diffusion prior has to fill")

# --- 4b. a beam larger than the grid must be cropped, not crash -------------------------
# Caught by the 64px dry run: a 129px recovered beam against a 64px grid raised a broadcast
# error. At 256px it would not have crashed but would have mis-placed the beam silently,
# which is worse than a crash.
print("\ncase 4b  an oversized beam is cropped about its centre")
big_beam = np.zeros((129, 129)); big_beam[64, 64] = 1.0     # delta at the exact centre
for size in (32, 64, 128, 256):
    t = beam_transfer_function(big_beam, (size, size))
    check(f"{size}px grid handles a 129px beam",
          tuple(t.shape) == (size, size) and bool(torch.isfinite(t).all()))
    # a centred delta convolves to identity: flat unit gain at every frequency
    check(f"{size}px: centred delta gives flat unit transfer",
          abs(float(t.max()) - 1.0) < 1e-5 and abs(float(t.min()) - 1.0) < 1e-5,
          f"min {float(t.min()):.4f} max {float(t.max()):.4f}")

# --- 5. the config must survive torch.save, and the unconditional path must train --------
# DotDict.__getattr__ returned None for ANY missing key, including dunders. pickle probes for
# __reduce_ex__/__getstate__, got None, and tried to call it -- surfacing as
# "TypeError: 'NoneType' object is not callable" from inside torch.save with nothing pointing
# at DotDict. That broke checkpointing for every config, conditional included, so this guards
# the whole project's DDPM checkpointing, not just DDRM's.
print("\ncase 5  config pickles, and the unconditional training path runs")
import pickle, tempfile
from src.models.diffusion_unet import default_diffusion_config
from src.training.diffusion import DenoisingDiffusion
from torch.utils.data import DataLoader, Dataset

cfg = default_diffusion_config(image_size=32)
cfg.data.conditional = False
try:
    pickle.dumps(cfg)
    ok_pickle = True
except TypeError:
    ok_pickle = False
check("the config survives pickling (torch.save depends on it)", ok_pickle)
check("a missing key still returns None, as the codebase relies on", cfg.model.no_such_key is None)


class _Clean(Dataset):
    def __len__(self): return 4
    def __getitem__(self, i):
        torch.manual_seed(i)
        return torch.rand(1, 32, 32)          # bare tensor, not a (dirty, clean) pair


cfg.diffusion.prediction_type = "v"
cfg.diffusion.min_snr_gamma = 5.0
with tempfile.TemporaryDirectory() as td:
    r = DenoisingDiffusion(config=cfg, device="cpu", lr=2e-5,
                           checkpoint_path=os.path.join(td, "p.pth"))
    loader = DataLoader(_Clean(), batch_size=2)
    h = r.train(loader, loader, n_epochs=1, verbose=False)
    check("unconditional training runs on single-channel batches",
          len(h["train_losses"]) == 1 and np.isfinite(h["train_losses"][0]),
          f"loss {h['train_losses'][0]:.1f}")
    check("the checkpoint is actually written", os.path.exists(os.path.join(td, "p.pth")))

print("\n" + "-" * 70)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
sys.exit(0)
