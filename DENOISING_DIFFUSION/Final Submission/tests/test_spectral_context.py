"""
2.5D spectral context (the physics-informed plan Phase 3a, option ii).

The dataset can hand the model a channel plus k neighbours along velocity instead of one
isolated channel. M1 and M2 are spectral SHAPE statistics and lag M0 badly, and the reason
is structural: every channel is denoised independently, so nothing uses the velocity axis
those two moments are computed over.

Three things have to hold or the change is worse than useless:

  1. k = 0 reproduces the old behaviour EXACTLY, or every existing checkpoint and published
     number silently changes meaning.
  2. Neighbours share the CENTRE channel's scale. Per-neighbour normalisation would map a
     bright channel and a faint one onto the same [0, 1] and destroy the relative amplitude
     along velocity, which is the entire signal being added.
  3. Ends of the cube clamp rather than wrap. The first and last channels are the line-free
     high-velocity ends and are physically unrelated to each other.

    PYTHONPATH=. python3 tests/test_spectral_context.py
"""
import os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from astropy.io import fits

from src.data.fits_cube_dataset import FITSChannelDataset

N_CHAN, SIZE = 12, 32
failures = []


def check(name, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


def make_cube_pair(tmp):
    """
    A cube whose channels have deliberately different amplitudes, so a scale bug shows up.
    Channel c is scaled by (c + 1), a 12x span from one end of the spectrum to the other.
    """
    rng = np.random.default_rng(0)
    base = rng.random((N_CHAN, SIZE, SIZE)).astype(np.float32)
    clean = base * np.arange(1, N_CHAN + 1, dtype=np.float32)[:, None, None]
    dirty = clean + rng.normal(0, 0.05, clean.shape).astype(np.float32)

    hdr = fits.Header({"BMAJ": 4.2629e-5, "BMIN": 3.2923e-5, "BPA": 16.9333,
                       "CDELT2": 0.01 / 3600.0})
    cp, dp = os.path.join(tmp, "clean.fits"), os.path.join(tmp, "dirty.fits")
    fits.PrimaryHDU(clean, header=hdr).writeto(cp)
    fits.PrimaryHDU(dirty, header=hdr).writeto(dp)
    return cp, dp, clean, dirty


def build(tmp, cp, dp, **kw):
    return FITSChannelDataset(
        cubes=[(cp, dp)],
        channel_sampler_fn=lambda n_channels, seed: list(range(n_channels)),
        target_size=SIZE, verbose=False, **kw)


print("=" * 70)
print("2.5D spectral context")
print("=" * 70)

with tempfile.TemporaryDirectory() as tmp:
    cp, dp, clean_raw, dirty_raw = make_cube_pair(tmp)

    base = build(tmp, cp, dp)
    ds1 = build(tmp, cp, dp, n_neighbors=1)
    ds2 = build(tmp, cp, dp, n_neighbors=2)

    # --- 1. backward compatibility ----------------------------------------------------
    d0, c0 = base[5]
    check("k=0 dirty is (1,H,W)", tuple(d0.shape) == (1, SIZE, SIZE), str(tuple(d0.shape)))
    check("k=0 clean is (1,H,W)", tuple(c0.shape) == (1, SIZE, SIZE), str(tuple(c0.shape)))

    d1, c1 = ds1[5]
    check("k=1 dirty is (3,H,W)", tuple(d1.shape) == (3, SIZE, SIZE), str(tuple(d1.shape)))
    check("k=2 dirty is (5,H,W)", tuple(ds2[5][0].shape) == (5, SIZE, SIZE))
    check("clean target stays 1 channel at k=1", tuple(c1.shape) == (1, SIZE, SIZE))

    check("k=1 centre channel is bit-identical to k=0",
          torch.equal(d1[1:2], d0), f"max diff {(d1[1:2] - d0).abs().max():.3e}")
    check("k=1 clean target is bit-identical to k=0", torch.equal(c1, c0))
    check("k=2 centre channel is bit-identical to k=0", torch.equal(ds2[5][0][2:3], d0))

    # --- 2. shared scale ---------------------------------------------------------------
    # The centre channel defines the scale, so IT must span [0,1]; the neighbours must not,
    # because channel c is 12x brighter at one end of the cube than the other.
    centre = d1[1]
    check("centre channel spans [0,1]",
          abs(float(centre.min())) < 1e-5 and abs(float(centre.max()) - 1.0) < 1e-5,
          f"[{float(centre.min()):.4f}, {float(centre.max()):.4f}]")
    lo_nb, hi_nb = d1[0], d1[2]
    check("neighbours are NOT independently normalised",
          not (abs(float(hi_nb.max()) - 1.0) < 1e-3 and abs(float(lo_nb.max()) - 1.0) < 1e-3),
          f"neighbour maxima {float(lo_nb.max()):.3f}, {float(hi_nb.max()):.3f}")
    check("brighter neighbour reads brighter than the centre",
          float(hi_nb.mean()) > float(centre.mean()),
          f"{float(hi_nb.mean()):.3f} vs {float(centre.mean()):.3f}")
    check("fainter neighbour reads fainter than the centre",
          float(lo_nb.mean()) < float(centre.mean()),
          f"{float(lo_nb.mean()):.3f} vs {float(centre.mean()):.3f}")

    # the exact scale: centre dirty channel's own (min, max), applied to all three
    lo, hi = float(dirty_raw[5].min()), float(dirty_raw[5].max())
    for j, ch in enumerate((4, 5, 6)):
        want = torch.from_numpy((dirty_raw[ch] - lo) / (hi - lo)).float()
        check(f"neighbour {ch} uses the centre's scale exactly",
              torch.allclose(d1[j], want, atol=1e-6),
              f"max diff {(d1[j] - want).abs().max():.3e}")

    # --- 3. clamping at the cube's ends ------------------------------------------------
    first = ds1[0][0]
    check("channel 0 clamps: [0,0,1], not a wrap to the last channel",
          torch.equal(first[0], first[1]) and not torch.equal(first[0], first[2]))
    last = ds1[N_CHAN - 1][0]
    check("last channel clamps: [n-2,n-1,n-1]",
          torch.equal(last[1], last[2]) and not torch.equal(last[0], last[1]))
    check("k=2 at channel 0 clamps twice", torch.equal(ds2[0][0][0], ds2[0][0][1]))

    # --- 4. augmentation moves the whole stack together --------------------------------
    aug = build(tmp, cp, dp, n_neighbors=1, augment=True)
    torch.manual_seed(3)
    da, ca = aug[5]
    plain, _ = ds1[5]
    found = None
    for kk in range(4):
        for fl in (False, True):
            t = torch.rot90(plain, kk, dims=(-2, -1))
            if fl:
                t = torch.flip(t, dims=(-1,))
            if torch.allclose(t, da, atol=1e-6):
                found = (kk, fl)
    check("augmented stack is one D4 element of the unaugmented one",
          found is not None, str(found))
    if found is not None:
        kk, fl = found
        t = torch.rot90(ds1[5][1], kk, dims=(-2, -1))
        if fl:
            t = torch.flip(t, dims=(-1,))
        check("clean target got the SAME D4 element as the dirty stack",
              torch.allclose(t, ca, atol=1e-6))

    # --- 5. the model needs no change --------------------------------------------------
    from src.models.unet import UNet
    net = UNet(in_channels=3, out_channels=1, base_channels=8, channel_multipliers=[1, 2],
               groups=8)
    with torch.no_grad():
        out = net(d1[None], torch.zeros(1, dtype=torch.long))
    check("UNet(in_channels=3) consumes the stack and returns 1 channel",
          tuple(out.shape) == (1, 1, SIZE, SIZE), str(tuple(out.shape)))

    # --- 6. one open per item, not 2k+1 ------------------------------------------------
    import src.data.fits_cube_dataset as mod
    opens = {"n": 0}
    real = mod.fits.open

    def counting_open(*a, **k):
        opens["n"] += 1
        return real(*a, **k)

    mod.fits.open = counting_open
    try:
        ds2[6]
    finally:
        mod.fits.open = real
    check("k=2 reads 5 neighbours in one open, plus one for clean",
          opens["n"] == 2, f"{opens['n']} opens")

print("\n" + "-" * 70)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("all checks passed")
sys.exit(0)
