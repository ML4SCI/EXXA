#!/usr/bin/env python
"""
End-to-end smoke test of the classical-baseline chain on SYNTHETIC FITS cubes.

Exercises exactly the path 07-classical-baselines.ipynb takes: split_cubes ->
FITSChannelDataset -> tune_on_validation -> denoise_cube -> generate_moment_maps
-> summarise_improvements, so a broken import or a shape/API mismatch is caught
here instead of halfway through a Kaggle session (which is how the missing
`torch.nn.functional` import in the DDPM notebook was found).

Cubes are tiny (24 channels, 64x64) so this runs in seconds on CPU.
"""
import os
import shutil
import sys
import tempfile

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

C, H, W = 24, 64, 64
CONTINUUM_N = 3
rng = np.random.default_rng(0)


def make_cube(seed: int):
    """A rotating-disk-ish cube: a moving Gaussian blob plus a static continuum."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    continuum = 0.4 * np.exp(-((yy - H / 2) ** 2 + (xx - W / 2) ** 2) / (2 * 14.0 ** 2))
    clean = np.empty((C, H, W), dtype=np.float32)
    for ch in range(C):
        # blob sweeps across the disk; brightest at line centre, absent at the edges
        frac = (ch - C / 2) / (C / 2)
        amp = float(np.exp(-(frac ** 2) / 0.18))
        cy = H / 2 + 9.0 * frac
        blob = amp * np.exp(-((yy - cy) ** 2 + (xx - W / 2) ** 2) / (2 * 5.0 ** 2))
        clean[ch] = continuum + blob
    dirty = clean + r.normal(0, 0.05, clean.shape).astype(np.float32)
    return clean, dirty


def write_cubes(root: str, folders):
    for folder, seed in folders:
        d = os.path.join(root, folder)
        os.makedirs(d, exist_ok=True)
        clean, dirty = make_cube(seed)
        hdr = fits.Header()
        # a velocity axis bettermoments can read (m/s), plus a beam
        hdr["CTYPE3"], hdr["CRVAL3"], hdr["CDELT3"], hdr["CRPIX3"] = "VELO-LSR", -2000.0, 200.0, 1.0
        hdr["CTYPE1"], hdr["CDELT1"], hdr["CRVAL1"], hdr["CRPIX1"] = "RA---SIN", -1e-5, 0.0, W / 2
        hdr["CTYPE2"], hdr["CDELT2"], hdr["CRVAL2"], hdr["CRPIX2"] = "DEC--SIN", 1e-5, 0.0, H / 2
        hdr["BPA"], hdr["BMAJ"], hdr["BMIN"] = 16.9333, 4.2629e-5, 3.2923e-5
        hdr["BUNIT"] = "Jy/beam"
        fits.writeto(os.path.join(d, "clean.fits"), clean, header=hdr, overwrite=True)
        fits.writeto(os.path.join(d, "dirty.fits"), dirty, header=hdr, overwrite=True)


print("=" * 66)
print("Classical-baseline chain: end-to-end smoke test (synthetic cubes)")
print("=" * 66)

tmp = tempfile.mkdtemp(prefix="exxa-smoke-")
try:
    # 6 cubes over 4 RunIDs so n_holdout=2 leaves a real train/val/holdout split
    write_cubes(tmp, [
        ("run_0001_00100_rt_00", 1), ("run_0001_00100_rt_01", 2),
        ("run_0002_00200_rt_00", 3), ("run_0003_00300_rt_00", 4),
        ("run_0004_00400_rt_00", 5), ("run_0004_00400_rt_01", 6),
    ])
    print(f"[setup] wrote 6 synthetic cubes ({C}x{H}x{W}) to a temp dir")

    from src.data.cube_split import split_cubes
    from src.data.fits_cube_dataset import FITSChannelDataset, continuum_of
    from src.evaluation.classical import (apply_filter, denoise_cube,
                                          summarise_improvements, tune_on_validation)
    from src.evaluation.moment_maps import generate_moment_maps
    print("[1] every module the notebook imports resolves")

    train, val, holdout = split_cubes(data_dir=tmp, n_holdout=2, val_fraction=0.25,
                                      seed=42, verbose=False)
    assert train and val and holdout, (len(train), len(val), len(holdout))
    print(f"[2] split OK: {len(train)} train / {len(val)} val / {len(holdout)} holdout")

    val_ds = FITSChannelDataset(val, n_samples=8, target_size=32, seed=42,
                                subtract_continuum=True, continuum_n=CONTINUUM_N,
                                verbose=False)
    d0, c0 = val_ds[0]
    assert d0.shape == (1, 32, 32) and c0.shape == (1, 32, 32), (d0.shape, c0.shape)
    print(f"[3] dataset OK: {len(val_ds)} channels, item shape {tuple(d0.shape)}")

    # augmentation must not fire on an eval split, and must fire when asked
    aug_ds = FITSChannelDataset(train, n_samples=8, target_size=32, seed=42,
                               subtract_continuum=True, continuum_n=CONTINUUM_N,
                               augment=True, verbose=False)
    import torch
    torch.manual_seed(0)
    seen = {aug_ds[0][0].numpy().tobytes() for _ in range(60)}
    assert len(seen) > 1, "augment=True produced a single orientation"
    assert len({val_ds[0][0].numpy().tobytes() for _ in range(10)}) == 1, \
        "eval split is not deterministic: augmentation leaked into val"
    print(f"[4] augmentation: {len(seen)} orientations on train, val stays deterministic")

    tuned = tune_on_validation(val_ds, max_channels=8, verbose=False)
    for m in ("gaussian", "median", "wiener", "none"):
        assert m in tuned and np.isfinite(tuned[m]["psnr"]), (m, tuned.get(m))
    best = max(("gaussian", "median", "wiener"), key=lambda m: tuned[m]["psnr"])
    print(f"[5] tuning OK: best={best} param={tuned[best]['param']} "
          f"PSNR {tuned[best]['psnr']:.2f} vs dirty {tuned['none']['psnr']:.2f}")
    assert tuned[best]["psnr"] > tuned["none"]["psnr"], "no filter beat the dirty control"

    # the notebook's section-5 loop, verbatim in structure
    import bettermoments as bm

    def mdiff(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        return float(np.nanmean(np.abs(a[mask] - b[mask])))

    def load_csub(path):
        with fits.open(path, memmap=False) as hdul:
            raw = np.ascontiguousarray(hdul[0].data).astype(np.float32)
            hdr = hdul[0].header.copy()
        return raw - continuum_of(raw, CONTINUUM_N)[None, :, :], hdr

    ho = holdout[0]
    dirty_csub, _ = load_csub(ho["dirty"])
    clean_csub, _ = load_csub(ho["clean"])
    _, velax = bm.load_cube(ho["dirty"])
    assert len(velax) == C, (len(velax), C)
    print(f"[6] cube load + velocity axis OK: {dirty_csub.shape}, velax {len(velax)} chans")

    cl = generate_moment_maps(None, data_velax=(clean_csub, velax))
    di = generate_moment_maps(None, data_velax=(dirty_csub, velax))
    den = denoise_cube(dirty_csub, best, tuned[best]["param"])
    assert den.shape == dirty_csub.shape and den.dtype == np.float32
    no = generate_moment_maps(None, data_velax=(den, velax))
    print(f"[7] moment maps OK: M0 shape {cl[0].shape}")

    row = {"cube": ho["folder"]}
    for nm, c, d, n in zip(("M0", "M1", "M2"), cl, di, no):
        dd, nn = mdiff(c, d), mdiff(c, n)
        row["imp_" + nm] = round(100.0 * (1 - nn / dd), 2) if dd > 0 else float("nan")
    print(f"[8] improvement formula OK: {row}")
    assert np.isfinite(row["imp_M0"]), "M0 improvement is not finite"

    s = summarise_improvements([row, row])
    assert s["M0"]["n"] == 2 and np.isfinite(s["M0"]["mean"])
    print(f"[9] summary OK: M0 {s['M0']['mean']:+.1f}% (n={s['M0']['n']})")

    # section 6's 256-round-trip control, in miniature
    import torch.nn.functional as F
    t = torch.from_numpy(dirty_csub)[:, None]
    small = F.interpolate(t, (32, 32), mode="bilinear", align_corners=False)[:, 0].numpy()
    filt = denoise_cube(small, best, tuned[best]["param"])
    back = F.interpolate(torch.from_numpy(filt)[:, None], (H, W),
                         mode="bilinear", align_corners=False)[:, 0].numpy()
    assert back.shape == dirty_csub.shape, back.shape
    rt = generate_moment_maps(None, data_velax=(back, velax))
    assert np.isfinite(mdiff(cl[0], rt[0]))
    print("[10] resolution round-trip control OK (shape and moments finite)")

    print("\n" + "=" * 66)
    print("Classical-baseline chain smoke test PASSED: notebook logic is sound")
    print("=" * 66)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
