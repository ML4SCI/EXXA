#!/usr/bin/env python
"""
Integration test for the checkpoint -> denoise-cube -> moment-map path.

Notebook 08 is a ~3 h GPU run. The parts most likely to break are not the training
(already covered by test_repeat_config) but the glue afterwards:
  repeat_config -> per-seed checkpoint -> load_net() rebuild -> denoise_cube_unet()
  -> bettermoments -> improvement formula -> channel_artifacts.
A shape or key mismatch anywhere there wastes the whole session, which is exactly
how the missing `torch.nn.functional` import in the DDPM notebook was found.

This reproduces that glue verbatim in miniature, on CPU, with no real data.
"""
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.cube_split import split_cubes
from src.data.fits_cube_dataset import FITSChannelDataset, continuum_of
from src.evaluation.artifacts import channel_artifacts, summarise
from src.evaluation.classical import summarise_improvements
from src.evaluation.moment_maps import generate_moment_maps
from src.models.unet import UNet
from src.training.sweep import repeat_config

C, H, W = 20, 48, 48
TARGET_SIZE = 24
CONTINUUM_N = 3
BS = 8
moments = ["M0", "M1", "M2"]

print("=" * 68)
print("Notebook 08 critical path: smoke test (synthetic cubes, CPU)")
print("=" * 68)


def make_pair(seed):
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    cont = 0.4 * np.exp(-((yy - H / 2) ** 2 + (xx - W / 2) ** 2) / (2 * 10.0 ** 2))
    clean = np.empty((C, H, W), np.float32)
    for ch in range(C):
        frac = (ch - C / 2) / (C / 2)
        amp = float(np.exp(-(frac ** 2) / 0.2))
        cy = H / 2 + 7.0 * frac
        clean[ch] = cont + amp * np.exp(-((yy - cy) ** 2 + (xx - W / 2) ** 2) / (2 * 4.0 ** 2))
    return clean, (clean + r.normal(0, 0.05, clean.shape)).astype(np.float32)


tmp = tempfile.mkdtemp(prefix="exxa-nb08-")
try:
    for i, folder in enumerate(["run_0001_00100_rt_00", "run_0001_00100_rt_01",
                                "run_0002_00200_rt_00", "run_0003_00300_rt_00"]):
        d = os.path.join(tmp, folder)
        os.makedirs(d, exist_ok=True)
        clean, dirty = make_pair(i + 1)
        hdr = fits.Header()
        hdr["CTYPE3"], hdr["CRVAL3"], hdr["CDELT3"], hdr["CRPIX3"] = "VELO-LSR", -1500.0, 150.0, 1.0
        hdr["BPA"], hdr["BMAJ"], hdr["BMIN"] = 16.9333, 4.2629e-5, 3.2923e-5
        fits.writeto(os.path.join(d, "clean.fits"), clean, header=hdr, overwrite=True)
        fits.writeto(os.path.join(d, "dirty.fits"), dirty, header=hdr, overwrite=True)
    print(f"[setup] 4 synthetic cubes ({C}x{H}x{W})")

    train, val, holdout = split_cubes(data_dir=tmp, n_holdout=1, val_fraction=0.34,
                                      seed=42, verbose=False)
    common = dict(n_samples=6, target_size=TARGET_SIZE, seed=42,
                  subtract_continuum=True, continuum_n=CONTINUUM_N, verbose=False)
    train_ds = FITSChannelDataset(train, **common)
    train_aug = FITSChannelDataset(train, augment=True, **common)
    val_ds = FITSChannelDataset(val, **common)
    print(f"[1] datasets: {len(train_ds)} train / {len(val_ds)} val / {len(holdout)} holdout")

    device = torch.device("cpu")
    ckpt_dir = os.path.join(tmp, "ck")
    CFG = dict(base_channels=8, channel_multipliers=(1, 2), lr=1e-3, alpha=0.88,
               batch_size=4, min_epochs=1, max_epochs=2, patience=1, sched_patience=2,
               use_beam=False)

    # the notebook's three-config loop, in miniature
    results = {}
    for name, ds in (("winner", train_ds), ("winner_aug", train_aug)):
        results[name] = repeat_config(ds, val_ds, device, n_seeds=2, base_seed=42,
                                      ckpt_dir=ckpt_dir, tag=name, verbose=False, **CFG)
    print(f"[2] repeat_config ran for {list(results)}; "
          f"winner PSNR {results['winner']['psnr']['mean']:.3f}"
          f"+/-{results['winner']['psnr']['std']:.3f}")

    # notebook cell: best-seed selection
    def best_seed_of(name):
        return int(max(results[name]["rows"], key=lambda r: r["psnr"])["seed"])

    bs_ = best_seed_of("winner")
    ckpt_path = os.path.join(ckpt_dir, f"winner_seed{bs_}.pth")
    assert os.path.exists(ckpt_path), ckpt_path
    print(f"[3] best-seed selection OK: seed {bs_} -> {os.path.basename(ckpt_path)}")

    # notebook cell: load_net, verbatim
    def load_net(p):
        ck = torch.load(p, map_location=device, weights_only=False)
        net = UNet(in_channels=1, out_channels=1, base_channels=ck["base_channels"],
                   channel_multipliers=ck["channel_multipliers"], time_emb_dim=128,
                   num_res_blocks=2, groups=math.gcd(8, ck["base_channels"]),
                   beam_dim=ck.get("beam_dim", 0)).to(device)
        net.load_state_dict(ck["model_state_dict"])
        net.eval()
        return net, ck

    net, ck = load_net(ckpt_path)
    print(f"[4] load_net rebuilt the architecture from checkpoint metadata "
          f"(base {ck['base_channels']}, mult {ck['channel_multipliers']}, ep {ck['epoch']})")

    # notebook cell: denoise_cube_unet, verbatim
    def denoise_cube_unet(ho, net):
        with fits.open(ho["dirty"], memmap=False) as h:
            raw = np.ascontiguousarray(h[0].data).astype(np.float32)
        csub = raw - continuum_of(raw, CONTINUUM_N)[None]
        Cc, Hh, Ww = csub.shape
        lo = csub.reshape(Cc, -1).min(axis=1)
        hi = csub.reshape(Cc, -1).max(axis=1)
        rng_ = hi - lo
        nz = rng_ > 0
        norm = np.zeros_like(csub)
        norm[nz] = (csub[nz] - lo[nz, None, None]) / rng_[nz, None, None]
        out = np.empty_like(csub)
        with torch.no_grad():
            for s in range(0, Cc, BS):
                t = torch.from_numpy(norm[s:s + BS])[:, None].to(device)
                t256 = F.interpolate(t, (TARGET_SIZE, TARGET_SIZE), mode="bilinear",
                                     align_corners=False)
                tz = torch.zeros(t256.size(0), dtype=torch.long, device=device)
                pred = net(t256, tz)
                back = F.interpolate(pred, (Hh, Ww), mode="bilinear",
                                     align_corners=False)[:, 0].cpu().numpy()
                for k in range(back.shape[0]):
                    ch = s + k
                    out[ch] = (back[k] * rng_[ch] + lo[ch]) if rng_[ch] > 0 else \
                        np.full((Hh, Ww), lo[ch], np.float32)
        return out, csub

    ho = holdout[0]
    den, dcsub = denoise_cube_unet(ho, net)
    assert den.shape == (C, H, W), den.shape
    assert np.isfinite(den).all(), "denoised cube has non-finite values"
    print(f"[5] denoise_cube_unet OK: {den.shape}, finite, "
          f"round-trip {TARGET_SIZE}->{H} applied")

    # notebook cell: moment maps + improvement formula
    import bettermoments as bm

    def mdiff(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.nanmean(np.abs(a[m] - b[m])))

    with fits.open(ho["clean"], memmap=False) as h:
        craw = np.ascontiguousarray(h[0].data).astype(np.float32)
    ccsub = craw - continuum_of(craw, CONTINUUM_N)[None]
    _, velax = bm.load_cube(ho["dirty"])
    cl = generate_moment_maps(None, data_velax=(ccsub, velax))
    di = generate_moment_maps(None, data_velax=(dcsub, velax))
    no = generate_moment_maps(None, data_velax=(den, velax))
    row = {"cube": ho["folder"], "seed": bs_}
    for nm, c_, d_, n_ in zip(moments, cl, di, no):
        dd, nn = mdiff(c_, d_), mdiff(c_, n_)
        row["imp_" + nm] = round(100.0 * (1 - nn / dd), 2) if dd > 0 else float("nan")
    assert all(np.isfinite(row["imp_" + m]) for m in moments), row
    print(f"[6] moment maps + improvement formula OK: {row}")

    s = summarise_improvements([row])
    assert s["M0"]["n"] == 1
    print(f"[7] summarise_improvements OK: M0 {s['M0']['mean']:+.1f}%")

    # notebook cell 10: artifact diagnostics over the val split
    rows = []
    with torch.no_grad():
        for i in range(len(val_ds)):
            d, c = val_ds[i]
            pred = net(d[None].to(device), torch.zeros(1, dtype=torch.long, device=device))
            cln = c[0].numpy()
            if cln.max() <= 0:
                continue
            rows.append(channel_artifacts(cln, d[0].numpy(), pred[0, 0].cpu().numpy()))
    a = summarise(rows)
    assert a and "frac_channels_with_blob" in a, a
    print(f"[8] artifact diagnostics OK over {a['n_channels']} channels: "
          f"overshoot {a['overshoot_mean']:.3f}, blobs/ch {a['blobs_per_channel']:.3f}")

    print("\n" + "=" * 68)
    print("Notebook 08 critical path PASSED: safe to spend the GPU session")
    print("=" * 68)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
