"""
Splits the 0.186 "loss/learning" term from the 2026-09-11 resolution-vs-loss test into its
two unseparated parts: DOMAIN (winner_aug_seed43 was trained on line emission and is being
run on a self-gravitating cube it has never seen) versus OBJECTIVE (MSE regresses to the
posterior mean and attenuates exactly the high-frequency velocity perturbation the wiggle
measures).

Same cube, same mask, same shared Keplerian model (fit on clean, reused for every method),
frac=0.05, channels 240-360 step 1 so the numbers drop
straight into that table. The only thing that changes between the two model rows is which
checkpoint runs.

  winner_aug_seed43  line-emission trained, single channel in / single channel out
  sg_k3_fresh        SG-trained (notebook 12), 7-channel neighbour stack in / 1 out

Each checkpoint is run through ITS OWN training preprocessing, read off the training code
rather than assumed:
  winner_aug   per-channel min-max, no continuum subtraction, resize to 256
  sg_k3_fresh  7 clamped neighbours, min-max shared from the CENTRE dirty channel across the
               whole stack, subtract_continuum=False (notebook 12's own setting), resize 256

Reading it: if sg_k3_fresh lands well above winner_aug's 0.760, the domain gap is a large
part of the 0.186 and the fix is to train on the target domain. If it lands near 0.760, the
domain is not the problem and the objective is, which points at spectral context / a
non-MSE objective instead.

Run: PYTHONPATH=.. python3 experiments/wiggle_domain_split.py
"""
import os, sys, math, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as Fn
from astropy.io import fits

from src.models.unet import UNet
from src.training.architectures import build_model
from src.data.fits_cube_dataset import continuum_of
from src.evaluation.moment_maps import generate_moment_maps, signal_mask
from src.evaluation.gi_wiggle import quadratic_moment1, compare_wiggles

SG = "self-gravitating cube and dirty cube/kinematic_data_v2"
UNET_CKPT = "models/08-seeds/winner_aug_seed43.pth"
K3_CKPT = "models/12-spectral/sg_k3_fresh.pth"
KIN_CKPT = "models/08-kinematic/kin_gamma0.pth"
SIZE, FRAC, MSTAR_BOUND, CONTINUUM_N = 256, 0.05, 50.0, 5
CH0, CH1 = 240, 361
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load_winner_aug():
    ck = torch.load(UNET_CKPT, map_location=dev, weights_only=False)
    net = UNet(in_channels=ck.get("in_channels", 1), out_channels=1,
               base_channels=ck["base_channels"], channel_multipliers=ck["channel_multipliers"],
               time_emb_dim=128, num_res_blocks=2, groups=math.gcd(8, ck["base_channels"]),
               beam_dim=ck.get("beam_dim", 0)).to(dev)
    net.load_state_dict(ck["model_state_dict"]); net.eval()
    return net


def load_k3():
    ck = torch.load(K3_CKPT, map_location=dev, weights_only=False)
    net = build_model("unet", base_channels=ck["base_channels"],
                      channel_multipliers=tuple(ck["channel_multipliers"]), use_beam=False,
                      n_neighbors=ck["n_neighbors"], out_channels=ck["out_channels"],
                      latent_dim=ck.get("latent_dim", 128)).to(dev)
    miss, unexp = net.load_state_dict(ck["model_state_dict"], strict=True)
    net.eval()
    return net, ck["n_neighbors"]


def load_kin_gamma0():
    """
    Notebook 08's 31-channel stack, kinematic_gamma=0 (the architecture control, no loss term).
    The only checkpoint in the project with a large measured win where there was headroom to
    win: 0.8155 mean resid_r against its line-emission holdouts' dirty at 0.4284. Never run on
    SG data. This asks whether the spectral-context mechanism transfers.
    """
    ck = torch.load(KIN_CKPT, map_location=dev, weights_only=False)
    net = build_model("unet", base_channels=ck["base_channels"],
                      channel_multipliers=tuple(ck["channel_multipliers"]), use_beam=False,
                      n_neighbors=ck["n_neighbors"], out_channels=ck["out_channels"],
                      latent_dim=ck.get("latent_dim", 128)).to(dev)
    net.load_state_dict(ck["model_state_dict"], strict=True)
    net.eval()
    return net, ck["n_neighbors"]


def denoise_stack31(net, csub_full, centres, k, continuum, batch=8):
    """
    Notebook 08's path: subtract_continuum=True, 2k+1 clamped neighbours, min-max shared from
    the CENTRE dirty channel, stack_target=True so the output is 2k+1 channels of which only
    the CENTRE (index k) is the prediction for this position. The continuum is added back at
    the end so the result lives in the same raw space as clean/dirty and drops straight into
    the standing comparison table.
    """
    nchan, H, W = csub_full.shape
    out = np.empty((len(centres), H, W), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(centres), batch):
            cen = np.asarray(centres[s:s + batch])
            nb = np.clip(cen[:, None] + np.arange(-k, k + 1)[None, :], 0, nchan - 1)
            stack = csub_full[nb]
            ref = csub_full[cen]
            lo = ref.reshape(len(cen), -1).min(axis=1)
            hi = ref.reshape(len(cen), -1).max(axis=1)
            rng = np.where((hi - lo) > 0, hi - lo, 1.0)
            n = (stack - lo[:, None, None, None]) / rng[:, None, None, None]
            t = torch.from_numpy(n).float().to(dev)
            t = Fn.interpolate(t, (SIZE, SIZE), mode="bilinear", align_corners=False)
            p = net(t, torch.zeros(t.size(0), dtype=torch.long, device=dev), None)
            p = Fn.interpolate(p, (H, W), mode="bilinear", align_corners=False).cpu().numpy()
            centre_out = p[:, k]                       # centre of the predicted stack
            for j in range(len(cen)):
                out[s + j] = centre_out[j] * rng[j] + lo[j] + continuum
    return out


def denoise_single(net, dirty, batch=8):
    """winner_aug's path: one channel in, per-channel min-max, resize to SIZE and back."""
    C, H, W = dirty.shape
    los = dirty.reshape(C, -1).min(axis=1); his = dirty.reshape(C, -1).max(axis=1)
    rng = his - los
    out = np.empty_like(dirty)
    with torch.no_grad():
        for s in range(0, C, batch):
            blk = dirty[s:s + batch]
            lo, hi = los[s:s + batch], his[s:s + batch]
            n = np.where((hi - lo)[:, None, None] > 0,
                         (blk - lo[:, None, None]) / np.where((hi - lo) > 0, hi - lo, 1)[:, None, None], 0)
            t = torch.from_numpy(n)[:, None].float().to(dev)
            t = Fn.interpolate(t, (SIZE, SIZE), mode="bilinear", align_corners=False)
            p = net(t, torch.zeros(t.size(0), dtype=torch.long, device=dev), None)
            b = Fn.interpolate(p, (H, W), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            for k in range(b.shape[0]):
                out[s + k] = b[k] * rng[s + k] + los[s + k] if rng[s + k] > 0 else np.full((H, W), los[s + k])
    return out


def denoise_stack_k(net, full_dirty, centres, k, batch=8):
    """
    notebook 12's path: 2k+1 clamped neighbours from the FULL cube, min-max shared from the
    CENTRE dirty channel (never per neighbour -- that would erase relative amplitude along
    velocity, which is the whole point of feeding neighbours), resize to SIZE, 1 channel out.
    """
    nchan, H, W = full_dirty.shape
    out = np.empty((len(centres), H, W), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(centres), batch):
            cen = np.asarray(centres[s:s + batch])
            nb = np.clip(cen[:, None] + np.arange(-k, k + 1)[None, :], 0, nchan - 1)
            stack = full_dirty[nb]                                    # (b, 2k+1, H, W)
            ref = full_dirty[cen]                                     # centre channel only
            lo = ref.reshape(len(cen), -1).min(axis=1)
            hi = ref.reshape(len(cen), -1).max(axis=1)
            rng = np.where((hi - lo) > 0, hi - lo, 1.0)
            n = (stack - lo[:, None, None, None]) / rng[:, None, None, None]
            t = torch.from_numpy(n).float().to(dev)
            t = Fn.interpolate(t, (SIZE, SIZE), mode="bilinear", align_corners=False)
            p = net(t, torch.zeros(t.size(0), dtype=torch.long, device=dev), None)
            b = Fn.interpolate(p, (H, W), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            for j in range(len(cen)):
                out[s + j] = b[j] * rng[j] + lo[j]
    return out


def main():
    with fits.open(f"{SG}/clean_sg.fits", memmap=True) as h:
        hdr, cdata = h[0].header, h[0].data[:]
    with fits.open(f"{SG}/dirty_sg.fits", memmap=True) as h:
        ddata = h[0].data[:]
    AU = abs(hdr["CDELT1"]) * 3600.0 * hdr.get("DIST_PC", 140.0)

    CH = list(range(CH0, CH1))
    velax = (hdr["CRVAL3"] + (np.array(CH) + 1 - hdr["CRPIX3"]) * hdr["CDELT3"]) * 1000.0
    clean = np.stack([np.asarray(cdata[c], np.float32) for c in CH]).astype(np.float64)
    dirty = np.stack([np.asarray(ddata[c], np.float32) for c in CH]).astype(np.float64)
    full_dirty = np.asarray(ddata, np.float64)

    print(f"device {dev} | {len(CH)} channels | full cube {full_dirty.shape}")

    t0 = time.time()
    wa = denoise_single(load_winner_aug(), dirty)
    print(f"  winner_aug_seed43 (line-emission trained): {(time.time()-t0)/60:.1f} min")

    t0 = time.time()
    k3net, k = load_k3()
    k3 = denoise_stack_k(k3net, full_dirty, CH, k)
    print(f"  sg_k3_fresh (SG trained, k={k}):            {(time.time()-t0)/60:.1f} min")

    t0 = time.time()
    kinnet, kk = load_kin_gamma0()
    cont = continuum_of(full_dirty, CONTINUUM_N)
    kin = denoise_stack31(kinnet, full_dirty - cont[None], CH, kk, cont)
    print(f"  kin_gamma0 (line-em, k={kk} stack):         {(time.time()-t0)/60:.1f} min")

    cubes = {"clean": clean, "dirty": dirty,
             "winner_aug (line-em)": wa, "sg_k3_fresh (SG)": k3,
             "kin_gamma0 (k=15 stack)": kin}

    m0, _, _ = generate_moment_maps("", data_velax=(clean, velax))
    mask = signal_mask(m0, frac=FRAC)
    rows = {t: quadratic_moment1(c, velax)[0] / 1000.0 for t, c in cubes.items()}

    cmp = compare_wiggles(rows, mask, AU, reference="clean")
    g = cmp["clean"]["geom"]
    flag = " DEGEN" if g["mstar_msun"] > 0.9 * MSTAR_BOUND else ""
    print(f"\n  shared model (fit on clean): mstar={g['mstar_msun']:.3f} incl={g['incl_deg']:.1f}"
          f" pa={g['pa_deg']:.1f} vsys={g['vsys']:.3f}{flag}")

    print(f"\n  {'method':24s} {'residRMS':>9s} {'raw r':>8s} {'resid r':>9s}")
    for tag in cubes:
        rms = cmp[tag]["rms_kms"]
        if tag == "clean":
            print(f"  {tag:24s} {rms:9.3f} {'--':>8s} {'--':>9s}")
            continue
        ok = np.isfinite(rows["clean"][mask]) & np.isfinite(rows[tag][mask])
        raw = float(np.corrcoef(rows["clean"][mask][ok], rows[tag][mask][ok])[0, 1])
        print(f"  {tag:24s} {rms:9.3f} {raw:8.4f} {cmp[tag]['corr']:9.4f}")

    fig, ax = plt.subplots(2, len(cubes), figsize=(4.8*len(cubes), 9))
    vm = np.nanpercentile(np.abs(rows["clean"][mask]), 98)
    vr = np.nanpercentile(np.abs(cmp["clean"]["residual"][mask]), 98)
    for i, tag in enumerate(cubes):
        im = ax[0, i].imshow(np.where(mask, rows[tag], np.nan), cmap="RdBu_r",
                             vmin=-vm, vmax=vm, origin="lower")
        ax[0, i].set_title(f"{tag}: M1"); plt.colorbar(im, ax=ax[0, i], fraction=0.046)
        im2 = ax[1, i].imshow(np.where(mask, cmp[tag]["residual"], np.nan), cmap="RdBu_r",
                              vmin=-vr, vmax=vr, origin="lower")
        ax[1, i].set_title(f"{tag}: residual (RMS {cmp[tag]['rms_kms']:.2f})")
        plt.colorbar(im2, ax=ax[1, i], fraction=0.046)
    plt.suptitle("Which trained checkpoint helps on the SG v2 cube? (frac=0.05, shared geometry)",
                 fontsize=13)
    plt.tight_layout()
    out = "results/self-gravitating/wiggle_domain_split.png"
    plt.savefig(out, dpi=120)
    print("\n  saved ->", out)


if __name__ == "__main__":
    main()
