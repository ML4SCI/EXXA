"""
Build clean/dirty cube pairs from the analytic GI disk (src/data/analytic_gi.py), laid out like sg_synth
so wiggle_scorecard.py reads them with `--only an01,...`.

  clean = emission (*) the hydro cubes' own beam (16.1 x 12.9 px, as in the sg_synth headers)
  dirty = beam_recovered (*) clean + beam_recovered (*) white noise     (same recipe as synthesize_sg_pairs.py:
          unit-sum recovered beam, post-beam noise sigma = NOISE_FRAC * in-signal RMS of a bright channel)
Also saves `<name>_vlos_truth.npy`: the exact unbeamed line-of-sight velocity on the sky grid.

Run: PYTHONPATH=. python3 experiments/make_analytic_gi_cubes.py            (the 8 cubes at 301 px, 1.99 au/px)
     PYTHONPATH=. python3 experiments/make_analytic_gi_cubes.py --native600   (randomised disks at 600 px, SG v2 geometry)

--native600 (notebook 14): 600 x 600 at 1.33 au/px (140 pc, 0.0095"/px, the SG v2 grid, where the recovered beam is defined), beam
16.8 x 10.5 px, 161 channels at 0.1 km/s. Disks are drawn from fixed seeds so every session regenerates the same ones:
a601-a612 train, a613 validation, a614-a615 held out and never trained on. Written to --out (default /kaggle/temp or /tmp), NOT
/kaggle/working, so they do not bloat the notebook Output; existing cubes are skipped, so a restarted session does not redo them.
"""
import os, sys, json
import numpy as np
from astropy.io import fits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data import analytic_gi as ag

OUT = os.path.join(ROOT, "self-gravitating cube and dirty cube/sg_analytic")
BEAM = os.path.join(ROOT, "results/self-gravitating/dirty_beam_recovered_v2.fits")
NOISE_FRAC = 0.35
TARGET_REL = 0.5
N, AU, NCH, DV, DIST = 301, 1.99, 161, 0.1, 175.178     # +-8 km/s: line-free edge channels, like the hydro cubes (172 ch)
# name: (md, m, beta, pitch_deg, incl_deg)  ms = 1, p = -1, rin = 10 au, rout = 290 au
DISKS = {"an01": (0.15, 2, 5.0, 13, 20), "an02": (0.35, 2, 5.0, 13, 30), "an03": (0.60, 2, 5.0, 13, 20),
         "an04": (0.35, 3, 5.0, 18, 20), "an05": (0.35, 2, 3.0, 13, 30), "an06": (0.60, 2, 3.0, 20, 20),
         # held out of every training set: different mass, arm number, cooling, pitch, inclination
         "an07": (0.45, 2, 4.0, 15, 25), "an08": (0.25, 3, 3.0, 18, 30),
         # FRESH test set of the improvement track: never trained or selected on; m = 4 and
         # i = 35-40 deg lie outside the training ranges on purpose. The "k" entries are no-wiggle twins (same disc, no spiral): invented ripple.
         "an09": (0.30, 2, 4.0, 16, 22), "an10": (0.50, 3, 5.0, 14, 28), "an11": (0.20, 2, 3.5, 11, 35), "an12": (0.55, 2, 6.0, 17, 40),
         "an13": (0.40, 3, 3.5, 21, 18), "an14": (0.28, 2, 4.5, 19, 32), "an15": (0.48, 4, 4.0, 15, 26), "an16": (0.65, 2, 3.0, 12, 24),
         "an09k": (0.30, 2, 4.0, 16, 22), "an12k": (0.55, 2, 6.0, 17, 40), "an15k": (0.48, 4, 4.0, 15, 26)}


def apply_beam(cube, beam):
    n = cube.shape[-1]
    pad = np.zeros((n, n)); h = beam.shape[0] // 2; c = n // 2
    pad[c - h:c + h + 1, c - h:c + h + 1] = beam
    k = np.fft.fft2(np.fft.ifftshift(pad))
    return np.real(np.fft.ifft2(np.fft.fft2(cube, axes=(-2, -1)) * k, axes=(-2, -1)))


def extra_disks(n, seed=700):
    """Candidate C: `n` extra TRAINING discs anx01.. with fresh random parameters, wider than an01-an06 (m 2 to 4, incl 12 to 45 deg)
    and every other one with a compact bright centre (rin 2 to 4 au instead of 10). Seed 700 is used nowhere else; none equals a test disc."""
    g = np.random.default_rng(seed)
    out = {}
    for k in range(1, n + 1):
        out[f"anx{k:02d}"] = (round(float(g.uniform(0.12, 0.65)), 3), int(g.choice([2, 2, 3, 3, 4])), round(float(g.uniform(2.5, 8.0)), 2),
                              round(float(g.uniform(10, 22)), 1), round(float(g.uniform(12, 45)), 1), round(float(g.uniform(2, 4)), 1) if k % 2 == 0 else 10.0)
    return out


def random_disks(n_train=12, n_val=1, n_hold=2, seed=600):
    """Disks across the range the hydro set covers: mass ratio, arm number, cooling, pitch angle, inclination."""
    g = np.random.default_rng(seed)
    out, k = {}, 601
    for group, count in (("train", n_train), ("val", n_val), ("hold", n_hold)):
        for _ in range(count):
            out[f"a{k}"] = (round(float(g.uniform(0.12, 0.65)), 3), int(g.choice([2, 2, 3])), round(float(g.uniform(2.5, 8.0)), 2),
                            round(float(g.uniform(10, 22)), 1), round(float(g.uniform(15, 35)), 1))
            k += 1
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--native600", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--extra", type=int, default=0, help="candidate C: also render this many extra training discs anx01..")
    args = ap.parse_args()
    if args.extra:
        _x = extra_disks(args.extra)
        assert not {v[:5] for v in _x.values()} & set(DISKS.values()), "an extra training disc equals a listed disc"
        DISKS = {**DISKS, **_x}
    if args.native600:
        N, AU, NCH, DV, DIST = 600, 1.33, 161, 0.1, 140.0
        BEAM_PX = (16.8, 10.5, 175.7)
        DISKS = random_disks()
        OUT = args.out or os.path.join("/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp", "sg_analytic600")
    os.makedirs(OUT, exist_ok=True)
    beam = fits.getdata(BEAM).astype(np.float64)
    beam /= beam.sum()
    rows = {}
    for k, (name, (md, m, beta, pitch, incl, *rin)) in enumerate(DISKS.items()):
        if os.path.exists(os.path.join(OUT, name, f"{name}_dirty.fits")):
            print(f"{name}: exists, skipped", flush=True)
            continue
        disk = ag.AnalyticGI(ms=1.0, md=md, p=-1.0, m=m, beta=beta, pitch_deg=pitch, incl_deg=float(incl), rin=rin[0] if rin else 10.0)
        clean, vt = ag.render_cube(disk, n=N, au_per_px=AU, n_chan=NCH, dv=DV, perturb=not name.endswith("k"),
                                   **({"beam_px": BEAM_PX} if args.native600 else {}))
        rng = np.random.default_rng(1000 + k)
        probe = clean[NCH // 2].astype(np.float64)
        sig_rms = float(np.sqrt(np.mean(probe[np.abs(probe) > 0.05 * np.abs(probe).max()] ** 2)))
        sigma = NOISE_FRAC * sig_rms
        gain = float(np.std(apply_beam(rng.normal(0, 1, (4, N, N)), beam)))
        # Compact emission makes rmsdiff/rms (the number synthesize_sg_pairs.py reports, 0.41-0.57 on the
        # hydro disks) come out 0.8-0.95 at NOISE_FRAC 0.35, so scale sigma per disk to hit TARGET_REL.
        c0 = clean[NCH // 2].astype(np.float64)
        rms_c = np.sqrt(np.mean(c0 ** 2))
        blur_rel = float(np.sqrt(np.mean((apply_beam(c0[None], beam)[0] - c0) ** 2)) / rms_c)
        noise_rel = float(np.sqrt(np.mean(apply_beam(rng.normal(0, sigma / gain, (1, N, N)), beam) ** 2)) / rms_c)
        scale = np.sqrt(max(TARGET_REL ** 2 - blur_rel ** 2, 1e-4)) / noise_rel
        sigma *= scale
        dirty = np.empty_like(clean)
        for s in range(0, NCH, 10):
            blk = clean[s:s + 10].astype(np.float64)
            dirty[s:s + 10] = apply_beam(blk, beam) + apply_beam(rng.normal(0, sigma / gain, blk.shape), beam)
        rel = float(np.sqrt(np.mean((dirty[NCH // 2] - clean[NCH // 2]) ** 2)) / rms_c)
        hdr = fits.Header()
        for key, val in dict(CDELT1=-AU / DIST / 3600.0, CDELT2=AU / DIST / 3600.0, CRPIX1=N // 2 + 1, CRPIX2=N // 2 + 1,
                             CRVAL3=0.0, CRPIX3=NCH // 2 + 1, CDELT3=DV, BUNIT="JY/BEAM", DIST_PC=DIST,
                             BMAJ=(BEAM_PX[0] if args.native600 else 16.1) * AU / DIST / 3600.0, BMIN=(BEAM_PX[1] if args.native600 else 12.9) * AU / DIST / 3600.0, BPA=175.7,
                             INCL_DEG=float(incl), AN_MD=md, AN_M=m, AN_BETA=beta, AN_PITCH=pitch,
                             SYNTHSIG=sigma, SYNTHREL=rel).items():
            hdr[key] = val
        d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
        fits.PrimaryHDU(clean, hdr).writeto(f"{d}/{name}_clean.fits", overwrite=True)
        fits.PrimaryHDU(dirty, hdr).writeto(f"{d}/{name}_dirty.fits", overwrite=True)
        np.save(f"{d}/{name}_vlos_truth.npy", vt.astype(np.float32))
        rows[name] = dict(md=md, m=m, beta=beta, pitch=pitch, incl=incl, noise_sigma=sigma, rmsdiff_over_rms=rel, blur_rel=blur_rel, noise_scale=float(scale))
        print(f"{name} md={md} m={m} beta={beta} pitch={pitch} i={incl}: rmsdiff/rms = {rel:.3f} (sg_synth 0.41-0.57)", flush=True)
    json.dump(rows, open(os.path.join(OUT, "disks.json"), "w"), indent=1)
