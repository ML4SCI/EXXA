"""
Young et al. 2026 Phantom GI discs (Zenodo 22045440, CC-BY-4.0) -> line-emission training / test cubes.

Per simulation: the LAST dump in its archive (the first is the pre-instability initial condition), one random view (inclination 10-60 deg, PA, in-plane
rotation, spin direction), 13CO with optical depth tau0 log-uniform 3-50, rendered by src/data/sph_cube.py on a 301 px grid sized so the disc (R95 of the
gas) fills 80 % of the field, 0.1 km/s channels covering the line plus 1 km/s of line-free edge, then a Gaussian beam of 5-12 px FWHM (axis ratio <= 1.3).
clean = rendered sky (*) beam (the beam is in clean, as in a CLEAN product: the noise-only / A = I regime); dirty = clean + beam-shaped noise at peak SNR 20
(the trainer redraws noise every round, so this only fixes the noise spectrum's shape). Written in the line-emission layout:
<out>/run_young_<name>/run_young_<name>_{clean,dirty}.fits with CDELT3 in km/s, BMAJ / BMIN / CDELT1 in deg (as if at 140 pc), INCL_DEG / DIST_PC / AU_PX.

Known limits (sph-validation v1-v3): matches MCFOST's velocity field at large scale (high-pass M1 corr 0.89 at 100 au, amplitude parity) but not below
~50 au; Young's discs are young and massive (protostellar) - used as a mass-ratio sweep for the GI wiggle, never as the test target population.

  python experiments/make_sph_cubes.py --out DIR [--runs a,b,...] [--seed 0] [--workers 16] [--keep-tar]
"""
import argparse, glob, json, math, os, shutil, sys, tarfile, time
import numpy as np
from astropy.io import fits

ROOT = os.environ.get("EXXA_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
from src.data import sph_cube as sc

RECORD = "https://zenodo.org/records/22045440/files/"
# 22 runs spanning mu = Md/M* 0.2-1.4, M* 0.1-1, R 50-200 au, archives <= ~700 MB. TEST = held out whole (never trained on).
TRAIN = ["M0.1_R50_Mu0.6", "M0.1_R100_Mu1.1", "M0.1_R200_Mu1.4", "M0.2_R50_Mu0.6", "M0.2_R100_Mu0.9", "M0.2_R200_Mu1.2",
         "M0.3_R50_Mu0.4", "M0.3_R100_Mu0.8", "M0.3_R200_Mu0.7", "M0.5_R50_Mu0.3", "M0.5_R100_Mu0.5", "M0.5_R200_Mu0.8",
         "M0.7_R100_mu0.4", "M0.7_R200_mu0.6", "M1_R50_Mu0.2", "M1_R200_Mu0.5"]
TEST = ["M0.1_R100_Mu1.0", "M0.2_R200_Mu1.0", "M0.3_R100_Mu0.7", "M0.5_R200_Mu0.5", "M0.7_R50_Mu0.3", "M1_R100_Mu0.4"]
# 2026-10-18: 8 more WHOLE simulations held out, never in TRAIN, so the GI tests do not rest on 9074: spread in M* (0.1 to 1), R (50 to 200 au) and
# mu (0.3 to 1.4); M0.5_R100_Mu0.3 and M1_R100_Mu0.3 are low-mu controls (weak or no GI arms). Archives 470 to 660 MB (the 76 MB runs look truncated).
TEST2 = ["M0.1_R100_Mu1.4", "M0.2_R100_Mu0.7", "M0.3_R200_Mu1.0", "M0.3_R50_Mu0.5", "M0.5_R100_Mu0.3", "M0.5_R200_Mu0.6", "M0.7_R200_mu0.5", "M1_R100_Mu0.3"]
# 2026-10-21: the FRESH test set of the improvement track, locked before any challenger: 8 runs never rendered, spread in M*, R, mu, full-size archives
TEST3 = ["M0.1_R50_Mu0.9", "M0.3_R200_Mu1.1", "M0.3_R50_Mu0.3", "M0.7_R100_mu0.3", "M0.5_R200_Mu0.9", "M0.7_R50_Mu0.2", "M1_R100_Mu0.5", "M1_R200_Mu0.6"]
PC140_AU_PER_DEG = 140.0 * 3600.0           # au per degree at 140 pc (headers only; the trainer uses pixel ratios)


def last_dump(tar_path, workdir):
    with tarfile.open(tar_path) as t:
        dumps = sorted((m for m in t.getmembers() if m.isfile() and os.path.basename(m.name).split("_")[-1].isdigit()),
                       key=lambda m: int(os.path.basename(m.name).split("_")[-1]))
        m = dumps[-1]
        m.name = os.path.basename(m.name)                                    # flatten; never extract paths from the archive
        try:
            t.extract(m, workdir, filter="data")
        except TypeError:                                                    # Python < 3.12: no filter argument (the name is already flattened)
            t.extract(m, workdir)
    return os.path.join(workdir, m.name), len(dumps)


def view_cube(d, rng, n=301):
    incl, pa, phi0, flip = rng.uniform(10, 60), rng.uniform(0, 360), rng.uniform(0, 360), bool(rng.integers(2))
    R = np.hypot(d["x"], d["y"])
    r95 = float(np.percentile(R, 95))
    au = 2 * r95 / (0.8 * n)
    X, Y, depth, v = sc.project(d, incl, pa, phi0, flip)
    vmax = float(np.abs(v).max()) + 1.0          # every particle + 1 km/s of line-free edge: the trainer removes the edge-channel mean as continuum
                                                  # (99.5th percentile left 47 % of the peak in the edge channels: the fast inner disc)
    velax = np.arange(-vmax, vmax + 1e-6, 0.1)
    tau0 = float(math.exp(rng.uniform(math.log(3), math.log(50))))
    sky = sc.render(d, n, au, velax, incl, pa_deg=pa, phi0_deg=phi0, flip=flip, tau0=tau0, smooth=True)
    bmaj = rng.uniform(5, 12); bmin = bmaj / rng.uniform(1.0, 1.3); bpa = rng.uniform(0, 180)
    clean = sc.gaussian_beam(sky, bmaj, bmin, bpa)
    white = rng.standard_normal(clean.shape).astype(np.float32)
    noise = sc.gaussian_beam(white, bmaj, bmin, bpa)
    noise *= np.float32(float(clean.max()) / 20.0 / max(float(noise.std()), 1e-30))
    meta = dict(incl=incl, pa=pa, phi0=phi0, flip=flip, tau0=tau0, au_px=au, r95_au=r95, bmaj_px=bmaj, bmin_px=bmin, bpa=bpa,
                n_chan=len(velax), vmax=vmax)
    return clean, clean + noise, velax, meta


def write_pair(out, name, clean, dirty, velax, meta):
    folder = os.path.join(out, f"run_young_{name}")
    os.makedirs(folder, exist_ok=True)
    h = fits.Header()
    deg = meta["au_px"] / PC140_AU_PER_DEG
    h["CTYPE1"], h["CDELT1"], h["CTYPE2"], h["CDELT2"] = "RA---SIN", -deg, "DEC--SIN", deg
    h["CTYPE3"], h["CRVAL3"], h["CDELT3"], h["CRPIX3"], h["CUNIT3"] = "VRAD", float(velax[0]), 0.1, 1.0, "km/s"
    h["BMAJ"], h["BMIN"], h["BPA"] = meta["bmaj_px"] * deg, meta["bmin_px"] * deg, meta["bpa"]
    h["BUNIT"] = "K"
    h["DIST_PC"], h["INCL_DEG"], h["AU_PX"], h["TAU0"] = 140.0, meta["incl"], meta["au_px"], meta["tau0"]
    h["ORIGIN"] = "Young+2026 Phantom (Zenodo 22045440) via src/data/sph_cube.py"
    for kind, cube in (("clean", clean), ("dirty", dirty)):
        fits.writeto(os.path.join(folder, f"run_young_{name}_{kind}.fits"), cube.astype(np.float32), h, overwrite=True)
    json.dump(meta, open(os.path.join(folder, "view.json"), "w"), indent=1)
    return folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default="", help="comma list; default TRAIN + TEST")
    ap.add_argument("--local", default="", help="use local .tar.gz files from this dir instead of downloading")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--keep-tar", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = [r for r in a.runs.split(",") if r] or TRAIN + TEST
    rng = np.random.default_rng(a.seed)
    report = {"train": [r for r in runs if r in TRAIN], "test": [r for r in runs if r in TEST or r in TEST2], "cubes": {}}
    work = os.path.join(a.out, "_work")
    os.makedirs(work, exist_ok=True)
    for name in runs:
        t0 = time.time()
        tar = os.path.join(a.local, f"{name}.tar.gz") if a.local else os.path.join(work, f"{name}.tar.gz")
        if not a.local and not os.path.exists(tar):
            import speedie_prep as sp
            sp.download_parallel(RECORD + f"{name}.tar.gz?download=1", tar, workers=a.workers)
        try:
            dump, ndump = last_dump(tar, work)
            d = sc.load_phantom(dump)
            clean, dirty, velax, meta = view_cube(d, rng)
        except Exception as e:                       # one unreadable archive must not stop the other 21
            print(f"{name:20s} FAILED: {type(e).__name__}: {e}", flush=True)
            report.setdefault("failed", {})[name] = f"{type(e).__name__}: {e}"
            json.dump(report, open(os.path.join(a.out, "sph_cubes_report.json"), "w"), indent=1, default=float)
            continue
        meta.update(run=name, dump=os.path.basename(dump), n_dumps=ndump, n_gas=int(len(d["m"])), mstar=d["mstar"], mdisc=float(d["m"].sum()),
                    temperature_source=d.get("T_source", "dump"),
                    mu=float(d["m"].sum() / d["mstar"]), split="test3" if name in TEST3 else ("test" if (name in TEST or name in TEST2) else "train"), minutes=(time.time() - t0) / 60)
        write_pair(a.out, name, clean, dirty, velax, meta)
        report["cubes"][name] = meta
        json.dump(report, open(os.path.join(a.out, "sph_cubes_report.json"), "w"), indent=1, default=float)
        print(f"{name:20s} {meta['split']:5s} mu {meta['mu']:.2f}  i {meta['incl']:4.1f}  {clean.shape}  {meta['au_px']:.2f} au/px  "
              f"beam {meta['bmaj_px']:.1f}x{meta['bmin_px']:.1f} px  tau0 {meta['tau0']:.0f}  [{meta['minutes']:.1f} min]", flush=True)
        os.remove(dump)
        if not a.keep_tar and not a.local:
            os.remove(tar)
    shutil.rmtree(work, ignore_errors=True)
    print("SPH CUBES DONE", flush=True)


if __name__ == "__main__":
    main()
