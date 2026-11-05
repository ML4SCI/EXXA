"""
DSHARP 12CO fiducial image(s) from the NRAO bulk server (https://bulk.cv.nrao.edu/almadata/lp/DSHARP/images/<Target>_CO.fits), through the unified denoiser. SPECTRAL TEST: the models were trained on 0.1 km/s channels, DSHARP has 0.35 km/s; each cube is run twice, native channels and interpolated to 0.1 km/s (--spec-interp 0.1), each with S5 injection (Speedie disc binned to 0.35 km/s into the cube's own noise). On Kaggle: fetch exoALMA fiducial cubes (Harvard Dataverse doi:10.7910/DVN/CFHWNH, `.image.fits` only) and run the unified denoiser on them
with experiments/alma_unified.py (INFERENCE ONLY: exoALMA is never trained on). One report, denoised cube and moment maps per cube in the Output.

"""
import os, sys, glob, json, shutil, subprocess, hashlib, time, urllib.request
os.environ.setdefault("EXXA_T0", str(time.time()))

REPO, BRANCH = "/tmp/EXXA", "midterm-prep"
PKG = f"{REPO}/DENOISING_DIFFUSION"
OUT = "/kaggle/working/dsharp"
CKPT = "unified_r4n_k15_long.pth,unified_r4n_k15_long_s11.pth,unified_r4n_k15_long_s23.pth,unified_r8_m1_r6.pth"   # recipe B (results/final/RECIPES.md)
# disk, line, inclination (deg), distance (pc): exoALMA I (Teague et al. 2025) values, for the S6 rotation-curve residual only
# disc, NRAO file stem, inclination (deg, Huang et al. 2018 Table 2, continuum ellipse fits), distance (pc, Gaia DR2 as used by DSHARP: to verify against
# Andrews et al. 2018b Table 1), centre crop in px (the 12CO disc plus sky around it: the Oct 17 MWC 758 tests showed tight fields lose flux).
# PA (deg) from the same table (continuum, 0-180: compare with the M1 fit modulo 180)
CUBES = [("HD_143006", "HD143006", 18.6, 165.0, 512, 169.0), ("AS_209", "AS209", 34.97, 121.0, 768, 85.76), ("HD_142666", "HD142666", 62.22, 148.0, 512, 162.11),
         ("HD_163296", "HD163296", 46.7, 101.0, 1024, 133.33), ("IM_Lup", "IMLup", 47.5, 158.0, 1280, 144.5)]
PART, NPART = 0, 2                                   # the launcher sets these to split the discs over several kernels
CUBES = CUBES[PART::NPART]
ZOOM = ["--target-beam-px", "10", "--max-scaled-px", "1024"]   # beam to about 10 px where the grid allows (capped for GPU memory)
INJECT_FOR = {"HD_143006", "AS_209"}   # S5 on two discs (runtime); the spectral test already did HD 142666
DATAVERSE, DOI = "https://dataverse.harvard.edu", "doi:10.7910/DVN/CFHWNH"
BUDGET_H = 11.0

def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)

if not os.path.exists(REPO):
    sh(["git", "clone", "--branch", BRANCH, "--depth", "1", "https://github.com/KrishanYadav333/EXXA.git", REPO], check=True)
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pytorch-msssim", "bettermoments"], check=True)
code = os.path.dirname(glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True)[0])
man = json.load(open(f"{code}/MANIFEST.json"))
for rel, info in man["files"].items():
    src, dst = f"{code}/{info['flat']}", f"{PKG}/{rel}"
    assert hashlib.sha256(open(src, "rb").read()).hexdigest() == info["sha256"], f"checksum mismatch {rel}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
assert "synth_noise" in open(f"{PKG}/experiments/alma_unified.py").read() and "spec-interp" in open(f"{PKG}/experiments/alma_unified.py").read() and "target-beam-px" in open(f"{PKG}/experiments/alma_unified.py").read(), "stale code overlay: alma_unified.py lacks --spec-interp / synth_noise (the exxa-wiggle-code dataset version attached is old)"
ckpts = [(glob.glob(f"/kaggle/input/**/{c}", recursive=True) or [None])[0] for c in CKPT.split(",")]
spd = glob.glob("/kaggle/input/**/GI_r0.5_clean.fits", recursive=True)      # S5 truth (Speedie, test-only), from the speedie-prep-fast Output
assert None not in ckpts, f"checkpoint not attached: {CKPT}"
os.chdir(PKG)
env = dict(os.environ, PYTHONPATH=PKG, EXXA_ROOT=PKG, PYTHONWARNINGS="ignore")
files = None
NRAO = "https://bulk.cv.nrao.edu/almadata/lp/DSHARP/images"
MODES = [("interp01", ["--spec-interp", "0.1", "--inject-snr", "6,8,12,24"])]   # native 0.35 km/s channels FAILED the S5 test; only the interpolated mode runs
for disk, stem, incl, dist, crop, pa in CUBES:
    name = f"{stem}_CO.fits"
    dst = f"/kaggle/temp/{name}"; os.makedirs("/kaggle/temp", exist_ok=True)
    t0 = time.time()
    sh(["curl", "-sL", "--retry", "5", "--retry-delay", "10", "-C", "-", "-o", dst, f"{NRAO}/{name}"], check=True)
    print(f"{name}: {os.path.getsize(dst) / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)
    for mode, extra in MODES:
        out = f"{OUT}/{disk}_{mode}"
        prev = glob.glob(f"/kaggle/input/**/dsharp/{disk}_{mode}/report.json", recursive=True)
        if prev:                                     # done in a previous session of this kernel
            shutil.copytree(os.path.dirname(prev[0]), out, dirs_exist_ok=True); print(f"{disk} {mode}: done in a previous session, kept", flush=True); continue
        if time.time() > float(os.environ["EXXA_T0"]) + BUDGET_H * 3600 - 1800:
            print(f"DEADLINE: {disk} {mode} not started; RESUME_NEEDED", flush=True); break
        r = sh([sys.executable, "-u", "experiments/alma_unified.py", "--cube", dst, "--ckpt", ",".join(ckpts), "--out", out,
                "--incl", str(incl), "--dist", str(dist), "--pa", str(pa), "--crop", str(crop)] + ZOOM + extra + (["--inject-dir", os.path.dirname(spd[0])] if spd and disk in INJECT_FOR else []), env=env)
        print(f"{disk} {mode}: exit {r.returncode}", flush=True)
    os.remove(dst)
print("DSHARP KERNEL DONE", flush=True)
