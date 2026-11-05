"""
On Kaggle: fetch exoALMA fiducial cubes (Harvard Dataverse doi:10.7910/DVN/CFHWNH, `.image.fits` only) and run the unified denoiser on them
with experiments/alma_unified.py (INFERENCE ONLY: exoALMA is never trained on). One report, denoised cube and moment maps per cube in the Output.

"""
import os, sys, glob, json, shutil, subprocess, hashlib, time, urllib.request
os.environ.setdefault("EXXA_T0", str(time.time()))

REPO, BRANCH = "/tmp/EXXA", "midterm-prep"
PKG = f"{REPO}/DENOISING_DIFFUSION"
OUT = "/kaggle/working/alma"
CKPT = "unified_r4n_k15_long.pth,unified_r4n_k15_long_s11.pth,unified_r4n_k15_long_s23.pth,unified_r8_m1_r6.pth"   # recipe B (results/final/RECIPES.md)
# disk, line, inclination (deg), distance (pc): exoALMA I (Teague et al. 2025) values, for the S6 rotation-curve residual only
# exoALMA I (Teague et al. 2025, arXiv:2504.18688): distance d from Table 1, |inclination| from Table 2 (discminer fits of 12CO); crop 512 px = 12.8 arcsec
# disk, |i| (deg), PA (deg, discminer, Table 2), distance (pc, Table 1)
CUBES = [(disk, "13CO", inc, dist, pa) for disk, inc, pa, dist in [
    ("MWC_758", 19.4, 240.3, 156), ("DM_Tau", 38.7, 335.7, 144), ("AA_Tau", 58.7, 272.7, 135), ("LkCa_15", 50.3, 62.1, 157), ("HD_34282", 58.3, 117.4, 309),
    ("CQ_Tau", 36.3, 235.1, 149), ("SY_Cha", 52.4, 346.2, 182), ("PDS_66", 31.9, 189.0, 98), ("HD_135344B", 16.1, 242.9, 135), ("HD_143006", 16.9, 168.9, 167),
    ("J1604", 6.0, 258.1, 145), ("J1615", 46.5, 325.4, 156), ("V4046_Sgr", 34.1, 255.7, 72), ("J1842", 39.4, 205.9, 151), ("J1852", 32.7, 117.1, 147)]]
PART, NPART = 0, 4                                   # the launcher sets these to split the discs over several kernels (CUBES[PART::NPART])
CUBES = CUBES[PART::NPART]
FIELD = ["--crop", "512"]                            # field of view
ZOOM = ["--target-beam-px", "10"]                    # review of 2026-10-16, "zoom in": beam to about 10 px, sky field and beam angle unchanged
INJECT_FOR = {"MWC_758", "HD_135344B", "DM_Tau", "LkCa_15"}   # S5 injection-recovery on three discs spanning inclination 16 / 39 / 50 deg; MWC 758 done in alma-exoalma-s5
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
assert "target-beam-px" in open(f"{PKG}/experiments/alma_unified.py").read(), "stale code overlay: alma_unified.py lacks --target-beam-px"
ckpts = [(glob.glob(f"/kaggle/input/**/{c}", recursive=True) or [None])[0] for c in CKPT.split(",")]
spd = glob.glob("/kaggle/input/**/GI_r0.5_clean.fits", recursive=True)      # S5 truth (Speedie, test-only), from the speedie-prep-fast Output
assert None not in ckpts, f"checkpoint not attached: {CKPT}"
os.chdir(PKG)
env = dict(os.environ, PYTHONPATH=PKG, EXXA_ROOT=PKG, PYTHONWARNINGS="ignore")
files = None
for disk, line, incl, dist, pa in CUBES:
    name = f"{disk}_{line}_fiducial.image.fits"
    out = f"{OUT}/{disk}_{line}"
    prev = glob.glob(f"/kaggle/input/**/alma/{disk}_{line}/report.json", recursive=True)
    if prev:                                         # done in a previous session of this kernel
        shutil.copytree(os.path.dirname(prev[0]), out, dirs_exist_ok=True); print(f"{name}: done in a previous session, kept", flush=True); continue
    if time.time() > float(os.environ["EXXA_T0"]) + BUDGET_H * 3600 - 1800:
        print(f"DEADLINE: {name} not started; RESUME_NEEDED", flush=True); break
    if files is None:
        req = urllib.request.Request(f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={DOI}",
                                     headers={"User-Agent": "Mozilla/5.0 (EXXA GSoC research; inference on public exoALMA data)"})
        with urllib.request.urlopen(req, timeout=60) as r:          # Dataverse answers 403 to the default Python-urllib agent (2026-10-13)
            files = {f["dataFile"]["filename"]: f["dataFile"] for f in json.load(r)["data"]["latestVersion"]["files"]}
    assert name in files, f"{name} not in the release; e.g. {sorted(files)[:5]}"
    dst = f"/kaggle/temp/{name}"; os.makedirs("/kaggle/temp", exist_ok=True)
    t0 = time.time()
    sh(["curl", "-sL", "--retry", "5", "--retry-delay", "10", "-C", "-", "-o", dst, f"{DATAVERSE}/api/access/datafile/{files[name]['id']}"], check=True)
    assert os.path.getsize(dst) == files[name]["filesize"], f"{name}: truncated download"
    print(f"{name}: {os.path.getsize(dst) / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)
    r = sh([sys.executable, "-u", "experiments/alma_unified.py", "--cube", dst, "--ckpt", ",".join(ckpts), "--out", out,
            "--incl", str(incl), "--dist", str(dist), "--pa", str(pa)] + FIELD + ZOOM + (["--inject-dir", os.path.dirname(spd[0])] if spd and disk in INJECT_FOR else []), env=env)
    print(f"{name}: exit {r.returncode}", flush=True)
    os.remove(dst)
print("ALMA KERNEL DONE", flush=True)
