"""
Kaggle script kernel: one denoiser trained on line emission + hydro SG + analytic GI disks, scored on PSNR, M0/M1/M2 and the
GI wiggle at 256/300/320/480/600 px. See experiments/train_unified.py for the design for why.

ARM selects the model: "wiener" (Wiener + learned correction) or "direct" (dirty + learned correction, ablation).
Code provenance: tracked files from a clone of `midterm-prep`; the new files ride in the private dataset `exxa-wiggle-code`
(MANIFEST.json, sha256). Output goes straight to /kaggle/working/unified_<ARM>.
"""
import os, sys, glob, json, shutil, subprocess, hashlib, time
os.environ.setdefault("EXXA_T0", str(time.time()))     # the session clock starts here, before clone and setup

ARM = "wiener"
K = 15                 # spectral window half-width (channels at 0.1 km/s): 3 = +-0.3 km/s, 15 = +-1.5 km/s like kin_gamma0
EXTRA = ["--snr", "3,150", "--chan-corr", "0.5", "--beam-max", "1.6", "--lr", "1.5e-4", "--clip-in", "10", "--fp32-stream", "--norm", "robust", "--crop", "256", "--batch", "4", "--seed", "31", "--rounds", "80", "--eval-ladder", "3,6,12,24", "--tag", "r6_k15_crop256"]             # extra train_unified.py flags, e.g. ["--snr", "10,150", "--chan-corr", "0.5", "--beam-max", "1.6", "--tag", "real_k3"]
MODE = "full"          # "smoke": 2 rounds x 40 steps, 1 line holdout + an07, two sizes ; "full": everything
BRANCH = "midterm-prep"
REPO = "/tmp/EXXA"     # not /kaggle/working: the clone must not land in the notebook Output
PKG = f"{REPO}/DENOISING_DIFFUSION"
OUT = f"/kaggle/working/unified_{ARM}_k{K}"

def resume_inputs(tags, out):
    """if a previous session of THIS run is attached as input (its own kernel output), copy its checkpoint / eval JSON / status
    for these tags into `out` and return {tag: checkpoint path or None}. Only files named for our own tags are taken, never another run's."""
    found = {}
    for tag in tags:
        st = [p for p in glob.glob(f"/kaggle/input/**/status_{tag}.json", recursive=True)]
        if not st:
            continue
        src = os.path.dirname(st[0])
        os.makedirs(out, exist_ok=True)
        for name in (f"status_{tag}.json", f"eval_{tag}.json", f"unified_{tag}.pth"):
            if os.path.exists(f"{src}/{name}"):
                shutil.copy(f"{src}/{name}", f"{out}/{name}")
        found[tag] = f"{out}/unified_{tag}.pth" if os.path.exists(f"{out}/unified_{tag}.pth") else None
        print(f"RESUME {tag}: previous session found in {src}: {json.load(open(st[0]))}", flush=True)
    return found


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)

if not os.path.exists(REPO):
    sh(["git", "clone", "--branch", BRANCH, "--depth", "1", "https://github.com/KrishanYadav333/EXXA.git", REPO])
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pytorch-msssim", "bettermoments"])
print(subprocess.run(["git", "-C", REPO, "log", "--oneline", "-1"], capture_output=True, text=True).stdout)

code = os.path.dirname(glob.glob("/kaggle/input/**/MANIFEST.json", recursive=True)[0])
man = json.load(open(f"{code}/MANIFEST.json"))
for rel, info in man["files"].items():
    src, dst = f"{code}/{info['flat']}", f"{PKG}/{rel}"
    assert hashlib.sha256(open(src, "rb").read()).hexdigest() == info["sha256"], f"checksum mismatch {rel}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
print("overlaid", list(man["files"]), "| staged from", man["local_head"], flush=True)

DATA = f"{PKG}/self-gravitating cube and dirty cube"
os.makedirs(DATA, exist_ok=True)
synth = os.path.dirname(glob.glob("/kaggle/input/**/run_9015_00370_rt_00", recursive=True)[0])
v2 = os.path.dirname(glob.glob("/kaggle/input/**/clean_sg.fits", recursive=True)[0])
for link, target in (("sg_synth", synth), ("kinematic_data_v2", v2)):
    if not os.path.lexists(f"{DATA}/{link}"):
        os.symlink(target, f"{DATA}/{link}")
le = {}
for p in glob.glob("/kaggle/input/**/run_00*", recursive=True):
    if os.path.isdir(p):
        le[os.path.dirname(p)] = le.get(os.path.dirname(p), 0) + 1
LE_DIR = max(le, key=le.get)
print("sg_synth ->", synth, "| v2 ->", v2, "| line emission ->", LE_DIR, f"({le[LE_DIR]} cubes)", flush=True)

os.chdir(PKG)
env = dict(os.environ, PYTHONPATH=PKG, EXXA_ROOT=PKG, LE_DIR=LE_DIR, PYTHONWARNINGS="ignore", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
sh([sys.executable, "experiments/make_analytic_gi_cubes.py"], env=env)

os.makedirs(OUT, exist_ok=True)
args = ["--arm", ARM, "--k", str(K), "--out", OUT] + EXTRA
TAG = EXTRA[EXTRA.index("--tag") + 1] if "--tag" in EXTRA else f"{ARM}_k{K}"
prev = resume_inputs([TAG], OUT)
if prev.get(TAG):
    args += ["--resume", prev[TAG]]
sph = glob.glob("/kaggle/input/**/run_young_*/view.json", recursive=True)
if sph:                # rendered GI discs attached (kernel source sph-cubes): train on the train split, score the held-out simulations
    args += ["--sph-dir", os.path.dirname(os.path.dirname(sph[0])), "--eval-sph"]
spd = glob.glob("/kaggle/input/**/GI_r0.5_clean.fits", recursive=True)
if spd:                # Speedie GI disk + Keplerian twin attached (kernel source speedie-prep-fast): score them after training (test only, never trained on)
    args += ["--speedie-dir", os.path.dirname(spd[0]), "--eval-speedie", "GI_r0.5,Kep_r0.5,GI_r1.5,Kep_r1.5"]
if MODE == "smoke":
    args += ["--rounds", "2", "--steps", "40", "--n-line-eval", "1", "--eval-sg", "an07", "--sizes", "256,600"]
t0 = time.time()
sh([sys.executable, "-u", "experiments/train_unified.py"] + args, env=env)
print(f"UNIFIED {ARM} DONE in {(time.time() - t0) / 60:.1f} min", flush=True)
