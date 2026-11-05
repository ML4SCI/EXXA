"""
FRESH-SET BASELINE of the improvement track: recipe B and candidate A (B + --central-fix) scored by the SAME kernel on
the fresh test set: analytic GI an09-an16 and no-wiggle twins an09k / an12k / an15k (both regimes, 600 px) and the eight Young TEST3 discs
(sph-cubes-test3 output, split "test3"). 
"""
import os, sys, glob, json, shutil, subprocess, hashlib, time
os.environ.setdefault("EXXA_T0", str(time.time()))

REPO, BRANCH = "/tmp/EXXA", "midterm-prep"
PKG = f"{REPO}/DENOISING_DIFFUSION"
OUT = "/kaggle/working/eval_fresh"

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
print("overlaid", list(man["files"]), "| staged from", man["local_head"], flush=True)

le = {}
for p in glob.glob("/kaggle/input/**/run_00*", recursive=True):
    if os.path.isdir(p):
        le[os.path.dirname(p)] = le.get(os.path.dirname(p), 0) + 1
LE_DIR = max(le, key=le.get)
DATA = f"{PKG}/self-gravitating cube and dirty cube"            # hydro + SG v2 + analytic
os.makedirs(DATA, exist_ok=True)
synth = os.path.dirname(glob.glob("/kaggle/input/**/run_9015_00370_rt_00", recursive=True)[0])
v2 = os.path.dirname(glob.glob("/kaggle/input/**/clean_sg.fits", recursive=True)[0])
for link, target in (("sg_synth", synth), ("kinematic_data_v2", v2)):
    if not os.path.lexists(f"{DATA}/{link}"):
        os.symlink(target, f"{DATA}/{link}")
_env0 = dict(os.environ, PYTHONPATH=PKG, EXXA_ROOT=PKG, LE_DIR=LE_DIR, PYTHONWARNINGS="ignore")
sh([sys.executable, "experiments/make_analytic_gi_cubes.py"], env=_env0, cwd=PKG, check=True)
find = lambda pat: (glob.glob(f"/kaggle/input/**/{pat}", recursive=True) or [None])[0]
FILES = ["unified_r4n_k15_long.pth", "unified_r4n_k15_long_s11.pth", "unified_r4n_k15_long_s23.pth"]
paths = [p for p in (find(f) for f in FILES) if p]     # every seed this account can see; the plain row below is the same ensemble
VARIANT = "B"
FILES = {"A": ["unified_r4n_k15_long.pth", "unified_r4n_k15_long_s11.pth", "unified_r4n_k15_long_s23.pth"],
         "B": ["unified_r4n_k15_long.pth", "unified_r4n_k15_long_s11.pth", "unified_r4n_k15_long_s23.pth", "unified_r8_m1_r6.pth"],
         "C": ["unified_r8_m1_r6.pth"]}[VARIANT]
find = lambda pat: (glob.glob(f"/kaggle/input/**/{pat}", recursive=True) or [None])[0]
paths = [find(f) for f in FILES]
assert all(paths), f"missing checkpoints: {dict(zip(FILES, paths))}"
TEST3 = "M0.1_R50_Mu0.9,M0.3_R200_Mu1.1,M0.3_R50_Mu0.3,M0.7_R100_mu0.3,M0.5_R200_Mu0.9,M0.7_R50_Mu0.2,M1_R100_Mu0.5,M1_R200_Mu0.6"
_v = glob.glob("/kaggle/input/**/run_young_M1_R200_Mu0.6/view.json", recursive=True)
if _v:
    SPH = os.path.dirname(os.path.dirname(_v[0]))
else:                                   # sph-cubes-test3 is not visible from this account: render TEST3 here, same command and seed
    SPH = "/kaggle/temp/sph_cubes"
    sh([sys.executable, "-u", "experiments/make_sph_cubes.py", "--out", SPH, "--workers", str(os.cpu_count()), "--seed", "3", "--runs", TEST3], env=_env0, cwd=PKG, check=True)
print("TEST3 cubes from", SPH, flush=True)
assert "--eval-sph-split" in open(f"{PKG}/experiments/train_unified.py").read() and "def central_flux_fix" in open(f"{PKG}/experiments/train_unified.py").read(), "stale code overlay (needs --eval-sph-split, central_flux_fix)"
assert '"an16"' in open(f"{PKG}/experiments/make_analytic_gi_cubes.py").read(), "stale code overlay: fresh analytic discs missing"
FRESH = "an09,an10,an11,an12,an13,an14,an15,an16,an09k,an12k,an15k"
common = ["--eval-ckpt", ",".join(paths), "--n-line-eval", "-1", "--eval-sg", FRESH, "--sizes", "600", "--gate", "40,120,0.25",
          "--sph-dir", SPH, "--eval-sph", "--eval-sph-split", "test3", "--wiener2", "--infer-beam-px", "10", "--infer-mode", "full"]
models = [("fresh_B", common), ("fresh_A_cfix", common + ["--central-fix"])]
sph = SPH; SPD = None
print("checkpoints:", paths, "| young sims:", bool(sph), "| speedie:", SPD, flush=True)

os.chdir(PKG)
env = dict(os.environ, PYTHONPATH=PKG, EXXA_ROOT=PKG, LE_DIR=LE_DIR, PYTHONWARNINGS="ignore")
prev = resume_inputs([n for n, _ in models], OUT)
for name, a in models:
    stf = f"{OUT}/status_{name}.json"
    if os.path.exists(stf) and json.load(open(stf)).get("eval_complete"):
        print(f"MODEL {name}: already complete in a previous session, skipped", flush=True)
        continue
    if a is None:
        print(f"MODEL {name}: checkpoint NOT FOUND, skipped", flush=True)
        continue
    t0 = time.time()
    r = sh([sys.executable, "-u", "experiments/train_unified.py", "--eval-only", *a, "--tag", name, "--out", OUT], env=env)
    print(f"MODEL {name} exit {r.returncode} in {(time.time() - t0) / 60:.1f} min", flush=True)
print("FINAL EVAL DONE", flush=True)
