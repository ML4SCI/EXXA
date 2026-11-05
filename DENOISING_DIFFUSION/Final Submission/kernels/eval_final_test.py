"""
THE ONE-TIME FINAL TEST of recipe B: the Speedie r1.5 disc and its Keplerian twin, rescaled to peak SNR 6 / 12 / 24, never used for any decision before. Run ONCE; whatever it gives is reported.
Kaggle kernel: the two inference-only levers TOGETHER (size-test: network on a ~10 px-beam grid, mode full; wiener2-test: second pass from
a Wiener built on the first pass), on the full decision set, gate on. VARIANT picks the ensemble: A = R4n-long seeds 3 + 11 + 23; B = A + R8-from-R6 (4 nets);
C = R8-from-R6 alone. Cases: 5 line holdouts, 5 held-out Young sims, 9074 (+ _ai), an07, an08, SG v2 at 256 / 600 px; Speedie GI / twin r0.5 + r1.5 native; ladder 3/6/12/24.

"""
import os, sys, glob, json, shutil, subprocess, hashlib, time
os.environ.setdefault("EXXA_T0", str(time.time()))

REPO, BRANCH = "/tmp/EXXA", "midterm-prep"
PKG = f"{REPO}/DENOISING_DIFFUSION"
OUT = "/kaggle/working/final_test"

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
sph = glob.glob("/kaggle/input/**/run_young_*/view.json", recursive=True)
SPD = os.path.dirname(glob.glob("/kaggle/input/**/GI_r0.5_clean.fits", recursive=True)[0])
common = ["--eval-ckpt", ",".join(paths), "--n-line-eval", "-1", "--eval-sg", "", "--sizes", "600", "--gate", "40,120,0.25", "--speedie-dir", SPD,
          "--eval-speedie", "", "--eval-ladder", "6,12,24", "--ladder-base", "GI_r1.5,Kep_r1.5",
          "--wiener2", "--infer-beam-px", "10", "--infer-mode", "full"]    # THE ONE-TIME FINAL TEST, as pre-registered: run once, never re-run
models = [("final_test_B", common)]
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
