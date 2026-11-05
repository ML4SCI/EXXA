"""
Plain script: PYTHONPATH=. python3 tests/test_checkpoint_eval.py

Fast checks on src/evaluation/checkpoint_eval.py that need no GPU and no data cubes: naming, discovery, and that
`describe` reads architecture from a checkpoint's own metadata. The full-cube validation (winner_aug_seed43 on
run_0002_00560_rt_00 reproducing the published v20 M0 +31.3 / M1 +77.2 / M2 +84.6) takes ~4 minutes on CPU and is

"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation import checkpoint_eval as ce

fails = []
def check(name, cond):
    print(("  OK    " if cond else "  FAIL  ") + name)
    if not cond: fails.append(name)

# labels keep the seed (it is identity) and drop only the notebook prefix and extension
check("label nb05 pth", ce.label_of("/x/nb05_winner_starlet_ft_seed42.pth") == "winner_starlet_ft_seed42")
check("label nb13 pth.tar", ce.label_of("nb13_ddpm_l1_ft.pth.tar") == "ddpm_l1_ft")
check("label best_models ckpt", ce.label_of("winner_aug_seed43.ckpt") == "winner_aug_seed43")
check("source nb05", ce.source_of("nb05_x.pth") == "nb05")
check("source best_models", ce.source_of("winner_aug_seed43.pth") == "best_models")
check("train size from label", [ce._train_size(l) for l in ("winner_aug_res320_seed43", "winner_aug_res480_seed43", "winner_aug_native600_seed43", "winner_mae_ft_seed42")] == [320, 480, 600, 256])
check("notebook 14 native arms are 600, not 256", [ce._train_size(l) for l in ("sweep_winner_600", "sweep_winner_aug_600", "sweep_winner_p10_600", "sweep_winner_seed600x")] == [600, 600, 600, 256])

root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_models")
if os.path.isdir(root):
    found = ce.discover([root])
    check("discover finds the best_models", {"winner_aug_seed43", "kin_gamma0", "sg_k3_fresh", "ddpm_seed42", "ddrm_prior"} <= set(found))
    try:
        import torch  # noqa: F401
        want = {"winner_aug_seed43": ("unet", "unet", 1, 1), "kin_gamma0": ("unet", "stack_kin", 31, 31),
                "sg_k3_fresh": ("unet", "stack_sg", 7, 1), "ddpm_seed42": ("diffusion", "ddpm", 1, 1),
                "ddrm_prior": ("diffusion", "ddrm", 1, 1)}
        for label, (kind, fam, i, o) in want.items():
            s = ce.describe(found[label])
            check(f"describe {label}: {kind}/{fam} {i}->{o}", (s.kind, s.family, s.in_channels, s.out_channels) == (kind, fam, i, o))
        check("ddrm is not scored, with a reason", not ce.describe(found["ddrm_prior"]).supported and ce.describe(found["ddrm_prior"]).why_not)
        check("K from in_channels", ce.describe(found["kin_gamma0"]).K == 15 and ce.describe(found["sg_k3_fresh"]).K == 3)
    except ImportError:
        print("  SKIP  describe (torch not installed)")
else:
    print("  SKIP  discovery (models/best_models not present)")

# the .para lookup must use the cube's DIRECTORY: ho["folder"] is a name, and a lookup there silently returned None
# (so the inclination was never fixed and the distance fell back to 140 pc)
ledir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Line Emission Data", "run_0002_00560_rt_00")
if os.path.isdir(ledir):
    check("para lookup finds the true inclination", abs((ce._para_value(ledir, "RT: imin") or 0) - 63.709) < 1e-3)
    check("para lookup finds the distance", abs((ce._para_value(ledir, "distance (pc)") or 0) - 100.247) < 1e-3)
    check("para lookup by folder NAME finds nothing (the bug)", ce._para_value("run_0002_00560_rt_00", "RT: imin") is None)
else:
    print("  SKIP  para lookup (Line Emission Data not present)")

# amplitude matching: rescale clean only when the two cubes are on clearly different scales
try:
    import numpy as np
    rng = np.random.default_rng(1)
    dirty = rng.normal(size=(6, 20, 20)).astype("f4") + 5.0
    def mk(clean):
        return ce.Case("t", "sg", clean.astype("f4"), dirty, np.arange(6.0), 1.0)
    c_far = mk(dirty / 300.0 + 0.001 * rng.normal(size=dirty.shape))
    k = ce.match_amplitude(c_far)
    check("amplitude: a ~300x mismatch is corrected", 250 < k < 350 and abs(float(c_far.clean.mean()) - float(dirty.mean())) < 0.5)
    c_near = mk(dirty * 0.99)
    before = c_near.clean.copy()
    check("amplitude: a 1% difference is left exactly as it was", ce.match_amplitude(c_near) == 1.0 and (c_near.clean == before).all())
except ImportError:
    print("  SKIP  amplitude (numpy missing)")

print("\nPASSED" if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)
