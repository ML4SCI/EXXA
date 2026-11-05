"""
Plain script: PYTHONPATH=. python3 tests/test_checkpoint_figures.py

Builds every figure in src/evaluation/checkpoint_figures.py from small synthetic artifacts and results, and checks each
file exists and is not blank. Catches a plotting bug in seconds instead of at the end of a Kaggle session. It does not
judge the science: that is what the real-run figures are for.
"""
import os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from src.evaluation import checkpoint_figures as cf

rng = np.random.default_rng(0)
H = W = 96; C = 30
y, x = np.mgrid[0:H, 0:W]
r = np.hypot(x - 48, y - 48)
disk = np.exp(-(r / 22) ** 2)
mask = disk > 0.05


def maps(noise):
    m0 = disk + noise * rng.normal(size=(H, W))
    m1 = 3000 * (x - 48) / 30 * disk + noise * 300 * rng.normal(size=(H, W))
    m2 = 400 * disk + noise * 50 * rng.normal(size=(H, W))
    return m0, m1, m2

def art(noise):
    m0, m1, m2 = maps(noise)
    return dict(m0=m0.astype("f4"), m1=m1.astype("f4"), m2=m2.astype("f4"), m1q=(m1 / 1000).astype("f4"),
                resid=(0.2 * noise * rng.normal(size=(H, W))).astype("f4"),
                chan=rng.normal(size=(3, H, W)).astype("f4") * noise + disk[None], spec=rng.normal(size=(3, C)).astype("f4"),
                ispec=(np.sin(np.linspace(0, 3, C)) * (1 + noise)).astype("f4"),
                invented=(rng.random((H, W)) * noise * 0.3).astype("f2"))

c0, d0 = maps(0.0), maps(0.6)
ref = dict(mask=mask, velax=np.linspace(-5000, 5000, C), chan_idx=np.array([8, 15, 22]), px=np.array([[48, 48], [48, 70], [8, 8]]),
           cx=48.0, cy=48.0, case=np.array("synthetic_case"), dirty_invented=(rng.random((H, W)) * 0.3).astype("f2"))
for tag, m in (("clean", c0), ("dirty", d0)):
    ref.update({f"{tag}_m0": m[0], f"{tag}_m1": m[1], f"{tag}_m2": m[2], f"{tag}_m1q": m[1] / 1000,
                f"{tag}_resid": 0.05 * rng.normal(size=(H, W)), f"{tag}_chan": rng.normal(size=(3, H, W)) * (0.0 if tag == "clean" else 0.5) + disk[None],
                f"{tag}_spec": rng.normal(size=(3, C)), f"{tag}_ispec": np.sin(np.linspace(0, 3, C)) + 0.1})

labels = [("winner_aug_seed43", "unet"), ("winner_mae_ft_seed42", "unet"), ("winner_starlet_p10_ft_seed42", "unet"),
          ("winner_hybrid_ft_seed42", "unet"), ("kin_gamma0", "stack_kin"), ("sg_k3_fresh", "stack_sg"), ("ddpm_l1_ft", "ddpm")]
fails = []
def check(name, cond):
    print(("  OK    " if cond else "  FAIL  ") + name)
    if not cond: fails.append(name)

with tempfile.TemporaryDirectory() as d:
    md = os.path.join(d, "maps"); os.makedirs(md)
    np.savez_compressed(os.path.join(md, "synthetic_case__REF.npz"), **ref)
    rows = []
    for i, (l, fam) in enumerate(labels):
        np.savez_compressed(os.path.join(md, f"synthetic_case__{l}.npz"), **art(0.1 + 0.1 * i))
        for case in ("synthetic_case", "run_b", "run_c"):
            rows.append(dict(checkpoint=l, family=fam, source="nb05", case=case, domain="line_emission" if case != "run_c" else "sg",
                             psnr=35 + i, ssim=0.99, M0=10 * i - 5, M1=60 + i, M2=40 + 5 * i, resid_r=0.5 + 0.05 * i, wiggle_gain=-0.1,
                             gradE_ratio=0.6 + 0.1 * i, lapvar_ratio=1, invented_blobs=0.3 * i, overshoot=1.0, dirty_resid_r=0.7))
    df = pd.DataFrame(rows)
    out = os.path.join(d, "out")
    paths = cf.build_all(md, df, out, figure_cases=None, log=lambda *a: None, extra=True)   # every figure kind
    lean = cf.build_all(md, df, out + '_lean', figure_cases=None, log=lambda *a: None)   # the default, lean set
    kinds = {os.path.basename(q) for q in lean}
    check('default build is the lean set: no chan / err_M0 sheets, but the core disk sheets, sharpness and invented are there',
          not any(k in n for n in kinds for k in ('_chan', '_err_M0')) and any('_sharpness' in n for n in kinds) and any('_invented' in n for n in kinds) and any('_M1' in n for n in kinds) and any('_err_M1' in n for n in kinds)
          and any('wiggle_classic' in n for n in kinds) and len(lean) < len(paths))
    names = sorted(os.path.basename(p) for p in paths)
    for need in ("nb16_table_line_emission.png", "nb16_scoreboard_line_emission.png", "nb16_table_sg.png", "nb16_per_cube.png",
                 "nb16_sheet_synthetic_case_M0.png", "nb16_sheet_synthetic_case_M1.png", "nb16_sheet_synthetic_case_M2.png",
                 "nb16_sheet_synthetic_case_err_M0.png", "nb16_sheet_synthetic_case_err_M1.png", "nb16_sheet_synthetic_case_wiggle.png", "nb16_sheet_synthetic_case_wiggle_classic_p01.png", "nb16_sheet_synthetic_case_wiggle_classic_p02.png",
                 "nb16_sheet_synthetic_case_chan0.png", "nb16_sheet_synthetic_case_chan1.png", "nb16_sheet_synthetic_case_chan2.png",
                 "nb16_sheet_synthetic_case_chan_err.png", "nb16_sheet_synthetic_case_sharpness.png", "nb16_sheet_synthetic_case_invented.png",
                 "nb16_sheet_synthetic_case_spectra.png", "nb16_sheet_synthetic_case_radial_power.png", "nb16_sheet_synthetic_case_integrated_spectrum.png",
                 "nb16_sheet_synthetic_case_calibration.png", "nb16_sheet_synthetic_case_error_hist.png", "nb16_sheet_synthetic_case_ensemble.png", "nb16_sheet_synthetic_case_topk.png"):
        check(f"built {need}", need in names or need[:-4] + "_p01.png" in names)   # a sheet with more than 6 checkpoints is paged
    check("loss x source matrix built when some arm names match", "nb16_loss_x_source.png" in names)
    for p in paths:
        ok = os.path.getsize(p) > 8000
        if not ok: check(f"non-blank {os.path.basename(p)}", False)
    check("every figure is non-trivial in size", all(os.path.getsize(p) > 8000 for p in paths))

    # loss x source with arm names it recognises
    df2 = df.copy()
    df2["checkpoint"] = df2["checkpoint"].replace({"winner_aug_seed43": "winner_hybrid_ft_seed42x"})
    rows2 = []
    for loss in ("hybrid", "mae", "wavelet", "starlet", "gradient"):
        for src in ("", "_p10", "_beam"):
            for case in ("c1", "c2"):
                rows2.append(dict(checkpoint=f"winner_{loss}{src}_ft_seed42", family="unet", case=case, domain="line_emission", psnr=40, ssim=.99, M0=30, M1=70, M2=60, resid_r=.5, wiggle_gain=0, gradE_ratio=.8, lapvar_ratio=1, invented_blobs=0, overshoot=1, dirty_resid_r=.7, source="nb05"))
        rows2.append(dict(checkpoint=f"winner_{loss}_fresh_seed42", family="unet", case="c1", domain="line_emission", psnr=40, ssim=.99, M0=30, M1=70, M2=60, resid_r=.5, wiggle_gain=0, gradE_ratio=.8, lapvar_ratio=1, invented_blobs=0, overshoot=1, dirty_resid_r=.7, source="nb05"))
    p = cf.fig_loss_source(pd.DataFrame(rows2), os.path.join(d, "ls.png"))
    check("loss x source matrix built from real-looking arm names", p is not None and os.path.getsize(p) > 8000)
    none = df.copy(); none["checkpoint"] = "kin_gamma0"
    check("loss x source matrix skipped, not crashed, when no arm name matches", cf.fig_loss_source(none, os.path.join(d, "n.png")) is None)

    # 34 checkpoints on one sheet left each disk unreadable; the sheet now pages, with the reference tiles repeated on every page
    yy, xx = np.mgrid[-1:1:96j, -1:1:96j]
    disk = np.where(xx ** 2 + yy ** 2 < 0.8, xx, np.nan)
    panels = [("CLEAN", disk), ("DIRTY", disk)] + [(f"ck{i}", disk * (1 + i / 40)) for i in range(34)]
    pages = cf.contact_sheet(panels, title="paging", path=os.path.join(d, "pg.png"), cmap="RdBu_r", vlim=(-1, 1), cbar_label="x", n_ref=2)
    check("34 checkpoints page into 6 sheets of 8 tiles (clean, dirty, 6 checkpoints)", len(pages) == 6 and all(os.path.getsize(q) > 5000 for q in pages) and pages[0].endswith("pg_p01.png"))
    one = cf.contact_sheet(panels[:6], title="one", path=os.path.join(d, "one.png"), cmap="RdBu_r", vlim=(-1, 1), cbar_label="x", n_ref=2)
    check("a small sheet stays one page under its own name", one == [os.path.join(d, "one.png")])

check("tile labels keep the seed", cf.short("sweep_winner_seed42") != cf.short("sweep_winner_seed43") and cf.short("winner_starlet_p10_ft_seed42") == "starlet_p10_ft s42")
dup = cf._drop_copies({"sweep_winner_aug_seed43": 1, "winner_aug_seed43": 1, "kin_gamma0": 1},
                      {"sweep_winner_aug_seed43": dict(psnr=1, M0=2, M1=3, M2=4), "winner_aug_seed43": dict(psnr=1, M0=2, M1=3, M2=4), "kin_gamma0": dict(psnr=5, M0=2, M1=3, M2=4)})
check("byte-copy checkpoints drawn once", sorted(dup) == ["kin_gamma0", "sweep_winner_aug_seed43"])

print("\nPASSED" if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)
