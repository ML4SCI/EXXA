"""
The final figure set (results/final/figs/): copies of the chosen archived figures plus three summary plots made from archived numbers.
Every plotted number is read from the run archives (no hand-typed values except where noted with their source). Run from this folder.
"""
import glob, json, os, shutil, statistics as st, subprocess
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run_dir(name):
    """Folder of a run under results/NN-.../ (runs are grouped by result)."""
    return glob.glob(f"results/[0-9]*/{name}")[0]


OUT = "results/final/figs"
os.makedirs(OUT, exist_ok=True)
COPY = [
    ("01_final_test_PV_GI_snr24.png", f"{run_dir('eval-final-test_v1_2026-10-18')}/figs/speedie_ladder_GI_r1.5_snr24_1024_pv.png"),
    ("02_final_test_PV_GI_snr12.png", f"{run_dir('eval-final-test_v1_2026-10-18')}/figs/speedie_ladder_GI_r1.5_snr12_1024_pv.png"),
    ("03_final_test_PV_twin_snr24.png", f"{run_dir('eval-final-test_v1_2026-10-18')}/figs/speedie_ladder_Kep_r1.5_snr24_1024_pv.png"),
    ("04_speedie_r05_PV.png", f"{run_dir('eval-final-b_v1_2026-10-17')}/figs_pv/speedie_GI_r0.5_1024_pv.png"),
    ("05_an07_PV.png", f"{run_dir('eval-final-b_v1_2026-10-17')}/figs_pv/an07_600_pv.png"),
    ("06_young_M1R100_PV.png", f"{run_dir('eval-final-b_v1_2026-10-17')}/figs_pv/run_young_M1_R100_Mu0.4_600_pv.png"),
    ("07_9074_PV.png", f"{run_dir('eval-final-b_v1_2026-10-17')}/figs_pv/9074_600_pv.png"),
    ("08_MWC758_S5_GI_snr12.png", f"{run_dir('mwc758-13co_zoom512_v2_2026-10-18')}/figs/S5_GI_r0.5_snr12.png"),
    ("09_MWC758_S5_GI_snr12_PV.png", f"{run_dir('mwc758-13co_zoom512_v2_2026-10-18')}/figs/S5_GI_r0.5_snr12_pv.png"),
    ("10_MWC758_S5_twin_snr12_PV.png", f"{run_dir('mwc758-13co_zoom512_v2_2026-10-18')}/figs/S5_Kep_r0.5_snr12_pv.png"),
    ("11_HD163296_real_PV.png", f"{run_dir('dsharp-final-hd163296_v1_2026-10-18')}/dsharp/HD_163296_interp01/HD163296_CO_pv.png"),
    ("12_HD163296_moments.png", f"{run_dir('dsharp-final-hd163296_v1_2026-10-18')}/dsharp/HD_163296_interp01/HD163296_CO_moments.png"),
    ("13_IMLup_real_PV.png", f"{run_dir('dsharp-final-p0_v1_2026-10-18')}/dsharp/IM_Lup_interp01/IMLup_CO_pv.png"),
    ("14_LkCa15_moments.png", f"{run_dir('exoalma-final-p3_v1_2026-10-18')}/alma/LkCa_15_13CO/LkCa_15_13CO_fiducial_moments.png"),
    ("15_planet_check_young_sector.png", f"{run_dir('eval-planet_v1_2026-10-18')}/planet_check/run_young_M1_R100_Mu0.4_600_sector6.5-8.5_planet.png"),
    # fresh set: discs never used for any choice
    ("19_fresh_young_M1R100_maps.png", f"{run_dir('eval-fresh-young_v1_2026-10-21')}/eval_fresh_young/figs_fresh_young_B/run_young_M1_R100_Mu0.5_600.png"),
    ("20_fresh_young_M1R100_PV.png", f"{run_dir('eval-fresh-young_v1_2026-10-21')}/eval_fresh_young/figs_fresh_young_B/run_young_M1_R100_Mu0.5_600_pv.png"),
    ("21_fresh_an10_deconv_PV.png", f"{run_dir('eval-fresh-b_v2_2026-10-21')}/eval_fresh/figs_fresh_B/an10_600_pv.png"),
]
for dst, src in COPY:
    shutil.copy(src, os.path.join(OUT, dst))

# 16: wiggle (rc_r) against SNR: final test + 9074 ladder (noise-only), model vs Wiener vs dirty
ft = json.load(open(f"{run_dir('eval-final-test_v1_2026-10-18')}/eval_final_test_B.json"))
lad = json.load(open(glob.glob(f"{run_dir('eval-ladder-sg_v1_2026-10-18')}/eval_*.json")[0]))
seeds = json.load(open(f"{run_dir('eval-final-test-seeds_v1_2026-10-21')}/eval_final_test_seeds_B.json"))   # B2: 4 noise realisations, SNR 12 / 24
sel = json.load(open(f"{run_dir('eval-ladder-seeds_v1_2026-10-20')}/eval_ladder_seeds_B.json"))       # selection disc r0.5, 8 / 8 / 6 noise draws
orc = json.load(open(f"{run_dir('oracle-speedie_v2_2026-10-15')}/oracle_speedie.json"))                  # oracle linear filter (knows clean's spectrum)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
for m, c in (("model", "C2"), ("wiener", "C1"), ("dirty", "C0")):
    v = {x: [sel[k]["wiggle"][m]["rc_r"] for k in sel if k.startswith(f"speedie_ladder_GI_r0.5_snr{x}@") or k.startswith(f"speedie_ladder_GI_r0.5_snr{x}_s")] for x in (6, 12, 24)}
    assert [len(v[x]) for x in (6, 12, 24)] == [8, 8, 6], {x: len(r) for x, r in v.items()}
    ax[1].errorbar([6, 12, 24], [st.mean(v[x]) for x in (6, 12, 24)], [st.stdev(v[x]) for x in (6, 12, 24)], fmt="o-", color=c, capsize=4, label=m)
ax[1].plot([6, 12, 24], [orc[f"speedie_ladder_GI_r0.5_snr{x}"]["oracle_linear"]["rc_r"] for x in (6, 12, 24)], "s--", color="k", label="oracle linear filter (knows the truth)")
ax[1].set_title("Selection disc: Speedie GI r0.5 (used for choices)\nmean +- sd over 6 to 8 noise draws; oracle: one draw", fontsize=10)
ax = [ax[0], ax[2], ax[1]]           # ax[1] below = the 9074 panel (right)
for m, c in (("model", "C2"), ("wiener", "C1"), ("dirty", "C0")):
    xs = [6, 12, 24]
    ax[0].plot(xs, [ft[f"speedie_ladder_GI_r1.5_snr{x}@1024"]["wiggle"][m]["rc_r"] for x in xs], "o:", color=c, mfc="none", alpha=0.7)
    v = {x: [seeds[k]["wiggle"][m]["rc_r"] for k in seeds if k.startswith(f"speedie_ladder_GI_r1.5_snr{x}@") or k.startswith(f"speedie_ladder_GI_r1.5_snr{x}_s")] for x in (12, 24)}
    assert all(len(r) == 4 for r in v.values()), {x: len(r) for x, r in v.items()}
    ax[0].errorbar([12, 24], [st.mean(v[x]) for x in (12, 24)], [st.stdev(v[x]) for x in (12, 24)], fmt="o-", color=c, capsize=4, label=m)
ax[0].plot([], [], "o:", color="grey", mfc="none", label="one-time run (open)")
for m, c in (("model", "C2"), ("wiener", "C1"), ("dirty", "C0")):
    xs2 = [11, 20, 40, 80]
    ax[1].plot(xs2, [lad[f"9074_ai_ladder_snr{x}@600"]["wiggle"][m]["rc_r"] for x in xs2], "o-", color=c, label=m)
ax[0].set_title("Final test: Speedie GI r1.5 (never used before)\nfilled: mean +- sd over 4 noise draws; open: the one-time run", fontsize=10); ax[1].set_title("Hydro disc 9074 (held out), noise-only, 600 px")
for a in ax:
    a.set_xscale("log"); a.set_xlabel("peak SNR"); a.set_ylabel("wiggle score rc_r"); a.set_ylim(-0.1, 1); a.grid(alpha=0.3); a.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/16_wiggle_vs_snr.png", dpi=110); plt.close(fig)

# 17: flux against the truth (AA Tau injection)
fx = json.load(open(f"{run_dir('aatau-s5flux_v1_2026-10-18')}/alma/AA_Tau_13CO/report.json"))["S5"]
fig, ax = plt.subplots(figsize=(6, 4.2))
for case, mk in (("GI_r0.5", "o-"), ("Kep_r0.5", "s--")):
    ks = sorted((k for k in fx if k.startswith(case)), key=lambda k: float(k.split("snr")[1]))
    xs = [float(k.split("snr")[1]) for k in ks]
    for m, c in (("model", "C2"), ("dirty", "C0")):
        ax.plot(xs, [fx[k]["flux_vs_truth"][m] for k in ks], mk, color=c, label=f"{m}, {'GI disc' if case.startswith('GI') else 'twin'}")
ax.axhline(1, color="k", lw=0.8); ax.set_xscale("log"); ax.set_xlabel("peak SNR"); ax.set_ylabel("line flux / true flux")
ax.set_title("Flux against the truth (injection into AA Tau's noise)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/17_flux_bias.png", dpi=110); plt.close(fig)

# 18: fitted vs published position angle, all 20 ALMA discs (refits applied)
import sys; sys.path.insert(0, "experiments")
import alma_table as at
rows = list(at.rows([p for d in sorted(glob.glob("results/05-real-alma/*final*")) for p in glob.glob(f"{d}/**/report.json", recursive=True) if "refit" not in p]))
fig, ax = plt.subplots(figsize=(6, 5))
for r in rows:
    if r["pa_fit"] is None or r["pa_lit"] is None:
        continue
    lit = r["pa_lit"] % 180; fit = r["pa_fit"]
    fit = fit if abs(fit - lit) <= 90 else (fit - 180 if fit > lit else fit + 180)
    ax.plot(lit, fit, "o", color="C3" if "interp" in r["disc"] else "C0")
    ax.annotate(r["disc"].replace("_13CO", "").replace("_interp01", ""), (lit, fit), fontsize=6, xytext=(3, 3), textcoords="offset points")
ax.plot([0, 180], [0, 180], "k-", lw=0.8)
ax.set_xlabel("published PA mod 180 (deg): exoALMA I (blue), Huang et al. 2018 (red)"); ax.set_ylabel("PA fitted on the raw M1 (deg)")
ax.set_title("Disc geometry on real data (20 discs)"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/18_pa_check.png", dpi=110); plt.close(fig)
# 00: recipe B inference pipeline (order as in train_unified.score_case / infer / denoise / Ensemble)
from matplotlib.patches import FancyBboxPatch
steps = [("noisy cube", "C7"), ("blind 3D Wiener\n(shape from the cube)", "C1"), ("size fix: resample\nso beam = 10 px*", "C7"),
         ("4 U-Nets: Wiener +\nlearned correction", "C2"), ("do-no-harm gate\n(blend to Wiener\nabove blind SNR 40)", "C3"), ("average 4 outputs,\nresample back", "C7"),
         ("pass 2*: Wiener rebuilt\nfrom pass 1, nets again", "C1"), ("denoised cube:\nM0 / M1 / M2, PV", "C2")]
fig, ax = plt.subplots(figsize=(16, 2.6)); ax.set_xlim(0, len(steps) * 2); ax.set_ylim(0, 2.2); ax.axis("off")
for i, (t, c) in enumerate(steps):
    ax.add_patch(FancyBboxPatch((i * 2 + 0.1, 0.55), 1.6, 1.1, boxstyle="round,pad=0.05", fc=c, alpha=0.25, ec=c))
    ax.text(i * 2 + 0.9, 1.1, t, ha="center", va="center", fontsize=9)
    if i < len(steps) - 1:
        ax.annotate("", (i * 2 + 2.08, 1.1), (i * 2 + 1.78, 1.1), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(0.1, 0.15, "* noise-only cubes (beam already in the clean cube). Beam-deconvolution cubes skip the size fix and pass 2. "
        "Recipe B: R4n-long seeds 3 / 11 / 23 + R8-from-R6 (results/final/RECIPES.md).", fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/00_pipeline.png", dpi=120); plt.close(fig)
print(sorted(os.listdir(OUT)))
