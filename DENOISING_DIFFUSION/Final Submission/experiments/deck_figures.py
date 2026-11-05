"""
Extra figures (results/final/figs/22_ to 28_): three charts made from archived numbers, plus copies of
the failure cases, so the figures show the bad results next to the good ones. Run from this folder (needs matplotlib).

"""
import glob, json, os, shutil
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run_dir(name):
    """Folder of a run under results/NN-.../ (runs are grouped by result)."""
    return glob.glob(f"results/[0-9]*/{name}")[0]


OUT = "results/final/figs"
INK, MUTED = "#10263b", "#5b6b7a"
C_DIRTY, C_WIEN, C_MODEL = "#9aa7b4", "#d98a1f", "#1467a8"
plt.rcParams.update({"font.size": 15, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def rc(d, k, arm):
    return d[k]["wiggle"][arm]["rc_r"]


def per_disc(path, keys, labels, title, fname, note=None, sort=True):
    d = json.load(open(path))
    rows = [(l, rc(d, k, "dirty"), rc(d, k, "wiener"), rc(d, k, "model")) for k, l in zip(keys, labels)]
    if sort:
        rows.sort(key=lambda r: r[3])
    fig, ax = plt.subplots(figsize=(11, 6.6))
    h = 0.26
    for i, (l, dv, wv, mv) in enumerate(rows):
        for j, (v, c) in enumerate(((dv, C_DIRTY), (wv, C_WIEN), (mv, C_MODEL))):
            ax.barh(i + (1 - j) * h, v, h * 0.9, color=c)
        ax.text(max(mv, wv, dv, 0) + 0.02, i, f"{mv:.2f}", va="center", color=INK, fontsize=14, fontweight="bold")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(min(-0.1, min(r[1] for r in rows) - 0.05), 1.08)
    ax.set_xlabel("wiggle score rc_r (1 = identical to the clean disc, 0 = no relation)")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontweight="bold", fontsize=16)
    mean = lambda i: sum(r[i] for r in rows) / len(rows)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_MODEL, C_WIEN, C_DIRTY)],
              labels=[f"recipe B, mean {mean(3):.2f}", f"Wiener, mean {mean(2):.2f}", f"noisy cube, mean {mean(1):.2f}"], loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3, frameon=False, fontsize=13)
    if note:
        fig.text(0.01, 0.005, note, color=MUTED, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(f"{OUT}/{fname}", dpi=130); plt.close(fig)


yk = ["run_young_M0.1_R50_Mu0.9@600", "run_young_M0.3_R200_Mu1.1@600", "run_young_M0.3_R50_Mu0.3@600", "run_young_M0.5_R200_Mu0.9@600",
      "run_young_M0.7_R100_mu0.3@600", "run_young_M0.7_R50_Mu0.2@600", "run_young_M1_R100_Mu0.5@600", "run_young_M1_R200_Mu0.6@600"]
yl = [k.replace("run_young_", "").replace("@600", "").replace("_", " ") for k in yk]
per_disc(f"{run_dir('eval-fresh-young_v1_2026-10-21')}/eval_fresh_young_B.json", yk, yl,
         "Eight fresh Young simulations: wiggle score per disc", "22_fresh_young_per_disc.png",
         "Source: eval-fresh-young_v1_2026-10-21. Two discs fail (M0.1 R50, M0.3 R200); the rest gain over the Wiener filter.")

ak = [f"an{n:02d}@600" for n in range(9, 17)]
per_disc(f"{run_dir('eval-fresh-b_v2_2026-10-21')}/eval_fresh_B.json", ak, [k.replace("@600", "") for k in ak],
         "Eight fresh analytic GI discs (beam undone): wiggle score per disc", "23_fresh_analytic_per_disc.png",
         "Source: eval-fresh-b_v2_2026-10-21. an15 (m = 4 spiral) is the weakest.")

# candidates against the written rule
CTRL, BAR, WIEN = 0.677, 0.727, 0.556
cand = [("A  central-flux fix", [None]), ("C  12 extra discs", [0.700, 0.663]), ("D  amplitude loss", [0.684, 0.698]), ("E  Young-trained", [0.690, 0.683, 0.665])]
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.axhspan(BAR, 0.78, color="#e6f2ea"); ax.axhline(BAR, color="#1e8e4e", lw=2); ax.axhline(CTRL, color=MUTED, lw=2, ls="--"); ax.axhline(WIEN, color=C_WIEN, lw=2, ls=":")
for i, (n, vals) in enumerate(cand):
    v = [x for x in vals if x is not None]
    ax.scatter([i] * len(v), v, s=140, color=C_MODEL, zorder=3, edgecolor="white", linewidth=2)
    if not v:
        ax.text(i, 0.60, "same wiggle as B on\nevery disc; rejected\non its M0 cost", ha="center", color=MUTED, fontsize=12)
ax.set_xticks(range(len(cand))); ax.set_xticklabels([c[0] for c in cand]); ax.set_xlim(-0.5, 3.5); ax.set_ylim(0.52, 0.78)
ax.text(3.48, BAR + 0.004, "adoption bar 0.727 (control + 0.05)", ha="right", color="#1e8e4e", fontsize=13)
ax.text(-0.47, CTRL + 0.004, "control (recipe B, 3 seeds) 0.677", ha="left", color=MUTED, fontsize=13)
ax.text(3.48, WIEN + 0.004, "Wiener 0.556", ha="right", color=C_WIEN, fontsize=13)
ax.set_ylabel("wiggle rc_r, fresh set, one dot per seed")
fig.suptitle("Four candidates against the rule written first: none reaches the bar", x=0.01, ha="left", color=INK, fontweight="bold", fontsize=16)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUT}/24_candidates_vs_rule.png", dpi=130); plt.close(fig)

# copies of failure cases (kept next to the good ones)
B = f"{run_dir('eval-fresh-b_v2_2026-10-21')}/eval_fresh/figs_fresh_B"; Y = f"{run_dir('eval-fresh-young_v1_2026-10-21')}/eval_fresh_young/figs_fresh_young_B"
for dst, src in [("25_fail_young_M0.3_R200_maps.png", f"{Y}/run_young_M0.3_R200_Mu1.1_600.png"), ("26_weak_an15_maps.png", f"{B}/an15_600.png"),
                 ("27_twin_an09k_PV.png", f"{B}/an09k_600_pv.png"), ("28_good_an12_maps.png", f"{B}/an12_600.png")]:
    shutil.copy(src, f"{OUT}/{dst}")
print("wrote", sorted(f for f in os.listdir(OUT) if f[:2] in ("22", "23", "24", "25", "26", "27", "28")))
