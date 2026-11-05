"""Tables for the improvement track: fresh analytic set, B vs candidate A, and B2 (final-test noise realisations).
Run from this folder: python experiments/fresh_tables.py [out.md]"""
import glob, json, sys, numpy as np

def run_dir(name):
    """Folder of a run under results/NN-.../ (runs are grouped by result)."""
    return glob.glob(f"results/[0-9]*/{name}")[0]
B = json.load(open(f"{run_dir('eval-fresh-b_v2_2026-10-21')}/eval_fresh_B.json")); A = json.load(open(f"{run_dir('eval-fresh-b_v2_2026-10-21')}/eval_fresh_A_cfix.json"))
out = []
p = lambda *a: out.append(" ".join(str(x) for x in a))
p("## Fresh analytic set, 600 px: recipe B vs candidate A (B + --central-fix)\n")
p("| case | PSNR B / A / Wiener | M0 B / A | M1 B / A | M2 B / A | rc_r B / A / Wiener / dirty | PV5 r B / Wiener / dirty | PV5 mag (m/s) clean / B / Wiener |")
p("|---|---|---|---|---|---|---|---|")
agg = {}
for k in B:
    b, a, w = B[k]["model"], A[k]["model"], B[k]["wiener"]
    wb, wa = B[k]["wiggle"], A[k]["wiggle"]
    m = lambda d, key: d.get(key, float("nan"))
    p(f"| {k} | {b['psnr']:.2f} / {a['psnr']:.2f} / {w['psnr']:.2f} | {b['M0']:.1f} / {a['M0']:.1f} | {b['M1']:.1f} / {a['M1']:.1f} | {b['M2']:.1f} / {a['M2']:.1f} | "
      f"{wb['model']['rc_r']:.3f} / {wa['model']['rc_r']:.3f} / {wb['wiener']['rc_r']:.3f} / {wb['dirty']['rc_r']:.3f} | "
      f"{m(wb['model'],'pv5_r'):.2f} / {m(wb['wiener'],'pv5_r'):.2f} / {m(wb['dirty'],'pv5_r'):.2f} | {m(wb['clean'],'pv5_mag_ms'):.0f} / {m(wb['model'],'pv5_mag_ms'):.0f} / {m(wb['wiener'],'pv5_mag_ms'):.0f} |")
    grp = ("twin" if "k" in k.split("@")[0].split("_")[0] else "gi") + ("_ai" if "_ai" in k else "_deconv")
    g = agg.setdefault(grp, {x: [] for x in ("psnrB", "psnrA", "dM0", "dM1", "dM2", "rcB", "rcA", "rcW", "pvB", "pvW")})
    g["psnrB"].append(b["psnr"]); g["psnrA"].append(a["psnr"]); g["dM0"].append(a["M0"] - b["M0"]); g["dM1"].append(a["M1"] - b["M1"]); g["dM2"].append(a["M2"] - b["M2"])
    g["rcB"].append(wb["model"]["rc_r"]); g["rcA"].append(wa["model"]["rc_r"]); g["rcW"].append(wb["wiener"]["rc_r"])
    g["pvB"].append(m(wb["model"], "pv5_r")); g["pvW"].append(m(wb["wiener"], "pv5_r"))
p("\n## Means by group (A minus B for moments)\n")
p("| group | n | PSNR B / A | dM0 / dM1 / dM2 (A - B, points; min) | rc_r B / A / Wiener | PV5 r B / Wiener |")
p("|---|---|---|---|---|---|")
for grp, g in agg.items():
    mu = {x: float(np.nanmean(v)) for x, v in g.items()}
    p(f"| {grp} | {len(g['psnrB'])} | {mu['psnrB']:.2f} / {mu['psnrA']:.2f} | {mu['dM0']:+.1f} / {mu['dM1']:+.1f} / {mu['dM2']:+.1f} ({min(g['dM0']):+.1f} / {min(g['dM1']):+.1f} / {min(g['dM2']):+.1f}) | "
      f"{mu['rcB']:.3f} / {mu['rcA']:.3f} / {mu['rcW']:.3f} | {mu['pvB']:.2f} / {mu['pvW']:.2f} |")

R = json.load(open(f"{run_dir('eval-final-test-seeds_v1_2026-10-21')}/eval_final_test_seeds_B.json"))
p("\n## B2: one-time final test (Speedie r1.5) over 4 noise realisations (realisation 0 = the original final-test noise)\n")
p("| disc, SNR | rc_r model | rc_r Wiener | rc_r dirty | PV5 r model | PV5 r Wiener | PV5 mag model (m/s) | PV5 mag Wiener | PV5 mag clean |")
p("|---|---|---|---|---|---|---|---|---|")
for base in ("GI_r1.5", "Kep_r1.5"):
    for snr in (12, 24):
        ks = [k for k in R if k.startswith(f"speedie_ladder_{base}_snr{snr}@") or k.startswith(f"speedie_ladder_{base}_snr{snr}_s")]
        f = lambda arm, key: [R[k]["wiggle"][arm][key] for k in ks]
        ms = lambda v: f"{np.mean(v):.2f} +- {np.std(v, ddof=1):.2f}" if abs(np.mean(v)) < 5 else f"{np.mean(v):.0f} +- {np.std(v, ddof=1):.0f}"
        p(f"| {base} SNR {snr} (n={len(ks)}; r0 = {R[[k for k in ks if '_s' not in k.split('snr')[1]][0]]['wiggle']['model']['rc_r']:.2f}) | {ms(f('model','rc_r'))} | {ms(f('wiener','rc_r'))} | {ms(f('dirty','rc_r'))} | "
          f"{ms(f('model','pv5_r'))} | {ms(f('wiener','pv5_r'))} | {ms(f('model','pv5_mag_ms'))} | {ms(f('wiener','pv5_mag_ms'))} | {np.mean(f('clean','pv5_mag_ms')):.0f} |")
print("\n".join(out))
open(sys.argv[1], "w").write("\n".join(out) + "\n") if len(sys.argv) > 1 else None

Y = run_dir('eval-fresh-young_v1_2026-10-21') + "/eval_fresh_young_{}.json"
import os
if os.path.exists(Y.format("B")):
    YB, YA = json.load(open(Y.format("B"))), json.load(open(Y.format("A_cfix")))
    out.append("\n## Fresh Young TEST3 (noise-only line cubes, 600 px): recipe B vs candidate A\n")
    out.append("| disc | PSNR B / A / Wiener | M0 B / A / Wiener | M1 B / A | M2 B / A | rc_r B / Wiener / dirty | PV5 r B / Wiener / dirty | PV5 mag clean / B / Wiener (m/s) |")
    out.append("|---|---|---|---|---|---|---|---|")
    cols = {x: [] for x in ("pB", "pA", "pW", "m0B", "m0A", "rB", "rW", "rD", "vB", "vW", "vD")}
    for k in YB:
        b, a, w, g = YB[k]["model"], YA[k]["model"], YB[k]["wiener"], YB[k]["wiggle"]
        out.append(f"| {k.replace('run_young_', '')} | {b['psnr']:.2f} / {a['psnr']:.2f} / {w['psnr']:.2f} | {b['M0']:.1f} / {a['M0']:.1f} / {w['M0']:.1f} | {b['M1']:.1f} / {a['M1']:.1f} | "
                   f"{b['M2']:.1f} / {a['M2']:.1f} | {g['model']['rc_r']:.3f} / {g['wiener']['rc_r']:.3f} / {g['dirty']['rc_r']:.3f} | "
                   f"{g['model']['pv5_r']:.2f} / {g['wiener']['pv5_r']:.2f} / {g['dirty']['pv5_r']:.2f} | {g['clean']['pv5_mag_ms']:.0f} / {g['model']['pv5_mag_ms']:.0f} / {g['wiener']['pv5_mag_ms']:.0f} |")
        for x, v in zip(cols, (b["psnr"], a["psnr"], w["psnr"], b["M0"], a["M0"], g["model"]["rc_r"], g["wiener"]["rc_r"], g["dirty"]["rc_r"], g["model"]["pv5_r"], g["wiener"]["pv5_r"], g["dirty"]["pv5_r"])):
            cols[x].append(v)
    mu = {x: float(np.mean(v)) for x, v in cols.items()}
    out.append(f"| **mean (n={len(YB)})** | {mu['pB']:.2f} / {mu['pA']:.2f} / {mu['pW']:.2f} | {mu['m0B']:.1f} / {mu['m0A']:.1f} | | | {mu['rB']:.3f} / {mu['rW']:.3f} / {mu['rD']:.3f} | {mu['vB']:.2f} / {mu['vW']:.2f} / {mu['vD']:.2f} | |")
    print("\n".join(out[-(len(YB) + 5):]))
    open(sys.argv[1], "a").write("\n".join(out[-(len(YB) + 5):]) + "\n") if len(sys.argv) > 1 else None
