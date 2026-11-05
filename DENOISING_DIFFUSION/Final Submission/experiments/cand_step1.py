"""Step 1: score candidate runs against the control (three seeds, single network) on the fresh set.
"Fresh mean" = an09 to an16 noise-only (_ai) + the 8 Young TEST3 discs; twins = an09k / an12k / an15k, both regimes, invented PV5 magnitude.
Run from this folder: python experiments/cand_step1.py [CAND_NAME=eval1.json,eval2.json ...]"""
import glob, json, sys, numpy as np

def run_dir(name):
    """Folder of a run under results/NN-.../ (runs are grouped by result)."""
    return glob.glob(f"results/[0-9]*/{name}")[0]


CTRL = sorted(glob.glob(f"{run_dir('eval-fresh-control_v1_2026-10-22')}/eval_fresh_control/eval_ctrl_r4n_s*.json"))
B = [f"{run_dir('eval-fresh-b_v2_2026-10-21')}/eval_fresh_B.json", f"{run_dir('eval-fresh-young_v1_2026-10-21')}/eval_fresh_young_B.json"]


def is_twin(k):
    return k.split("@")[0].split("_")[0].endswith("k")


def score(paths):
    d = {}
    for p in paths:
        d.update(json.load(open(p)))
    fresh = [k for k in d if k.startswith("run_young_") or ("_ai@" in k and not is_twin(k))]
    twins = [k for k in d if is_twin(k)]
    assert len(fresh) == 16 and len(twins) == 6, (len(fresh), len(twins), paths)
    f = lambda ks, g: float(np.mean([g(d[k]) for k in ks]))
    return {"rc_r": f(fresh, lambda c: c["wiggle"]["model"]["rc_r"]), "psnr": f(fresh, lambda c: c["model"]["psnr"]),
            **{m: f(fresh, lambda c, m=m: c["model"][m]) for m in ("M0", "M1", "M2")},
            "twin_pv5_mag": f(twins, lambda c: c["wiggle"]["model"]["pv5_mag_ms"]),
            "wiener_rc_r": f(fresh, lambda c: c["wiggle"]["wiener"]["rc_r"])}


def verdict(s, c):
    r1 = s["rc_r"] - c["rc_r"] >= 0.05
    r2 = abs(s["psnr"] - c["psnr"]) <= 0.5 and all(abs(s[m] - c[m]) <= 2 for m in ("M0", "M1", "M2"))
    r3 = s["twin_pv5_mag"] - c["twin_pv5_mag"] <= 3
    return r1, r2, r3


rows = {f"control {p.split('_')[-1][:-5]}": score([p]) for p in CTRL}
ctrl = {k: float(np.mean([r[k] for r in rows.values()])) for k in next(iter(rows.values()))}
rows["control mean"] = ctrl
rows["recipe B (ensemble)"] = score(B)
cands = [a.split("=") for a in sys.argv[1:]]
for name, ps in cands:
    rows[name] = score(ps.split(","))
print("| run | rc_r fresh | PSNR | M0 / M1 / M2 | twin PV5 mag (m/s) | Wiener rc_r | rules 1 / 2 / 3 |")
print("|---|---|---|---|---|---|---|")
for name, s in rows.items():
    v = " / ".join("pass" if x else "FAIL" for x in verdict(s, ctrl)) if name in dict(cands) else ""
    print(f"| {name} | {s['rc_r']:.3f} | {s['psnr']:.2f} | {s['M0']:.1f} / {s['M1']:.1f} / {s['M2']:.1f} | {s['twin_pv5_mag']:.1f} | {s['wiener_rc_r']:.3f} | {v} |")

if __name__ == "__main__" and not cands:   # self-check: a copy of the control passes rule 2 and 3, fails rule 1
    assert verdict(ctrl, ctrl) == (False, True, True)
