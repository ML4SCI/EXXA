"""
One report for the FROZEN final model, from eval JSONs written by train_unified.py.

  python experiments/final_report.py --final final=eval_final.json --seeds s3=a.json s11=b.json s23=c.json --out results/final/RESULTS.md
         [--ladder final=eval_final.json] [--gate gate=eval_gate.json] [--apply-gate 40,120,0.25]

For each of the five sizes: the final (averaged) model next to dirty and Wiener (their rows come from the final file), per group (line, hydro, analytic, sgv2,
young_test, speedie); then, per group and size, the RANGE of every metric across the individual seeds (min .. max), because a single seed is not evidence. The SNR ladder (3 / 6 / 12 / 24) comes from ladder_report.py's table,
and the gate sweep (a * model + (1 - a) * Wiener) is printed per case if the file has it. Every number is a mean over the cubes of the group, never over seeds.
"""
import argparse, json, os, subprocess, sys, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_models as cm

SIZES = (256, 300, 320, 480, 600)


def gated(res, gate):
    """Apply the do-no-harm gate to a result file that stored a * model + (1 - a) * Wiener for a in {0.25, 0.5, 0.75} (`--gate-sweep`): per case, a = gate_alpha(snr_est);
    the stored mix nearest to a replaces the model's moments and wiggle. Cases without stored mixes keep the plain model. Returns (new results, list of changes)."""
    lo, hi, floor = gate
    out, changes = {}, []
    for key, v in res.items():
        if "failed" in v or "mix" not in v or "snr_est" not in v:
            out[key] = v; continue
        a = float(min(1.0, max(floor, 1.0 - (1.0 - floor) * (v["snr_est"] - lo) / (hi - lo))))
        stored = sorted(float(x) for x in v["mix"])
        near = min(stored + [1.0], key=lambda x: abs(x - a))
        if near == 1.0:
            out[key] = v; continue
        w = dict(v); w["model"] = {**v["model"], **v["mix"][f"{near:g}"]}
        w["wiggle"] = {**v["wiggle"], "model": v["wiggle"][f"mix{near:g}"]}
        out[key] = w
        changes.append((key, v["snr_est"], a, near))
    return out, changes


def load(spec):
    return {n: json.load(open(p)) for n, p in (x.split("=", 1) for x in spec)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", nargs=1, required=True)
    ap.add_argument("--seeds", nargs="*", default=[])
    ap.add_argument("--ladder", nargs="*", default=[])
    ap.add_argument("--gate", nargs="*", default=[])
    ap.add_argument("--apply-gate", default="", help="lo,hi,floor: apply the do-no-harm gate to the final file (needs --gate-sweep mixes)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    final, seeds = load(a.final), load(a.seeds)
    (fname, fres), = final.items()
    changes = []
    if a.apply_gate:
        fres, changes = gated(fres, tuple(float(x) for x in a.apply_gate.split(",")))
        seeds = {}                                  # per-seed files have no mixes: their range is for the PLAIN model, say so in the table
        fname = fname + "+gate"
    L = [f"# Final model `{fname}`: results at all five sizes", "",
         "Moments are signal-masked, noise-clipped improvements over dirty. `model` = the averaged final model; dirty and Wiener rows are computed on the same cubes.", ""]
    for size in SIZES:
        refs = {"dirty": cm.summarise(fres, size, "dirty"), "wiener": cm.summarise(fres, size, "wiener"), fname: cm.summarise(fres, size, "model")}
        L += [f"## {size} px", ""]
        for g in cm.GROUPS:
            if not any(g in t for t in refs.values()):
                continue
            L += [f"### {g}", "", "| model | n | " + " | ".join(l for _, l in cm.METRICS) + " |", "|" + "---|" * (len(cm.METRICS) + 2)]
            for name, t in refs.items():
                if g in t:
                    r = t[g]
                    L.append(f"| {name} | {r['n']} | " + " | ".join(cm.fmt(r[k]) for k, _ in cm.METRICS) + " |")
            if seeds:
                rng = {}
                for sn, sres in seeds.items():
                    t = cm.summarise(sres, size, "model").get(g)
                    if t:
                        for k, _ in cm.METRICS:
                            if t[k] is not None:
                                rng.setdefault(k, []).append(t[k])
                if rng:
                    L.append(f"| seed range ({len(seeds)} seeds) | | " + " | ".join(f"{min(rng[k]):.2f} .. {max(rng[k]):.2f}" if k in rng else "-" for k, _ in cm.METRICS) + " |")
            L.append("")
    if changes:
        L += ["## Where the gate changed the model", "", "| case | blind SNR est | gate a | mix used |", "|---|---|---|---|"]
        L += [f"| {k} | {e:.0f} | {x:.2f} | {n:g} |" for k, e, x, n in sorted(changes, key=lambda c: -c[1])] + [""]
    for spec in a.ladder:
        n, p = spec.split("=", 1)
        L += [f"## SNR ladder (Speedie GI r0.5 with its own noise rescaled): `{n}`", "", subprocess.run([sys.executable, os.path.join(HERE, "ladder_report.py"), f"{n}={p}"],
                                                                                                          capture_output=True, text=True).stdout, ""]
    for n, res in load(a.gate).items():
        L += [f"## Gate sweep: `{n}`", "", "| case | blind SNR est | a | M0 | M1 | M2 | PSNR |", "|---|---|---|---|---|---|---|"]
        for key, v in sorted(res.items()):
            if "failed" in v or "mix" not in v:
                continue
            rows = [("1 (model)", v["model"])] + [(a_, m) for a_, m in sorted(v["mix"].items())] + [("0 (Wiener)", v["wiener"])]
            for a_, m in rows:
                L.append(f"| {key} | {v.get('snr_est', float('nan')):.0f} | {a_} | {m.get('M0', float('nan')):.0f} | {m.get('M1', float('nan')):.0f} | {m.get('M2', float('nan')):.0f} | {m.get('psnr', float('nan')):.1f} |")
        L.append("")
    txt = "\n".join(L)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w").write(txt + "\n")
    print(txt[:3000])


if __name__ == "__main__":
    main()
