"""
One table for every model the project has trained, from the eval_*.json files that train_unified.py writes (one per model, one key per
`case@size`). Groups of held-out cubes, per size:

  line      the 5 line-emission holdouts (run_00xx): PSNR, M0/M1/M2 %, rotation-curve wiggle, line-centre error
  hydro     9074 (deconvolution task) and 9074_ai (noise-only task)
  analytic  an07, an08 (both tasks)
  sgv2      SG v2
  speedie   GI disk and its Keplerian twin at native pixels (the twin must stay WIGGLE-FREE: reported as the model's excess rotation-curve rms)

For every model the same file also holds dirty and Wiener on the same cubes, so those are added as reference rows once.
Selection rule: by wiggle, line-centre error and M0/M1/M2 on held-out cubes, never by PSNR alone.

Usage: python experiments/compare_models.py --size 600 name=path/eval_x.json [name2=path2.json ...] [--out results/comparison.md]
"""
import json, sys, argparse, statistics as st

GROUPS = {"line": lambda c: c.startswith("run_00"), "hydro": lambda c: c in ("9074", "9074_ai"),
          "analytic": lambda c: c in ("an07", "an08", "an07_ai", "an08_ai"), "sgv2": lambda c: c == "sgv2",
          "young_test": lambda c: c.startswith("run_young_"),      # held-out WHOLE Young et al. 2026 simulations (R5 and later evals)
          "speedie_GI": lambda c: c.startswith("speedie_GI"), "speedie_Kep": lambda c: c.startswith("speedie_Kep")}


def cases_at(res, size):
    out = {}
    for key, v in res.items():
        if "failed" in v:
            continue
        case, s = key.rsplit("@", 1)
        if case.startswith("speedie_") or int(s) == size:
            out[case] = v
    return out


def pick(v, who, k):
    """metric k of `who` ('model' | 'wiener' | 'dirty') in one case record."""
    if k in ("psnr", "M0", "M1", "M2"):
        if who == "dirty":
            return v["dirty_psnr"] if k == "psnr" else 0.0
        return v[who].get(k)
    g = v["wiggle"].get(who)
    return None if g is None else g.get(k)


METRICS = [("psnr", "PSNR"), ("M0", "M0%"), ("M1", "M1%"), ("M2", "M2%"), ("rc_r", "rc_r"), ("spine_r", "spine_r"), ("dv_med_ms", "dv m/s")]   # spine_r: Terry et al. 2022 channel-map spine


def summarise(res, size, who):
    rows = {}
    cs = cases_at(res, size)
    for g, f in GROUPS.items():
        sel = [v for c, v in cs.items() if f(c)]
        if not sel:
            continue
        row = {"n": len(sel)}
        for k, _ in METRICS:
            vals = [x for x in (pick(v, who, k) for v in sel) if x is not None and x == x]
            row[k] = st.mean(vals) if vals else None
        if g == "speedie_Kep":     # excess rotation-curve rms over what clean has (clean is the floor): spurious wiggle in m/s
            ex = [v["wiggle"][who]["rc_rms_ms"] - v["wiggle"][who]["rc_rms_clean_ms"] for v in sel
                  if v["wiggle"].get(who) and "rc_rms_ms" in v["wiggle"][who]]
            row["spurious_ms"] = st.mean(ex) if ex else None
        rows[g] = row
    return rows


def fmt(x, w=6, d=2):
    return f"{x:{w}.{d}f}" if isinstance(x, (int, float)) else " " * (w - 1) + "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=path/to/eval.json")
    ap.add_argument("--size", type=int, default=600)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    data = {}
    for m in a.models:
        name, path = m.split("=", 1)
        data[name] = json.load(open(path))
    lines = [f"# Model comparison at {a.size} px (speedie cases at native pixels)", ""]
    first = next(iter(data.values()))
    refs = {"dirty": summarise(first, a.size, "dirty"), "wiener": summarise(first, a.size, "wiener")}
    table = dict(refs)
    for name, res in data.items():
        table[name] = summarise(res, a.size, "model")
    for g in GROUPS:
        if not any(g in t for t in table.values()):
            continue
        lines += [f"## {g}", "", "| model | n | " + " | ".join(lbl for _, lbl in METRICS) + (" | spurious m/s |" if g == "speedie_Kep" else " |"),
                  "|" + "---|" * (len(METRICS) + 2 + (1 if g == "speedie_Kep" else 0))]
        for name, t in table.items():
            if g not in t:
                continue
            r = t[g]
            lines.append(f"| {name} | {r['n']} | " + " | ".join(fmt(r[k]) for k, _ in METRICS) + (f" | {fmt(r.get('spurious_ms'))} |" if g == "speedie_Kep" else " |"))
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
