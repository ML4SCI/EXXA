#!/usr/bin/env python
"""
src/evaluation/sweep_analysis.py
================================
Correlation analysis for hyperparameter-sweep results, with a control for how
long each run actually trained.

Why the control matters
-----------------------
The 12-run sweep reported that `alpha` correlates most strongly with PSNR
(r = +0.62) and concluded the Bayesian follow-up should focus there. That reading
ignores `epochs_run`: with early stopping, two runs of the same configuration can
train for very different numbers of epochs, and longer runs score better
(r = +0.65 between `epochs_run` and PSNR -- larger than any hyperparameter).

`alpha` and `epochs_run` are themselves correlated (r = +0.54), so the raw
correlations are confounded. Partialling out `epochs_run` moves the ranking
substantially: `base_channels` rises from +0.43 to +0.82 (its effect was masked,
because wider models early-stopped sooner) while `alpha` falls from +0.62 to
+0.43. Model width, not loss weighting, looks like the dominant driver.

Version 16 made the same point the hard way: the sweep's best configuration
scored 37.11 dB during the sweep and 30.28 dB when retrained under a different
seed, because early stopping fired at epoch 21 instead of 38.

Caveat that belongs with every number this module prints: n is 12. Partial
correlations on 12 points are noisy, and nothing here is a significance test.
The intended use is to decide what the next sweep should vary, and to stop the
"alpha matters most" claim being repeated as established.
"""

import csv
import math
from typing import Dict, List, Optional, Sequence

# Columns treated as candidate predictors. `epochs_run` is included so it can be
# ranked alongside the hyperparameters -- the whole point is that it competes.
DEFAULT_PREDICTORS = ("alpha", "epochs_run", "base_channels", "log10_lr", "use_beam")


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson r; NaN when either input has no spread."""
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def partial_corr(x: Sequence[float], y: Sequence[float], z: Sequence[float]) -> float:
    """
    Correlation between x and y with the linear effect of z removed.

    Standard first-order partial correlation:
        (r_xy - r_xz r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))

    Returns NaN when z explains x or y almost perfectly, because the partial is
    then numerically meaningless. The guard is applied to each correlation
    separately rather than to the product: at r_yz = 1 - 1e-16 the product still
    rounds to ~1e-8, comfortably above any small epsilon, and the function would
    hand back a number like -1.8e-8 that looks like a real "no effect" result
    instead of an undefined one.
    """
    rxy, rxz, ryz = _pearson(x, y), _pearson(x, z), _pearson(y, z)
    if not all(map(math.isfinite, (rxy, rxz, ryz))):
        return float("nan")
    # 1 - r^2 < 1e-10 means |r| > 0.99999999995: the control and the variable are
    # collinear to within floating-point noise. A legitimately strong r = 0.999
    # leaves 1 - r^2 = 2e-3 and passes.
    if (1 - rxz ** 2) < 1e-10 or (1 - ryz ** 2) < 1e-10:
        return float("nan")
    denom = math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / denom


def load_sweep_csv(path: str) -> List[dict]:
    """
    Read a `run_sweep` CSV, keeping only completed runs and deriving predictors.

    Rows whose `best_val_loss` is the literal "failed" (how `run_sweep` records an
    OOM or crash) are dropped -- they carry no metrics and would poison the means.
    Adds `log10_lr` because learning rate is sampled log-uniformly, so its linear
    correlation is only meaningful in log space, and casts `use_beam` to 0/1.
    """
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("best_val_loss", "")).strip().lower() == "failed":
                continue
            try:
                out = {
                    "run": int(r["run"]),
                    "psnr": float(r["psnr"]),
                    "ssim": float(r["ssim"]),
                    "mse": float(r["mse"]),
                    "alpha": float(r["alpha"]),
                    "epochs_run": float(r["epochs_run"]),
                    "best_epoch": float(r["best_epoch"]),
                    "base_channels": float(r["base_channels"]),
                    "log10_lr": math.log10(float(r["lr"])),
                    "use_beam": 1.0 if str(r["use_beam"]).strip() == "True" else 0.0,
                    "channel_multipliers": r.get("channel_multipliers", ""),
                }
            except (KeyError, ValueError):
                continue
            rows.append(out)
    return rows


def analyse(
    rows: List[dict],
    target: str = "psnr",
    predictors: Sequence[str] = DEFAULT_PREDICTORS,
    control: Optional[str] = "epochs_run",
) -> Dict[str, dict]:
    """
    Rank predictors by raw and control-adjusted correlation with `target`.

    Args:
        rows: output of `load_sweep_csv`.
        target: metric to correlate against (the fixed sweep metric, normally psnr).
        predictors: columns to rank.
        control: column whose linear effect is partialled out; None to skip.
            Defaults to `epochs_run`, the training-duration confound.

    Returns:
        {predictor: {"raw": r, "partial": r_or_None, "shift": partial-raw}}
        The control column itself is reported with its partial taken against the
        strongest other predictor, so it can be ranked without being compared to
        itself.
    """
    y = [r[target] for r in rows]
    out: Dict[str, dict] = {}

    for p in predictors:
        if p not in rows[0]:
            continue
        x = [r[p] for r in rows]
        raw = _pearson(x, y)
        if control is None or p == control:
            out[p] = {"raw": raw, "partial": None, "shift": None}
        else:
            par = partial_corr(x, y, [r[control] for r in rows])
            out[p] = {"raw": raw, "partial": par,
                      "shift": (par - raw) if math.isfinite(par) else None}

    # rank the control against the strongest remaining predictor, so it is not
    # left unranked just because it is the thing being controlled for
    if control and control in out:
        others = [(p, v["raw"]) for p, v in out.items()
                  if p != control and math.isfinite(v["raw"])]
        if others:
            strongest = max(others, key=lambda kv: abs(kv[1]))[0]
            par = partial_corr([r[control] for r in rows], y, [r[strongest] for r in rows])
            out[control]["partial"] = par
            out[control]["partial_against"] = strongest
            out[control]["shift"] = (par - out[control]["raw"]) if math.isfinite(par) else None
    return out


def format_report(rows: List[dict], result: Dict[str, dict], target: str = "psnr",
                  control: Optional[str] = "epochs_run") -> str:
    """Render `analyse` output as a fixed-width table with the caveat attached."""
    n = len(rows)
    lines = [
        f"sweep correlation analysis  (n = {n} completed runs, target = {target})",
        "",
        f"{'predictor':<18}{'raw r':>10}{'partial r':>12}{'shift':>10}",
        "-" * 50,
    ]
    ranked = sorted(result.items(),
                    key=lambda kv: abs(kv[1]["partial"] if kv[1]["partial"] is not None
                                       and math.isfinite(kv[1]["partial"]) else kv[1]["raw"]),
                    reverse=True)
    for p, v in ranked:
        par = "-" if v["partial"] is None or not math.isfinite(v["partial"]) else f"{v['partial']:+.3f}"
        sh = "-" if v["shift"] is None else f"{v['shift']:+.3f}"
        note = ""
        if p == control and v.get("partial_against"):
            note = f"  (vs {v['partial_against']})"
        lines.append(f"{p:<18}{v['raw']:>+10.3f}{par:>12}{sh:>10}{note}")
    lines += [
        "-" * 50,
        f"partial r removes the linear effect of '{control}'." if control else "",
        "",
        f"CAVEAT: n = {n}. Partial correlations on this few points are noisy and none",
        "of these are significance tests. Use them to choose what the next sweep",
        "varies, not as established effect sizes.",
    ]
    return "\n".join(l for l in lines if l is not None)


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--csv", default="results/sweep_results.csv")
    ap.add_argument("--target", default="psnr")
    ap.add_argument("--out", default=None, help="optional CSV to write the table to")
    a = ap.parse_args()

    rows = load_sweep_csv(a.csv)
    if not rows:
        raise SystemExit(f"no completed runs found in {a.csv}")
    res = analyse(rows, target=a.target)
    print(format_report(rows, res, target=a.target))

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["predictor", "raw_r", "partial_r_controlling_epochs_run", "shift", "n"])
            for p, v in res.items():
                w.writerow([p, round(v["raw"], 4),
                            "" if v["partial"] is None else round(v["partial"], 4),
                            "" if v["shift"] is None else round(v["shift"], 4),
                            len(rows)])
        print(f"\nsaved -> {a.out}")
