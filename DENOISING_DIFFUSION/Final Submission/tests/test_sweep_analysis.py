#!/usr/bin/env python
"""
Tests for the sweep correlation analysis.

The load-bearing case is [4]: a predictor that is related to the target ONLY
through the control must have its partial correlation collapse to ~0. If that
fails, the module cannot support its central claim -- that `alpha`'s apparent
importance was partly training duration in disguise.

  1. partial_corr reproduces a hand-checkable value.
  2. partial_corr is NaN when the control explains a variable perfectly.
  3. a genuine direct effect SURVIVES the control (guards against a method that
     just shrinks everything toward zero).
  4. a pure confound is dissolved by the control.
  5. a SUPPRESSED effect is revealed by the control (the base_channels case).
  6. load_sweep_csv drops the "failed" rows run_sweep writes on OOM.
  7. load_sweep_csv derives log10_lr and casts use_beam to 0/1.
  8. analyse ranks the control against another predictor, not itself.
  9. format_report states n and keeps the small-sample caveat attached.
"""

import csv
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.sweep_analysis import (analyse, format_report, load_sweep_csv,
                                          partial_corr, _pearson)

print("=" * 66)
print("Sweep correlation analysis test")
print("=" * 66)

# [1] hand-checkable: y = z exactly, x independent-ish -> partial(x,y|z) is 0/0 guarded
z = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.0, 4.0, 6.0, 8.0, 10.0]        # y = 2z, perfectly collinear with z
x = [5.0, 3.0, 4.0, 1.0, 2.0]
assert not math.isfinite(partial_corr(x, y, z)), "perfect collinearity must not yield a number"
print("[1]/[2] perfect collinearity with the control -> NaN, not a spurious value")

# Synthetic data for [3]-[5]. All three carry genuine noise: a target that is an
# exact linear function of the control is the degenerate case guarded in [2], not a
# confound, and would make these cases vacuous.
import random

rnd = random.Random(0)
n = 200
zz = [rnd.gauss(0, 1) for _ in range(n)]                       # the control
noise = lambda s: [rnd.gauss(0, s) for _ in range(n)]

# [3] a real direct effect must SURVIVE the control.
# `direct` is independent of the control; both feed the target at similar strength.
direct = [rnd.gauss(0, 1) for _ in range(n)]
e3 = noise(0.5)
target = [1.0 * direct[i] + 1.0 * zz[i] + e3[i] for i in range(n)]
raw = _pearson(direct, target)
par = partial_corr(direct, target, zz)
assert par > raw and par > 0.7, f"direct effect not preserved: raw {raw:+.3f} partial {par:+.3f}"
print(f"[3] direct effect survives the control: raw {raw:+.3f} -> partial {par:+.3f}")

# [4] THE KEY CASE -- a pure confound must collapse.
# `conf` is related to the target ONLY through the control: it tracks zz, and the
# target is driven by zz plus its own independent noise.
e4a, e4b = noise(0.3), noise(1.0)
conf = [zz[i] + e4a[i] for i in range(n)]
target_via_z = [3.0 * zz[i] + e4b[i] for i in range(n)]
raw_c = _pearson(conf, target_via_z)
par_c = partial_corr(conf, target_via_z, zz)
assert abs(raw_c) > 0.7, f"confound should look strong before control: {raw_c:+.3f}"
assert abs(par_c) < 0.2, f"confound was not dissolved: raw {raw_c:+.3f} partial {par_c:+.3f}"
print(f"[4] pure confound dissolved: raw {raw_c:+.3f} -> partial {par_c:+.3f}")

# [5] a SUPPRESSED effect must be revealed -- the base_channels case, where wider
# models genuinely help but early-stop sooner, so the raw correlation understates them.
e5a, e5b = noise(1.0), noise(0.5)
width = [rnd.gauss(0, 1) for _ in range(n)]
ctrl = [-2.0 * width[i] + e5a[i] for i in range(n)]            # anti-correlated with width
tgt = [1.5 * width[i] + 1.0 * ctrl[i] + e5b[i] for i in range(n)]
raw_w = _pearson(width, tgt)
par_w = partial_corr(width, tgt, ctrl)
assert par_w > raw_w + 0.2, f"suppression not revealed: raw {raw_w:+.3f} partial {par_w:+.3f}"
print(f"[5] suppressed effect revealed: raw {raw_w:+.3f} -> partial {par_w:+.3f}")

# [6]/[7] CSV parsing: drop failed rows, derive log10_lr and 0/1 beam
tmp = tempfile.mkdtemp(prefix="exxa-sweepan-")
csv_path = os.path.join(tmp, "s.csv")
fields = ["run", "base_channels", "channel_multipliers", "lr", "alpha", "sched_patience",
          "use_beam", "batch_size", "best_epoch", "epochs_run", "best_val_loss",
          "psnr", "ssim", "mse", "wall_time_s"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for i in range(4):
        w.writerow({"run": i, "base_channels": 16 + 16 * i, "channel_multipliers": "1x2x4",
                    "lr": 0.001, "alpha": 0.5 + 0.1 * i, "sched_patience": 5,
                    "use_beam": "True" if i % 2 else "False", "batch_size": 32,
                    "best_epoch": 20 + i, "epochs_run": 25 + i, "best_val_loss": 0.003,
                    "psnr": 30.0 + i, "ssim": 0.98, "mse": 0.0007, "wall_time_s": 500})
    # the crash row run_sweep writes: metrics blank, best_val_loss == "failed"
    w.writerow({"run": 99, "base_channels": 64, "channel_multipliers": "1x2x4", "lr": 0.001,
                "alpha": 0.7, "sched_patience": 5, "use_beam": "False", "batch_size": 8,
                "best_epoch": "", "epochs_run": "", "best_val_loss": "failed",
                "psnr": "", "ssim": "", "mse": "", "wall_time_s": ""})

rows = load_sweep_csv(csv_path)
assert len(rows) == 4, f"failed row not dropped: {len(rows)}"
assert all(r["run"] != 99 for r in rows)
print(f"[6] load_sweep_csv kept {len(rows)} completed runs and dropped the failed row")

assert abs(rows[0]["log10_lr"] - math.log10(0.001)) < 1e-12, rows[0]["log10_lr"]
assert sorted({r["use_beam"] for r in rows}) == [0.0, 1.0], {r["use_beam"] for r in rows}
print("[7] derived log10_lr and cast use_beam to 0/1")

# [8] the control is ranked against another predictor, not itself
res = analyse(rows, target="psnr")
assert "epochs_run" in res and res["epochs_run"]["partial"] is not None
assert res["epochs_run"].get("partial_against") not in (None, "epochs_run"), res["epochs_run"]
print(f"[8] control ranked against '{res['epochs_run']['partial_against']}', not itself")

# [9] the report carries n and the caveat -- the numbers must not travel without it
rep = format_report(rows, res, target="psnr")
assert "n = 4" in rep and "CAVEAT" in rep and "significance tests" in rep, rep
print("[9] report states n and keeps the small-sample caveat")

import shutil
shutil.rmtree(tmp, ignore_errors=True)

print("\n" + "=" * 66)
print("All sweep-analysis tests PASSED")
print("=" * 66)
