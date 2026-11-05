#!/usr/bin/env python
"""
Tests for `repeat_config` -- the multi-seed harness that puts error bars on a
configuration, so "37.11 dB beats 32.95 dB" can be checked against seed noise.

  1. it runs exactly n_seeds times and reports the seeds it used.
  2. different seeds actually produce different models (otherwise the std is a
     meaningless zero and the harness is decorative).
  3. std is the ddof=1 sample std of the per-seed values.
  4. a single seed yields std 0.0 rather than NaN or a crash.
  5. the crash-safe CSV gets one row per seed, with a header.
  6. per-seed checkpoints are written so moment maps can be run per seed.
  7. model objects are dropped from the returned rows (holding n_seeds models is
     what makes a repeat loop OOM).
  8. the harness is reproducible: same base_seed -> same metrics.
"""

import csv
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.sweep import repeat_config

print("=" * 64)
print("Multi-seed repeat harness test")
print("=" * 64)


class TinyDS(torch.utils.data.Dataset):
    """Small deterministic (dirty, clean) pairs -- enough to train 2 epochs on CPU."""

    def __init__(self, n=8, size=32):
        g = torch.Generator().manual_seed(0)
        self.clean = torch.rand(n, 1, size, size, generator=g)
        self.dirty = (self.clean + 0.1 * torch.randn(n, 1, size, size, generator=g)).clamp(0, 1)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, i):
        return self.dirty[i], self.clean[i]


ds = TinyDS()
device = torch.device("cpu")
CFG = dict(base_channels=8, channel_multipliers=(1, 2), lr=1e-3, alpha=0.8,
           batch_size=4, min_epochs=1, max_epochs=2, patience=1)

tmp = tempfile.mkdtemp(prefix="exxa-repeat-")
try:
    csv_path = os.path.join(tmp, "repeats.csv")
    ckpt_dir = os.path.join(tmp, "ckpts")

    out = repeat_config(ds, ds, device, n_seeds=3, base_seed=100, out_csv=csv_path,
                        ckpt_dir=ckpt_dir, tag="unit", verbose=False, **CFG)

    # [1] correct number of runs, correct seeds
    assert out["n_seeds"] == 3 and len(out["rows"]) == 3, out["n_seeds"]
    assert [r["seed"] for r in out["rows"]] == [100, 101, 102], out["rows"]
    print(f"[1] ran 3 seeds: {[r['seed'] for r in out['rows']]}")

    # [2] seeds must actually diverge -- identical PSNRs would mean the seed is ignored
    psnrs = out["psnr"]["values"]
    assert len(set(round(p, 9) for p in psnrs)) > 1, f"seed had no effect: {psnrs}"
    print(f"[2] seeds diverge: PSNRs {', '.join(f'{p:.4f}' for p in psnrs)}")

    # [3] std is the ddof=1 sample std
    assert abs(out["psnr"]["std"] - float(np.std(psnrs, ddof=1))) < 1e-9, out["psnr"]
    assert abs(out["psnr"]["mean"] - float(np.mean(psnrs))) < 1e-9, out["psnr"]
    print(f"[3] mean/std correct (ddof=1): {out['psnr']['mean']:.4f} "
          f"+/- {out['psnr']['std']:.4f}")

    # [4] n_seeds=1 -> std 0.0, not NaN
    one = repeat_config(ds, ds, device, n_seeds=1, base_seed=7, tag="single",
                        verbose=False, **CFG)
    assert one["psnr"]["std"] == 0.0 and np.isfinite(one["psnr"]["mean"]), one["psnr"]
    print(f"[4] single seed: std {one['psnr']['std']} (finite mean "
          f"{one['psnr']['mean']:.4f})")

    # [5] crash-safe CSV: header plus one row per seed
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3, f"expected 3 CSV rows, got {len(rows)}"
    assert rows[0]["tag"] == "unit" and rows[0]["seed"] == "100", rows[0]
    assert all(r["psnr"] for r in rows), rows
    print(f"[5] CSV has {len(rows)} rows with populated metrics")

    # [6] one checkpoint per seed, and it is loadable
    for s in (100, 101, 102):
        p = os.path.join(ckpt_dir, f"unit_seed{s}.pth")
        assert os.path.exists(p), f"missing checkpoint {p}"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ck and "epoch" in ck, list(ck)
    print("[6] per-seed checkpoints written and loadable (moment maps can run per seed)")

    # [7] no model objects retained (OOM guard)
    assert all("model" not in r for r in out["rows"]), "model leaked into rows"
    print("[7] model objects dropped from returned rows")

    # [8] reproducible for a fixed base_seed
    again = repeat_config(ds, ds, device, n_seeds=2, base_seed=100, tag="again",
                          verbose=False, **CFG)
    assert np.allclose(again["psnr"]["values"], psnrs[:2], atol=1e-6), \
        (again["psnr"]["values"], psnrs[:2])
    print("[8] reproducible: same base_seed reproduces the same per-seed metrics")

    # [9] schedule kwargs (min_epochs / max_epochs / patience) forward to train_unet.
    # Notebook 08 puts the schedule INSIDE each config dict so its `winner_p10` arm can
    # differ from `winner` on patience alone. If **config silently dropped or overrode
    # patience, that arm would be a duplicate of `winner` and the early-stopping question
    # would go unanswered while still costing an hour of GPU. Asserted on the forwarded
    # kwargs rather than on epochs_run, because a run that never plateaus stops at
    # max_epochs under any patience and would make the check vacuously pass.
    from src.training import sweep as _sweep

    seen = []
    _real = _sweep.train_unet

    def _spy(train_ds, val_ds, dev, **kw):
        seen.append(kw)
        return {"model": None, "best_val_loss": 0.1, "best_epoch": 3, "epochs_run": 5,
                "train_losses": [1.0, 0.5], "val_losses": [1.0, 0.5],
                "wall_time_s": 1.0, "psnr": 30.0, "ssim": 0.9, "mse": 0.001}

    _sweep.train_unet = _spy
    try:
        sched = dict(min_epochs=20, max_epochs=60, patience=5)
        arm_a = dict(CFG, **sched)
        arm_b = dict(arm_a, patience=10)
        _sweep.repeat_config(ds, ds, device, n_seeds=1, base_seed=1, tag="a",
                             verbose=False, **arm_a)
        _sweep.repeat_config(ds, ds, device, n_seeds=1, base_seed=1, tag="b",
                             verbose=False, **arm_b)
    finally:
        _sweep.train_unet = _real

    assert seen[0]["patience"] == 5 and seen[1]["patience"] == 10, seen
    assert seen[0]["max_epochs"] == 60 and seen[0]["min_epochs"] == 20, seen[0]
    differing = {k for k in set(seen[0]) | set(seen[1])
                 if seen[0].get(k) != seen[1].get(k)}
    assert differing == {"patience"}, f"arms differ on more than patience: {differing}"
    print("[9] schedule kwargs forward; a patience-only arm differs on exactly one axis")

    # [10] resume: a session killed part-way must not retrain finished seeds.
    # The 12-run repeat is several GPU-hours and Kaggle times sessions out, so this
    # is the difference between losing one arm and losing the whole run.
    rcsv = os.path.join(tmp, "resume.csv")
    rck = os.path.join(tmp, "rck")
    first = repeat_config(ds, ds, device, n_seeds=2, base_seed=42, out_csv=rcsv,
                          ckpt_dir=rck, tag="armA", verbose=False, **CFG)
    second = repeat_config(ds, ds, device, n_seeds=3, base_seed=42, out_csv=rcsv,
                           ckpt_dir=rck, tag="armA", verbose=False, **CFG)
    assert len(second["rows"]) == 3, second["rows"]
    # the reused rows must carry the ORIGINAL metrics -- if they were silently
    # retrained the numbers would drift and the "resume" would be a lie
    for r0 in first["rows"]:
        got = [r for r in second["rows"] if r["seed"] == r0["seed"]][0]
        assert abs(got["psnr"] - r0["psnr"]) < 1e-9, (r0["psnr"], got["psnr"])
    # a different tag must NOT be skipped just because seeds collide
    other = repeat_config(ds, ds, device, n_seeds=1, base_seed=42, out_csv=rcsv,
                          ckpt_dir=rck, tag="armB", verbose=False, **CFG)
    assert len(other["rows"]) == 1 and other["rows"][0]["tag"] == "armB", other["rows"]
    # resume=False forces a genuine re-run
    forced = repeat_config(ds, ds, device, n_seeds=1, base_seed=42, out_csv=rcsv,
                           ckpt_dir=rck, tag="armA", resume=False, verbose=False, **CFG)
    assert len(forced["rows"]) == 1
    print(f"[10] resume reused {len(first['rows'])} seeds with identical metrics; "
          "tags isolated; resume=False still retrains")

    print("\n" + "=" * 64)
    print("All multi-seed repeat tests PASSED")
    print("=" * 64)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
