#!/usr/bin/env python
"""
src/data/cube_split.py
======================
Cube-level (NOT channel-level) train / val / holdout split for line-emission data.

Rules for the split:
  - Split at the CUBE level, never the channel level.
  - Hold out 3-4 ENTIRE cubes for inference only: never in train OR val. These are
    reserved for end-to-end testing: denoise every channel, then build moment maps.

Leakage guard: grouping by RunID:
  Each RunID can have several RT variants (e.g. run_0002 -> rt_00, rt_01, rt_04),
  which are near-identical parameter variants of the same
  underlying simulation. To avoid train/holdout leakage, ALL cubes sharing a RunID are
  assigned to the SAME split (group-level split). So `n_holdout` counts RunID groups
  held out, and every cube under those RunIDs is reserved.

A "cube" here is one (clean, dirty) FITS pair living in a run subfolder.
"""

import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

NAME_RE = re.compile(r"run_(\d+)_(\d+)_rt_(\d+)", re.IGNORECASE)


def _find_pair(folder_path: str) -> Tuple[str, str]:
    """Return (clean_path, dirty_path) for a run subfolder; '' if missing."""
    clean = dirty = ""
    for fn in sorted(os.listdir(folder_path)):
        low = fn.lower()
        if not (low.endswith(".fits") or low.endswith(".fit")):
            continue
        full = os.path.join(folder_path, fn)
        if "clean" in low:
            clean = full
        elif "dirty" in low:
            dirty = full
    return clean, dirty


def list_cubes(data_dir: str) -> List[dict]:
    """Discover all valid cube pairs under `data_dir`.

    Returns a list of dicts: {folder, run_id, clean, dirty}. Folders missing either
    file are skipped (with a warning).
    """
    cubes = []
    for folder in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, folder)
        if not os.path.isdir(fpath):
            continue
        m = NAME_RE.search(folder)
        if not m:
            print(f"[cube_split] WARN: folder '{folder}' does not match run_<id>_<step>_rt_<pp>; skipping")
            continue
        clean, dirty = _find_pair(fpath)
        if not clean or not dirty:
            print(f"[cube_split] WARN: '{folder}' missing clean or dirty FITS; skipping")
            continue
        cubes.append({
            "folder": folder,
            "run_id": m.group(1),
            "clean": clean,
            "dirty": dirty,
        })
    return cubes


def split_cubes(
    data_dir: str = "data/Line Emission Data",
    n_holdout: int = 3,
    val_fraction: float = 0.2,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Split line-emission cubes into train / val / inference-only holdout, by RunID group.

    Args:
        data_dir: root folder of per-run subfolders.
        n_holdout: number of RunID groups to reserve ENTIRELY for inference.
        val_fraction: fraction of the remaining (non-holdout) RunID groups used for validation.
        seed: RNG seed for reproducible splits.
        verbose: print a summary.

    Returns:
        (train_cubes, val_cubes, holdout_cubes), each a list of cube dicts
        {folder, run_id, clean, dirty}. Holdout cubes share NO RunID with train/val.
    """
    cubes = list_cubes(data_dir)
    if not cubes:
        raise FileNotFoundError(f"No valid cube pairs found under {data_dir}")

    # group cubes by RunID (leakage guard: variants of a run stay together)
    by_run: Dict[str, List[dict]] = defaultdict(list)
    for c in cubes:
        by_run[c["run_id"]].append(c)
    run_ids = sorted(by_run.keys())

    if n_holdout >= len(run_ids):
        raise ValueError(
            f"n_holdout ({n_holdout}) must be < number of distinct RunIDs ({len(run_ids)})"
        )

    rng = np.random.default_rng(seed)
    shuffled = list(run_ids)
    rng.shuffle(shuffled)

    holdout_runs = sorted(shuffled[:n_holdout])
    remaining_runs = shuffled[n_holdout:]

    # val split over the REMAINING run groups (cube-level, by RunID)
    n_val = int(round(len(remaining_runs) * val_fraction))
    n_val = max(1, n_val) if len(remaining_runs) > 1 else 0   # keep >=1 val if possible
    val_runs = sorted(remaining_runs[:n_val])
    train_runs = sorted(remaining_runs[n_val:])

    def gather(run_list):
        out = []
        for rid in run_list:
            out.extend(by_run[rid])
        return out

    train_cubes = gather(train_runs)
    val_cubes = gather(val_runs)
    holdout_cubes = gather(holdout_runs)

    # safety assertion: holdout RunIDs disjoint from train/val RunIDs
    tv_runs = set(train_runs) | set(val_runs)
    assert not (set(holdout_runs) & tv_runs), "LEAKAGE: holdout RunID overlaps train/val!"

    if verbose:
        print("=" * 70)
        print("CUBE-LEVEL SPLIT (grouped by RunID: no channel-level leakage)")
        print("=" * 70)
        print(f"  data_dir          : {data_dir}")
        print(f"  total cubes       : {len(cubes)}  across {len(run_ids)} distinct RunIDs")
        print(f"  seed={seed}  n_holdout={n_holdout}  val_fraction={val_fraction}")
        print("-" * 70)
        print(f"  TRAIN   : {len(train_cubes):2d} cubes | RunIDs {train_runs}")
        print(f"  VAL     : {len(val_cubes):2d} cubes | RunIDs {val_runs}")
        print(f"  HOLDOUT : {len(holdout_cubes):2d} cubes | RunIDs {holdout_runs}  <-- inference only, NEVER trained/validated")
        print("-" * 70)
        print("  HOLDOUT cube folders (reserved for moment-map evaluation):")
        for c in holdout_cubes:
            print(f"    - {c['folder']}")
        print("=" * 70)

    return train_cubes, val_cubes, holdout_cubes


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cube-level train/val/holdout split")
    ap.add_argument("--data-dir", default="data/Line Emission Data")
    ap.add_argument("--n-holdout", type=int, default=3)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train, val, holdout = split_cubes(
        data_dir=args.data_dir, n_holdout=args.n_holdout,
        val_fraction=args.val_fraction, seed=args.seed,
    )

    # explicit final accounting
    print("\nFINAL COUNTS")
    print(f"  train   cubes: {len(train)}")
    print(f"  val     cubes: {len(val)}")
    print(f"  holdout cubes: {len(holdout)}")
    print(f"  TOTAL        : {len(train) + len(val) + len(holdout)}")
    print("\nHOLDOUT RunIDs:", sorted({c['run_id'] for c in holdout}))
