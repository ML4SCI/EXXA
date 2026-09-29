#!/usr/bin/env python
"""
src/data/inspect_line_emission.py
=================================
Empirical inspection of the line-emission FITS dataset.

For each run subfolder under the data root it:
  - records the folder name and its parsed naming pattern,
  - identifies the clean and dirty FITS files,
  - opens one clean/dirty pair at a time (memory-mapped, header+shape only where
    possible), printing cube shape, dtype, min/max, and channel count,
  - checks the channel count against the expected ~201,
  - closes each file before opening the next (no bulk load into memory),
  - prints a summary table at the end.

Run:
    python -m src.data.inspect_line_emission
    python -m src.data.inspect_line_emission --data-dir "data/Line Emission Data"
"""

import argparse
import os
import re
from typing import Optional

import numpy as np
from astropy.io import fits

EXPECTED_CHANNELS = 201
# folder naming pattern hypothesis: run_<RunID>_<StepID>_rt_<PP>
NAME_RE = re.compile(r"run_(\d+)_(\d+)_rt_(\d+)", re.IGNORECASE)


def parse_name(folder: str) -> str:
    """Return a human-readable parse of the folder name, or 'unrecognised'."""
    m = NAME_RE.search(folder)
    if not m:
        return "unrecognised pattern"
    run_id, step_id, rt = m.groups()
    return f"RunID={run_id}, StepID={step_id}, RT={rt}"


def find_pair(folder_path: str):
    """Return (clean_path, dirty_path) for FITS files in a folder; either may be None."""
    clean = dirty = None
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


def inspect_fits(path: str) -> Optional[dict]:
    """Open a single FITS file (memmap), read shape/dtype/min/max, then close it."""
    if path is None or not os.path.exists(path):
        return None
    info = {}
    # memmap=True avoids reading the whole cube; we only pull header + reduced stats.
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[0]
        hdr = hdu.header
        shape = tuple(hdu.shape)               # (NAXIS3, NAXIS2, NAXIS1) for a cube
        info["shape"] = shape
        info["dtype"] = str(hdu.data.dtype)
        # channel count = first (slowest) axis for an (C, H, W) cube; fall back to NAXIS3
        info["n_channels"] = shape[0] if len(shape) == 3 else None
        info["bunit"] = hdr.get("BUNIT", "")
        info["ctype3"] = hdr.get("CTYPE3", "")
        # min/max computed via nan-aware reduction (memmap keeps it off-heap)
        data = hdu.data
        info["min"] = float(np.nanmin(data))
        info["max"] = float(np.nanmax(data))
        info["n_nan"] = int(np.isnan(data).sum())
    return info


def main():
    ap = argparse.ArgumentParser(description="Inspect line-emission FITS dataset")
    ap.add_argument("--data-dir", default="data/Line Emission Data",
                    help="root folder containing per-run subfolders")
    args = ap.parse_args()

    root = args.data_dir
    print(f"Data root: {os.path.abspath(root)}")
    if not os.path.isdir(root):
        print(f"ERROR: not a directory: {root}")
        return

    subfolders = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    print(f"Subfolders found: {len(subfolders)}\n")

    rows = []
    for folder in subfolders:
        fpath = os.path.join(root, folder)
        parsed = parse_name(folder)
        clean, dirty = find_pair(fpath)

        print("=" * 78)
        print(f"FOLDER: {folder}")
        print(f"  parsed: {parsed}")
        print(f"  clean : {os.path.basename(clean) if clean else 'MISSING'}")
        print(f"  dirty : {os.path.basename(dirty) if dirty else 'MISSING'}")

        clean_info = inspect_fits(clean)   # opened, read, closed
        dirty_info = inspect_fits(dirty)   # opened, read, closed (one at a time)

        for label, info in (("clean", clean_info), ("dirty", dirty_info)):
            if info is None:
                print(f"  [{label}] no file")
                continue
            ok = info["n_channels"] == EXPECTED_CHANNELS
            flag = "OK" if ok else f"!! expected ~{EXPECTED_CHANNELS}"
            print(f"  [{label}] shape={info['shape']} dtype={info['dtype']} "
                  f"channels={info['n_channels']} ({flag})")
            print(f"           range=[{info['min']:.5g}, {info['max']:.5g}] "
                  f"NaNs={info['n_nan']} BUNIT={info['bunit']} CTYPE3={info['ctype3']}")

        ref = dirty_info or clean_info
        rows.append({
            "folder": folder,
            "parsed": parsed,
            "clean": os.path.basename(clean) if clean else "-",
            "dirty": os.path.basename(dirty) if dirty else "-",
            "shape": ref["shape"] if ref else None,
            "channels": ref["n_channels"] if ref else None,
        })

    # ---- summary table ----
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    hdr = f"{'folder':<28} {'shape':<20} {'chan':>5} {'clean?':>7} {'dirty?':>7}"
    print(hdr)
    print("-" * len(hdr))
    chan_counts = {}
    for r in rows:
        chan = r["channels"]
        chan_counts[chan] = chan_counts.get(chan, 0) + 1
        print(f"{r['folder']:<28} {str(r['shape']):<20} {str(chan):>5} "
              f"{('yes' if r['clean'] != '-' else 'NO'):>7} "
              f"{('yes' if r['dirty'] != '-' else 'NO'):>7}")

    print("-" * len(hdr))
    print(f"Total subfolders     : {len(rows)}")
    print(f"Channel-count tally   : {chan_counts}")
    n_expected = sum(1 for r in rows if r["channels"] == EXPECTED_CHANNELS)
    print(f"Matching ~{EXPECTED_CHANNELS} channels : {n_expected}/{len(rows)}")
    missing = [r["folder"] for r in rows if r["clean"] == "-" or r["dirty"] == "-"]
    print(f"Folders missing a pair: {missing if missing else 'none'}")


if __name__ == "__main__":
    main()
