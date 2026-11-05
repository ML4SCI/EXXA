#!/usr/bin/env python
"""
src/evaluation/recover_version.py
=================================
Recover a Kaggle run's outputs from the commit its GitHub integration pushed.

Several notebook versions completed, wrote their CSVs to the kernel Output tab, and were
never downloaded: the artifacts are gone, but the *executed notebook* was pushed to git
with its outputs still embedded. That commit holds every figure as base64 PNG and the whole
stdout, which is enough to reconstruct most of what the run produced.

Figures are named from the ``savefig`` call in their own cell rather than by output order.
Ordering by position mislabelled three of notebook 05 v17's four figures: it writes
``sweepwinner_loss.png``, not ``unet_line_emission_loss.png``, and writes no
``moment_map_holdout_summary.png`` at all.

CSVs cannot be recovered this way; only what the notebook printed. Where a printed table
carries the same numbers, reconstruct it by hand and name the file ``.RECONSTRUCTED.csv``
so it is never mistaken for the artifact the run actually wrote.

Usage:
    python -m src.evaluation.recover_version <commit> <notebook.ipynb> <dest-dir>
"""

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Dict


def recover(commit: str, notebook_path: str, dest: str, verbose: bool = True,
            overwrite: bool = False) -> Dict:
    """Extract every embedded figure and the full stdout of a pushed Kaggle run.

    Existing files are left alone unless ``overwrite=True``. A recovered figure is the
    notebook's *display* copy, which is not the same file as the ``savefig`` output that
    was downloaded from the Output tab: lower fidelity and a different size. Filling a
    gap with it is useful; silently replacing a genuine downloaded artifact with it is a
    quiet loss, which is what happened to notebook 05 v16 before this guard existed.
    """
    raw = subprocess.run(["git", "show", f"{commit}:{notebook_path}"],
                         capture_output=True, text=True).stdout
    if not raw:
        raise FileNotFoundError(f"{notebook_path} not found in {commit}")
    nb = json.loads(raw)
    os.makedirs(dest, exist_ok=True)

    files, log, skipped = {}, [], []
    for ci, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        # The name a figure was saved under, taken from its own cell. Two forms occur:
        # a literal inside savefig(...), and a variable assigned earlier in the same cell
        # (notebook 07 uses `fig_path = os.path.join(OUT_DIR, '...png')` then
        # `plt.savefig(fig_path, ...)`), which a literal-only pattern misses entirely.
        names = []
        for arg in re.findall(r"savefig\(\s*([^,)]+)", src):
            arg = arg.strip()
            lit = re.match(r"""(?:os\.path\.join\([^,]+,\s*)?['"]([^'"]+\.png)['"]""", arg)
            if lit:
                names.append(lit.group(1))
                continue
            if re.fullmatch(r"\w+", arg):        # a variable: resolve it in this cell
                m = re.search(
                    rf"""{arg}\s*=\s*(?:os\.path\.join\([^,]+,\s*)?['"]([^'"]+\.png)['"]""",
                    src)
                if m:
                    names.append(m.group(1))
        imgs = []
        for o in cell.get("outputs", []):
            if o.get("output_type") == "stream":
                log.append("".join(o.get("text", [])))
            p = o.get("data", {}).get("image/png")
            if p:
                imgs.append(p)
        for j, payload in enumerate(imgs):
            blob = base64.b64decode(payload)
            if j < len(names):
                base = os.path.basename(names[j])
                how = f"named from savefig('{names[j]}')"
            else:
                # displayed but never saved: still a real output of the run, and
                # dropping it would silently lose a figure
                base = f"unnamed_cell{ci}_{j}.png"
                how = "displayed with no savefig call; named after its cell"
            # A notebook can write the same filename twice: 05 v7 and v9 each contain TWO
            # runs (a line-emission section and an appended continuum section) that both
            # save moment_maps_holdout.png. Keying on the basename alone kept only the
            # last and silently discarded the first.
            stem, ext = os.path.splitext(base)
            n = 1
            while base in files:
                n += 1
                base = f"{stem}__{n}{ext}"
            out = os.path.join(dest, base)
            if os.path.exists(out) and not overwrite:
                skipped.append(base)
                files[base] = {"kept_existing": True}
                continue
            with open(out, "wb") as f:
                f.write(blob)
            files[base] = {
                "bytes": len(blob),
                "sha256_16": hashlib.sha256(blob).hexdigest()[:16],
                "cell": ci,
                "provenance": f"extracted from {commit}; {how}",
            }

    text = "\n".join(log)
    if os.path.exists(os.path.join(dest, "run_log.txt")) and not overwrite:
        skipped.append("run_log.txt")
        if verbose:
            print(f"kept {len(skipped)} existing file(s): {skipped}")
        return files
    with open(os.path.join(dest, "run_log.txt"), "w") as f:
        f.write(text)
    files["run_log.txt"] = {"bytes": len(text),
                            "provenance": f"stdout captured in {commit}"}
    if verbose:
        n_new = sum(1 for v in files.values() if not v.get("kept_existing"))
        print(f"recovered {n_new - 1} figure(s) + run log -> {dest}")
        if skipped:
            print(f"   kept existing (not overwritten): {skipped}")
        for n, v in sorted(files.items()):
            if not v.get("kept_existing"):
                print(f"   {v['bytes']:>9,}  {n}")
    return files


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    recover(sys.argv[1], sys.argv[2], sys.argv[3])
