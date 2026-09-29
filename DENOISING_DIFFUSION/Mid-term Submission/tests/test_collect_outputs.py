#!/usr/bin/env python
"""
Smoke test for the per-notebook output bundler.

The point of the bundle is provenance: which notebook wrote a file, under which commit.
A collector that quietly drops a file, or that reports success for a result that was
never written, would be worse than the flat results/ directory it replaces.
"""

import json
import os
import tempfile

from src.evaluation.collect_outputs import collect_outputs

print("=" * 60)
print("Output Collector Smoke Test")
print("=" * 60)

tmp = tempfile.mkdtemp()
src_a = os.path.join(tmp, "results")
src_b = os.path.join(tmp, "kaggle")
os.makedirs(os.path.join(src_a, "checkpoints"))
os.makedirs(src_b)


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


write(os.path.join(src_a, "seed_repeats.csv"), "tag,psnr\nv12,38.6\n")
write(os.path.join(src_a, "seed_spread.png"), "PNGDATA")
write(os.path.join(src_a, "checkpoints", "winner_seed42.pth"), "WEIGHTS")
write(os.path.join(src_b, "seed_moment_summary.csv"), "config,imp_M0\nv12,74.9\n")

out_root = os.path.join(tmp, "out")
patterns = ["seed_repeats.csv", "seed_moment_summary.csv", "seed_spread.png",
            "winner_seed*.pth", "does_not_exist.csv"]

run_dir = collect_outputs("08-seeds-and-augmentation", patterns,
                          root=out_root, roots=(src_a, src_b), verbose=False,
                          extra={"headline_psnr": 38.66})

# [1] the run folder is namespaced by notebook and carries a version tag in its path
rel = os.path.relpath(run_dir, out_root)
nb_dir, run_name = os.path.split(rel)
assert nb_dir == "08-seeds-and-augmentation", rel
assert "_" in run_name and run_name.startswith("20"), run_name
print(f"[1] versioned path OK: {rel}")

# [2] every pattern that matched something was copied, including one found in a
#     different search root and one matched by a wildcard
names = sorted(os.listdir(run_dir))
for expect in ("seed_repeats.csv", "seed_moment_summary.csv", "seed_spread.png",
               "winner_seed42.pth", "manifest.json"):
    assert expect in names, (expect, names)
print(f"[2] collected across roots and wildcards: {[n for n in names if n != 'manifest.json']}")

man = json.load(open(os.path.join(run_dir, "manifest.json")))

# [3] a pattern that matched NOTHING must be reported, not silently dropped, otherwise
#     a run that failed to produce a result looks identical to one that succeeded
assert man["missing_patterns"] == ["does_not_exist.csv"], man["missing_patterns"]
print(f"[3] missing result reported: {man['missing_patterns']}")

# [4] provenance actually recorded, and checksums are per-file (not a constant)
assert man["notebook"] == "08-seeds-and-augmentation"
assert man["git_sha"] and man["utc"].endswith("Z"), man
assert man["extra"]["headline_psnr"] == 38.66
sums = {v["sha256_16"] for v in man["files"].values()}
assert len(sums) == len(man["files"]), "checksums collide, not hashing content"
print(f"[4] manifest OK: sha {man['git_sha']}, {len(man['files'])} distinct checksums")

# [5] contents survive the copy byte-for-byte
assert open(os.path.join(run_dir, "seed_repeats.csv")).read() == "tag,psnr\nv12,38.6\n"
print("[5] file contents preserved")

# [6] a second run must NOT overwrite the first: that is the whole reason for versioning
run2 = collect_outputs("08-seeds-and-augmentation", ["seed_repeats.csv"],
                       root=out_root, roots=(src_a, src_b), verbose=False)
runs = sorted(os.listdir(os.path.join(out_root, "08-seeds-and-augmentation")))
assert run2 != run_dir, "second run reused the first run's folder"
assert len(runs) == 2, runs
assert os.path.exists(os.path.join(run_dir, "seed_spread.png")), "first run was clobbered"
print(f"[6] second run kept separate: {runs}")

print("\n" + "=" * 60)
print("All output-collector tests PASSED")
print("=" * 60)
