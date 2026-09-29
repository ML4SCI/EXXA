#!/usr/bin/env python
"""
src/evaluation/collect_outputs.py
=================================
Bundle one notebook's artifacts into a versioned, self-describing folder.

Results used to land in a single flat `results/` directory with no record of which
notebook produced them or which revision of the code was running, so
`moment_map_holdout_summary_ddpm.csv` from a run with the unmasked M2 metric was
indistinguishable from one written after the noise-clip fix, and the only way to tell
was to remember. Each run now writes

    <root>/<notebook_id>/<UTC timestamp>_<git sha>/

with a manifest recording the commit, the time, and a checksum per file. Nothing
overwrites anything, so two runs can be compared directly instead of by recollection.
"""

import glob as _glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from typing import Dict, Iterable, List, Optional, Sequence

# Where a notebook's files might be found. Kaggle notebooks write some artifacts to the
# top level of /kaggle/working (so they survive as notebook Output) and others into the
# git clone underneath it, so both have to be searched.
DEFAULT_ROOTS: Sequence[str] = ("../results", "../experiments", "/kaggle/working")


def git_sha(short: bool = True) -> str:
    """Current commit, or 'nogit' when the code is not running from a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        sha = out.stdout.strip().split("\n")[0]
        return sha or "nogit"
    except Exception:
        return "nogit"


def kaggle_version() -> Optional[int]:
    """This kernel's Kaggle version number, if the platform will tell us.

    Kaggle does not document an environment variable for it, and in practice none of the
    KAGGLE_* variables carries it: the version only becomes visible afterwards, in the
    commit message its GitHub integration writes ('Kaggle Notebook | <name> | Version N').
    So this returns None on Kaggle far more often than not. Probed anyway because it
    costs nothing and the platform may start exposing it.
    """
    for key in ("KAGGLE_KERNEL_VERSION", "KAGGLE_NOTEBOOK_VERSION", "KERNEL_VERSION"):
        v = os.environ.get(key)
        if v and v.strip().isdigit():
            return int(v.strip())
    return None


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def _resolve(patterns: Iterable[str], roots: Sequence[str]) -> Dict[str, str]:
    """Map basename -> newest matching path, searching each root recursively.

    Keyed by basename so the same artifact found in two roots (the /kaggle/working copy
    and the in-clone original) is collected once, taking whichever was written last.
    """
    found: Dict[str, str] = {}
    for pat in patterns:
        hits: List[str] = []
        if os.path.isabs(pat):
            hits = _glob.glob(pat, recursive=True)
        else:
            for r in roots:
                hits += _glob.glob(os.path.join(r, "**", pat), recursive=True)
                hits += _glob.glob(os.path.join(r, pat))
        for h in hits:
            if not os.path.isfile(h):
                continue
            b = os.path.basename(h)
            if b not in found or os.path.getmtime(h) > os.path.getmtime(found[b]):
                found[b] = h
    return found


def collect_outputs(
    notebook_id: str,
    patterns: Sequence[str],
    *,
    root: Optional[str] = None,
    roots: Sequence[str] = DEFAULT_ROOTS,
    extra: Optional[dict] = None,
    stamp: Optional[str] = None,
    sha: Optional[str] = None,
    version: Optional[int] = None,
    move: bool = False,
    verbose: bool = True,
) -> str:
    """
    Copy this notebook's artifacts into a versioned run folder and write a manifest.

    Args:
        notebook_id: folder name for this notebook, e.g. '09-architecture-comparison'.
        patterns: glob patterns (basenames are fine) naming what this notebook produces.
            Patterns matching nothing are recorded in the manifest as `missing` rather
            than passed over silently: a result that did not get written is exactly
            what you want to notice here.
        root: destination root. Defaults to /kaggle/working/outputs on Kaggle (top level,
            so the bundle is unambiguously part of the notebook Output and is not buried
            in the 5000-file git clone) and ../results locally.
        roots: directories to search for the artifacts.
        extra: anything else worth recording: key config, headline metrics.
        version: Kaggle version number, if known. Prefixes the run folder with 'v<N>_'.
            Left None it is probed from the environment and usually stays None.
        stamp / sha: override the run tag. Only for filing results produced BEFORE this
            collector existed, where "now" and "current HEAD" would both be wrong; a live
            run must leave these unset so the tag reflects what actually ran.
        move: git mv the files instead of copying, so history follows them. Used when
            reorganising an existing flat results/ directory, not by a live run.

    Returns:
        Path to the run folder.
    """
    on_kaggle = os.path.exists("/kaggle")
    if root is None:
        root = "/kaggle/working/outputs" if on_kaggle else "../results"

    sha = sha or git_sha()
    stamp = stamp or time.strftime("%Y-%m-%dT%H%M%S", time.gmtime())
    version = version if version is not None else kaggle_version()
    # 'v<N>_' when the version is known, plain timestamp otherwise: an invented or
    # guessed version number in a path is worse than none, since the whole point of the
    # tag is that it can be trusted without checking
    prefix = f"v{version}_" if version is not None else ""

    # Uniqueness is enforced by the filesystem, not by clock resolution. Two collections
    # inside the same second would otherwise share a folder and merge: silently mixing
    # two runs' artifacts, the exact failure this versioning exists to prevent.
    base = os.path.join(root, notebook_id, f"{prefix}{stamp}_{sha}")
    run_dir, n = base, 1
    while os.path.exists(run_dir):
        n += 1
        run_dir = f"{base}-{n}"
    os.makedirs(run_dir)

    found = _resolve(patterns, roots)
    matched_names = set(found)
    missing = [p for p in patterns
               if not any(_glob.fnmatch.fnmatch(n, p) or n == p for n in matched_names)]

    files = {}
    for name, src in sorted(found.items()):
        dst = os.path.join(run_dir, name)
        if move:
            r = subprocess.run(["git", "mv", src, dst], capture_output=True, text=True)
            if r.returncode != 0:          # untracked, or not a checkout
                shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)
        files[name] = {
            "bytes": os.path.getsize(dst),
            "sha256_16": _sha256(dst),
            "source": src,
        }

    manifest = {
        "notebook": notebook_id,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": sha,
        "kaggle_version": version,
        "kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        "on_kaggle": on_kaggle,
        "python": platform.python_version(),
        "files": files,
        "missing_patterns": missing,
    }
    if extra:
        manifest["extra"] = extra
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    if verbose:
        total = sum(v["bytes"] for v in files.values())
        print(f"collected {len(files)} file(s), {total / 2**20:.1f} MiB -> {run_dir}")
        for n, v in sorted(files.items()):
            print(f"   {v['bytes'] / 2**20:8.2f} MiB  {n}")
        if missing:
            print(f"   NOT FOUND (not written this run?): {missing}")
    return run_dir
