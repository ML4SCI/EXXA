"""
Build trainable (dirty, clean) pairs from the September self-gravitating clean cubes.

Why this exists: in the original run_sg_* pairs the dirty cubes differ from the clean ones by only 0.4 to 7 percent
RMS (the line-emission training set differs by about 50 percent), too little noise to train a denoiser. The batch
does provide five new self-gravitating disks as clean targets, with stated ground truth in the `.para` files.

So we apply the corruption ourselves:

    dirty = beam (*) clean + beam (*) white_noise

The beam is the one recovered from the v2 pair's cross-spectrum. Noise is convolved with the
same beam because in a Jy/beam map the noise has been through the instrument too -- Phase 0
established that on 2026-08-20, and a flat white term there was one of the things that made
its early verdicts wrong.

**Honest about what this is.** Convolution plus correlated Gaussian noise is not an
interferometer. There is no uv sampling, no phase error, no deconvolution residual structure.
It is a controlled approximation, and it is exactly the forward model DDRM already assumes,
which is the point: `A` here is known EXACTLY because we applied it, rather than estimated
from a pair at 0.80 held-out correlation. CASA simobserve is the higher-fidelity version of
this step and is the natural follow-up.

Output is named `run_<id>_<step>_rt_<pp>` so `src/data/cube_split.py` discovers it unchanged.

Run: PYTHONPATH=.. python3 experiments/synthesize_sg_pairs.py
"""
import os, sys, glob, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from astropy.io import fits

from src.evaluation.forward_operator import apply_beam

SRC = "self-gravitating cube and dirty cube/_v3_extract/kinematic_data"
BEAM = "results/self-gravitating/dirty_beam_recovered_v2.fits"
OUT = "self-gravitating cube and dirty cube/sg_synth"

# Noise sigma as a multiple of the clean cube's in-signal RMS. Tuned so the resulting
# clean-vs-dirty difference lands in the line-emission training set's 0.41-0.57 band.
NOISE_FRAC = 0.35
CHUNK = 40          # channels per pass, keeps the 601x600x600 cube off the heap
SEED = 42

# Synthetic run ids: 9xxx marks a pair generated here. The trailing digits
# keep the source run recognisable, and each disk gets a DIFFERENT id so cube_split's
# leakage-safe grouping treats them as independent disks.
NAME_MAP = {
    "run_sg_00019_rt_00019":  "run_9019_00019_rt_00",
    "run_sg_15_00370_rt_rt00": "run_9015_00370_rt_00",
    "run_sg_25_00370_rt_00":   "run_9025_00370_rt_00",
    "run_sg_32_00020_rt_00.":  "run_9032_00020_rt_00",
    "run_sg_74_00025_rt_00":   "run_9074_00025_rt_00",
}


def trim_padding(cube, n=45):
    lead = 0
    for i in range(1, min(n, cube.shape[0])):
        if np.array_equal(cube[i], cube[0]):
            lead = i
        else:
            break
    trail = 0
    for i in range(2, min(n, cube.shape[0])):
        if np.array_equal(cube[-i], cube[-1]):
            trail = i - 1
        else:
            break
    return (lead + 1 if lead else 0), cube.shape[0] - (trail + 1 if trail else 0)


beam = fits.getdata(BEAM).astype(np.float64)
_raw_sum = float(beam.sum())
# Normalise to unit DC gain. The recovered beam sums to ~355 (the Jy/pixel -> Jy/beam area
# factor, and the 129 px crop truncating the negative bowl), so convolving with it as-is
# multiplies the flux scale by 355 and the clean/dirty difference becomes a scale factor
# rather than a corruption. That also breaks the dataset's shared-scale normalisation, which
# divides BOTH cubes by the DIRTY channel's range: the target would land ~355x below the
# input. Unit sum keeps dirty and clean on one scale so the pair differs only by blur+noise.
beam = beam / _raw_sum
print(f"beam {beam.shape}, raw sum {_raw_sum:.3f} -> normalised to unit DC gain, "
      f"peak now {beam.max():.5g}")
os.makedirs(OUT, exist_ok=True)

rows = []
for cpath in sorted(glob.glob(f"{SRC}/*/*clean*.fits")):
    src_run = os.path.basename(os.path.dirname(cpath))
    if src_run not in NAME_MAP:
        continue
    dst_run = NAME_MAP[src_run]
    dst_dir = os.path.join(OUT, dst_run)
    os.makedirs(dst_dir, exist_ok=True)

    with fits.open(cpath, memmap=True) as h:
        hdr = h[0].header.copy()
        data = h[0].data
        lo, hi = trim_padding(data)
        nch = hi - lo
        H, W = data.shape[1], data.shape[2]
        print(f"\n{src_run} -> {dst_run}")
        print(f"  {data.shape}, keeping channels {lo}-{hi} ({nch})")

        # Noise scale from the signal region of a bright channel, not the whole map: a cube
        # that is mostly empty sky would otherwise get a uselessly small sigma.
        probe = np.asarray(data[(lo + hi) // 2], np.float64)
        thr = 0.05 * np.nanmax(np.abs(probe))
        sig_rms = float(np.sqrt(np.nanmean(probe[np.abs(probe) > thr] ** 2)))
        sigma = NOISE_FRAC * sig_rms
        print(f"  in-signal RMS {sig_rms:.5g} -> noise sigma {sigma:.5g}")

        clean_out = np.empty((nch, H, W), dtype=np.float32)
        dirty_out = np.empty((nch, H, W), dtype=np.float32)
        rng = np.random.default_rng(SEED + abs(hash(dst_run)) % 10000)

        # Convolving white noise with a unit-sum kernel suppresses its variance by a factor
        # that depends on the kernel, so calibrate AFTER convolution: draw once, measure, and
        # rescale to the level actually wanted. Otherwise the achieved noise level silently
        # tracks the beam normalisation instead of NOISE_FRAC.
        _probe_noise = apply_beam(rng.normal(0.0, 1.0, size=(1, H, W)), beam)
        _gain = float(np.std(_probe_noise))
        noise_scale = sigma / max(_gain, 1e-30)
        print(f"  beam noise gain {_gain:.5g} -> pre-convolution draw scaled by {noise_scale:.5g}")

        for s in range(0, nch, CHUNK):
            e = min(s + CHUNK, nch)
            block = np.asarray(data[lo + s:lo + e], np.float64)
            conv = apply_beam(block, beam)
            noise = apply_beam(rng.normal(0.0, noise_scale, size=block.shape), beam)
            clean_out[s:e] = block.astype(np.float32)
            dirty_out[s:e] = (conv + noise).astype(np.float32)
            print(f"    channels {s}-{e}", end="\r")

    # difference achieved, on the brightest channel, comparable to the table in
    k = int(np.argmax([np.std(clean_out[i]) for i in range(0, nch, max(1, nch // 40))])
            * max(1, nch // 40))
    a, b = clean_out[k].astype(np.float64), dirty_out[k].astype(np.float64)
    rel = float(np.sqrt(np.mean((b - a) ** 2)) / max(np.sqrt(np.mean(a ** 2)), 1e-30))

    hdr["CRPIX3"] = hdr.get("CRPIX3", 1.0) - lo      # keep the velocity axis honest after trim
    hdr["HISTORY"] = "dirty synthesized: beam*clean + beam*N(0,sigma)"
    hdr["SYNTHSRC"] = (src_run, "source run of the September batch")
    hdr["SYNTHBM"] = (os.path.basename(BEAM), "forward operator applied")
    hdr["SYNTHNRM"] = (_raw_sum, "beam divided by this to give unit DC gain")
    hdr["SYNTHSIG"] = (float(sigma), "target noise sigma AFTER beam convolution")
    hdr["SYNTHREL"] = (rel, "achieved rmsdiff/rms_clean at brightest channel")

    fits.PrimaryHDU(clean_out, header=hdr).writeto(
        os.path.join(dst_dir, f"{dst_run}_clean.fits"), overwrite=True)
    fits.PrimaryHDU(dirty_out, header=hdr).writeto(
        os.path.join(dst_dir, f"{dst_run}_dirty.fits"), overwrite=True)

    print(f"  wrote {dst_run}: rmsdiff/rms_clean = {rel:.4f}")
    rows.append(dict(src=src_run, dst=dst_run, channels=nch, shape=[nch, H, W],
                     sigma=float(sigma), rel_diff=rel))

print("\n" + "=" * 70)
print(f"{'run':24s} {'channels':>9} {'rmsdiff/rms':>12}   (training set: 0.41-0.57)")
for r in rows:
    print(f"{r['dst']:24s} {r['channels']:9d} {r['rel_diff']:12.4f}")

with open("results/self-gravitating/sg_synth_manifest.json", "w") as f:
    json.dump(dict(noise_frac=NOISE_FRAC, beam=BEAM, seed=SEED, runs=rows), f, indent=2)
print("\nmanifest -> results/self-gravitating/sg_synth_manifest.json")
print(f"cubes    -> {OUT}/")
