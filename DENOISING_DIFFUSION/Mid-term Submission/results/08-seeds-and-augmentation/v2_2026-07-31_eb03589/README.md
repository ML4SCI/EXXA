# Notebook 08, code commit `eb03589` (2026-07-31)

This run trained the four arms across seeds 42, 43 and 44. That gave the twelve checkpoints
that every later notebook reuses. The notebook's output never recorded a Kaggle version number,
so `v2_` in the folder name is just an ordering label and not a verified version.

## Files

The executed notebook, with its full output, is
[`../../../notebooks/08-seeds-and-augmentation.ipynb`](../../../notebooks/08-seeds-and-augmentation.ipynb).
It is not duplicated here.

| file | what it is |
|---|---|
| `figure_cell24.png` | the per-seed spread plot |
| `seed_repeats.csv` | PSNR, SSIM and MSE for all twelve checkpoints, which [`../../../MODELS.md`](../../../MODELS.md) tabulates |

## Why this run still matters

v2 and v4 use the same twelve checkpoints. v4 reused v2's weights and only scored them again, so
their PSNR, SSIM and MSE rows are identical. Only the evaluation code differs:

```
v2 (eb03589)   raw:   no noise clip, no signal mask
v4 (1ca611f)   clip:  3-sigma noise clip added
```

That makes v2's moment scores the ones that can sit next to V12, because V12's published
+69.8 / +17.5 / +20.1 was measured on the same raw metric. v2 is the only seed-validated result
in the project that can be compared with the reference checkpoint without changing the metric.
On that footing `winner_aug` scores **M0 +87.5% ± 4.2** over three seeds, the highest M0 here
and with the lowest variance.

The artifact diagnostics from this run cannot be used, though. It reports 0.0% invented
structure for all four arms. That is a suspicious zero: the background mask selected no pixels,
so the detector could not fire. The overshoot values (0.89 to 0.93, all below 1.0) come from the
same problem. Use v4's rates instead, which are 22.3 to 39.0%.

So the rule for this run is simple. Take its moment scores and take v4's artifact rates, and
never mix a moment number from one with the other.
