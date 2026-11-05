# 07. Image sizes and moment maps

On held-out line-emission cubes the final model's quality does not depend on image size:

| size (px) | 256 | 300 | 320 | 480 | 600 |
|---|---|---|---|---|---|
| PSNR, final model (dB) | 51.2 | 51.8 | 51.7 | 51.7 | 51.7 |
| PSNR, Wiener (dB) | 30.5 | | | | 30.6 |

M0, M1 and M2 improve over the noisy cube by about 62, 88.5 and 90.5 percent at every size (62.0 to 62.7, 88.4 to 88.6 and 90.0 to 90.9 across the five sizes) (signal-masked, noise-clipped). The full table, the SNR ladder and the gate are in `../final/RESULTS.md`; the inference settings in `../final/RECIPES.md`. The do-no-harm gate matters on bright cubes: on a test cube at SNR about 135 the ungated model changed M0 and M2 by -76 and -257 percent, the gated one by -4 and +50 percent.

## Runs in this folder

- [`eval-final-b_v1_2026-10-17/`](eval-final-b_v1_2026-10-17/): the final model on every held-out case at five sizes (`eval-final-b_v1_2026-10-17/eval_final_B.json`)

Where a run folder has its own README, it states the Kaggle run, its settings and its outcome.
