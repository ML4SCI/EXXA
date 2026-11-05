# r9-k15-s61_v1_2026-10-17

Kaggle kernel `unified-r9-s61` (script kernel: the kernel source and log stay on Kaggle; figures stay in the kernel Output).
Kaggle version: 1 (one session, 644 min, train and eval complete). Checkpoint: `unified_r9_k15_s61.pth` (349 MB, zip-verified).

**What:** R9 = the R6 recipe (R4n flags, 256 px crops, batch 4, seed 61) with `--train-beam-px 8,12`: every redraw is resampled so the beam spans 8 to 12 px, the beam scale where the size fix runs the network. A training-side version of the size fix.
**Evaluated WITH both inference levers** (`--infer-beam-px 10 --infer-mode full --wiener2`, no gate in the table; the log shows the size fix running on the 480 / 600 px line cubes). So the like-for-like baselines are recipes A and B (`results/09-design-choices/combo-a_v1_2026-10-15/` and `combo-b`), NOT the lever-free R4n / R6 evaluations.

**Result (`experiments/compare_models.py --size 600`; R9 s61 = ONE model, A = 3-seed average, B = 4-model average, both with the same levers):**
- line, 600 px: PSNR 51.37 (A 51.40, B 51.74); M0 / M1 / M2 56.6 / 89.9 / 90.3 (A 60.9 / 88.2 / 90.8, B 61.8 / 88.6 / 90.9); rc_r 0.74 (0.77, 0.76); spine_r 0.72 (0.73, 0.75).
- hydro (9074 and 9074_ai, 600 px): rc_r 0.40 (A 0.37, B 0.38); spine_r 0.41 (0.72, 0.72); M0 / M1 / M2 23 / 34 / 51 (42 / 46 / 65). Clearly worse.
- analytic an07 / an08: M0 / M1 / M2 53.7 / 62.5 / 69.2 (61.4 / 71.1 / 83.0), rc_r 0.80 (0.84).
- SG v2: spine_r 0.62 (0.52).
- Speedie ladder GI r0.5, rc_r at SNR 6 / 12 / 24: 0.14 / 0.61 / 0.75 (the lever-free R4n seed 3: 0.11 / 0.60 / 0.81). Twin invented wiggle at SNR 6 / 12 / 24: 42 / 13 / 5 m/s (R4n seed 3 without levers: 82 / 5 / minus 1).

**Verdict against the pre-set rule (adopt only on a clear gain on 9074 or the Speedie ladder):** not adopted. 9074 is worse (spine 0.41 vs 0.72), the Speedie SNR 12 wiggle is 0.61 vs 0.60 (no gain). What it shows: one R9 model with the levers equals the 3-seed average A on line PSNR (51.37 vs 51.40) and M1, so training at the beam scale gives about what averaging gives, but it costs the hydro and analytic cubes. Kept and documented; seed 62 (second R9 seed) runs today.
