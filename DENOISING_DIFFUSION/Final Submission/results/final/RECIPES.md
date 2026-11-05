# Inference recipes (2026-10-16: store A, use B)

Both use the same inference flags; they differ only in which checkpoints are averaged.

| | B (USED: the deliverable) | A (STORED: the 3-model alternative) |
|---|---|---|
| checkpoints averaged | `unified_r4n_k15_long.pth`, `unified_r4n_k15_long_s11.pth`, `unified_r4n_k15_long_s23.pth`, `unified_r8_m1_r6.pth` | the first three only |
| kernel sources | `unified-r4n-k15-long`, `unified-r4n-k15-long-s11`, `unified-r4n-k15-long-s23`, `unified-r8-m1-r6` | the first three |
| inference cost | 4 networks (~1.33x A) | 3 networks |
| line @600 PSNR, M0/M1/M2 | 51.7 dB, 62/89/91 % | 51.4 dB, 61/88/91 % |
| Speedie GI native M0, spine | 39 %, 0.39 | 28 %, 0.34 |
| twin invented wiggle, SNR 6 | +25 m/s | +34 m/s |

Both columns are recomputed from `results/09-design-choices/combo-a_v1_2026-10-15/eval_combo_A.json` and `combo-b_v1_2026-10-15/eval_combo_B.json` (the five line holdouts at 600 px, the Speedie r0.5 disk and its twin at SNR 6, where the invented wiggle is the model's rotation-curve rms minus the truth's).

Flags (both): `--gate 40,120,0.25 --wiener2 --infer-beam-px 10 --infer-mode full`; the size fix applies to noise-only cubes only (`train_unified.infer(..., ai)`).
ALMA (`experiments/alma_unified.py`): the same by default (`--infer-beam-px 10`, two-pass Wiener on, gate 40,120,0.25); pass the four checkpoints of B to `--ckpt`.
The four checkpoints are described in [`MODELS.md`](../../MODELS.md). If the Oct 17 R6 seeds are adopted they join B's list; A stays as stored here.

## FROZEN 2026-10-18 03:00 IST: recipe B

Recipe B is the frozen deliverable, exactly as above (R4n-long seeds 3, 11, 23 + R8-from-R6 at M1 weight 0.01, averaged; do-no-harm gate 40 / 120 / 0.25; two-pass Wiener; size fix `--infer-beam-px 10 --infer-mode full` on noise-only cubes only). No candidate from the Oct 16 to 17 batch met its pre-set adoption rule: R6 s32 (9074 per-pixel 0.38, rule 0.43), R6 s33 (restarted from scratch on Oct 18 as `unified-r6-s33b` for the alternative-checkpoint question only; the first run stopped at round 67 of 80; two resumes failed because Kaggle no longer attaches the version-1 output; partial checkpoint kept, `unified_r6_k15_s33_round67.pth`; not pursued: with s32 below the rule a third seed cannot make the gain consistent), R9 s61 / s62 (no gain on 9074 or the Speedie ladder; SNR 12 wiggle 0.61 / 0.32), R8 weights 0.003 / 0.03 (SNR 12 wiggle 0.30 / 0.42 against 0.64 for 0.01). Recipe A stays stored as the 3-model alternative; R6 (seeds 31, 32) is the documented kinematics alternative.
After this line the one-time final test (Speedie r1.5 ladder at SNR 6 / 12 / 24, `kernels/eval_final_test.py`) is run once on this recipe and reported whatever it gives.

## DECISION 2026-10-19: ship one recipe, B

Recipe B (four averaged checkpoints, gate, two-pass Wiener, size fix) is the single deliverable; the figures and all real-data results use it. Recipe A is stored and documented as the 3-network alternative (S3: on four real exoALMA discs A agrees with B to 3 to 4 m/s, flux within 1.6 percent); R6 (three seeds) is documented as the kinematics alternative, not shipped (46.9 vs 51.7 dB, 9074 noise-only 0.38 vs 0.45, its 9074 deconvolution gain not reproduced across seeds). Evidence: (R6 three seeds) and 2026-10-19 (S3 / S4), results/final/ANALYSIS.md sections 6 and 8.
