# r6-k15-s32_v1-v2_2026-10-17

Kaggle kernel `unified-r6-s32`. v1 (2026-10-16 15:40 IST) trained rounds 1-70 and stopped at its 12 h deadline;
v2 (2026-10-17 02:30 IST, resumed with its own v1 output attached) resumed at round 70, finished 80 and evaluated (231 min). The log here is v2's; v1's ended at
"DEADLINE: training stopped after round 70/80". Checkpoint `unified_r6_k15_s32.pth` (349 MB, zip-verified). Evaluated WITHOUT the inference levers, like R6 seed 31 and R4n seed 3.

**What:** R6 recipe (R4n flags, 256 px crops, batch 4), seed 32: does R6's real-GI wiggle gain on 9074 repeat with another seed?

**Result (raw model, 600 px unless said; vs R6 seed 31 and R4n seed 3, all lever-free):**
- 9074 per-pixel rc_r: 0.38 (R6 s31 0.43, R4n 0.34); spine 0.53 (0.54, 0.60). At 300 px: rc_r 0.42 (0.45, 0.30).
- 9074 noise-only (`_ai`): rc_r 0.38 (0.37, 0.40).
- line 600 px: PSNR 46.85 (45.76, 49.30); M0 / M1 / M2 48.4 / 83.9 / 81.6; rc_r 0.58 (0.64, 0.67).
- Speedie ladder rc_r SNR 6 / 12 / 24: 0.18 / 0.57 / 0.78 (R4n 0.11 / 0.60 / 0.81); twin invented wiggle 31 / 7 / 0 m/s (R6 s31 293 / 12 / 5).

**Verdict against the pre-set rule:** not met (0.38). Across two seeds R6's per-pixel gain on the
deconvolution 9074 at 600 px is 0.40 vs R4n's 0.34, smaller than seed 31 alone suggested; at 300 px both seeds keep it (0.45, 0.42 vs 0.30). R6 stays the documented alternative, not part of B;
seed 33 (main account, running) is the third look.
