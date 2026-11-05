# r6-k15-s33_v1-v3_2026-10-17_stopped

Kaggle kernel `unified-r6-s33`. R6 recipe (R4n flags, 256 px crops, batch 4), seed 33, the third R6 seed.
v1 (2026-10-17 05:55 IST) trained rounds 1 to 67 of 80 and stopped at its 12 h deadline (RESUME_NEEDED, as designed); its checkpoint and status were downloaded the same day.
v2 (16:47 IST) and v3 (21:47 IST), both resumed with the kernel's own output attached, crashed at startup: `IndexError` on `glob("/kaggle/input/**/MANIFEST.json")`, i.e. no inputs were mounted (v2's log was identical). After v2 failed, the kernel's "latest output" is an empty failed version, so the v1 output can no longer be attached as a source.
**Not pursued (2026-10-18):** R6 seeds join recipe B only if they repeat seed 31's 9074 gain (0.43 per pixel); seed 32 did not (0.38), so a third seed cannot make the gain consistent, and the alternative-checkpoint question is answered by seeds 31 and 32. The partial checkpoint is kept: `unified_r6_k15_s33_round67.pth` (349 MB, zip-verified), round 67 of 80, never evaluated.
