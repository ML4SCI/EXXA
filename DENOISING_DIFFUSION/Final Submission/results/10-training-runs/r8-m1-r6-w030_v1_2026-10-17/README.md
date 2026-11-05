# r8-m1-r6-w030_v1_2026-10-17

Kaggle kernel `unified-r8-w030` v1, 2026-10-17, 20 rounds (script kernel; the kernel source and log stay on Kaggle; figures stay in the kernel Output).
Checkpoint `unified_r8_m1_r6_w030.pth` (349 MB, zip-verified).
**What:** R8 from R6 (line-centre / M1 loss fine-tune, warm start from R6 seed 31) with the M1 loss weight 0.03 instead of 0.01 (the weight in `unified_r8_m1_r6.pth`, which is in recipe B). Evaluated without the inference levers, like R6.
**Result:** Speedie SNR 12 wiggle rc_r worse than with weight 0.01 (0.30 for w 0.003, 0.42 for w 0.03, against 0.64); 9074 not better. Not adopted.
