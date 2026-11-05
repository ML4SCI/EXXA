# unified-wiener-real-k15 v1 (R3b) and unified-wiener-real-sgv2 v1 (R3c): COMPLETE, 12000 steps each, zero skipped steps

R3b: K=15 (+-1.5 km/s), lr 1.5e-4, `--clip-in 10`, realistic noise (peak SNR 10-150, channel kernel a ~ U(0, 0.5), beam x U(1, 1.6)), no SG v2. 247 min.
R3c (`../real-k3-sgv2_v1_2026-10-10/`): K=3, lr 3e-4, same noise, SG v2 in training. 179 min.
Neither used `--fp32-stream`; neither overflowed (R3 `unified-wiener-real` v1, same code as R3c minus SG v2, did at step 2013: the overflow is path-dependent).
Comparison tables at 600 and 256 px: v1 (unified-wiener, merged with its eval-speedie scores) against both. IMPORTANT: the "wiener" row in those tables comes from v1's file
(channel-independent Wiener); the new runs' own JSONs hold the velocity-aware Wiener, which scores differently on Speedie (GI r0.5 rc_r 0.59 against 0.85, PSNR 34.7 against
32.4).  Figures: Speedie GI / twin r0.5 and 9074 at 600 px for each model.
Checkpoints: `unified_real_k15.pth`, `unified_real_k3_sgv2.pth` in the kernel Outputs.
