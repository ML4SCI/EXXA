# unified-wiener-k15 v1 (Kaggle), FAILED at round 5/40

Round 2 of the unified model: Wiener + learned correction, window +-1.5 km/s (K=15, 31 channels in, 63 with the
Wiener copy), deconvolution and noise-only tasks (p_ai 0.5), planned 40 rounds x 300 steps. Launched 2026-10-10 05:44 IST.

Died at step 1532 (33 min): `FloatingPointError: non-finite loss at step 1532: inputs finite=True`.
The batch was finite, so the inf/NaN came from the fp16-autocast forward pass. The last full round had
L1 0.030, d1 0.018, d2 0.009, no sign of divergence before it. The trainer raised on the first non-finite step.

What survived: `unified_wiener_k15.pth` at round 5 in the kernel Output (not scored, not used).

Fix (train_unified.py): a non-finite step is now skipped (gradients dropped, loss scale halved) and the run
aborts only after 50 in a row or if the inputs themselves are non-finite. The round mean is taken over finite
steps and the log prints the running count of skipped steps. Checked locally on MPS by injecting two NaN
losses into a smoke run: both skipped, run completed. Relaunched as v2.

Files: the README only (the log and script stay in the kernel Output).
