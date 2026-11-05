# unified-wiener-k15 v3 (Kaggle), FAILED at step 5032 (round 13/40, ~80 min of training)

Same config as v1 and v2 (K=15, 31-channel window, deconvolution and noise-only tasks, p_ai 0.5, lr 3e-4, 12000 steps), with the skip-bad-step
trainer. Got past step 1532 (where v1 and v2 died) but the run had already diverged there:

- Rounds 1-5: L1 0.046, 0.037, 0.033, 0.029, 0.030 (healthy).
- Step 1532: first non-finite loss; 13 skipped steps in round 6; round 6 L1 0.052 and it never came back (0.045-0.054 for rounds 6-12).
- Rounds 12-13: 23 more non-finite steps per round, then, at step ~5000, alternating non-finite and CUDA-OOM steps (51 in a row) and the abort.
- Inputs in the bad batches: most -3..5 (the normal range), but a group at -30..+40 (about 10 x the usual); those overflow fp16 in the 63-channel input conv.

The OOM steps in the last rounds are a consequence (the graph of a failed step), not the cause. Round-5 weights were overwritten by later (degraded) rounds in the
kernel Output, so the saved `unified_wiener_k15.pth` is the round-13 model and not useful.

Fix (train_unified.py `--clip-in`): clamp the network input to +-10, lower the learning rate to 1.5e-4, keep the skip logic; relaunched with realistic noise as
`unified-wiener-real-k15`.
