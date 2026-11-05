# unified-wiener-real v1 (Kaggle, R3), FAILED at step 2242 (round 7/40)

R3 of K=3, lr 3e-4, realistic noise (peak SNR 10-150, channel kernel a ~ U(0, 0.5), beam x U(1, 1.6)), velocity-aware Wiener,
12000 steps planned, 19 training cubes. Rounds 1-6 healthy and better than unified-wiener v1 (L1 0.0139 at round 6 against 0.018-0.025 there).
Step 2013: first non-finite loss on ORDINARY inputs (range -1..2.8); round 7: 70 non-finite steps of 300, L1 0.0257; then 51 in a row, abort.

Cause (a probe of the saved round-7 weights, fp32, ordinary real batches): weights finite, output max 0.67, but the decoder shortcut convs
reach 1.27e5 (dec_blocks.2.0.shortcut) and 9.9e4 (dec_blocks.3.0.shortcut); fp16 max is 65504, so the fp16-autocast forward overflows to inf.
The U-Net's residual stream (out = h + shortcut(x), src/models/unet.py) is not normalised and grows with training. Not the data, not the
noise recipe, not the learning rate alone.

Fix (train_unified.py `--fp32-stream`, new runs only): shortcut convs, down/up-sampling convs and the residual add in fp32; verified
numerically identical to the original blocks (max difference 0.0) with the same state-dict keys. Relaunch as unified-wiener-real-fp32 (control) and
unified-wiener-real-fp32-sgv2 (R3c, with SG v2). The k15 failures (v1-v3) are most likely the same mechanism.
