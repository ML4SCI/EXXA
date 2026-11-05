# Notebook 05, Kaggle Version 21 (code `6f5c798`)

This is the run where sections 7 and 8 finally finished and kept their output. Before it, the
CSV write in section 7 failed on a missing `n_background_px` column, and it failed after the whole
300-channel analysis had already run. The numbers were printed and then lost. The fix is in
`6f5c798`, and this run saved them.

Nothing was trained here and there are no new moment scores. Three checkpoints came back from
v19's Output and twelve from notebook 08, and `winner_patch` was scored from its stored weights,
which gives 15 rows over 6 arms. The moment table is the same as in the run before it. What is
new is all in sections 7 and 8.

## Artifact diagnostics for `sweep_winner_p10`, seed 42 (300 validation channels)

```
PEAK OVERSHOOT   denoised.max / clean.max
  mean 1.418 | median 0.848 | p90 1.054 | max 91.580
  channels overshooting by >10%: 8%

NEGATIVE FLOOR LEAK   denoised.min   (clean floor ~0)
  mean -0.13108 | most negative -0.21807

INVENTED STRUCTURE   (background above 20% of clean peak, blobs >= 20 px)
  channels with >=1 fake blob: 33% | blobs/channel 0.97
  mean invented background area: 6.6486% | worst channel: 99.9587%

  SNR split at median 3.9
    low-SNR  half: blobs/channel 1.63 | invented area 13.2590% | overshoot 1.992
    high-SNR half: blobs/channel 0.31 | invented area 0.0382% | overshoot 0.844
```

## What the numbers say

**Invented structure switches on rather than growing gradually.** The scatter plot in
`figure_cell24.png` has two clusters. About 25 channels sit near 100% invented background area,
and nearly everything else sits near 0%, with almost nothing in between. Channels do not get
steadily worse as the SNR drops; they flip. Every channel in the failing cluster has an SNR below
about 0.5. So the 6.65% mean is misleading as a "typical" value. What is really happening is that
about 8% of channels fail almost completely.

**The floor leak is systematic.** `denoised.min` clusters around -0.10 and -0.155, and no channel
gets near zero, where the clean floor is. Every channel comes out with a negative pedestal. It is
the same kind of failure as the DDPM's positive pedestal, only on the other side of zero, and it
also drives the overshoot statistic.

**The worst channels are where the metric stops meaning much.** The three in `figure_cell26.png`
report `SNR 0.0` and `invented 99.1-99.96%` over about 65,500 background pixels. "Invented" is
measured against 20% of the clean peak, and on a channel with no signal that peak is just a noise
spike, so the bar is close to zero and any pedestal clears it. These are real failures, but the
percentage is not a meaningful size. It is the same lesson as a suspicious zero: when a
diagnostic gives an extreme number, check what it was divided by before believing it.

## Why the SNR split matters

The gap between the two halves is the clearest sign of the faint-channel problem in this
project: 347 times in invented area and 5.3 times in blob rate, at the same split. It fits the
idea that M2 suffers from truncated line wings, because the model's inventions and the line wings
sit in the same faint regime.

An earlier run (v18) reported 1.580 against 0.213 for the same split. That was a different
checkpoint from a different run, so it is a separate measurement and not a correction of this
one.
