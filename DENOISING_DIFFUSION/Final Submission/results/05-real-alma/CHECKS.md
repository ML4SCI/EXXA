# ALMA checks S1 to S7

A real cube has no truth, so no single number says "good". These seven checks were fixed before the final real-data runs and are applied by `experiments/alma_unified.py`.

A real cube has no truth, so no single number says "good". The bar is qualitative: a pipeline that works
and gives scientific insight. These checks make that claim defensible. **S1, S6 and a visual inspection in DS9 were requested; S2 to S5
are additions [added], there to stop an artefact being reported as science.**

| id | check | what it answers | how |
|---|---|---|---|
| S1 | Noise, flux, M1 shift | Did it remove noise without changing the signal? | Built. Off-line rms, M0 total ratio, M1 shift on the brightest 10% of pixels, against the channel width. |
| S2 **[added]** | Invented structure | Does it draw structure into pure noise? | Run on line-free planes, count connected pixels above a fixed threshold. Known risk here: 08 v2 measured 22 to 39% invented blobs at a fixed threshold after its own detector wrongly returned 0%, so check the denominator. |
| S3 **[added]** | Model agreement | Which features are robust? | Compare aug43, starlet, wavelet. A feature all three keep is more credible than one only a single model draws. |
| S4 **[added]** | Pixel-scale robustness | Is the result an artefact of our resampling? | Rerun at P1, P2 and plus or minus 25%. If moments move by more than the model does, the resampling is the story. |
| S5 **[added]** | Injection-recovery on real noise | The only test with a known answer on real noise | Inject one of our held-out clean cubes (holdout RunIDs 0002, 0025, 0026) into line-free exoALMA planes at several peak SNRs, denoise, score PSNR and M0/M1/M2 with the project's own protocol so numbers compare with the project's other results. Doubles as a degradation curve on real noise. Limit: injected signal skips CLEAN, so it tests robustness to real noise statistics, not to CLEAN artefacts. |
| S6 | Kinematics | Are the velocity features kept? | Keplerian residual map of raw against denoised: correlation of the two, and residual amplitude ratio. Then compare with the published exoALMA kinematic analyses (Izquierdo et al. 2024, the discminer analysis cited in exoALMA I, and the companion papers). |
| S7 optional | Independent reference | Is the denoised cube closer to something real? | exoALMA also ships a 0.3" set at about 0.25 K noise. Smooth the fiducial and the denoised cube to 0.3" and compare each with it. |

### Provisional thresholds

Placeholders anchored to exoALMA's quoted velocity precision of about 10 m/s, to be **revised after the first
run**, not treated as pass marks: median |M1 shift| on bright pixels of at most 20 m/s and 95th percentile at most
50 m/s; M0 total ratio within 5% of 1; off-line rms reduced without invented structure (S2 near the raw rate).
If the first run lands far outside these, the first suspect is preprocessing (pixel scale, normalization), not
the model.

### Revised thresholds (2026-10-19, after the final runs; evidence in `results/RUNS.md` and the archived reports under `results/05-real-alma/`)

1. **M1: the S1 shift against the raw M1 is dropped as a pass mark; a truth-based mark replaces it.** Reason: the raw M1 is
   itself noisy, so the shift measures the raw cube's error, not the model's. On every S5 injection the shift
   (`m1_shift_model_vs_dirty_bright_median_ms`, 41 to 48 m/s at SNR 12 on exoALMA) is close to the dirty cube's own error
   against the truth (50 to 67 m/s), while the model's error is 9 to 27 m/s. New mark: **S5 line-centre error of the model
   at most half the dirty cube's at peak SNR 12 and above.** exoALMA (MWC 758, HD 135344B, DM Tau, LkCa 15, AA Tau): pass
   everywhere (0.17 to 0.40 of dirty at SNR 12, GI disc and twin). DSHARP: AS 209 passes (0.38 GI, 0.31 twin); HD 143006 fails on the GI
   disc at SNR 12 (20 vs 34 m/s, 0.59) and passes on the twin (0.43). The real-data shift stays in the reports as a description.
2. **Flux: within 5 percent of 1, measured outside 3 beam FWHM of the star**, the inner 3 beams reported separately.
   Reason: the excess is concentrated on the unresolved centre (den / raw 1.16 to 1.66 within 1 beam, 0.97 to 1.24 at
   1 to 3 beams), where the truth-based injection (a disc with a central cavity) gives no calibration
   (`experiments/alma_flux_by_radius.py`). Outside 3 beams: 13 of 15 exoALMA discs pass (0.965 to 1.034); DM Tau (0.946)
   and HD 143006 (0.938) fail by 0.4 and 1.2 points. Truth-based (S5 flux / true flux, whole disc): AA Tau 1.016 / 1.025 at
   SNR 12 and 1.017 / 1.023 at 24 (pass); DSHARP passes at SNR 24 (0.99 to 1.02), AS 209 fails at SNR 12 (1.06 / 1.07).
   The central excess (+5 to +66 percent) is a reported limit: fluxes from the denoised cube are not used within 3 beams.
3. **S2 unchanged** (invented structure in line-free planes at most the raw rate): pass on all 20 discs (at most 1e-4;
   HD 143006 DSHARP 4e-3, where those planes hold the disc's own fast inner gas).
4. **S3 / S4 marks** (new): a recipe or a +-25 percent zoom changes M1 by less than 20 m/s and flux by less than 5 percent:
   pass (A 3 to 4 m/s, 1.6 percent; zoom 4 to 9 m/s, 1.2 percent; R6, not shipped, 10 to 11 m/s, 4 percent).

---
