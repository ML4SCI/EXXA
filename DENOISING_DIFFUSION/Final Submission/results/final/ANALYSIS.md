# Recipe B: analysis results (final model, frozen 2026-10-18 03:00)

The companion of `RESULTS.md` (the generated five-size tables). Every number here is copied from the run archive named next to it and from the
 rc_r = correlation of the rotation-curve-subtracted M1 residual
with the truth's (the 2D wiggle score); PV r = correlation of the quadratically detrended minor-axis M1 curve with the truth's (`experiments/pv_curve.py`,
the Speedie et al. 2024 measurement; 1 px slice or 5 px slit); PV magnitude = standard deviation of that detrended curve (m/s). Peak SNR = peak of the
clean cube over the noise rms. Model = recipe B (`RECIPES.md`).

## 1. The one-time final test (Speedie r1.5, never used for any decision)

Source: `results/01-final-test/eval-final-test_v1_2026-10-18/` (`eval-final-test`, run once). The headline wiggle-against-SNR curve.

| Speedie r1.5 | SNR 6 | SNR 12 | SNR 24 |
|---|---|---|---|
| rc_r: model / Wiener / dirty | 0.06 / 0.05 / 0.00 | **0.44** / 0.17 / 0.03 | **0.70** / 0.41 / 0.16 |
| PV r (1 px = 5 px): model / Wiener / dirty | 0.16 / 0.04 / 0.03 | **0.35** / 0.01 / 0.12 | **0.71** / 0.27 / 0.64 |
| PV magnitude, GI disc (clean 63 m/s): model / Wiener / dirty | 102 / 197 / 1130 | 94 / 72 / 847 | 60 / 32 / 88 |
| PV magnitude, no-wiggle twin (clean 3 m/s): model / Wiener / dirty | 81 / 233 / 1004 | 84 / 24 / 336 | 49 / 27 / 75 |
| selection disc r0.5, rc_r model (Wiener) | 0.14 (0.07) | 0.60 (0.18) | 0.83 (0.43) |

Selection optimism: 0.13 to 0.16 in rc_r between the selection disc and the final test, while Wiener is the same on both. **Error bars (2026-10-21, `results/01-final-test/eval-final-test-seeds_v1_2026-10-21/`):** over 4 noise realisations of the same test (realisation 0 = the one above) rc_r is 0.59 +- 0.10 at SNR 12 and 0.79 +- 0.06 at SNR 24 (Wiener 0.18 +- 0.03 / 0.42 +- 0.02), PV5 r 0.68 +- 0.24 / 0.85 +- 0.10; the reported draw was the least favourable of the four, so most of that optimism was the noise draw (gap to the selection ladder 0.09 / 0.05). On the twin the model leaves a
ripple of about +-20 m/s plus inner-edge swings; at SNR 12 a PV magnitude alone cannot separate a wiggle from that ripple, the PV correlation can.

## 2. PV curves on the other held-out GI discs (selection set, eval-final-b)

Source: `results/07-image-sizes/eval-final-b_v1_2026-10-17/`. PV r, model / Wiener / dirty, 1 px slice (5 px slit):
Speedie GI r0.5 0.87 / 0.81 / 0.23 (0.89 / 0.83 / 0.31); an07 at 600 px 0.90 / 0.03 / 0.07 (0.91 / 0.02 / 0.10); Young M1 R100 at 600 px 0.80 / 0.45 /
0.73 (0.88 / 0.45 / 0.81): on Young M1 R100, a strong-wiggle disc at moderate SNR, the noisy cube's own PV curve is nearly as good as the model's, so the
gain there is over Wiener, not over dirty. Invented wiggle on the no-wiggle twin (PV, SNR 12): about 20 m/s.

## 3. Eight more held-out Young discs (weak wiggles)

Source: `results/09-design-choices/eval-young-test2_v1_2026-10-18/` (reproduced to the last digit by a duplicate run). Clean minor-axis wiggles of 1 to 14 m/s
(PV magnitude). Means at 600 px, model / Wiener / dirty: rc_r 0.35 / 0.24 / 0.20, PV r (5 px) 0.30 / 0.07 / 0.33. Moments and PSNR strong on every disc
(PSNR 49.5 to 53.2 dB vs Wiener 31.2 to 33.8; M1 89 to 94 percent). Two near face-on discs (14.3 and 12.3 deg, from the render log) give nothing. Statement: the model recovers GI
wiggles whose minor-axis amplitude is tens of m/s or more (Speedie, Young M0.7 R50 and M1 R100, an07); about 10 m/s or less, or nearly face-on, is not
recovered by it or by anything else tested.

## 3b. Detection limit

Source: `results/final/detection_limit.md`. **Statement.** Wiggles with a minor-axis PV magnitude of about 20 m/s or more are recovered reliably at blind SNR 13 to 18 (an07, an08, Young M1 R100, Young M0.7 R50: PV r 0.83 to 0.92, rc_r 0.66 to 0.87), and the 63 to 72 m/s Speedie wiggle partly at peak SNR 12 (PV r 0.35 to 0.44) and well at 24 (0.71 to 0.79); nothing is recovered at peak SNR 6 or below. Below about 15 m/s recovery is inconsistent (11 Young discs, 1 to 14 m/s: PV r minus 0.03 to 0.78, median 0.29, against the noisy cube's median 0.22): that is the limit.

## 4. Hydro disc 9074: SNR ladder, PV, generalisation

Sources: `results/04-hydro-9074/eval-ladder-sg_v1_2026-10-18/` (rc_r) and `results/04-hydro-9074/eval-ladder-sg_v2_2026-10-18/` (beam-matched PV). 9025 is a TRAINING disc (diagnostic only).

| rc_r, 600 px | native | SNR 11 | 20 | 40 | 80 |
|---|---|---|---|---|---|
| 9074 noise-only: model / dirty / Wiener | 0.45 / 0.31 / 0.13 | 0.40 / 0.15 / 0.09 | 0.48 / 0.32 / 0.09 | 0.57 / 0.48 / 0.22 | 0.70 / 0.61 / 0.62 |
| 9025 noise-only (training disc): model / dirty | 0.35 / 0.08 | 0.57 / 0.13 | 0.66 / 0.29 | 0.78 / 0.42 | 0.82 / 0.57 |
| 9074 deconvolution: model / dirty | 0.35 / 0.13 | 0.36 / 0.21 | 0.38 / 0.34 | 0.56 / 0.47 | 0.64 / 0.63 |
| 9074 PV 5 px vs beam-matched truth (deconv.): model / dirty | 0.45 / 0.51 | 0.47 / 0.53 | 0.37 / 0.46 | 0.33 / 0.45 | 0.21 / 0.40 |
| 9074 PV 5 px vs truth (noise-only): model / Wiener | 0.27 / 0.77 | 0.36 / 0.34 | 0.43 / 0.62 | 0.53 / 0.51 | 0.80 / 0.78 |

Reading: noise-limited at native SNR (9.5) and also a generalisation gap of 0.12 to 0.21 against a training disc at the same SNR on the ladder; the model keeps 57 to 72
percent of the wiggle's amplitude at every SNR; in the deconvolution regime at high SNR it equals dirty. On 9074's minor axis the model is worse than Wiener in the noise-only case (0.27 against 0.77) and not above dirty in the deconvolution case (negative result): the PV advantage on GI discs does not carry over to this hydro disc. The beam-only oracle reaches rc_r 0.93.

## 5. Planet check (two spots flagged in the review)

Source: `results/08-compact-features/eval-planet_v1_2026-10-18/planet_check/`. Young M1 R100 at 7.6 o'clock: a compact 171 m/s feature in clean (4.6 times the
ring scatter, one beam wide) that no method recovers (model amplitude 0.03). run_0002_00560 (rendered with planets' emission off, gravity kept): no
isolated kink stands out; the strongest localised feature on the 64 deg view is inner-disc systematics, which the model keeps (amplitude 0.82).

## 6. One checkpoint or two: R6 (three seeds) against B

Sources: `results/10-training-runs/r6-k15-crop256_v2_2026-10-15/`, `results/10-training-runs/r6-k15-s32_v1-v2_2026-10-17/`, `results/10-training-runs/r6-k15-s33b_v2_2026-10-18/`, `results/07-image-sizes/eval-final-b_v1_2026-10-17/`.

| | 9074 @600 rc_r deconv. | 9074 noise-only | line @600 PSNR | Speedie r0.5 SNR 12 / 24 | an07 @600 |
|---|---|---|---|---|---|
| R6, 3 seeds (mean +- spread) | 0.40 +- 0.03 | 0.38 +- 0.02 | 46.9 +- 1.2 | 0.60 / 0.80 | 0.83 |
| recipe B | 0.35 | **0.45** | **51.7** | 0.60 / **0.83** | **0.87** |

Recommendation: one checkpoint (B); R6 documented as the alternative (decision of the author).

## 7. Real ALMA: exoALMA (15 discs, 13CO) and DSHARP (5 discs, 12CO)

Setting: 512 px field, beam resampled to about 10 px (`--target-beam-px 10`; DSHARP capped at a 1024 px grid), DSHARP channels interpolated to 0.1 km/s;
continuum subtracted only where both far sides of the line show the same structure (V4046 Sgr only: a 1 sigma per channel point source at the star).
Sources: `results/05-real-alma/exoalma-final-p0_v1_2026-10-18/` to `results/05-real-alma/exoalma-final-p3_v1_2026-10-18/`, `results/05-real-alma/dsharp-final-p0_v1_2026-10-18/`, `results/05-real-alma/dsharp-final-p1_v1_2026-10-18/`, `results/05-real-alma/dsharp-final-hd163296_v1_2026-10-18/` (all in `results/05-real-alma/`); geometry refits (HD 142666, HD 143006
13CO, PDS 66, HD 34282) replace the kernel's S6 where present. Table from `experiments/alma_table.py` (PA difference modulo 180; M* with the inclination
fixed at the published value; no truth on real data, so S6 and PV are agreement with the RAW cube):

| disc | blind SNR | beam px (zoom) | flux | noise red. | S2 | S6 rc_r | PV5 r | PV5 mag den / raw (m/s) | PA sky fit / lit (diff mod 180) | M* | S5 GI rc_r | cont. sub |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HD_163296_interp01 | 28 | 8.0 (1.00) | 0.991 | 21x | 2e-05 | 0.98 | 0.97 | 41 / 47 | 132.7 / 133.3 (0.6) | 1.71 | - | False |
| HD_142666_interp01 | 7 | 7.7 (1.29) | 1.030 | 10x | 1e-04 | 0.69 | -0.06 | 60 / 272 | 162.1 / 162.1 (0.0) | 1.47 | - | False |
| HD_143006_interp01 | 8 | 16.5 (1.00) | 1.006 | 2x | 4e-03 | 0.84 | 0.88 | 12 / 44 | 165.1 / 169.0 (3.9) | 1.56 | -0.20 / -0.21 / -0.19 / -0.15 | False |
| IM_Lup_interp01 | 13 | 12.2 (1.00) | 0.977 | 45x | 0e+00 | 0.93 | 0.94 | 49 / 87 | 144.4 / 144.5 (0.1) | 1.02 | - | False |
| AS_209_interp01 | 12 | 9.5 (1.05) | 1.030 | 11x | 1e-04 | 0.85 | 0.63 | 30 / 45 | 84.9 / 85.8 (0.9) | 1.24 | 0.26 / 0.55 / 0.86 / 0.95 | False |
| HD_135344B_13CO | 15 | 6.0 (1.67) | 1.045 | 183x | 0e+00 | 0.77 | 0.87 | 25 / 53 | 61.7 / 242.9 (1.2) | 1.60 | 0.86 / 0.93 / 0.95 | False |
| HD_34282_13CO | 15 | 6.0 (1.67) | 1.060 | 88x | 1e-07 | 0.85 | 0.51 | 12 / 16 | 117.1 / 117.4 (0.3) | 1.68 | - | False |
| MWC_758_13CO | 11 | 6.0 (1.67) | 1.037 | 196x | 0e+00 | 0.76 | 0.79 | 36 / 54 | 62.0 / 240.3 (1.7) | 1.34 | 0.85 / 0.93 / 0.95 | False |
| V4046_Sgr_13CO | 18 | 6.0 (1.67) | 1.075 | 62x | 1e-06 | 0.77 | 0.68 | 20 / 33 | 75.7 / 255.7 (0.0) | 1.67 | - | True |
| CQ_Tau_13CO | 9 | 6.0 (1.67) | 1.141 | 100x | 0e+00 | 0.24 | 0.58 | 31 / 38 | 52.7 / 235.1 (2.4) | 1.30 | - | False |
| DM_Tau_13CO | 9 | 6.0 (1.67) | 1.059 | 51x | 2e-06 | 0.49 | -0.17 | 6 / 14 | 156.1 / 335.7 (0.4) | 0.48 | 0.68 / 0.84 / 0.90 | False |
| HD_143006_13CO | 7 | 6.0 (1.67) | 1.142 | 41x | 2e-06 | 0.43 | 0.88 | 21 / 26 | 167.4 / 168.9 (1.5) | 1.31 | - | False |
| J1842_13CO | 7 | 6.0 (1.67) | 1.147 | 76x | 7e-07 | 0.64 | 0.69 | 35 / 39 | 26.8 / 205.9 (0.9) | 1.04 | - | False |
| AA_Tau_13CO | 8 | 6.0 (1.67) | 1.090 | 82x | 1e-07 | 0.80 | 0.38 | 26 / 27 | 93.2 / 272.7 (0.5) | 0.74 | - | False |
| J1604_13CO | 18 | 6.0 (1.67) | 0.992 | 224x | 0e+00 | 0.08 | 0.64 | 9 / 28 | 89.8 / 258.1 (11.7) | 0.43 | - | False |
| J1852_13CO | 13 | 6.0 (1.67) | 1.065 | 143x | 3e-08 | 0.81 | -0.32 | 9 / 17 | 118.1 / 117.1 (1.0) | 0.96 | - | False |
| SY_Cha_13CO | 6 | 6.0 (1.67) | 1.073 | 166x | 0e+00 | 0.73 | 0.24 | 18 / 50 | 163.6 / 346.2 (2.6) | 0.66 | - | False |
| J1615_13CO | 12 | 6.0 (1.67) | 1.022 | 80x | 1e-07 | 0.79 | 0.54 | 10 / 26 | 145.8 / 325.4 (0.4) | 1.14 | - | False |
| LkCa_15_13CO | 11 | 6.0 (1.67) | 1.018 | 124x | 0e+00 | 0.77 | 0.65 | 12 / 21 | 61.2 / 62.1 (0.9) | 1.15 | 0.75 / 0.88 / 0.92 | False |
| PDS_66_13CO | 7 | 6.0 (1.67) | 1.291 | 35x | 1e-05 | 0.31 | 0.75 | 66 / 83 | 9.5 / 189.0 (0.5) | 1.21 | - | False |

**DSHARP S5 rows above are superseded** by the rerun with correct framing and clean noise (`results/05-real-alma/dsharp-s5fix_v1_2026-10-19/`): GI rc_r at SNR 6 / 8 / 12 / 24, model vs Wiener: HD 143006 0.00 / 0.10 / 0.21 / 0.85 vs -0.04 / 0.01 / 0.12 / 0.56; AS 209 0.01 / 0.16 / 0.41 / 0.93 vs 0.05 / 0.19 / 0.65 / 0.89 (the model loses to Wiener at SNR 12 on AS 209); model flux / truth 1.35 to 1.41 at SNR 6, 1.19 to 1.25 at SNR 8, 0.99 to 1.07 at 12 to 24.

Findings: (a) **Geometry**: 19 of 20 discs land within 0.0 to 3.9 deg of the published PA (exoALMA I discminer / Huang et al. 2018),
with M* close to exoALMA I's; J1604 (6 deg) is untestable. (b) **Injection-recovery in real noise** (GI rc_r, model at SNR 12 / 24 / 36): MWC 758
0.85 / 0.93 / 0.95, HD 135344B 0.86 / 0.93 / 0.95, LkCa 15 0.75 / 0.88 / 0.92, DM Tau 0.68 / 0.84 / 0.90 (Wiener 0.16 to 0.89, dirty 0.04 to 0.48);
line-centre error 12 to 27 m/s (dirty 26 to 67); DSHARP (correct framing): AS 209 0.01 / 0.16 / 0.41 / 0.93, HD 143006 0.00 / 0.10 / 0.21 / 0.85 at SNR 6 / 8 / 12 / 24, Wiener equal or better below SNR 24. (c) **Flux**: biased
high by the model, measured against the truth (AA Tau injection, `results/05-real-alma/aatau-s5flux_v1_2026-10-18/`): model / true flux 1.085 / 1.039 / 1.016 /
1.017 at SNR 6 / 8 / 12 / 24 (dirty 0.98 to 1.00); the real-disc ratios above (1.02 to 1.29, highest on the faintest) follow the same trend. (d) **PV on
real discs**: the denoised minor-axis curve is flatter than the raw one (5 px magnitude 49 vs 87 m/s on IM Lup, 6 vs 14 on DM Tau); part is noise removal,
part looks like lost structure (IM Lup + half, DM Tau step at 100 au), so the denoised PV curve is a lower bound on real minor-axis structure; at high
SNR (HD 163296, 28) raw and denoised agree (r 0.97). PV agreement is not meaningful on PDS 66 (29 to 58 au profile) or at blind SNR 6 to 8 (raw curve
noise-dominated). (e) S2 (invented structure in line-free planes above 4 sigma): at most 4e-3 (HD 143006 DSHARP, where those planes hold real emission),
otherwise at most 1e-4.

## 7b. S6 against published kinematics, flux by radius, pass marks

**Published rotation curves** (exoALMA VI 13CO, all 15 discs; `results/05-real-alma/s6-published-rotation_2026-10-19/`): median |ours - published| 64 m/s from the
raw M1, 49 m/s from the denoised M1 (closer on 11 of 15); rotation-curve substructure correlation median 0.43 raw, 0.57 denoised (higher on 11 of 15).
The remaining offset grows with inclination (thin-disc fit against exoALMA VI's elevated surface) and is the same raw or denoised.
**Flux by radius:** den / raw 1.16 to 1.66 within 1 beam of the star, 0.97 to 1.24 at 1 to 3 beams, 0.94 to 1.03 outside 3 beams (all 15 discs). The
excess is the unresolved centre; flux from the denoised cube is used only outside 3 beams. V4046 Sgr: 1.022 outside 3 beams.
**Pass marks**: truth-based line-centre error at most half of dirty's at SNR 12+ (exoALMA all pass; DSHARP HD 143006
fails); flux within 5 percent outside 3 beams (13 / 15; DM Tau 0.946, HD 143006 0.938 fail); S2 all pass; S3 / S4 pass.

## 7c. Flux root cause, S7, error bars, cross-survey (2026-10-21)

Sources: `results/05-real-alma/alma-kepc-mwc758_v1_2026-10-19/`, `results/05-real-alma/alma-s7-lowres_v1_2026-10-19/`, `results/05-real-alma/alma-seeds-mwc758_v1_2026-10-19/`, `results/05-real-alma/alma-hd143006-12co_v1_2026-10-19/`, `results/01-final-test/eval-ladder-seeds_v1_2026-10-20/`.
- **Central flux excess = model bias:** analytic compact-centre disc in MWC 758 noise: model / true flux within 1 beam 1.30 to 1.39 on the first noise draw (1.28 to 1.41 over three draws) at SNR 6 / 12 / 24 (dirty 1.01 to 1.06 on the first draw, 0.94 to 1.06 over three); faint outer flux under-represented at low SNR (0.41 beyond 6 beams at SNR 6). Denoised flux is used only outside 3 beams of the star.
- **S7 (independent 0.3 arcsec images, 4 discs): no improvement shown.** Denoised rms difference 1.2 to 1.6 sigma_low vs raw 0.9 to 1.1; M1 difference 14 to 37 vs 8 to 12 m/s; the test favours raw (shared visibilities), and the independent image agrees with raw at the star. Reported as inconclusive-to-unfavourable.
- **Error bars:** MWC 758 injection (5 noise realisations): GI rc_r SNR 12 / 24 = 0.844 +- 0.004 / 0.926 +- 0.002 (Wiener 0.638 +- 0.008 / 0.831 +- 0.001); line-centre error 20.0 +- 0.2 / 14.6 +- 0.2 m/s. Speedie r0.5 selection ladder (GI, 8 / 8 / 6 realisations at SNR 6 / 12 / 24): rc_r 0.24 +- 0.11 / 0.68 +- 0.03 / 0.84 +- 0.01; PV 5 px r 0.53 +- 0.22 / 0.74 +- 0.15 / 0.88 +- 0.05. The one-time final test (r1.5) is a single realisation by design.
- **Invented wiggle on the no-wiggle twin (Speedie r0.5 twin, 4 realisations, `results/02-spurious-wiggle/eval-ladder-twin_v1_2026-10-21/`):** PV 5 px magnitude of the model 27 +- 5 / 18 +- 3 m/s at SNR 12 / 24 (clean 11.4; Wiener 33 +- 11 / 14 +- 6; dirty 510 / 113); 2D residual above clean 22.2 +- 0.9 / 18.9 +- 0.3 m/s (Wiener 46.5 / 23.5). A PV magnitude below about 25 to 30 m/s at SNR 12 (20 at SNR 24) cannot be told from this ripple.
- **Cross-survey, HD 143006 12CO (exoALMA vs DSHARP):** rotation-curve substructure agreement between the two independent observations 0.20 (raw) to 0.80 (denoised); one disc, smooth curves, no significance test.

## 8. S3 (recipes) and S4 (pixel scale) on four real exoALMA discs

Sources: `results/05-real-alma/alma-s3-recipes_v1_2026-10-18/`, `results/05-real-alma/alma-s4-pixscale_v1_2026-10-18/`. Discs: MWC 758, LkCa 15, HD 135344B, J1615. Agreement with B's final run:

| variant | flux var / B | median abs dM1 (m/s) | PV 5 px agreement (r) | MWC 758 S5 GI rc_r, SNR 12 / 24 / 36 |
|---|---|---|---|---|
| recipe A | 1.006 to 1.016 | 3 to 4 | 0.94 to 1.00 | 0.85 / 0.93 / 0.95 (B: the same) |
| R6 | 0.961 to 1.017 | 10 to 11 | 0.81 to 0.98 | 0.74 / 0.89 / 0.93 |
| beam 7.5 px | 0.988 to 1.001 | 7 to 9 | 0.84 to 0.99 | 0.85 / 0.93 / 0.95 |
| beam 12.5 px | 0.993 to 0.998 | 4 to 5 | 0.81 to 0.99 | 0.85 / 0.93 / 0.95 |

Reading: the real-data result barely depends on the recipe (A) or on the zoom (plus or minus 25 percent); R6 differs more and recovers less of an injected wiggle. Where PV agreement falls to 0.81 to 0.84 the disc's PV curve is noise-dominated (PV against raw 0.54 to 0.65 for B).

## 9. Improvement track: fresh test set

Source: `results/03-fresh-set/eval-fresh-b_v2_2026-10-21/fresh_tables.md`, `results/06-improvement-attempts/alma-kepc-cfix_v1_2026-10-21/`. Discs never used for any choice; 600 px; gate inactive.

| fresh analytic, mean of 8 GI discs | rc_r B / Wiener / dirty | PV5 r B / Wiener | PSNR B / Wiener (dB) |
|---|---|---|---|
| deconvolution | 0.786 / 0.470 / 0.40 | 0.87 / 0.61 | 42.4 / 27.5 |
| noise-only (blind SNR 19 to 34) | 0.796 / 0.778 / 0.55 | 0.87 / 0.89 | 50.3 / 30.5 |

| Young TEST3, 8 discs (noise-only line cubes) | 0.561 / 0.333 / 0.246 | 0.52 / 0.44 | 51.1 / 32.6 |

On fresh discs B keeps its wiggle advantage in the deconvolution regime and on the fresh Young simulations (without the two near face-on discs, which give nothing for any method: 0.74 vs 0.44); on the smooth analytic discs in the noise-only regime at blind SNR 19 to 34 it ties Wiener on the wiggle while staying 20 dB better in PSNR. Invented PV5 magnitude on the three twins: 8 to 17 m/s (deconvolution; Wiener 2 to 6), 6 to 9 m/s (noise-only; Wiener 4 to 7). Sources: `results/03-fresh-set/eval-fresh-b_v2_2026-10-21/`, `results/03-fresh-set/eval-fresh-young_v1_2026-10-21/`.

**Candidate A (central flux fix): does not replace B; effect depends on the disc.** Known-truth compact centre (KepC): flux within 1 beam / truth 0.91 to 1.03 (B 1.28 to 1.41). Fresh Young simulations: M0 +8.8 points over B, better on all 8, PSNR +0.7 dB. Fresh analytic discs, noise-only: M0 -6.5 (worst -11.4). M1, M2 and every wiggle score identical to B in all cases. It cannot meet adoption rule (1) (wiggle +0.05) by construction; rule (2) passes averaged over rule (1)'s set (+1.1 M0 points) and fails on the analytic group alone. It is shipped as the optional flag `--central-fix`; by default central flux is taken from the input cube.
