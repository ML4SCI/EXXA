# Results

Every number in the top-level README comes from a folder here. Each numbered folder collects the runs behind one result, with a README that states the result, shows its figures and lists the runs; each run folder keeps the evaluation JSON and reports the run produced, with its own README where it has one. Logs, per-case figures, kernel sources, FITS outputs and checkpoints stayed in the Kaggle Output and are not in this repository; where a README names one of them, it describes what the original run produced. The networks of the final model are described in [`../MODELS.md`](../MODELS.md).

| folder | result |
|---|---|
| [`01-final-test/`](01-final-test/) | the one-time held-out test, noise realisations, the linear limit |
| [`02-spurious-wiggle/`](02-spurious-wiggle/) | the Keplerian twin, spurious wiggle and the detection limit |
| [`03-fresh-set/`](03-fresh-set/) | eight new analytic disks, three twins and eight Young simulations, disk by disk |
| [`04-hydro-9074/`](04-hydro-9074/) | the held-out hydrodynamic disk, the hard case |
| [`05-real-alma/`](05-real-alma/) | 15 exoALMA and 5 DSHARP disks: geometry, rotation curves, injection, flux, uncertainty |
| [`06-improvement-attempts/`](06-improvement-attempts/) | four candidates against a rule fixed beforehand |
| [`07-image-sizes/`](07-image-sizes/) | PSNR and moment maps at five image sizes |
| [`08-compact-features/`](08-compact-features/) | planet-like features and the HD 142666 kink test |
| [`09-design-choices/`](09-design-choices/) | how each inference setting was tested before it was kept |
| [`10-training-runs/`](10-training-runs/) | the runs that trained the four final networks, and the alternatives |

Summaries across results: [`final/ANALYSIS.md`](final/ANALYSIS.md) (analysis with error bars), [`final/RESULTS.md`](final/RESULTS.md) (the five-size table), [`final/RECIPES.md`](final/RECIPES.md) (inference settings), [`final/detection_limit.md`](final/detection_limit.md). All figures are in [`final/figs/`](final/figs/). [`RUNS.md`](RUNS.md) lists every run of the final phase in launch order.

The denoised cubes, moment-map FITS files and per-case images stayed in the Kaggle outputs; only reports and tables are kept here. Model checkpoints (the `unified_*.pth` files named in the run READMEs) are not committed either; see [`../MODELS.md`](../MODELS.md).
