# eval-fresh-b v2, 2026-10-21

Kaggle `eval-fresh-b` v2 (complete; v1 failed in 1 min, TEST3 input not visible from the other account). Fresh test set of the improvement track: recipe B (`fresh_B`) and candidate A = B + `--central-fix` (`fresh_A_cfix`) scored by the same kernel on analytic GI an09 to an16
and the no-wiggle twins an09k / an12k / an15k at 600 px, deconvolution and noise-only (`_ai`) regimes. Gate inactive on every case (gate_a 1.0,
blind SNR 14.8 to 33.5). **The Young TEST3 part did not run**: the in-kernel render failed (`sarracen` not installed, all 8 FAILED, the kernel did
not stop); it is scored by `eval-fresh-young`. Tables: `fresh_tables.md` (from `experiments/fresh_tables.py`).

Figures: a moment / wiggle panel and a minor-axis PV plot per case stayed in the kernel Output; a selection is in [`../../final/figs/`](../../final/figs/).
