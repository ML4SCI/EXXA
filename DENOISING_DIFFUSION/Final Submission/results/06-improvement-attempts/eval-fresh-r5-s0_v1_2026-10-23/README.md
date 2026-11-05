# eval-fresh-r5-s0 v1, 2026-10-23

Kaggle `eval-fresh-r5-s0` v1 (launched 2026-10-22 ~22:45 IST, complete before 04:40 IST 2026-10-23). Candidate E: R5 seed(s) 0
(existing checkpoints, `unified_r5_k15_s*.pth`), each as a single network through the control's evaluation (`eval-fresh-control`
kernel with only the checkpoints changed). Young TEST3 rendered in-kernel with the same command and seed as the control, which also rendered
in-kernel; Wiener rc_r is identical on every disc, so the cubes match. `eval_fresh_r5/` = JSONs and per-case figures. Table: `python experiments/cand_step1.py`.
