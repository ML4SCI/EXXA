# unified-ft-kin, Kaggle Version 1 (2026-10-09 23:27 IST to 10 ~02:55): fine-tune kin_gamma0_mae_ft and kin_gamma0_starlet_ft on the mixed data

Private GPU script kernel `unified-ft-kin`; code `experiments/train_unified.py --init <ckpt> --lr 1e-4 --rounds 20 --steps 300 --before` (legacy arm: each checkpoint keeps its own 31-channel
input / 31-channel output format; dirty input, no Wiener prior), data mix of line emission, hydro and analytic disks (each simulation served as the deconvolution task or, 50 %, the noise-only task), fresh noise
every round, sizes 256/300/320/480/600. Per checkpoint: 6000 steps in ~48 min, scoring ~60 min, 110 min total. Source checkpoints: `kin_gamma0_{mae,starlet}_ft` as `.ckpt` in `exxa-unified-ft-sources`.

The evaluation JSONs stay in the kernel Output: one per checkpoint after fine-tuning (12 cases x 5 sizes + native) and one for the starting checkpoint under the same protocol (7 cases at 300 and 600 px).
6 cases failed to score per checkpoint (the empty-spine-track bug, fixed afterwards). Checkpoints: `ft_<ckpt>/unified_ft_<ckpt>.pth` in the kernel Output.
Caveat: "before" covers fewer cases than "after"; compare only matched cases (experiments/compare_models.py prints group means over different subsets).
