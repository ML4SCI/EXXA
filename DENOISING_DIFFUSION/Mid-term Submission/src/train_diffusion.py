#!/usr/bin/env python
"""
src/train_diffusion.py
======================
Entry point for training the conditional DDPM (scaled DiffusionUNet) on the
EXXA protoplanetary-disk dataset.

Pipeline
--------
1. Load ``data/dirty.npy`` / ``data/clean.npy`` (both already in [0, 1]).
2. 80/20 train/val split (``random_state=42``, matching the Week-3 notebooks).
3. Patch-based loaders via :class:`src.data.dataset.AstroDataset`
   (``parse_patches=True``, patch_size=64, n patches per image).
4. Train :class:`src.training.diffusion.DenoisingDiffusion`
   (ch=64, ch_mult=[1,2,2,4], attn@16: fits a 4 GB RTX 2050).
5. Save loss curve + best checkpoint, then DDIM-evaluate PSNR/SSIM/MSE.

Examples
--------
    # quick smoke test (tiny subset, 1 epoch)
    python -m src.train_diffusion --epochs 1 --limit 16 --no-plot

    # real run
    python -m src.train_diffusion --epochs 40 --batch-size 4 --patch-n 4
"""

import argparse
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data.dataset import create_dataloaders
from src.models.diffusion_unet import default_diffusion_config
from src.training.diffusion import DenoisingDiffusion


def parse_args():
    p = argparse.ArgumentParser(description="Train conditional DDPM for EXXA denoising")
    p.add_argument("--dirty", default="data/dirty.npy")
    p.add_argument("--clean", default="data/clean.npy")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4, help="images per batch (effective patches = batch_size * patch_n)")
    p.add_argument("--patch-n", type=int, default=4, help="random patches extracted per image")
    p.add_argument("--image-size", type=int, default=64, help="patch size (must match model resolution)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta-schedule", default="linear", choices=["linear", "quad", "const", "jsd", "sigmoid"])
    p.add_argument("--num-workers", type=int, default=0, help="0 required on Windows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="use only the first N images (debug)")
    p.add_argument("--sampling-timesteps", type=int, default=25, help="DDIM steps for evaluation")
    p.add_argument("--eval-batches", type=int, default=4, help="val batches to DDIM-evaluate (None=all)")
    p.add_argument("--checkpoint", default="results/checkpoints/diffusion_best.pth.tar")
    p.add_argument("--loss-plot", default="results/diffusion_loss.png")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    dp = p.add_mutually_exclusive_group()
    dp.add_argument("--data-parallel", dest="data_parallel", action="store_true", default=None,
                    help="force nn.DataParallel across all visible GPUs (default: auto when >1 GPU)")
    dp.add_argument("--no-data-parallel", dest="data_parallel", action="store_false",
                    help="force single-GPU even if multiple are visible")
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_loss_plot(train_losses, val_losses, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train", marker="o", ms=3)
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="val", marker="s", ms=3)
    plt.xlabel("epoch")
    plt.ylabel("noise-estimation loss (sum over CHW, mean over batch)")
    plt.title("Conditional DDPM training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"=> loss curve saved to {path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    print(f"=> device: {args.device}")
    print(f"=> loading {args.dirty} / {args.clean}")
    dirty = np.load(args.dirty).astype(np.float32)
    clean = np.load(args.clean).astype(np.float32)
    if args.limit:
        dirty, clean = dirty[: args.limit], clean[: args.limit]
    print(f"=> data: dirty {dirty.shape}, clean {clean.shape}")

    d_tr, d_val, c_tr, c_val = train_test_split(
        dirty, clean, test_size=0.2, random_state=args.seed, shuffle=True
    )
    print(f"=> split: train {d_tr.shape[0]}, val {d_val.shape[0]} images")

    train_loader, val_loader = create_dataloaders(
        dirty_train=d_tr, clean_train=c_tr,
        dirty_val=d_val, clean_val=c_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        parse_patches=True,
        patch_size=args.image_size,
        n_patches=args.patch_n,
    )

    config = default_diffusion_config(image_size=args.image_size)
    config.diffusion.num_diffusion_timesteps = args.timesteps
    config.diffusion.beta_schedule = args.beta_schedule

    diffusion = DenoisingDiffusion(
        config=config,
        device=args.device,
        lr=args.lr,
        checkpoint_path=args.checkpoint,
        data_parallel=args.data_parallel,
    )
    print(f"=> GPUs in use: {diffusion.num_gpus} "
          f"({'DataParallel' if diffusion.data_parallel else 'single'})")

    results = diffusion.train(train_loader, val_loader, n_epochs=args.epochs)

    print("\n=> training done")
    print(f"   best val loss : {results['best_val_loss']:.4f}")
    print(f"   checkpoint    : {args.checkpoint}")

    if not args.no_plot:
        save_loss_plot(results["train_losses"], results["val_losses"], args.loss_plot)

    if not args.no_eval and val_loader is not None:
        print(f"\n=> DDIM evaluation ({args.sampling_timesteps} steps, "
              f"{'all' if args.eval_batches is None else args.eval_batches} val batches)")
        # evaluate with the best (EMA) weights
        diffusion.load_checkpoint(args.checkpoint)
        metrics = diffusion.evaluate(
            val_loader,
            sampling_timesteps=args.sampling_timesteps,
            max_batches=args.eval_batches,
            use_ema=True,
        )
        print(f"   n={metrics['n']}  PSNR={metrics['psnr']:.4f} dB  "
              f"SSIM={metrics['ssim']:.4f}  MSE={metrics['mse']:.6f}")


if __name__ == "__main__":
    main()
