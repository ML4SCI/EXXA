"""
src/train_autoencoder.py
========================
Trains DenoisingAutoencoder on dirty->clean astronomical image pairs.

Usage
-----
    python src/train_autoencoder.py

Outputs
-------
    results/checkpoints/autoencoder_best.pth: best model checkpoint
    results/autoencoder_loss.png: train/val loss curves
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project root on sys.path so imports work when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.autoencoder import DenoisingAutoencoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PATCH_SIZE  = 64
BATCH_SIZE  = 16
NUM_WORKERS = 0        # Windows: spawn-based multiprocessing breaks DataLoader
EPOCHS      = 30
LR          = 1e-3
VAL_SPLIT   = 0.20
SEED        = 42

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset: one random 64x64 patch per sample per epoch
# ---------------------------------------------------------------------------
class PatchDataset(Dataset):
    """
    Extracts one random PATCH_SIZE x PATCH_SIZE patch per image per epoch.

    dirty : np.ndarray  (N, H, W)  float32
    clean : np.ndarray  (N, H, W)  float32
    """

    def __init__(self, dirty: np.ndarray, clean: np.ndarray, patch_size: int = 64):
        super().__init__()
        assert dirty.dtype == np.float32 and clean.dtype == np.float32
        self.dirty      = dirty
        self.clean      = clean
        self.patch_size = patch_size
        self._h, self._w = dirty.shape[1], dirty.shape[2]
        assert self._h >= patch_size and self._w >= patch_size

    def __len__(self):
        return len(self.dirty)   # one patch per image per epoch

    def __getitem__(self, idx):
        ps = self.patch_size

        # Random top-left corner
        r = np.random.randint(0, self._h - ps + 1)
        c = np.random.randint(0, self._w - ps + 1)

        dirty_patch = self.dirty[idx, r:r+ps, c:c+ps]
        clean_patch = self.clean[idx, r:r+ps, c:c+ps]

        # Per-patch normalisation to [0, 1] using dirty range
        lo, hi = dirty_patch.min(), dirty_patch.max()
        if hi > lo:
            dirty_patch = (dirty_patch - lo) / (hi - lo)
            clean_patch = np.clip((clean_patch - lo) / (hi - lo), 0.0, 1.0)

        # Add channel dim -> (1, H, W)
        return (
            torch.from_numpy(dirty_patch[np.newaxis]),
            torch.from_numpy(clean_patch[np.newaxis]),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train() if train else model.eval()
    total_loss = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for dirty, clean in loader:
            dirty = dirty.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            pred = model(dirty)
            loss = criterion(pred, clean)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * dirty.size(0)

    return total_loss / len(loader.dataset)


def plot_loss_curves(train_losses, val_losses, save_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Train MSE", linewidth=2, color="#4C9EEB")
    ax.plot(epochs, val_losses,   label="Val MSE",   linewidth=2,
            color="#E8715A", linestyle="--")

    best_ep  = int(np.argmin(val_losses)) + 1
    best_val = min(val_losses)
    ax.axvline(best_ep, color="gray", linestyle=":", linewidth=1.2,
               label=f"Best epoch ({best_ep})")
    ax.scatter([best_ep], [best_val], color="#E8715A", zorder=5, s=80)
    ax.annotate(f"Best: {best_val:.5f}", xy=(best_ep, best_val),
                xytext=(best_ep + 0.8, best_val + 5e-4),
                fontsize=9, color="#E8715A")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("DenoisingAutoencoder: Training Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # 1. Load data
    print("Loading data ...")
    clean = np.load(os.path.join(DATA_DIR, "clean.npy")).astype(np.float32)
    dirty = np.load(os.path.join(DATA_DIR, "dirty.npy")).astype(np.float32)
    print(f"  dirty : {dirty.shape}  dtype={dirty.dtype}")
    print(f"  clean : {clean.shape}  dtype={clean.dtype}")

    # 2. Train / val split
    indices = np.arange(len(dirty))
    train_idx, val_idx = train_test_split(indices, test_size=VAL_SPLIT,
                                          random_state=SEED)
    print(f"  Train: {len(train_idx)} images  |  Val: {len(val_idx)} images")

    # 3. Datasets & DataLoaders
    train_ds = PatchDataset(dirty[train_idx], clean[train_idx], PATCH_SIZE)
    val_ds   = PatchDataset(dirty[val_idx],   clean[val_idx],   PATCH_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # 4. Model, loss, optimiser
    model     = DenoisingAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {total_params:,}\n")

    # 5. Training loop
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    ckpt_path = os.path.join(CKPT_DIR, "autoencoder_best.pth")

    print(f"{'Epoch':>6}  {'Train MSE':>12}  {'Val MSE':>12}  {'LR':>10}  {'Time':>7}")
    print("-" * 57)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - t0
        improved = val_loss < best_val_loss
        marker   = " <<" if improved else ""

        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  "
              f"{current_lr:>10.2e}  {elapsed:>5.1f}s{marker}")

        if improved:
            best_val_loss = val_loss
            torch.save({
                "epoch":             epoch,
                "model_state_dict":  model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":          val_loss,
                "train_loss":        train_loss,
            }, ckpt_path)

    print(f"\nTraining complete.")
    print(f"Best val MSE : {best_val_loss:.6f}")
    print(f"Checkpoint   : {ckpt_path}")

    # 6. Loss curve
    plot_loss_curves(train_losses, val_losses,
                     os.path.join(RESULTS_DIR, "autoencoder_loss.png"))


if __name__ == "__main__":
    main()
