"""
src/evaluate_all.py
====================
Evaluates every denoising method on the same 100 validation samples
and prints a single unified comparison table (PSNR / SSIM / MSE).

Methods evaluated
-----------------
  Classical
    1. Noisy input (baseline)
    2. Gaussian σ=1
    3. Gaussian σ=2
    4. Median 3×3
    5. Wiener

  Neural network (patch-level inference on full 600×600 images via sliding window)
    6. Autoencoder: MSE-only          (results/checkpoints/autoencoder_best.pth)
    7. Autoencoder: HybridLoss        (results/checkpoints/autoencoder_hybrid_best.pth)
    8. VAE: MSE+SSIM+KL               (results/checkpoints/vae_best.pth)

All neural models use the same sliding-window patch tiling strategy:
  - 64×64 patches, stride 32 (50 % overlap)
  - Per-patch min-max normalisation (same as training)
  - Patches stitched back to 600×600 via weighted averaging (linear blend in overlap)

Evaluation is on the same 100 val-split indices used in training (random_state=42).

Usage
-----
    python src/evaluate_all.py

Output
------
    Console: unified table
    results/evaluation_report.txt : same table saved to disk
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from scipy.ndimage  import gaussian_filter, median_filter
from scipy.signal   import wiener
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity   as ssim_fn

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SCRIPT_DIR)          # DENOISING_DIFFUSION/
DATA_DIR    = os.path.join(ROOT_DIR, "data")
CKPT_DIR    = os.path.join(ROOT_DIR, "results", "checkpoints")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, ROOT_DIR)

N_SAMPLES  = 100
PATCH_SIZE = 64
STRIDE     = 32          # 50 % overlap → smooth stitching
SEED       = 42
BATCH_INFER= 32          # number of patches per GPU batch during inference


# --------------------------------------------------------------------------- #
# Metrics helper
# --------------------------------------------------------------------------- #
def metrics(clean: np.ndarray, denoised: np.ndarray) -> dict:
    """Returns PSNR, SSIM, MSE for a single 600×600 float32 image pair."""
    denoised = np.clip(denoised.astype(np.float32), 0.0, 1.0)
    clean    = clean.astype(np.float32)
    p = psnr_fn(clean, denoised, data_range=1.0)
    s = ssim_fn(clean, denoised, data_range=1.0)
    m = float(np.mean((clean - denoised) ** 2))
    return {"PSNR": p, "SSIM": s, "MSE": m}


# --------------------------------------------------------------------------- #
# Sliding-window inference for neural models
# --------------------------------------------------------------------------- #
def infer_full_image(
    model,
    dirty_img: np.ndarray,
    device,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
    is_vae: bool = False,
) -> np.ndarray:
    """
    Runs model on a full 600×600 image using overlapping patches.

    Returns a float32 (600, 600) numpy array in [0, 1].
    """
    H, W = dirty_img.shape
    out_sum   = np.zeros((H, W), dtype=np.float64)
    out_weight= np.zeros((H, W), dtype=np.float64)

    # Collect all patch positions
    rows = list(range(0, H - patch_size + 1, stride))
    cols = list(range(0, W - patch_size + 1, stride))
    # Make sure last row/col is always included
    if rows[-1] + patch_size < H:
        rows.append(H - patch_size)
    if cols[-1] + patch_size < W:
        cols.append(W - patch_size)

    patches, positions, norms = [], [], []

    for r in rows:
        for c in cols:
            dp = dirty_img[r:r+patch_size, c:c+patch_size].astype(np.float32)
            lo, hi = dp.min(), dp.max()
            if hi > lo:
                dp_n = (dp - lo) / (hi - lo)
            else:
                dp_n = dp.copy()
            patches.append(dp_n)
            positions.append((r, c))
            norms.append((lo, hi))

    # Batch inference
    model.eval()
    patch_preds = []
    with torch.no_grad():
        for start in range(0, len(patches), BATCH_INFER):
            batch = np.stack(patches[start:start+BATCH_INFER])          # (B, H, W)
            t = torch.from_numpy(batch[:, np.newaxis]).to(device)        # (B,1,H,W)
            if is_vae:
                pred, _, _ = model(t)
            else:
                pred = model(t)
            patch_preds.extend(pred.squeeze(1).cpu().numpy())

    # Stitch: linear weight (1 everywhere in this simple version, edge blending via overlap average)
    for pred, (r, c), (lo, hi) in zip(patch_preds, positions, norms):
        out_sum   [r:r+patch_size, c:c+patch_size] += pred.astype(np.float64)
        out_weight[r:r+patch_size, c:c+patch_size] += 1.0

    stitched = (out_sum / np.maximum(out_weight, 1e-8)).astype(np.float32)
    return np.clip(stitched, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Load data ──────────────────────────────────────────────────────────── #
    print("Loading data ...")
    clean_all = np.load(os.path.join(DATA_DIR, "clean.npy")).astype(np.float32)
    dirty_all = np.load(os.path.join(DATA_DIR, "dirty.npy")).astype(np.float32)

    indices = np.arange(len(dirty_all))
    _, val_idx = train_test_split(indices, test_size=0.20, random_state=SEED)

    # Take first N_SAMPLES from val set (deterministic)
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(val_idx, size=min(N_SAMPLES, len(val_idx)), replace=False)
    chosen.sort()
    print(f"Evaluating on {len(chosen)} val samples  (indices {chosen[0]}..{chosen[-1]})")

    # ── Load neural models ─────────────────────────────────────────────────── #
    from src.models.autoencoder import DenoisingAutoencoder
    from src.models.vae         import DenoisingVAE

    def load_ae(ckpt_name: str):
        m = DenoisingAutoencoder().to(device)
        ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name), map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        return m

    def load_vae(ckpt_name: str):
        ckpt = torch.load(os.path.join(CKPT_DIR, ckpt_name), map_location=device)
        m = DenoisingVAE(latent_dim=ckpt.get("latent_dim", 128)).to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        return m

    print("Loading checkpoints ...")
    ae_mse    = load_ae("autoencoder_best.pth")
    ae_hybrid = load_ae("autoencoder_hybrid_best.pth")
    vae_model = load_vae("vae_best.pth")
    print("  All checkpoints loaded.")

    # ── Accumulate metrics ─────────────────────────────────────────────────── #
    methods = [
        "Noisy input",
        "Gaussian  sigma=1",
        "Gaussian  sigma=2",
        "Median  3x3",
        "Wiener",
        "AE  MSE-only  (30ep)",
        "AE  HybridLoss (30ep)",
        "VAE  MSE+SSIM+KL (30ep)",
    ]
    accum = {m: {"PSNR": [], "SSIM": [], "MSE": []} for m in methods}

    print(f"\nRunning evaluation on {len(chosen)} samples ...")
    t_start = time.time()

    for i, idx in enumerate(chosen):
        c = clean_all[idx]   # (600, 600) float32
        d = dirty_all[idx]   # (600, 600) float32

        def acc(method, denoised):
            r = metrics(c, denoised)
            accum[method]["PSNR"].append(r["PSNR"])
            accum[method]["SSIM"].append(r["SSIM"])
            accum[method]["MSE"].append(r["MSE"])

        # Classical
        acc("Noisy input",        d)
        acc("Gaussian  sigma=1",  gaussian_filter(d, sigma=1.0))
        acc("Gaussian  sigma=2",  gaussian_filter(d, sigma=2.0))
        acc("Median  3x3",        median_filter(d, size=3))
        acc("Wiener",             wiener(d).astype(np.float32))

        # Neural: patch-tiled on full 600×600
        acc("AE  MSE-only  (30ep)",      infer_full_image(ae_mse,    d, device))
        acc("AE  HybridLoss (30ep)",     infer_full_image(ae_hybrid, d, device))
        acc("VAE  MSE+SSIM+KL (30ep)",   infer_full_image(vae_model, d, device, is_vae=True))

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1:>3}/{len(chosen)}]  {elapsed:.0f}s elapsed")

    elapsed_total = time.time() - t_start
    print(f"\nEvaluation complete in {elapsed_total:.1f}s")

    # ── Build table ────────────────────────────────────────────────────────── #
    col_w = [28, 10, 8, 12]
    sep   = "+" + "+".join("-" * w for w in col_w) + "+"
    hdr   = (f"| {'Method':<{col_w[0]-2}} | {'PSNR (dB)':>{col_w[1]-2}} "
             f"| {'SSIM':>{col_w[2]-2}} | {'MSE':>{col_w[3]-2}} |")

    lines = []
    lines.append(f"\nUnified Comparison: {len(chosen)} Val Samples")
    lines.append(sep)
    lines.append(hdr)
    lines.append(sep.replace("-", "="))

    # Sort by PSNR descending
    sorted_methods = sorted(methods, key=lambda m: np.mean(accum[m]["PSNR"]), reverse=True)

    best_psnr = max(np.mean(accum[m]["PSNR"]) for m in methods)
    best_ssim = max(np.mean(accum[m]["SSIM"]) for m in methods)
    best_mse  = min(np.mean(accum[m]["MSE"])  for m in methods)

    for m in sorted_methods:
        p = np.mean(accum[m]["PSNR"])
        s = np.mean(accum[m]["SSIM"])
        ms= np.mean(accum[m]["MSE"])

        p_star  = " *" if abs(p - best_psnr) < 1e-6 else "  "
        s_star  = " *" if abs(s - best_ssim) < 1e-6 else "  "
        ms_star = " *" if abs(ms- best_mse)  < 1e-6 else "  "

        row = (f"| {m:<{col_w[0]-2}} | {p:>{col_w[1]-4}.4f}{p_star} "
               f"| {s:>{col_w[2]-4}.4f}{s_star} | {ms:>{col_w[3]-4}.6f}{ms_star} |")
        lines.append(row)
        lines.append(sep)

    lines.append("* = best in column")

    output = "\n".join(lines)
    print(output)

    # Save to file
    report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"\nSaved -> {report_path}")


if __name__ == "__main__":
    main()
