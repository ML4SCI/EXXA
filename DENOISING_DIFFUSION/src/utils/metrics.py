"""Image quality metrics and baseline denoiser comparison utility."""

from __future__ import annotations

from typing import Dict

import numpy as np
from skimage.metrics import structural_similarity


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Squared Error between pred and target."""
    return float(np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2))


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio in dB.

    Args:
        pred: Predicted image, values in [0, data_range]
        target: Ground-truth image, values in [0, data_range]
        data_range: Value range of the images (default 1.0)

    Returns:
        PSNR in dB, or inf if pred == target exactly.
    """
    err = mse(pred, target)
    if err == 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / err))


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """
    Structural Similarity Index (SSIM).

    Args:
        pred: Predicted image
        target: Ground-truth image
        data_range: Value range of the images (default 1.0)

    Returns:
        SSIM score in [-1, 1].
    """
    p = pred.astype(np.float64)
    t = target.astype(np.float64)

    # skimage expects (H, W) or (H, W, C)
    if p.ndim == 4:          # (B, C, H, W) -> average over batch
        scores = [
            structural_similarity(
                p[i].transpose(1, 2, 0),
                t[i].transpose(1, 2, 0),
                data_range=data_range,
                channel_axis=-1,
            )
            for i in range(p.shape[0])
        ]
        return float(np.mean(scores))

    if p.ndim == 3:          # (C, H, W) -> (H, W, C)
        p = p.transpose(1, 2, 0)
        t = t.transpose(1, 2, 0)
        return float(structural_similarity(p, t, data_range=data_range, channel_axis=-1))

    # (H, W)
    return float(structural_similarity(p, t, data_range=data_range))


def compare_denoisers(
    noisy: np.ndarray,
    target: np.ndarray,
    outputs: Dict[str, np.ndarray],
    data_range: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple denoised outputs against a clean target.

    Args:
        noisy: Noisy input image (used as the baseline entry "noisy")
        target: Clean ground-truth image
        outputs: Mapping of denoiser name -> denoised image
        data_range: Value range of the images (default 1.0)

    Returns:
        Dict mapping each name (plus "noisy" baseline) to
        {"psnr": float, "ssim": float, "mse": float}.

    Example:
        >>> results = compare_denoisers(noisy, clean, {
        ...     "gaussian": gaussian_filtered,
        ...     "ddpm": ddpm_output,
        ... })
        >>> print(results["ddpm"]["psnr"])
    """
    all_outputs = {"noisy": noisy, **outputs}
    return {
        name: {
            "psnr": psnr(img, target, data_range=data_range),
            "ssim": ssim(img, target, data_range=data_range),
            "mse": mse(img, target),
        }
        for name, img in all_outputs.items()
    }
