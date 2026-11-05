"""
Loss functions for EXXA denoising pipeline.

HybridLoss : alpha * MSE + beta * (1 - SSIM)
VAELoss    : alpha * MSE + beta * (1 - SSIM) + gamma * KL
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class HybridLoss(nn.Module):
    """
    Weighted combination of MSE loss and SSIM-based loss.

    Tanmay's ratio: total_loss = 0.8 * MSE + 0.2 * (1 - SSIM)
    MSE anchors pixel-level accuracy; SSIM guides structural fidelity.

    Args:
        alpha (float): Weight for MSE loss.  Default: 0.8
        beta  (float): Weight for SSIM loss. Default: 0.2
    """

    def __init__(self, alpha: float = 0.8, beta: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_fn = nn.MSELoss()

    def forward(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the hybrid loss.

        Args:
            output: Model prediction tensor  (B, C, H, W), range [0, 1]
            target: Ground-truth tensor      (B, C, H, W), range [0, 1]

        Returns:
            total_loss: alpha * mse_loss + beta * ssim_loss
            mse_loss:   Raw MSE component
            ssim_loss:  Raw SSIM component  (1 - SSIM)
        """
        mse_loss = self.mse_fn(output, target)
        ssim_loss = 1.0 - ssim(output, target, data_range=1.0, size_average=True)
        total_loss = self.alpha * mse_loss + self.beta * ssim_loss
        return total_loss, mse_loss, ssim_loss


if __name__ == "__main__":
    torch.manual_seed(42)
    pred = torch.rand(2, 1, 64, 64)
    gt   = torch.rand(2, 1, 64, 64)

    # Tanmay's ratio: 0.8 MSE + 0.2 (1-SSIM)
    criterion = HybridLoss(alpha=0.8, beta=0.2)
    total, mse, ssim_l = criterion(pred, gt)

    print(f"Total loss : {total.item():.6f}  (0.8*MSE + 0.2*SSIM_loss)")
    print(f"MSE  loss  : {mse.item():.6f}")
    print(f"SSIM loss  : {ssim_l.item():.6f}  (1 - SSIM score)")


# ---------------------------------------------------------------------------
# VAE Loss
# ---------------------------------------------------------------------------

class VAELoss(nn.Module):
    """
    Combined loss for Variational Autoencoder denoising.

    total = alpha * MSE + beta * (1 - SSIM) + gamma * KL

    The KL divergence term regularises the latent space toward N(0, I).
    For convolutional latents (B, C, H_z, W_z) the KL is averaged per element.

    Args:
        alpha (float): MSE weight.         Default: 0.8
        beta  (float): SSIM loss weight.   Default: 0.2
        gamma (float): KL weight.          Default: 0.001
    """

    def __init__(
        self,
        alpha: float = 0.8,
        beta:  float = 0.2,
        gamma: float = 0.001,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.mse_fn = nn.MSELoss()

    def forward(
        self,
        output:  torch.Tensor,
        target:  torch.Tensor,
        mu:      torch.Tensor,
        log_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            output  : Reconstructed image  (B, 1, H, W), range [0, 1]
            target  : Clean ground truth   (B, 1, H, W), range [0, 1]
            mu      : Latent mean          (B, C, H_z, W_z)
            log_var : Latent log-variance  (B, C, H_z, W_z)

        Returns:
            total_loss : alpha*MSE + beta*(1-SSIM) + gamma*KL
            mse_loss   : raw MSE component
            ssim_loss  : raw SSIM component  (1 - SSIM score)
            kl_loss    : KL divergence per element, averaged
        """
        mse_loss  = self.mse_fn(output, target)
        ssim_loss = 1.0 - ssim(output, target, data_range=1.0, size_average=True)

        # KL(N(mu, var) || N(0, 1)) = -0.5 * mean(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())

        total_loss = (
            self.alpha * mse_loss
            + self.beta  * ssim_loss
            + self.gamma * kl_loss
        )
        return total_loss, mse_loss, ssim_loss, kl_loss


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(42)
    pred    = torch.rand(2, 1, 64, 64)
    gt      = torch.rand(2, 1, 64, 64)
    mu_     = torch.randn(2, 128, 8, 8)
    log_v   = torch.randn(2, 128, 8, 8)

    vae_criterion = VAELoss(alpha=0.8, beta=0.2, gamma=0.001)
    total, mse, ssim_l, kl = vae_criterion(pred, gt, mu_, log_v)
    print("VAELoss sanity check:")
    print(f"  Total loss : {total.item():.6f}")
    print(f"  MSE   loss : {mse.item():.6f}")
    print(f"  SSIM  loss : {ssim_l.item():.6f}")
    print(f"  KL    loss : {kl.item():.6f}")



def spectral_moment1(x: torch.Tensor, velax: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Intensity-weighted mean velocity along the channel axis, differentiable.

    ``x`` is (B, C, H, W) with C the spectral axis; ``velax`` is (C,) in km/s. Returns
    (B, H, W).

    Not an unbiased M1 and does not need to be. It is computed identically on the prediction
    and on the target, so a bias from the finite channel window cancels in their difference.
    What it has to be is SENSITIVE to a shift of the line peak, which is exactly the quantity
    a per-channel denoiser perturbs.

    The denominator is clamped rather than masked: a spaxel with no line contributes a
    near-zero numerator and denominator, and clamping keeps the gradient finite there instead
    of producing NaNs that poison the whole batch.
    """
    w = x.clamp(min=0.0)                       # negative "weights" make the mean meaningless
    num = (w * velax.view(1, -1, 1, 1)).sum(dim=1)
    den = w.sum(dim=1).clamp(min=eps)
    return num / den


# ---------------------------------------------------------------------------
# MAE / wavelet / starlet losses (2026-09-15 mentee loss sweep)
# ---------------------------------------------------------------------------

class MAELoss(nn.Module):
    """
    HybridLoss with L1 in place of MSE.

    total = alpha * MAE + beta * (1 - SSIM)

    Same alpha/beta convention as HybridLoss so it drops into train_unet's existing
    `alpha` sweep param unchanged.
    """

    def __init__(self, alpha: float = 0.8, beta: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mae_fn = nn.L1Loss()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mae_loss = self.mae_fn(output, target)
        ssim_loss = 1.0 - ssim(output, target, data_range=1.0, size_average=True)
        total_loss = self.alpha * mae_loss + self.beta * ssim_loss
        return total_loss, mae_loss, ssim_loss


def _haar_dwt2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One level of a 2D Haar DWT, per-channel (depthwise), stride 2. Returns LL, LH, HL, HH."""
    c = x.shape[1]
    ll = x.new_tensor([[1, 1], [1, 1]]) / 4
    lh = x.new_tensor([[1, 1], [-1, -1]]) / 4
    hl = x.new_tensor([[1, -1], [1, -1]]) / 4
    hh = x.new_tensor([[1, -1], [-1, 1]]) / 4
    kernel = torch.stack([ll, lh, hl, hh]).unsqueeze(1).repeat(c, 1, 1, 1)  # (4c,1,2,2)
    out = F.conv2d(x, kernel, stride=2, groups=c)
    out = out.view(x.shape[0], c, 4, out.shape[-2], out.shape[-1])
    return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]


class WaveletLoss(nn.Module):
    """
    Multi-level Haar-DWT L1 loss (a "pix2pix-style" high-frequency loss).

    total = alpha * MSE(output, target) + beta * sum_levels L1(detail_pred, detail_target)

    Detail (LH/HL/HH) sub-bands penalise edge/texture mismatch directly, which is the
    smoothing symptom the mentee is chasing -- MSE alone rewards a blurred mean. Levels
    beyond `n_levels` stop mattering once the sub-band is a handful of pixels; 3 covers
    256px down to 32px, still meaningful detail.

    No pywt dependency: implemented as a fixed-kernel depthwise conv2d.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, n_levels: int = 3):
        super().__init__()
        self.alpha, self.beta, self.n_levels = alpha, beta, n_levels
        self.mse_fn = nn.MSELoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mse_loss = self.mse_fn(output, target)
        detail_loss = wavelet_detail(output, target, self.n_levels)
        total_loss = self.alpha * mse_loss + self.beta * detail_loss
        return total_loss, mse_loss, detail_loss


_B3_SPLINE_1D = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def _starlet_transform(x: torch.Tensor, n_scales: int) -> list[torch.Tensor]:
    """
    A trous (undecimated) starlet transform, the standard multiscale wavelet in
    astronomical image processing (isotropic, no downsampling -- unlike Haar DWT this
    keeps every scale at full resolution, which matters for per-pixel-aligned loss).

    Returns [w_1, ..., w_n, c_n]: n detail (wavelet) planes plus the final smooth
    approximation, each same shape as `x`. The B3-spline kernel is separable and its
    support doubles each scale ("a trous" = holes punched between taps), giving a
    dyadic scale ladder without resampling.
    """
    b, c = x.shape[0], x.shape[1]
    k1d = _B3_SPLINE_1D.to(device=x.device, dtype=x.dtype)
    planes = []
    cj = x
    for j in range(n_scales):
        hole = 2 ** j
        pad = 2 * hole
        kh = torch.zeros(4 * hole + 1, device=x.device, dtype=x.dtype)
        kh[::hole] = k1d
        kernel_row = kh.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
        kernel_col = kh.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
        smoothed = F.conv2d(F.pad(cj, (pad, pad, 0, 0), mode="reflect"), kernel_row, groups=c)
        smoothed = F.conv2d(F.pad(smoothed, (0, 0, pad, pad), mode="reflect"), kernel_col, groups=c)
        planes.append(cj - smoothed)
        cj = smoothed
    planes.append(cj)
    return planes


class StarletLoss(nn.Module):
    """
    Starlet (a trous B3-spline) multiscale loss -- ExoALMA-style wavelet loss, the one
    already validated on ALMA data (mentee, 2026-09-12 meeting).

    total = alpha * MSE(output, target) + beta * sum_scales L1(w_pred, w_target)

    Isotropic and shift-invariant unlike Haar, which matters for line-emission cubes
    where the disk structure has no preferred axis. `n_scales` follows the wavelet
    literature default (4) for 256px images; halve it for 480/600px inputs if the finest
    scale gets too expensive, but the mentee's current smoothing complaint is at coarse
    scale, so keep the low end covered.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, n_scales: int = 4):
        super().__init__()
        self.alpha, self.beta, self.n_scales = alpha, beta, n_scales
        self.mse_fn = nn.MSELoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mse_loss = self.mse_fn(output, target)
        detail_loss = starlet_detail(output, target, self.n_scales)
        total_loss = self.alpha * mse_loss + self.beta * detail_loss
        return total_loss, mse_loss, detail_loss


def _sobel(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude, per-channel (depthwise). Kernel built via `x.new_tensor`
    so it always matches the input's device/dtype -- no buffer to keep in sync."""
    c = x.shape[1]
    gx = x.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    gy = x.new_tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    ex = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), gx, groups=c)
    ey = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), gy, groups=c)
    return torch.sqrt(ex ** 2 + ey ** 2 + 1e-12)


def wavelet_detail(pred: torch.Tensor, target: torch.Tensor, n_levels: int = 3) -> torch.Tensor:
    """Detail-only term behind WaveletLoss -- multi-level Haar LH/HL/HH L1, summed."""
    detail = pred.new_zeros(())
    po, pt = pred, target
    for _ in range(n_levels):
        if min(po.shape[-2:]) < 2:
            break
        llo, lho, hlo, hho = _haar_dwt2(po)
        llt, lht, hlt, hht = _haar_dwt2(pt)
        detail = detail + sum(F.l1_loss(a, b) for a, b in ((lho, lht), (hlo, hlt), (hho, hht)))
        po, pt = llo, llt
    return detail


def starlet_detail(pred: torch.Tensor, target: torch.Tensor, n_scales: int = 4) -> torch.Tensor:
    """Detail-only term behind StarletLoss -- mean L1 across a trous wavelet planes."""
    wp, wt = _starlet_transform(pred, n_scales), _starlet_transform(target, n_scales)
    return sum(F.l1_loss(a, b) for a, b in zip(wp, wt)) / len(wp)


def gradient_detail(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Detail-only term behind GradientLoss -- Sobel-magnitude L1."""
    return F.l1_loss(_sobel(pred), _sobel(target))


# Detail-only functions, (pred, target) -> scalar, no MSE/pixel term mixed in. Used by
# diffusion.py's noise_estimation_loss to add a detail-preserving term on the predicted-clean
# estimate, where "hybrid"/"mae" make no sense (there is no separate SSIM/pixel split to
# reweight on a noise-prediction target).
LOSS_REGISTRY_DETAIL_ONLY = {
    "wavelet":  wavelet_detail,
    "starlet":  starlet_detail,
    "gradient": gradient_detail,
}


class GradientLoss(nn.Module):
    """
    Sobel-gradient L1 loss (the "pix2pix-style" edge loss the mentee read about,
    2026-09-12 meeting). pix2pix's own trick is an adversarial loss for high-frequency
    realism plus L1 for low-frequency correctness; this keeps that same split without a
    discriminator -- MSE anchors the low-frequency (smooth) content, the Sobel term
    anchors edges directly, which is the exact thing per-channel smoothing erases.

    total = alpha * MSE(output, target) + beta * L1(sobel(output), sobel(target))
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.mse_fn = nn.MSELoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mse_loss = self.mse_fn(output, target)
        edge_loss = gradient_detail(output, target)
        total_loss = self.alpha * mse_loss + self.beta * edge_loss
        return total_loss, mse_loss, edge_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    pred = torch.rand(2, 1, 64, 64)
    gt = torch.rand(2, 1, 64, 64)
    crits = [("MAE", MAELoss()), ("Wavelet", WaveletLoss()), ("Starlet", StarletLoss()),
             ("Gradient", GradientLoss())]
    for name, crit in crits:
        t, a, b = crit(pred, gt)
        print(f"{name:<8} total {t.item():.6f}  primary {a.item():.6f}  detail {b.item():.6f}")
        assert torch.isfinite(t), f"{name} produced a non-finite loss"
    print("losses.py sanity check passed")


class KinematicLoss(nn.Module):
    """
    HybridLoss plus a penalty on velocity-field error.

    Motivation, measured (2026-08-28): the U-Net improves pixel metrics and M0 substantially
    while DEGRADING the GI wiggle, the kinematic diagnostic (residual correlation 0.804
    against 0.891 for doing nothing). It is optimising pixel accuracy, and the wiggle is a
    sub-channel velocity perturbation that per-channel smoothing shifts. Nothing in the
    objective ever asked it to preserve velocity structure.

    L = alpha*MSE + beta*(1-SSIM) + gamma*|M1(pred) - M1(target)|

    The M1 term needs several channels at once, which is why this only works with a dataset
    yielding channel stacks (``FITSChannelDataset(n_neighbors=k)``) and a model whose
    out_channels matches its in_channels.

    Args:
        alpha, beta: as HybridLoss.
        gamma: weight on the velocity term. Its scale differs from the pixel terms (km/s
            against normalised intensity), so this needs tuning rather than a default of 1.
        velax: (C,) channel velocities in km/s.
    """

    def __init__(self, alpha: float = 0.8, beta: float = 0.2, gamma: float = 1.0,
                 velax: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.mse_fn = nn.MSELoss()
        self.register_buffer("velax", velax if velax is not None else torch.zeros(1))

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mse_loss = self.mse_fn(output, target)
        # SSIM over the channel stack: treat channels as a batch of single-channel images so
        # the window is spatial, not spectral.
        B, C, H, W = output.shape
        ssim_loss = 1.0 - ssim(output.reshape(B * C, 1, H, W).clamp(0, 1),
                               target.reshape(B * C, 1, H, W).clamp(0, 1),
                               data_range=1.0, size_average=True)

        v = self.velax.to(output.device)
        kin_loss = (spectral_moment1(output, v) - spectral_moment1(target, v)).abs().mean()

        total = self.alpha * mse_loss + self.beta * ssim_loss + self.gamma * kin_loss
        return total, mse_loss, ssim_loss, kin_loss
