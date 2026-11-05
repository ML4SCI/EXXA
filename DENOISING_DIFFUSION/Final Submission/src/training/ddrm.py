"""
DDRM: Denoising Diffusion Restoration Models (Kawar et al. 2022), specialised to a
convolution operator.

Why this exists, and why only now. DDRM was refuted for this project's line-emission training
data on 2026-08-20: Phase 0 measured `A = I` there (both clean and dirty cubes are Jy/beam,
so no operator sits between network input and target), and DDRM has nothing to invert when
`A = I`. That refutation stands.

What changed is the self-gravitating cube. The 2026-08-27 ablation convolved the CLEAN cube
with the recovered beam, added no noise and ran no model, and reproduced the entire loss of
the GI wiggle: residual RMS fell 1.394 -> 0.174 and correlation with the true residual fell
to 0.116, statistically indistinguishable from the real dirty cube's 0.111. The beam ERASES
the kinematic signal on that cube. A denoiser cannot undo that even in principle -- removing
noise cannot restore structure a convolution destroyed -- but a measurement-consistency prior
can, because it reconstructs what the instrument could not measure rather than cleaning what
it did. That is what DDRM is for, and it is why this file exists.

## The specialisation that makes it cheap

DDRM works in the spectral domain of `A` and normally needs its SVD. For a convolution `A`
is diagonal in the Fourier basis, so the SVD is free:

    A       = FFT -> multiply by the beam's transfer function -> IFFT
    A^T     = the same operation (a real symmetric kernel is its own transpose)
    singular values = |FFT(beam)|, one per spatial frequency

No decomposition is computed, and each reverse step costs two FFTs rather than a matrix solve.

## What the algorithm actually does

At every reverse step, each Fourier mode is handled according to how well the instrument
measured it, which is exactly `sigma_y / s_k` (noise level over that mode's singular value):

  * modes the beam passes strongly (`s_k` large) are pinned close to the measurement, because
    the data constrains them,
  * modes the beam suppressed to nothing (`s_k` ~ 0) are left entirely to the diffusion prior,
    because the telescope genuinely carries no information about them,
  * modes in between are blended in proportion.

That is the whole idea: the prior fills in only what the instrument could not see, and never
overrides what it could.
"""

from typing import Optional, Tuple

import numpy as np
import torch


def beam_transfer_function(beam: np.ndarray, shape: Tuple[int, int],
                           device=None) -> torch.Tensor:
    """
    |FFT(beam)| on the grid `shape`, i.e. the singular values of the convolution operator.

    `beam` is centred (as `estimate_beam_from_pair` returns it); it is zero-padded to `shape`
    and ifftshift-ed so its peak sits at the origin, which is what makes the FFT real and
    positive for a symmetric kernel rather than carrying a linear phase ramp.
    """
    H, W = shape
    bh, bw = beam.shape

    # A beam larger than the target grid must be CROPPED about its centre, not slotted in.
    # Slicing without this raises a broadcast error when the beam is bigger (caught by the
    # 64px dry run against a 129px beam) and, worse, would silently mis-place the beam in
    # the cases where it happens to fit. Crop symmetrically so the peak stays centred.
    if bh > H or bw > W:
        cy_b, cx_b = bh // 2, bw // 2
        half_h, half_w = min(H, bh) // 2, min(W, bw) // 2
        beam = beam[cy_b - half_h:cy_b - half_h + min(H, bh),
                    cx_b - half_w:cx_b - half_w + min(W, bw)]
        bh, bw = beam.shape

    pad = np.zeros((H, W), dtype=np.float64)
    h, w = bh // 2, bw // 2
    cy, cx = H // 2, W // 2
    pad[cy - h:cy - h + bh, cx - w:cx - w + bw] = beam
    pad = np.fft.ifftshift(pad)
    s = np.abs(np.fft.fft2(pad))
    t = torch.from_numpy(s).float()
    return t.to(device) if device is not None else t


def apply_operator(x: torch.Tensor, transfer: torch.Tensor) -> torch.Tensor:
    """A(x): convolve a (B, 1, H, W) batch by multiplying with the transfer function."""
    X = torch.fft.fft2(x)
    return torch.real(torch.fft.ifft2(X * transfer[None, None]))


def ddrm_steps(y: torch.Tensor, seq, model, b: torch.Tensor, transfer: torch.Tensor,
               sigma_y: float, eta: float = 0.85, eta_b: float = 1.0,
               prediction_type: str = "eps", x_cond: Optional[torch.Tensor] = None,
               generator: Optional[torch.Generator] = None):
    """
    DDRM reverse process for a convolution operator, mirroring `generalized_steps`' signature
    so `DenoisingDiffusion.sample` can dispatch to either.

    Args:
        y: the measurement (the dirty image), (B, 1, H, W), in the model's own [-1, 1] scale.
        seq: the timestep subsequence, ascending, as `generalized_steps` takes it.
        model: the trained noise-prediction network.
        b: the beta schedule.
        transfer: |FFT(beam)| from `beam_transfer_function`, the singular values.
        sigma_y: measurement noise standard deviation, in the SAME [-1, 1] scale as `y`.
            This is the one parameter that must be got right: it sets the boundary between
            "the instrument measured this mode" and "the prior must invent it".
        eta: DDIM stochasticity for the unmeasured subspace.
        eta_b: how hard measured modes are pulled toward the data. 1.0 means "trust the
            measurement fully where the beam passed it", which is the standard choice.
        prediction_type: "eps" or "v", matching what the model was trained with. A v-trained
            model decoded as eps produces plausible-looking noise rather than an error, so it
            is threaded through explicitly rather than defaulted.
        x_cond: optional conditioning image for a conditional model. The project's existing
            DDPM is conditional on the dirty image, so this is passed for compatibility.

    Returns (xs, x0_preds), matching `generalized_steps`.
    """
    from src.training.diffusion import compute_alpha

    with torch.no_grad():
        n = y.size(0)
        dev = y.device
        s = transfer[None, None]                       # (1, 1, H, W) singular values

        # Modes the beam suppresses below the noise are unmeasured: the data says nothing
        # about them and the prior owns them entirely. This threshold IS the algorithm's
        # boundary between restoration and generation.
        measured = s > (sigma_y + 1e-8)

        Y = torch.fft.fft2(y)
        # Least-squares estimate in the measured subspace: divide out the beam where it is
        # safe to, leave zero where it is not.
        Xhat = torch.where(measured, Y / torch.clamp(s, min=1e-8), torch.zeros_like(Y))

        seq_next = [-1] + list(seq[:-1])
        # Initialise: the measured modes start from the data, the rest from pure noise.
        at_T = compute_alpha(b, (torch.ones(n) * seq[-1]).long().to(dev))
        noise = torch.randn(y.shape, device=dev, generator=generator)
        Xt = torch.where(measured,
                         Xhat + torch.fft.fft2(noise) * torch.sqrt(
                             torch.clamp(1 - at_T, min=0.0)) / torch.clamp(s, min=1e-8),
                         torch.fft.fft2(noise))
        xt = torch.real(torch.fft.ifft2(Xt))

        xs, x0_preds = [xt.cpu()], []
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(dev)
            next_t = (torch.ones(n) * j).to(dev)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(dev)

            inp = torch.cat([x_cond, xt], dim=1) if x_cond is not None else xt
            out = model(inp, t)
            if prediction_type == "v":
                x0_t = at.sqrt() * xt - (1 - at).sqrt() * out
            else:
                x0_t = (xt - out * (1 - at).sqrt()) / at.sqrt()
            x0_t = x0_t.clamp(-1.0, 1.0)
            x0_preds.append(x0_t.cpu())

            # --- the DDRM step, per Fourier mode --------------------------------------
            X0 = torch.fft.fft2(x0_t)
            sig_next = torch.sqrt(torch.clamp(1 - at_next, min=0.0))

            # Unmeasured modes: ordinary DDIM, the prior is the only source of information.
            noise = torch.randn(y.shape, device=dev, generator=generator)
            X_unmeasured = X0 + torch.fft.fft2(noise) * sig_next

            # Measured modes: blend the prior's prediction toward the data, weighted by how
            # well this mode was measured (sigma_y / s_k against the current noise level).
            frac = torch.clamp(sig_next / torch.clamp(sigma_y / torch.clamp(s, min=1e-8),
                                                      min=1e-8), max=1.0)
            X_measured = (1 - eta_b * frac) * X0 + eta_b * frac * Xhat
            X_measured = X_measured + torch.fft.fft2(noise) * sig_next * eta * (1 - frac)

            Xt_next = torch.where(measured, X_measured, X_unmeasured)
            xt_next = torch.real(torch.fft.ifft2(Xt_next))
            xs.append(xt_next.cpu())

    return xs, x0_preds
