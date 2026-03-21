"""DDPM wrapper combining U-Net backbone and noise scheduler."""

import torch
import torch.nn as nn
from typing import Tuple

from .unet import UNet
from .noise_scheduler import NoiseScheduler


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.

    Wraps the U-Net backbone and noise scheduler into a single module.
    Exposes training loss computation and inference sampling.

    Args:
        unet: U-Net model that predicts noise given (x_t, t)
        scheduler: NoiseScheduler managing beta/alpha values
        loss_type: Loss function to use — "l1" or "l2"

    Example:
        >>> unet = UNet(in_channels=1, out_channels=1)
        >>> scheduler = NoiseScheduler(timesteps=1000)
        >>> model = DDPM(unet, scheduler)
        >>> x0 = torch.randn(2, 1, 64, 64)
        >>> loss = model.training_loss(x0)
    """

    def __init__(
        self,
        unet: UNet,
        scheduler: NoiseScheduler,
        loss_type: str = "l2",
    ) -> None:
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.loss_type = loss_type

    def training_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Compute training loss for a batch of clean images.

        Samples random timesteps and noise, runs forward diffusion,
        predicts noise with U-Net, and computes loss against true noise.

        Args:
            x0: Clean image batch, shape (B, C, H, W)

        Returns:
            Scalar loss tensor
        """
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate a clean image from pure Gaussian noise.

        Args:
            shape: Output shape (B, C, H, W)
            device: Target device

        Returns:
            Generated image, shape (B, C, H, W)
        """
        raise NotImplementedError

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """Alias for training_loss — used during the training loop."""
        return self.training_loss(x0)
