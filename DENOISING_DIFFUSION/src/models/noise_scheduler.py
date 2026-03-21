"""Noise scheduler for the forward and reverse diffusion process."""

import torch
from typing import Tuple


class NoiseScheduler:
    """
    Manages the noise schedule for the diffusion process.

    Precomputes beta, alpha, and alpha_cumprod values used in both
    the forward (noising) and reverse (denoising) diffusion steps.

    Args:
        timesteps: Total number of diffusion steps T
        beta_schedule: Schedule type — "linear" or "cosine"
        beta_start: Starting beta value (used for linear schedule)
        beta_end: Ending beta value (used for linear schedule)

    Example:
        >>> scheduler = NoiseScheduler(timesteps=1000, beta_schedule="linear")
        >>> x0 = torch.randn(2, 1, 64, 64)
        >>> t = torch.tensor([100, 500])
        >>> xt, noise = scheduler.q_sample(x0, t)
        >>> xt.shape
        torch.Size([2, 1, 64, 64])
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: add noise to clean image at timestep t.

        Samples x_t ~ q(x_t | x_0) using the closed-form expression:
            x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * eps

        Args:
            x0: Clean image, shape (B, C, H, W)
            t: Timestep indices, shape (B,)
            noise: Optional pre-sampled noise; sampled from N(0,I) if None

        Returns:
            (x_t, noise) tuple — noisy image and the noise that was added
        """
        raise NotImplementedError

    def p_sample(
        self,
        model: torch.nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reverse diffusion: denoise one step from x_t to x_{t-1}.

        Args:
            model: U-Net that predicts noise given (x_t, t)
            xt: Noisy image at timestep t, shape (B, C, H, W)
            t: Current timestep indices, shape (B,)

        Returns:
            Denoised image x_{t-1}, shape (B, C, H, W)
        """
        raise NotImplementedError

    def p_sample_loop(
        self,
        model: torch.nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Full reverse diffusion loop from pure noise to clean image.

        Args:
            model: Trained U-Net
            shape: Output shape (B, C, H, W)
            device: Target device

        Returns:
            Generated clean image, shape (B, C, H, W)
        """
        raise NotImplementedError
