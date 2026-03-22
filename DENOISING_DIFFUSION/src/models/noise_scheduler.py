"""Noise scheduler for the forward and reverse diffusion process."""

import math
import torch
import torch.nn.functional as F
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
        if beta_schedule not in {"linear", "cosine"}:
            raise ValueError(f"Unknown beta_schedule '{beta_schedule}'. Choose 'linear' or 'cosine'.")

        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end

        betas = (
            self._get_betas_linear() if beta_schedule == "linear"
            else self._get_betas_cosine()
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register("betas", betas)
        self.register("alphas", alphas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())

    def register(self, name: str, tensor: torch.Tensor) -> None:
        """Store a precomputed tensor as an attribute."""
        setattr(self, name, tensor)

    def _get_betas_linear(self) -> torch.Tensor:
        """Linear beta schedule from beta_start to beta_end."""
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps, dtype=torch.float64)

    def _get_betas_cosine(self, s: float = 8e-3) -> torch.Tensor:
        """
        Cosine beta schedule (Nichol & Dhariwal, 2021).

        Produces a smoother schedule that avoids too much noise at
        the start and end of the diffusion process.

        Args:
            s: Small offset to prevent beta from being too small near t=0
        """
        steps = self.timesteps + 1
        t = torch.linspace(0, self.timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((t / self.timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(0, 0.999)

    def _extract(self, tensor: torch.Tensor, t: torch.Tensor, shape: Tuple) -> torch.Tensor:
        """Extract values at timestep t and reshape to broadcast over (B, C, H, W)."""
        out = tensor.to(t.device)[t].float()
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

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
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
        return xt, noise

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
