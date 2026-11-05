"""
src/models/vae.py
=================
Variational Autoencoder (VAE) for astronomical image denoising.

Architecture mirrors DenoisingAutoencoder but adds:
  - Encoder outputs mu and log_var instead of a deterministic bottleneck
  - Reparameterization trick: z = mu + eps * std  (differentiable sampling)
  - Decoder takes the sampled z and reconstructs the clean patch
  - forward() returns (output, mu, log_var) for loss computation

Input/Output
------------
    Input  : (B, 1, 64, 64) dirty patch, values in [0, 1]
    Output : (B, 1, 64, 64) reconstructed clean patch, values in [0, 1]
    Also   : mu, log_var tensors (B, latent_dim, 8, 8) for KL divergence

Tensor shapes at each stage (latent_dim=128)
---------------------------------------------
    Input          (B,   1, 64, 64)
    enc1           (B,  32, 64, 64)  ConvBlock(1, 32)
    pool           (B,  32, 32, 32)
    enc2           (B,  64, 32, 32)  ConvBlock(32, 64)
    pool           (B,  64, 16, 16)
    enc3           (B, 128, 16, 16)  ConvBlock(64, 128)
    pool           (B, 128,  8,  8)
    pre_latent     (B, 256,  8,  8)  ConvBlock(128, 256)
    mu             (B, 128,  8,  8)  Conv2d(256, 128, 1)
    log_var        (B, 128,  8,  8)  Conv2d(256, 128, 1)
    z (sampled)    (B, 128,  8,  8)
    up3            (B, 128, 16, 16)  ConvTranspose2d(128, 128, 2, 2)
    dec3           (B, 128, 16, 16)
    up2            (B,  64, 32, 32)
    dec2           (B,  64, 32, 32)
    up1            (B,  32, 64, 64)
    dec1           (B,  32, 64, 64)
    out + sigmoid  (B,   1, 64, 64)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 conv layers with BatchNorm + ReLU. Preserves spatial dims."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DenoisingVAE(nn.Module):
    """
    Variational Autoencoder for protoplanetary disk image denoising.

    Args:
        latent_dim (int): Number of channels in the latent space (mu/log_var maps).
                          Spatial dims of the latent map are patch_size / 8 = 8x8.
                          Default: 128  -> latent tensor is (B, 128, 8, 8).
    """

    def __init__(self, latent_dim: int = 128, base_channels: int = 32,
                 linear_head: bool = False):
        """
        Args:
            latent_dim: channels in the latent mu/log_var maps.
            base_channels: width of the first encoder stage; the ladder is
                (c, 2c, 4c) with an 8c pre-latent trunk, so the default 32
                reproduces the original fixed 32/64/128/256 architecture.
            linear_head: drop the output sigmoid.

        `linear_head` is required for line-emission data and defaults to False
        only so continuum-era results stay reproducible. Under the shared
        dirty-scale normalisation the clean target can exceed 1, which a sigmoid
        cannot represent: the same defect that made the U-Net's Moment-0
        catastrophic before 5ed8fc6. Comparing a sigmoid VAE against a
        linear-head U-Net would penalise the VAE for the normalisation scheme
        rather than for its architecture.

        The latent map is spatial, so the network is resolution-agnostic: the
        "8x8" in the class docstring is the 64x64 case, and a 256x256 input
        simply gives a 32x32 latent.
        """
        super().__init__()
        self.latent_dim = latent_dim
        c = base_channels

        # ------------------------------------------------------------------ #
        # Encoder
        # ------------------------------------------------------------------ #
        self.enc1 = ConvBlock(1, c)
        self.enc2 = ConvBlock(c, 2 * c)
        self.enc3 = ConvBlock(2 * c, 4 * c)
        self.pool = nn.MaxPool2d(2)

        # Pre-latent feature extraction (shared trunk before mu/log_var split)
        self.pre_latent = ConvBlock(4 * c, 8 * c)

        # Latent distribution parameters: 1x1 convs preserve spatial dims
        self.conv_mu      = nn.Conv2d(8 * c, latent_dim, 1)
        self.conv_log_var = nn.Conv2d(8 * c, latent_dim, 1)

        # ------------------------------------------------------------------ #
        # Decoder
        # ------------------------------------------------------------------ #
        self.up3  = nn.ConvTranspose2d(latent_dim, 4 * c, 2, stride=2)
        self.dec3 = ConvBlock(4 * c, 4 * c)

        self.up2  = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = ConvBlock(2 * c, 2 * c)

        self.up1  = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1 = ConvBlock(c, c)

        self.out_conv = nn.Conv2d(c, 1, 1)
        self.sigmoid  = nn.Identity() if linear_head else nn.Sigmoid()

    # ---------------------------------------------------------------------- #
    # Reparameterization
    # ---------------------------------------------------------------------- #
    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample z ~ N(mu, exp(log_var)) using the reparameterization trick.

        z = mu + eps * std,  eps ~ N(0, I)

        This makes sampling differentiable: gradients flow through mu and
        log_var, not through the stochastic sampling step.

        Args:
            mu      : Mean map      (B, latent_dim, H_z, W_z)
            log_var : Log-variance  (B, latent_dim, H_z, W_z)

        Returns:
            z       : Sampled latent (B, latent_dim, H_z, W_z)
        """
        if self.training:
            std = torch.exp(0.5 * log_var)   # sigma = exp(log_var / 2)
            eps = torch.randn_like(std)       # eps ~ N(0, I)
            return mu + eps * std
        else:
            # At inference, use the mean directly (no sampling noise)
            return mu

    # ---------------------------------------------------------------------- #
    # Forward pass
    # ---------------------------------------------------------------------- #
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x : Dirty patch tensor (B, 1, 64, 64), range [0, 1]

        Returns:
            output  : Reconstructed clean patch (B, 1, 64, 64)
            mu      : Latent mean map            (B, latent_dim, 8, 8)
            log_var : Latent log-variance map    (B, latent_dim, 8, 8)
        """
        # Encode
        e1 = self.enc1(x)               # (B,  32, 64, 64)
        e2 = self.enc2(self.pool(e1))   # (B,  64, 32, 32)
        e3 = self.enc3(self.pool(e2))   # (B, 128, 16, 16)
        h  = self.pre_latent(self.pool(e3))  # (B, 256,  8,  8)

        # Latent distribution
        mu      = self.conv_mu(h)        # (B, 128,  8,  8)
        log_var = self.conv_log_var(h)   # (B, 128,  8,  8)

        # Sample
        z = self.reparameterize(mu, log_var)  # (B, 128,  8,  8)

        # Decode
        d3 = self.dec3(self.up3(z))     # (B, 128, 16, 16)
        d2 = self.dec2(self.up2(d3))    # (B,  64, 32, 32)
        d1 = self.dec1(self.up1(d2))    # (B,  32, 64, 64)

        output = self.sigmoid(self.out_conv(d1))  # (B, 1, 64, 64)

        return output, mu, log_var


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingVAE(latent_dim=128).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"DenoisingVAE parameters: {total_params:,}")

    x = torch.randn(2, 1, 64, 64).to(device)
    output, mu, log_var = model(x)
    print(f"Input     : {x.shape}")
    print(f"Output    : {output.shape}")
    print(f"mu        : {mu.shape}")
    print(f"log_var   : {log_var.shape}")
    print("Forward pass: OK")
