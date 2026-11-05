import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DenoisingAutoencoder(nn.Module):
    """
    Convolutional denoising autoencoder.

    Args:
        base_channels: width of the first encoder stage. The ladder is
            (c, 2c, 4c) with an 8c bottleneck, so the default 32 reproduces the
            original fixed 32/64/128/256 architecture exactly.
        linear_head: drop the output sigmoid.

    `linear_head` matters for line-emission work and defaults to False only to
    keep the continuum-era results reproducible. Under the shared dirty-scale
    normalisation the clean target can exceed 1 (it is normalised by the DIRTY
    channel's min/max, and the clean peak may be brighter), so a sigmoid output
    cannot represent the target at all. Removing it from the U-Net was the second
    half of the fix in 5ed8fc6 that took Moment-0 from -6402% to positive. An
    autoencoder left with its sigmoid is not a fair comparison against that U-Net:
    it would lose for a reason unrelated to architecture.

    Fully convolutional, so any input size divisible by 8 works (256x256 gives a
    32x32 bottleneck); nothing here is tied to the 64x64 patches of the
    continuum era.
    """

    def __init__(self, base_channels: int = 32, linear_head: bool = False):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = ConvBlock(1, c)
        self.enc2 = ConvBlock(c, 2 * c)
        self.enc3 = ConvBlock(2 * c, 4 * c)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(4 * c, 8 * c)

        # Decoder
        self.up3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2)
        self.dec3 = ConvBlock(4 * c, 4 * c)

        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = ConvBlock(2 * c, 2 * c)

        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1 = ConvBlock(c, c)

        self.out = nn.Conv2d(c, 1, 1)
        self.sigmoid = nn.Identity() if linear_head else nn.Sigmoid()

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decode
        d3 = self.dec3(self.up3(b))
        d2 = self.dec2(self.up2(d3))
        d1 = self.dec1(self.up1(d2))

        return self.sigmoid(self.out(d1))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingAutoencoder().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    x = torch.randn(2, 1, 64, 64).to(device)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print("Forward pass: OK")
