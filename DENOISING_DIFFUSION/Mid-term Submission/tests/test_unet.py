#!/usr/bin/env python
"""
Test U-Net model architecture and output shapes.
"""

import torch
from src.models.unet import UNet, create_model

print("=" * 60)
print("U-Net Model Test")
print("=" * 60)

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[1] Device: {device}")

# Create model
print("\n[2] Creating U-Net model...")
model = create_model(
    in_channels=2,  # Noisy + Clean
    out_channels=1,  # Noise prediction
    base_channels=64,
    device=device,
)

print(f"Model created successfully!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
print("\n[3] Testing forward pass...")
batch_size = 4
input_shape = (batch_size, 2, 608, 608)
timesteps = torch.randint(0, 1000, (batch_size,))

x = torch.randn(input_shape).to(device)
t = timesteps.to(device)

print(f"Input shape: {x.shape}")
print(f"Timesteps shape: {t.shape}")

# Forward pass
with torch.no_grad():
    output = model(x, t)

print(f"\n✓ Forward pass successful!")
print(f"Output shape: {output.shape}")
print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

# Verify shapes
print("\n[4] Verifying output shapes...")
expected_output_shape = (batch_size, 1, 608, 608)
assert output.shape == expected_output_shape, \
    f"Output shape mismatch! Expected {expected_output_shape}, got {output.shape}"
print(f"✓ Output shape matches expected: {expected_output_shape}")

# Test with different input sizes
print("\n[5] Testing with different input sizes...")
test_sizes = [
    (4, 2, 256, 256),
    (2, 2, 512, 512),
    (1, 2, 600, 600),
]

for size in test_sizes:
    x_test = torch.randn(size).to(device)
    t_test = torch.randint(0, 1000, (size[0],)).to(device)
    
    with torch.no_grad():
        out_test = model(x_test, t_test)
    
    print(f"Input {size} → Output {tuple(out_test.shape)} ✓")

# Memory usage estimate
print("\n[6] Memory analysis...")
num_params = sum(p.numel() for p in model.parameters())
param_memory = num_params * 4 / (1024**2)  # Assuming float32
print(f"Trainable parameters: {num_params:,}")
print(f"Parameter memory (float32): {param_memory:.2f} MB")

# Batch size scaling
print("\n[7] Batch size estimates (GPU memory ~12GB):")
for batch in [1, 2, 4, 8, 16]:
    x_est = torch.randn(batch, 2, 608, 608).to(device)
    t_est = torch.randint(0, 1000, (batch,)).to(device)
    
    with torch.no_grad():
        out_est = model(x_est, t_est)
    
    print(f"  Batch size {batch}: ✓")

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
print("\nModel architecture:")
print(f"  Input: (B, 2, 608, 608) - [Noisy, Clean] channels")
print(f"  Output: (B, 1, 608, 608) - Noise prediction")
print(f"  Parameters: {num_params:,}")
print(f"  Time conditioning: Sinusoidal (dim=128)")
print(f"  Encoder levels: 4 (with downsampling)")
print(f"  Bottleneck: 2 residual blocks")
print(f"  Decoder levels: 4 (with upsampling + skip connections)")
