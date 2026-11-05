"""
Image preprocessing utilities for astronomical observations.
"""

import numpy as np
import torch
from typing import Tuple, Optional


def normalize_image(
    image: np.ndarray,
    method: str = "minmax",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """
    Normalize image to [0, 1] or [-1, 1] range.
    
    Args:
        image: Input image array
        method: Normalization method:
            - "minmax": Scale to [0, 1] using min/max
            - "percentile": Scale to [0, 1] using 1st/99th percentiles (robust)
            - "zscore": Standardize to zero mean, unit variance
            - "tanh": Scale to [-1, 1] using tanh transformation
        vmin: Minimum value for clipping (optional, computed if None)
        vmax: Maximum value for clipping (optional, computed if None)
        
    Returns:
        Normalized image
        
    Example:
        >>> img = np.random.randn(64, 64)
        >>> img_norm = normalize_image(img, method="percentile")
        >>> print(img_norm.min(), img_norm.max())  # Should be close to 0, 1
    """
    if method == "minmax":
        if vmin is None:
            vmin = image.min()
        if vmax is None:
            vmax = image.max()
            
        if vmax - vmin < 1e-8:
            return np.zeros_like(image)
            
        return (image - vmin) / (vmax - vmin)
    
    elif method == "percentile":
        if vmin is None:
            vmin = np.percentile(image, 1)
        if vmax is None:
            vmax = np.percentile(image, 99)
            
        if vmax - vmin < 1e-8:
            return np.zeros_like(image)
            
        image_clipped = np.clip(image, vmin, vmax)
        # The input is already clipped to [vmin, vmax], so the result is [0, 1] up to
        # rounding -- but in float32 the division alone can land at 1.0000001. Clip the
        # output too, so the documented [0, 1] range holds exactly.
        return np.clip((image_clipped - vmin) / (vmax - vmin), 0.0, 1.0)
    
    elif method == "zscore":
        mean = image.mean()
        std = image.std()
        if std < 1e-8:
            return np.zeros_like(image)
        return (image - mean) / std
    
    elif method == "tanh":
        # Map to [-1, 1] using tanh transformation
        # First z-score normalize, then apply tanh
        mean = image.mean()
        std = image.std()
        if std < 1e-8:
            return np.zeros_like(image)
        return np.tanh((image - mean) / (3 * std))
    
    else:
        raise ValueError(
            f"Unknown normalization method: {method}. "
            f"Choose from: minmax, percentile, zscore, tanh"
        )


def denormalize_image(
    image: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """
    Reverse min-max normalization.
    
    Args:
        image: Normalized image in [0, 1] range
        vmin: Original minimum value
        vmax: Original maximum value
        
    Returns:
        Denormalized image in original value range
        
    Example:
        >>> img = np.random.randn(64, 64)
        >>> vmin, vmax = img.min(), img.max()
        >>> img_norm = normalize_image(img)
        >>> img_denorm = denormalize_image(img_norm, vmin, vmax)
        >>> np.allclose(img, img_denorm)  # True
    """
    return image * (vmax - vmin) + vmin


def data_transform(X: torch.Tensor) -> torch.Tensor:
    """
    Transform data from [0, 1] to [-1, 1] range (for diffusion models).
    
    Args:
        X: Input tensor in [0, 1] range
        
    Returns:
        Tensor in [-1, 1] range
    """
    return 2 * X - 1.0


def inverse_data_transform(X: torch.Tensor) -> torch.Tensor:
    """
    Inverse transform data from [-1, 1] to [0, 1] range.
    
    Args:
        X: Input tensor in [-1, 1] range
        
    Returns:
        Tensor in [0, 1] range, clamped
    """
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def pad_to_multiple(
    image: np.ndarray,
    multiple: int = 16,
    mode: str = "reflect",
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Pad image so dimensions are multiples of given value.
    
    Useful for ensuring images work with U-Net architectures that
    require specific dimension constraints.
    
    Args:
        image: Input image, shape (H, W) or (C, H, W)
        multiple: Pad dimensions to be multiples of this value
        mode: Padding mode ("reflect", "constant", "edge", "wrap")
        
    Returns:
        (padded_image, padding_amounts) where padding_amounts is
        (pad_top, pad_bottom, pad_left, pad_right)
        
    Example:
        >>> img = np.random.randn(63, 67)  # Not multiples of 16
        >>> img_padded, padding = pad_to_multiple(img, 16)
        >>> print(img_padded.shape)  # (64, 80) - both multiples of 16
    """
    if image.ndim == 2:
        h, w = image.shape
        c = None
    elif image.ndim == 3:
        c, h, w = image.shape
    else:
        raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")
    
    # Calculate padding needed
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Apply padding
    if image.ndim == 2:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
    else:
        padding = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
    
    padded = np.pad(image, padding, mode=mode)
    
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def unpad_image(
    image: np.ndarray,
    padding: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Remove padding from image.
    
    Args:
        image: Padded image
        padding: (pad_top, pad_bottom, pad_left, pad_right)
        
    Returns:
        Unpadded image
    """
    pad_top, pad_bottom, pad_left, pad_right = padding
    
    if image.ndim == 2:
        h, w = image.shape
        return image[pad_top : h - pad_bottom, pad_left : w - pad_right]
    elif image.ndim == 3:
        c, h, w = image.shape
        return image[:, pad_top : h - pad_bottom, pad_left : w - pad_right]
    else:
        raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")
