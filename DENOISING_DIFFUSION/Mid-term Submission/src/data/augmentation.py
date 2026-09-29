"""
Data augmentation utilities for astronomical images.

Provides physics-aware augmentation techniques that preserve the
physical properties of astronomical observations while increasing
training dataset diversity.
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Callable, List, Optional


class RandomHorizontalFlip:
    """
    Randomly flip image horizontally.
    
    Args:
        p: Probability of flipping (default: 0.5)
        
    Example:
        >>> transform = RandomHorizontalFlip(p=0.5)
        >>> img = np.random.randn(64, 64)
        >>> img_flipped = transform(img)
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply random horizontal flip.
        
        Args:
            img: Input image, shape (H, W) or (C, H, W)
            
        Returns:
            Flipped or original image
        """
        if np.random.rand() < self.p:
            if img.ndim == 2:
                return np.fliplr(img).copy()
            elif img.ndim == 3:
                return np.flip(img, axis=2).copy()
        return img


class RandomVerticalFlip:
    """
    Randomly flip image vertically.
    
    Args:
        p: Probability of flipping (default: 0.5)
        
    Example:
        >>> transform = RandomVerticalFlip(p=0.5)
        >>> img = np.random.randn(64, 64)
        >>> img_flipped = transform(img)
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply random vertical flip.
        
        Args:
            img: Input image, shape (H, W) or (C, H, W)
            
        Returns:
            Flipped or original image
        """
        if np.random.rand() < self.p:
            if img.ndim == 2:
                return np.flipud(img).copy()
            elif img.ndim == 3:
                return np.flip(img, axis=1).copy()
        return img


def get_augmentation_transforms(
    augment: bool = True,
    image_size: int = 64,
) -> transforms.Compose:
    """
    Get data augmentation pipeline for astronomical images.
    
    Args:
        augment: Whether to apply augmentations (if False, just basic transforms)
        image_size: Target image size for resizing  
        
    Returns:
        torchvision.transforms.Compose object
        
    Note:
        Astronomical images require careful augmentation to preserve
        physical meaning. We use:
        - 90° rotations (preserves isotropy)
        - Horizontal/vertical flips (preserves symmetry)
        - NO color jittering (intensity has physical meaning)
        - NO random crops (spatial structure matters)
        - NO perspective transforms (distorts geometry)
        
    Example:
        >>> transform = get_augmentation_transforms(augment=True)
        >>> img = np.random.randn(64, 64)
        >>> img_aug = transform(img)
    """
    transform_list = []
    
    if augment:
        # Random 90-degree rotations (0, 90, 180, 270)
        # Physics: Disk observations are typically isotropic
        transform_list.append(RandomRotation90())
        
        # Random horizontal and vertical flips
        # Physics: No preferred orientation in most cases
        transform_list.append(RandomHorizontalFlip(p=0.5))
        transform_list.append(RandomVerticalFlip(p=0.5))
    
    # Convert numpy to tensor
    transform_list.append(transforms.Lambda(lambda x: torch.from_numpy(x).float()))
    
    return transforms.Compose(transform_list)


class RandomRotation90:
    """
    Randomly rotate image by 0, 90, 180, or 270 degrees.
    
    This is physics-aware: rotations by 90° preserve the regular grid
    structure and don't introduce interpolation artifacts.
    
    Example:
        >>> transform = RandomRotation90()
        >>> img = np.random.randn(64, 64)
        >>> img_rotated = transform(img)
    """
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply random 90-degree rotation.
        
        Args:
            img: Input image, shape (H, W) or (C, H, W)
            
        Returns:
            Rotated image
        """
        k = np.random.randint(0, 4)  # 0, 1, 2, or 3
        
        if img.ndim == 2:
            # (H, W) -> rotate in (0, 1) axes
            return np.rot90(img, k=k, axes=(0, 1)).copy()
        elif img.ndim == 3:
            # (C, H, W) -> rotate in (1, 2) axes
            return np.rot90(img, k=k, axes=(1, 2)).copy()
        else:
            raise ValueError(f"Expected 2D or 3D image, got {img.ndim}D")


class AddGaussianNoise:
    """
    Add Gaussian noise to image (for data augmentation).
    
    This simulates varying noise levels in observations, helping the
    model generalize to different observation conditions.
    
    Args:
        sigma_range: (min_sigma, max_sigma) for random noise level
        
    Example:
        >>> transform = AddGaussianNoise(sigma_range=(0.01, 0.05))
        >>> img = np.random.randn(64, 64)
        >>> img_noisy = transform(img)
    """
    
    def __init__(self, sigma_range: tuple = (0.01, 0.05)):
        self.sigma_range = sigma_range
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Add random Gaussian noise."""
        sigma = np.random.uniform(*self.sigma_range)
        noise = np.random.randn(*img.shape) * sigma
        return  (img + noise).astype(img.dtype)


class RandomIntensityScale:
    """
    Randomly scale image intensity by a factor.
    
    This simulates different exposure times or telescope sensitivities.
    
    Args:
        scale_range: (min_scale, max_scale) for random intensity scaling
        
    Note:
        Only use this if you're normalizing images during training!
        Otherwise it changes the physical flux values.
        
    Example:
        >>> transform = RandomIntensityScale(scale_range=(0.8, 1.2))
        >>> img = np.random.randn(64, 64)
        >>> img_scaled = transform(img)
    """
    
    def __init__(self, scale_range: tuple = (0.8, 1.2)):
        self.scale_range = scale_range
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Scale image intensity."""
        scale = np.random.uniform(*self.scale_range)
        return (img * scale).astype(img.dtype)


def create_training_transforms(
    config: dict,
    include_noise: bool = False,
    include_intensity_scale: bool = False,
) -> transforms.Compose:
    """
    Create comprehensive training augmentation pipeline.
    
    Args:
        config: Configuration dictionary
        include_noise: Whether to add Gaussian noise augmentation
        include_intensity_scale: Whether to add intensity scaling
        
    Returns:
        Composed transform
        
    Example:
        >>> config = {"data": {"image_size": 64}}
        >>> transform = create_training_transforms(config, include_noise=True)
    """
    transform_list = []
    
    # Geometric augmentations (always physics-aware)
    transform_list.append(RandomRotation90())
    transform_list.append(RandomHorizontalFlip(p=0.5))
    transform_list.append(RandomVerticalFlip(p=0.5))
    
    # Noise augmentation (optional)
    if include_noise:
        transform_list.append(AddGaussianNoise(sigma_range=(0.01, 0.05)))
    
    # Intensity scaling (optional, use with caution)
    if include_intensity_scale:
        transform_list.append(RandomIntensityScale(scale_range=(0.9, 1.1)))
    
    # Convert to tensor
    transform_list.append(transforms.Lambda(lambda x: torch.from_numpy(x).float()))
    
    return transforms.Compose(transform_list)


def create_validation_transforms() -> transforms.Compose:
    """
    Create validation transform (no augmentation).
    
    Returns:
        Transform that only converts to tensor
        
    Example:
        >>> transform = create_validation_transforms()
        >>> img = np.random.randn(64, 64)
        >>> img_tensor = transform(img)
    """
    return transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x).float())
    ])


# Physics constraints to remember:
# 
# DO USE:
# - 90° rotations (maintains regular grid)
# - Horizontal/vertical flips (preserves symmetry)
# - Mild noise addition (simulates observation conditions)
# 
# DO NOT USE:
# - Arbitrary angle rotations (creates interpolation artifacts)
# - Color jittering (intensity = physical flux)
# - Random crops that remove context (spatial structure matters)
# - Perspective transforms (distorts physical geometry)
# - Elastic deformations (changes actual structure)
#
# CAUTION:
# - Intensity scaling (only if normalizing during training)
# - Gaussian blur (may remove real features)
