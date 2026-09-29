"""
PyTorch Dataset class for astronomical observations.

Handles loading and batching of paired noisy/clean observations
from FITS files or numpy arrays.
"""

# `X | Y` annotations below are evaluated at runtime on Python < 3.10 and raise
# TypeError there. Deferring annotation evaluation keeps the module importable on
# 3.9, so the test suite runs locally before a Kaggle session is spent.
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List
from .preprocessing import normalize_image, denormalize_image, pad_to_multiple
from .augmentation import get_augmentation_transforms


class AstroDataset(Dataset):
    """
    Dataset for loading paired noisy and clean astronomical observations.
    
    Supports loading from:
    - NumPy files (.npy, .npz)
    - FITS files (via fits_loader)
    - In-memory numpy arrays
    
    Supports patch-based training for memory efficiency on large images.
    
    Args:
        dirty: Array or path to noisy observations, shape (N, H, W)
        clean: Array or path to clean observations, shape (N, H, W)
        augmentation: Optional augmentation transforms
        normalize: Whether to normalize images to [-1, 1]
        padding: Target size for padding (will pad to multiple of this)
        patch_size: Size of patches to extract (e.g., 64). If None, use full images.
        n: Number of random patches to extract per image (default 1)
        parse_patches: Whether to extract and stack patches or use full image
    """
    
    def __init__(
        self,
        dirty: np.ndarray | str | Path,
        clean: np.ndarray | str | Path,
        augmentation: Optional[dict] = None,
        normalize: bool = True,
        padding: int = 32,
        split: Optional[str] = None,
        split_ratio: float = 0.8,
        patch_size: Optional[int] = None,
        n: int = 1,
        parse_patches: bool = False,
    ):
        self.dirty = self._load_data(dirty)
        self.clean = self._load_data(clean)
        self.normalize = normalize
        self.padding = padding
        self.patch_size = patch_size
        self.n_patches = n
        self.parse_patches = parse_patches
        
        # Apply split if specified
        if split is not None:
            self.dirty, self.clean = self._apply_split(split, split_ratio)
        
        # Apply augmentation
        self.augmentation = augmentation or {}
        self.transforms = get_augmentation_transforms(**self.augmentation)
        
        assert len(self.dirty) == len(self.clean), \
            f"Mismatched lengths: dirty={len(self.dirty)}, clean={len(self.clean)}"
        
        # Check spatial dimensions match
        assert self.dirty.shape == self.clean.shape, \
            f"Mismatched shapes: dirty={self.dirty.shape}, clean={self.clean.shape}"
        
    def _load_data(self, data: np.ndarray | str | Path) -> np.ndarray:
        """Load data from array or file."""
        if isinstance(data, np.ndarray):
            return data.astype(np.float32)
        
        path = Path(data)
        if path.suffix == ".npy":
            return np.load(path).astype(np.float32)
        elif path.suffix == ".npz":
            loaded = np.load(path)
            # Assume single array or take 'data' key
            key = list(loaded.keys())[0] if len(loaded.files) > 0 else None
            return loaded[key].astype(np.float32)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _apply_split(
        self,
        split: str,
        ratio: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split data into train/val/test."""
        n = len(self.dirty)
        n_train = int(n * ratio)
        
        if split == "train":
            return self.dirty[:n_train], self.clean[:n_train]
        elif split == "val":
            return self.dirty[n_train:], self.clean[n_train:]
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def __len__(self) -> int:
        # Always return number of images, not patches
        return len(self.dirty)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Get a single sample."""
        dirty_img = self.dirty[idx]  # (H, W)
        clean_img = self.clean[idx]  # (H, W)
        img_id = f"{idx:05d}"
        
        if self.parse_patches and self.patch_size:
            # Get random patch coordinates
            h, w = dirty_img.shape
            i_list, j_list, th, tw = self.get_params(h, w, (self.patch_size, self.patch_size), self.n_patches)
            
            # Stack patches: (n_patches, 2, patch_size, patch_size)
            patches = []
            for i, j in zip(i_list, j_list):
                dirty_patch = dirty_img[i:i+th, j:j+tw]
                clean_patch = clean_img[i:i+th, j:j+tw]
                combined = np.stack([dirty_patch, clean_patch], axis=0)  # (2, H, W)
                patches.append(combined)
            
            combined = np.stack(patches, axis=0)  # (n_patches, 2, H, W)
            img_tensor = torch.from_numpy(combined).float()
        else:
            # Full-image: stack dirty and clean into 2-channel tensor
            # Add channel dimension
            dirty_img = np.expand_dims(dirty_img, axis=0)  # (1, H, W)
            clean_img = np.expand_dims(clean_img, axis=0)  # (1, H, W)
            
            # Apply augmentation
            if self.transforms:
                dirty_img = self.transforms(dirty_img)
                clean_img = self.transforms(clean_img)
            
            # Normalize
            if self.normalize:
                dirty_img = normalize_image(dirty_img)
                clean_img = normalize_image(clean_img)
            
            # Pad if needed
            if self.padding:
                dirty_img, _ = pad_to_multiple(dirty_img, self.padding)
                clean_img, _ = pad_to_multiple(clean_img, self.padding)
            
            # Stack dirty and clean: (2, H, W)
            combined = np.concatenate([dirty_img, clean_img], axis=0)
            img_tensor = torch.from_numpy(combined).float()
        
        return img_tensor, img_id
    
    @staticmethod
    def get_params(h: int, w: int, target_size: Tuple[int, int], n: int) -> Tuple[List[int], List[int], int, int]:
        """
        Generate n random crop coordinates.
        
        Args:
            h: Image height
            w: Image width
            target_size: (target_h, target_w) for crops
            n: Number of crops to generate
        
        Returns:
            (i_list, j_list, target_h, target_w) - lists of top-left corners and target size
        """
        target_h, target_w = target_size
        
        # Ensure we don't crop beyond bounds
        max_i = max(0, h - target_h)
        max_j = max(0, w - target_w)
        
        i_list = [np.random.randint(0, max_i + 1) if max_i > 0 else 0 for _ in range(n)]
        j_list = [np.random.randint(0, max_j + 1) if max_j > 0 else 0 for _ in range(n)]
        
        return i_list, j_list, target_h, target_w


def create_dataloaders(
    dirty_train: np.ndarray | str,
    clean_train: np.ndarray | str,
    dirty_val: Optional[np.ndarray | str] = None,
    clean_val: Optional[np.ndarray | str] = None,
    config: Optional[dict] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    augmentation: Optional[dict] = None,
    normalize: bool = True,
    padding: int = 32,
    parse_patches: bool = False,
    patch_size: Optional[int] = None,
    n_patches: int = 1,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        dirty_train: Training noisy data
        clean_train: Training clean data
        dirty_val: Validation noisy data (optional)
        clean_val: Validation clean data (optional)
        config: Config dict with keys like 'training', 'data' (optional)
        batch_size: Batch size for loading
        num_workers: Number of parallel workers
        augmentation: Augmentation config dict
        normalize: Whether to normalize to [-1, 1]
        padding: Pad images to multiple of this
        parse_patches: Whether to extract patches
        patch_size: Size of patches if parse_patches=True
        n_patches: Number of patches to extract
    
    Returns:
        (train_loader, val_loader) tuple
    """
    # Extract config if provided
    if config is not None:
        batch_size = config.get("training", {}).get("batch_size", batch_size)
        num_workers = config.get("data", {}).get("num_workers", num_workers)
        n_patches = config.get("training", {}).get("patch_n", n_patches)
        patch_size = config.get("data", {}).get("image_size", patch_size)
        normalize = config.get("data", {}).get("normalize", normalize)
    
    train_dataset = AstroDataset(
        dirty=dirty_train,
        clean=clean_train,
        augmentation=augmentation,
        normalize=normalize,
        padding=padding,
        patch_size=patch_size,
        n=n_patches,
        parse_patches=parse_patches,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = None
    if dirty_val is not None and clean_val is not None:
        val_dataset = AstroDataset(
            dirty=dirty_val,
            clean=clean_val,
            augmentation=None,  # No augmentation on validation
            normalize=normalize,
            padding=padding,
            patch_size=patch_size,
            n=n_patches,
            parse_patches=parse_patches,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return train_loader, val_loader
