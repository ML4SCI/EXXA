"""
Data loading and preprocessing utilities for astronomical observations.

This module provides classes and functions for:
- Loading FITS files from ALMA/VLT observations
- Loading synthetic training data from simulations
- Data augmentation techniques (physics-aware)
- Train/val/test splitting utilities
- PyTorch Dataset and DataLoader creation

The legacy exports below are resolved lazily (PEP 562). They belong to the
continuum/patch era that the move to full-image line emission deprecated, and importing them eagerly
made every line-emission import pull in `torchvision` and the `X | Y` type syntax
of Python 3.10+: neither of which the line-emission pipeline uses. That coupling
broke `import src.data.fits_cube_dataset` on interpreters older than 3.10 even
though the module itself is compatible, which blocked running the test suite
locally before pushing to Kaggle. Resolving on first attribute access keeps
`from src.data import AstroDataset` working and leaves the line-emission path
dependency-free.
"""

import importlib

_LAZY = {
    "AstroDataset": ".dataset",
    "create_dataloaders": ".dataset",
    "FITSLoader": ".fits_loader",
    "load_fits_file": ".fits_loader",
    "normalize_image": ".preprocessing",
    "denormalize_image": ".preprocessing",
    "get_augmentation_transforms": ".augmentation",
}

__all__ = list(_LAZY)


def __getattr__(name):
    """Import the owning submodule on first access to one of the legacy names."""
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name], __name__), name)
        globals()[name] = value          # cache so later lookups skip this path
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
