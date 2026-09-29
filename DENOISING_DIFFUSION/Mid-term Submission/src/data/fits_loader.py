"""
FITS file loader for astronomical observations.

Handles loading FITS files from ALMA, VLT, and other telescopes,
with proper handling of headers, WCS coordinates, and data units.
"""

import numpy as np
from pathlib import Path
from typing import Union, Tuple, Dict, Optional, List
from astropy.io import fits
from astropy.wcs import WCS
import warnings


class FITSLoader:
    """
    Loader for FITS files from astronomical observations.
    
    Supports loading intensity maps from ALMA/VLT observations,
    handling headers, coordinate systems, and data normalization.
    
    Args:
        normalize: Whether to normalize loaded images (default: True)
        target_shape: If provided, resize images to this shape (H, W)
        
    Example:
        >>> loader = FITSLoader(normalize=True)
        >>> data, header = loader.load("observation.fits")
        >>> print(data.shape, data.min(), data.max())
    """
    
    def __init__(
        self,
        normalize: bool = True,
        target_shape: Optional[Tuple[int, int]] = None,
    ):
        self.normalize = normalize
        self.target_shape = target_shape
        
    def load(
        self, filepath: Union[str, Path], hdu_index: int = 0
    ) -> Tuple[np.ndarray, fits.Header]:
        """
        Load a FITS file and return data and header.
        
        Args:
            filepath: Path to FITS file
            hdu_index: Index of HDU to load (default: 0 for primary HDU)
            
        Returns:
            (data, header) tuple where:
            - data: numpy array of image data, shape (H, W) or (C, H, W)
            - header: astropy fits.Header object with metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If FITS file is invalid or empty
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"FITS file not found: {filepath}")
            
        try:
            with fits.open(filepath) as hdul:
                # Get data and header
                data = hdul[hdu_index].data
                header = hdul[hdu_index].header
                
                if data is None:
                    raise ValueError(f"No data in HDU {hdu_index} of {filepath}")
                
                # Convert to float32 for ML
                data = data.astype(np.float32)
                
                # Handle different dimensionalities
                # FITS files can be (H, W), (N, H, W), or (N, M, H, W)
                if data.ndim == 4:
                    # (N, M, H, W) -> take first slice
                    warnings.warn(
                        f"4D FITS data detected, taking first slice [0, 0]. "
                        f"Original shape: {data.shape}"
                    )
                    data = data[0, 0]
                elif data.ndim == 3:
                    # (N, H, W) -> take first channel or handle as multi-channel
                    if data.shape[0] == 1:
                        data = data[0]  # Single channel, remove dimension
                    # else: keep as multi-channel (C, H, W)
                elif data.ndim > 4:
                    raise ValueError(
                        f"Unsupported FITS dimensionality: {data.ndim}D. "
                        f"Expected 2D, 3D, or 4D. Shape: {data.shape}"
                    )
                
                # Handle NaN and infinite values
                if np.any(np.isnan(data)):
                    warnings.warn(f"NaN values detected in {filepath}, replacing with 0")
                    data = np.nan_to_num(data, nan=0.0)
                    
                if np.any(np.isinf(data)):
                    warnings.warn(f"Inf values detected in {filepath}, clipping")
                    data = np.nan_to_num(data, posinf=np.finfo(np.float32).max, 
                                        neginf=np.finfo(np.float32).min)
                
                # Normalize if requested
                if self.normalize:
                    data = self._normalize(data)
                    
                # Resize if requested
                if self.target_shape is not None:
                    data = self._resize(data, self.target_shape)
                    
                return data, header
                
        except Exception as e:
            raise ValueError(f"Failed to load FITS file {filepath}: {str(e)}")
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data to [0, 1] range using robust scaling.
        
        Uses percentile-based normalization to handle outliers better
        than simple min-max scaling.
        
        Args:
            data: Input array
            
        Returns:
            Normalized array in [0, 1] range
        """
        # Use 1st and 99th percentiles for robust normalization
        vmin = np.percentile(data, 1)
        vmax = np.percentile(data, 99)
        
        if vmax - vmin < 1e-8:
            warnings.warn("Image has very low dynamic range, returning zeros")
            return np.zeros_like(data)
        
        # Clip and normalize
        data = np.clip(data, vmin, vmax)
        data = (data - vmin) / (vmax - vmin)
        
        return data
    
    def _resize(self, data: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Resize data to target shape using interpolation.
        
        Args:
            data: Input array, shape (H, W) or (C, H, W)
            target_shape: Target (height, width)
            
        Returns:
            Resized array
        """
        from scipy.ndimage import zoom
        
        if data.ndim == 2:
            # (H, W) -> (target_H, target_W)
            zoom_factors = (target_shape[0] / data.shape[0], 
                          target_shape[1] / data.shape[1])
            data = zoom(data, zoom_factors, order=1)  # Bilinear interpolation
        elif data.ndim == 3:
            # (C, H, W) -> (C, target_H, target_W)
            zoom_factors = (1.0,  # Don't change channels
                          target_shape[0] / data.shape[1],
                          target_shape[1] / data.shape[2])
            data = zoom(data, zoom_factors, order=1)
        
        return data
    
    def load_batch(
        self, filepaths: List[Union[str, Path]], hdu_index: int = 0
    ) -> Tuple[np.ndarray, List[fits.Header]]:
        """
        Load multiple FITS files into a batch.
        
        Args:
            filepaths: List of paths to FITS files
            hdu_index: Index of HDU to load
            
        Returns:
            (data_batch, headers) tuple where:
            - data_batch: (N, H, W) or (N, C, H, W) array
            - headers: List of fits.Header objects
            
        All images must have the same shape (set target_shape if needed).
        """
        data_list = []
        headers = []
        
        for filepath in filepaths:
            data, header = self.load(filepath, hdu_index)
            data_list.append(data)
            headers.append(header)
        
        # Stack into batch
        try:
            data_batch = np.stack(data_list, axis=0)
        except ValueError as e:
            shapes = [d.shape for d in data_list]
            raise ValueError(
                f"Cannot stack images with different shapes: {shapes}. "
                f"Set target_shape in FITSLoader to resize all images to same size."
            ) from e
        
        return data_batch, headers


def load_fits_file(
    filepath: Union[str, Path],
    normalize: bool = True,
    hdu_index: int = 0,
) -> Tuple[np.ndarray, fits.Header]:
    """
    Convenience function to load a single FITS file.
    
    Args:
        filepath: Path to FITS file
        normalize: Whether to normalize data to [0, 1]
        hdu_index: HDU index to load
        
    Returns:
        (data, header) tuple
        
    Example:
        >>> data, header = load_fits_file("observation.fits")
        >>> print(f"Loaded {data.shape} image")
        >>> print(f"Telescope: {header.get('TELESCOP', 'Unknown')}")
    """
    loader = FITSLoader(normalize=normalize)
    return loader.load(filepath, hdu_index)


def load_fits_directory(
    directory: Union[str, Path],
    pattern: str = "*.fits",
    normalize: bool = True,
    target_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, List[fits.Header], List[Path]]:
    """
    Load all FITS files from a directory.
    
    Args:
        directory: Path to directory containing FITS files
        pattern: Glob pattern for matching files (default: "*.fits")
        normalize: Whether to normalize images
        target_shape: Target shape for resizing (required if images have different sizes)
        
    Returns:
        (data_batch, headers, filepaths) tuple where:
        - data_batch: (N, H, W) array of images
        - headers: List of FITS headers
        - filepaths: List of file paths (same order as data)
        
    Example:
        >>> data, headers, paths = load_fits_directory(
        ...     "observations/", target_shape=(256, 256)
        ... )
        >>> print(f"Loaded {len(data)} images from {len(paths)} files")
    """
    directory = Path(directory)
    filepaths = sorted(directory.glob(pattern))
    
    if len(filepaths) == 0:
        raise FileNotFoundError(
            f"No FITS files found in {directory} with pattern '{pattern}'"
        )
    
    loader = FITSLoader(normalize=normalize, target_shape=target_shape)
    data_batch, headers = loader.load_batch(filepaths)
    
    return data_batch, headers, filepaths
