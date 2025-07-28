# 🌌 FOUNDATION_MODELS for Exoplanet Characterization

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)  

Self‑supervised pipeline using **Masked Autoencoders (MAE)** to learn latent representations from protoplanetary disk continuum images, for downstream exoplanet characterization tasks.

---

## 🗂️ Project Structure

```text
FOUNDATION_MODELS/
│
├── Scripts/
│   ├── ClusterMetric.py          # Cluster analysis metrics
│   ├── features.py               # Radial & elliptical feature engineering
│   ├── image_augmentations.py    # Image augmentation pipeline
│   ├── LatentSpace.py            # `LatentSpaceAnalyzer` class for clustering and visualizing 
│   ├── MAE.py                    # Masked Autoencoder model definition
│   ├── PatchEmbeddings.py        # Extended PatchEmbed with engineered features
│   ├── Training.py               # Training utilities (train loop, schedulers)
│   └── utils.py                  # Positional embeddings, dataset loader, etc.
│
└── Notebooks/
    └── combined_notebook.ipynb   # End‑to‑end demo using all Scripts modules
```

---

## 🧩 Key Components

* **Feature Engineering (`features.py`)**

  * Radial & elliptical feature extraction centered on disk geometry
  * Gradient-strength, symmetry, and texture descriptors

* **Image Augmentations (`image_augmentations.py`)**

  * Random flips, rotations, Gaussian noise, intensity shifts
  * Improves model robustness to observational variations

* **Masked Autoencoder (`MAE.py`)**

  * Learns to reconstruct masked patches, encouraging contextual understanding
  * Configurable encoder/decoder depths, attention heads, masking ratio

* **Patch Embeddings (`PatchEmbeddings.py`)**

  * Extends ViT PatchEmbed to incorporate engineered features alongside raw pixels

* **Latent Space Analysis (`LatentSpace.py`)**

  * t‑SNE / UMAP projections for visualizing learned embeddings
  * Clustering analysis via `ClusterMetric.py`

* **Training Utilities (`Training.py`)**

  * Standard PyTorch training loop, learning‑rate schedulers, checkpointing

---

## 🪐 Author

**Tanmay Singhal**
*Exploring the universe, one patch at a time.*

```

Feel free to adjust badges, add a `requirements.txt`, or link to datasets and pre-trained checkpoints as needed.
```