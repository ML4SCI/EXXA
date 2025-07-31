# Midterm Work

This folder contains the notebooks developed up to the **Midterm phase** of the project .

## 📁 Notebooks:

### 1. `Preprocessing_Original.ipynb`

- Loads the simulated transmission spectra dataset (98k samples, 269 wavelength bins).
- Normalizes spectra to remove scale-based biases for machine learning and quantum encoding.
- Dimensionality reduction using Autoencoder.
- Clustering to analyze data distribution with Heatmaps to analyse clusters.
- Pairplots to analyze atmospheric parameters and latent features.

### 2. `Quantum_Encoding_Testing.ipynb`

- Uses the latent space features from `Preprocessing_Original.ipynb` to test different Quantum Encoding methods - **Angle Encoding**, **Amplitude Encoding**, and **Qsample Encoding** to prepare data for downsttream tasks involving QML Algorithms.
- Implements K-Means clustering and plotting heatmaps to check atmospheric parameter distributions across clusters for encoded data.

### 3. `Clustering_Test.ipynb`

- Applies various clustering algorithms to both the autoencoder latent features and quantum-encoded representations.
- Visualizes clusters to assess structure and separability in the feature space for both the classical and quantum representations.
