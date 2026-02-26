# EXXA — Exoplanet Exploration with Machine Learning

EXXA is a collection of machine learning projects for exoplanet science, developed under [ML4SCI](https://ml4sci.org/) as part of [Google Summer of Code](https://summerofcode.withgoogle.com/).

The repository hosts contributions that apply deep learning, quantum machine learning, and computer vision techniques to problems in exoplanet atmosphere characterization, protoplanetary disk denoising, planetary system architecture prediction, and more.

## Project Structure

| Directory | Description |
|-----------|-------------|
| [`ANOMALY_DETECTION/`](ANOMALY_DETECTION/) | Anomaly detection in exoplanet atmospheric spectra |
| [`ATMOSPHERE_CHARACTERIZATION/`](ATMOSPHERE_CHARACTERIZATION/) | ML models for identifying chemical species and atmospheric properties from observed spectra |
| [`DENOISING_DIFFUSION/`](DENOISING_DIFFUSION/) | Denoising astronomical observations of protoplanetary disks |
| [`DUST_CONTINUUM_APPROCH/`](DUST_CONTINUUM_APPROCH/) | Dust continuum analysis approaches for disk characterization |
| [`EQUIVARIANT_NETWORKS_PLANETARY_SYSTEMS_ARCHITECTURES/`](EQUIVARIANT_NETWORKS_PLANETARY_SYSTEMS_ARCHITECTURES/) | Equivariant vision networks for predicting planetary system architectures |
| [`FOUNDATION_MODELS_FOR_EXOPLANET_CHARACTERIZATION/`](FOUNDATION_MODELS_FOR_EXOPLANET_CHARACTERIZATION/) | Foundation models for exoplanet detection and analysis |
| [`KINEMATIC_APPROACH/`](KINEMATIC_APPROACH/) | Kinematic approaches to exoplanet characterization |
| [`NEURAL_NETWORK_CLASSIFIER/`](NEURAL_NETWORK_CLASSIFIER/) | Neural network classifiers for exoplanet data |
| [`QUANTUM_MACHINE_LEARNING_FOR_EXOPLANET_CHARACTERIZATION/`](QUANTUM_MACHINE_LEARNING_FOR_EXOPLANET_CHARACTERIZATION/) | Quantum ML techniques for exoplanet atmosphere classification |
| [`TIME_SERIES_APPROACH/`](TIME_SERIES_APPROACH/) | Time series methods for exoplanet signal analysis |

Each subdirectory corresponds to a GSoC project and contains its own README with detailed instructions.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/ML4SCI/EXXA.git
   cd EXXA
   ```

2. Navigate to the project of interest and follow its README for setup and usage instructions. Most projects use Python with PyTorch or TensorFlow and provide Jupyter notebooks for experimentation.

## Related GSoC 2026 Projects

- Equivariant Vision Networks for Predicting Planetary Systems' Architectures
- Denoising Astronomical Observations of Protoplanetary Disks
- Exoplanet Atmosphere Characterization
- Foundation Models for Exoplanet Characterization
- Quantum Machine Learning for Exoplanet Characterization

For more details, see the [ML4SCI GSoC 2026 project list](https://ml4sci.org/gsoc/2026/summary.html).

## Contributing

Contributions are welcome! If you would like to contribute, please:

1. Fork this repository
2. Create a feature branch (`git checkout -b my-feature`)
3. Commit your changes and push to your fork
4. Open a pull request against the `main` branch

For questions or discussions, reach out via the [ML4SCI Gitter channel](https://matrix.to/#/#ML4SCI_general:gitter.im) or email [ml4-sci@cern.ch](mailto:ml4-sci@cern.ch).

## Acknowledgments

This project is part of [Machine Learning for Science (ML4SCI)](https://ml4sci.org/), supported by Google Summer of Code. We thank all GSoC contributors and mentors who have made this work possible.
