# EXXA

## Overview

EXXA is a collection of machine learning approaches for problems related to exoplanet discovery, characterization, and astrophysical signal analysis. The repository contains multiple independent research directions exploring how modern ML techniques can be applied to astronomical datasets.

Each directory focuses on a different scientific or modeling approach, ranging from classical neural networks to diffusion models and quantum machine learning.

## Repository Structure

| Directory                                               | Description                                                                                        |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| ANOMALY_DETECTION                                       | Detects unusual patterns in astronomical observations that may indicate rare astrophysical events. |
| ATMOSPHERE_CHARACTERIZATION                             | Uses machine learning to infer atmospheric properties of exoplanets from observational data.       |
| DENOISING_DIFFUSION                                     | Diffusion-based models for reconstructing or generating astrophysical signals.                     |
| DUST_CONTINUUM_APPROCH                                  | Models dust continuum emissions in astrophysical systems.                                          |
| EQUIVARIANT_NETWORKS_PLANETARY_SYSTEMS_ARCHITECTURES    | Uses equivariant neural networks to model planetary system structures.                             |
| FOUNDATION_MODELS_FOR_EXOPLANET_CHARACTERIZATION        | Experiments with foundation models for large-scale exoplanet analysis.                             |
| KINEMATIC_APPROACH                                      | Uses kinematic methods to analyze motion and dynamics in astrophysical datasets.                   |
| NEURAL_NETWORK_CLASSIFIER                               | Standard neural network classifiers for astronomical signal classification tasks.                  |
| QUANTUM_MACHINE_LEARNING_FOR_EXOPLANET_CHARACTERIZATION | Explores quantum machine learning techniques for exoplanet analysis.                               |
| TIME_SERIES_APPROACH                                    | Time-series modeling of astronomical observations such as light curves.                            |

## Requirements

Typical dependencies include

Python 3.8+
NumPy
SciPy
PyTorch or TensorFlow
Matplotlib
AstroPy
Jupyter Notebook

Install dependencies using

pip install -r requirements.txt

If a directory has its own environment or dependency file, follow the instructions within that module.

## Usage

Most modules are implemented as research notebooks and scripts. Navigate to the relevant directory and run the corresponding notebook.

Example

cd TIME_SERIES_APPROACH
jupyter notebook

Each module contains experiments, models, and evaluation workflows specific to the problem it addresses.

## Goals

The project explores how machine learning can assist in

exoplanet detection
atmospheric parameter estimation
astronomical signal denoising
rare event detection
physical system modeling

## Contributing

Contributions are welcome. If you want to extend an approach or add a new ML method for astrophysical analysis, open a pull request with a clear description of the contribution.

## License

Specify the license for this repository here.
