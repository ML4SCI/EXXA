# Protoplanetary Disk Classification with Equivariant Neural Networks

This project aims to classify protoplanetary disks to determine whether they contain forming planets or not. The classification task uses multi-channel images from simulated protoplanetary disks, using advanced neural network architectures that incorporate equivariance to rotational and translational transformations.

## Project Overview

The documentation is present here: https://medium.com/@murariu.alexandra2002/gsoc-with-ml4sci-equivariant-vision-networks-for-predicting-planetary-systems-architectures-576f13e8d403 (work in progress)

This project utilizes two primary models:

- **EquivariantHybridModel**: Combines a pre-trained ResNet-18 for feature extraction with an equivariant layer to handle rotational symmetries.
- **E2SteerableCNN**: Utilizes the e2cnn library to create a fully equivariant convolutional neural network based on SE(2) symmetries.

Additionally, an active learning framework and a multi-agent system using reinforcement learning (RL) are implemented to optimize channel selection and improve model performance.

## Requirements

- Python 3.6+
- PyTorch
- PyTorch Lightning
- e3nn
- e2cnn
- wandb
- pandas
- numpy
- Pillow

## Installation

1. Clone the repository:
2. Install the required packages
    ```bash
    pip install -r requirements.txt
    ```

## Running the project
1. Simple active learning
    ```bash
    python . --method simple --model EquivariantHybridModel --epochs 10
    ```
    or
    ```bash
    python . --method simple --model E2SteerableCNN --epochs 10
    ```

2. Multi-agent system approach
    ```bash
    python . --method mas --model EquivariantHybridModel --epochs 10 --n_agents 3
    ```

### Explanation of parameters
`--method:` The method to use for training (`simple` for simple active learning, `mas` for multi-agent system).

`--model:` The model architecture to use (`EquivariantHybridModel` or `E2SteerableCNN`).

`--epochs`: The number of epochs to train the model (default is `10`).

`--n_agents:` The number of agents to use for the multi-agent system (default is `3`).

