# Exoplanet Detection in Protoplanetary Disks: GSoC

This folder contains my GSoC contribution, which involves detecting exoplanets in protoplanetary disks. The project uses multiple deep learning approaches that focus on equivariance and ensembling to handle the rotational symmetries present in astronomical images. The best-performing approach achieved 96% accuracy using a VGG16-based architecture and a meta-learning boosting approach on model probabilities.

## 1. Models

### 1. Simple Equivariant Model (rotational group)
The code is present here: [EquivariantHybridModel](models/models.py), and an example usage is present in this notebook: [equivariant-hybrid-approach.ipynb](notebooks/equivariant-hybrid-approach.ipynb).

- **Architecture**: 2 stacked equivariant convolutional layers followed by fully connected layers.
    - Input: `e2nn.FieldType` on the N=4/8/16 rotation group, which treats the input image as trivial representation under rotation.
    - First equivariant layer: `e2nn.R2Conv` with a `5x5` kernel, which maps the input representation to a higher-dimensional one, with 10 feature channels.
    - First activation: `e2nn.ReLU` (equivariant ReLU)
    - Second equivariant layer `e2nn.R2Conv` with a 3x3 kernel, which preserves the number of output channels.
    - Second activation: `e2nn.ReLU`
    - Fully-connected layer: with 2 possible output classes

- **Outcome**: While the equivariant layers improved rotational invariance, this architecture was too slim and didn't perform as well as other models.

### 2. Steerable Equivariant Model (continuous rotation group)
The code is present here: [EquivariantSteerableModel](models/models.py), and an example usage is present in this notebook: [equivariant-steerable-approach.ipynb](notebooks/equivariant-steerable-approach.ipynb).

- **Architecture**: This model applies steerable convolutions based on the continuous rotation group SO(2) (infinite possible rotations), allowing it to learn symmetries at any angle.
    - **Input FieldType**: The input image is encoded as a `FieldType` with trivial representations, supporting 3 channels (RGB).
    - **Block 1**: 
        - Convolution using `e2nn.R2Conv` with a 7x7 kernel.
        - Batch normalization (`e2nn.IIDBatchNorm2d`) and an equivariant activation `e2nn.FourierELU`.
    - **Block 2**: 
        - Another convolution using `e2nn.R2Conv` (5x5 kernel) and similar activation and normalization.
        - This layer also applies an antialiased average pooling layer (`e2nn.PointwiseAvgPoolAntialiased`) for dimensionality reduction.
    - **Block 3**: 
        - A third convolution with a 5x5 kernel, again followed by batch normalization and FourierELU activation.
        - Pooling applied to further reduce feature size.
    - **Invariant Map**: The final `e2nn.R2Conv` layer produces invariant features across rotations.
    - **Fully Connected Layer**: After pooling, the model applies a fully connected layer to predict the class probabilities.
  
- **Outcome**: By using continuous rotational symmetries (SO(2)), the model achieves robustness to any rotational transformation. However, due to the higher complexity of steerable filters, this model was more computationally intensive. Performance-wise, it was stronger than the simple equivariant model but still didn't outperform the more advanced VGG16-based model.

### 3. Equivariant VGG16 Model

The code is present here: [EquivariantVgg16](models/models.py), and an example usage is present in this notebook: [equivariant-vgg16-approach.ipynb](notebooks/equivariant-vgg16-approach.ipynb).

- **Architecture**: This model modifies the standard VGG16 architecture to use equivariant layers for rotation invariance. The following changes were made to the original VGG16:
    - Each convolutional layer is replaced by `e2nn.R2Conv` layers that are equivariant under a rotational group.
    - We use a rotation group \( C_4 \) for 90-degree rotational symmetry. The input is treated as a trivial representation under this group.
    - The ReLU activation functions are replaced by their equivariant counterparts: `e2nn.ReLU`.
    - Max pooling is replaced by `e2nn.PointwiseAvgPoolAntialiased` to ensure smooth equivariant pooling.
    - Batch normalization is replaced by `e2nn.IIDBatchNorm2d`, which maintains equivariance.

- **Fully-Connected Layers**: After the equivariant feature extraction, the fully connected layers remain the same as in VGG16. The output passes through the classifier layers consisting of two hidden layers and a final output layer with a softmax activation for binary classification.

    - **Equivariant Feature Extraction**:
      - The modified VGG16 has multiple layers of equivariant convolutions with the `C_4` group for 90-degree rotations.
    - **Fully Connected Layers**:
      - The output features are flattened and passed through three linear layers with ReLU activations and dropout.

- **Outcome**: This model, despite being computationally heavy, showed the best performance out of all approaches, especially when dealing with complex rotational variations. It successfully leveraged the deep architecture of VGG16, making it a robust solution for classifying protoplanetary disk images.

## 2. Dataset 
- File: [dataset.py](dataset/`dataset.py`): contains the `PlanetaryDataset` class for loading and processing planetary images across different velocity channels, as well as utilities for balancing and splitting the dataset.
- The dataset is hosted on Kaggle. Please reach out if interested.

## 3. Utilities
- File [metadata.py](utilities/metadata.py): contains sweep configurations, as well as the channel subset split.
- File [metrics.py](utilities/metrics.py): contains methods useful for plotting evaluation metrics.
- File [training.py](utilities/training.py): contains various functions and utilities for training, evaluation, and ensemble learning.
 
 ## Other notebooks
 - File [metalearner-on-vgg16-base-models.ipynb](notebooks/metalearner-on-vgg16-base-models.ipynb): In this file, I ran meta-boosters on top of the predictions of the base eqVGG16 models and compared the results.
 - File [save-predictions.ipynb](notebooks/save_predictions.ipynb): I saved the predictions after loading and running 4 models at a time on the test dataset, due to computational limitations.
 - File [voting-classifiers.ipynb](notebooks/voting-classifiers.ipynb): In this file, I used hard and soft voting classifiers on the predictions saved in the aforementioned file.

 ## Installation

To install this project and its dependencies, you can use `pip`:

### Option 1: Install from Source

1. Clone the repository:

   ```bash
   git clone https://github.com/ML4SCI/EXXA.git
   cd EQUIVARIANT_NETWORKS_PLANETARY_SISTEMS_ARCHITECTURES/final_submission
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. For development, install the additional tools as well:
    ```bash
    pip install -e .[dev]
    ```