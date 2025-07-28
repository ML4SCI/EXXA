import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


class PlanetaryDataset(Dataset):
    """
    A PyTorch Dataset class for loading planetary images across different velocity channels.

    Args:
        data_dir (str): Directory containing the run folders for each image.
        csv_file (str): CSV file with metadata about the runs.
        channels (list): List of velocity channels to load.
        transform (callable, optional): Optional transform to apply to the images.
    """

    def __init__(self, data_dir, csv_file, channels, transform=None):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.channels = channels

        self.data["label"] = self.data.apply(
            lambda row: 0 if row["n"] == 0 else 1,
            axis=1,  # binary classification of planet presence
        )

    def __len__(self):
        return len(self.data) * len(self.channels)

    def __getitem__(self, idx):
        run_idx = idx // len(self.channels)
        channel_idx = idx % len(self.channels)
        run = int(self.data.iloc[run_idx]["run"])
        channel = self.channels[channel_idx]
        run_folder = os.path.join(self.data_dir, f"Run_{run}")

        if not os.path.isdir(run_folder):
            print(
                f"Warning: Run directory {run_folder} does not exist. Skipping run {run}."
            )
            return None

        image = None
        for step_folder in os.listdir(run_folder):
            step_path = os.path.join(run_folder, step_folder)
            if os.path.isdir(step_path):
                pattern = re.compile(f"13co_chan_{channel}_.*\\.png$")
                found_image = False
                for img_file in os.listdir(step_path):
                    if pattern.match(img_file):
                        img_path = os.path.join(step_path, img_file)
                        if os.path.isfile(img_path):
                            image = Image.open(img_path).convert("RGB")
                            if self.transform:
                                image = self.transform(image)
                            found_image = True
                            break
                if not found_image:
                    print(
                        f"Warning: No image found for channel {channel} in run {run}, step {step_folder}"
                    )
            if image is not None:
                break  # Process only the first step folder for each run

        if image is None:
            print(
                f"Warning: No valid images found for run {run}. Skipping run, channel was: {channel}."
            )
            return None

        label = torch.tensor(self.data.iloc[run_idx]["label"], dtype=torch.long)
        return image, label


def balance_dataset(dataset):
    """
    Balance the dataset by undersampling the majority class to match the minority class.
    """
    labels = [dataset[i][1].item() for i in range(len(dataset))]

    # indices for each class
    class_0_indices = [i for i, label in enumerate(labels) if label == 0]
    class_1_indices = [i for i, label in enumerate(labels) if label == 1]

    # Undersample the majority class
    if len(class_0_indices) > len(class_1_indices):
        class_0_indices = np.random.choice(
            class_0_indices, len(class_1_indices), replace=False
        )
    else:
        class_1_indices = np.random.choice(
            class_1_indices, len(class_0_indices), replace=False
        )

    balanced_indices = np.concatenate([class_0_indices, class_1_indices])
    np.random.shuffle(balanced_indices)

    return Subset(dataset, balanced_indices)


def stratified_split(dataset, val_size=0.2, random_seed=42):
    """
    Returns train and validation indices with class balance preserved.
    If val_size is 0, it returns the entire dataset as the training set.
    """
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    indices = np.arange(len(dataset))

    if val_size == 0:
        # If val_size is 0, return all indices for training and an empty validation set
        return indices, []

    # Use sklearn's train_test_split with stratify parameter to split the data while maintaining class balance
    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, stratify=labels, random_state=random_seed
    )

    return train_indices, val_indices
