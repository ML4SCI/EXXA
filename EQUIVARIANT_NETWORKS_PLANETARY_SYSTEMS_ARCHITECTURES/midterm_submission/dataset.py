import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class PlanetaryDataset(Dataset):
    def __init__(self, data_dir, csv_file, channels, transform=None):
        self.data_dir = data_dir
        self.data = pd.read_csv(csv_file)
        self.channels = channels
        self.transform = transform

        self.data["label"] = self.data.apply(
            lambda row: 0 if row["n"] == 0 else 1, axis=1
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        run = int(self.data.iloc[idx]["run"])
        run_folder = os.path.join(self.data_dir, f"Run_{run}")

        if not os.path.isdir(run_folder):
            print(
                f"Warning: Run directory {run_folder} does not exist. Skipping run {run}."
            )
            return None

        images = []
        for step_folder in os.listdir(run_folder):
            step_path = os.path.join(run_folder, step_folder)
            if os.path.isdir(step_path):
                step_images = []
                for channel in self.channels:
                    pattern = re.compile(f"13co_chan_{channel}_.*\\.png$")
                    found_image = False
                    for img_file in os.listdir(step_path):
                        if pattern.match(img_file):
                            img_path = os.path.join(step_path, img_file)
                            if os.path.isfile(img_path):
                                image = Image.open(img_path).convert("RGB")
                                if self.transform:
                                    image = self.transform(image)
                                step_images.append(image)
                                found_image = True
                                break
                    if not found_image:
                        print(
                            f"Warning: No image found for channel {channel} in run {run}, step {step_folder}"
                        )
                if len(step_images) == len(self.channels):
                    images.extend(step_images)
                else:
                    print(
                        f"Warning: Not enough images found in step folder {step_folder} for run {run}. Skipping step folder."
                    )
            break  # Process only the first step folder for each run

        if len(images) == 0:
            print(
                f"Warning: No valid images found for run {run}. Skipping run, channels were: {self.channels}."
            )
            return None

        images = torch.stack(images, dim=0)  # Shape: [num_channels, 3, height, width]
        images = images.permute(1, 0, 2, 3)  # Shape: [3, num_channels, height, width]
        images = images.reshape(
            3 * len(self.channels), images.size(2), images.size(3)
        )  # Shape: [3 * num_channels, height, width]

        label = torch.tensor(self.data.iloc[idx]["label"], dtype=torch.long)
        return images, label
