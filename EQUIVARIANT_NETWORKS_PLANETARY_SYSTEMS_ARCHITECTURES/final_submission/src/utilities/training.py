from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomAffine,
    ToTensor,
    Resize,
)
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models.models import (
    MetaLearner,
    create_steerable_model,
    create_equivariant_hybrid_model,
)
from utilities.training import balance_dataset, stratified_split
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from dataset.dataset import PlanetaryDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import random
from utilities.metrics import plot_confusion_matrix_and_roc
import wandb
import torch


TRAIN_TRANSFORM = Compose(
    [
        RandomHorizontalFlip(p=0.5),
        RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ToTensor(),
        Resize((224, 224)),
    ]
)

VAL_TEST_TRANSFORM = Compose([ToTensor(), Resize((224, 224))])


def run_model_sequentially(models, images, device):
    """
    Runs a list of models sequentially on the input images.

    Args:
        models (list): List of models to run sequentially.
        images (torch.Tensor): Input image tensor.
        device (str): The device to run the models on (e.g., 'cuda' or 'cpu').

    Returns:
        features (list): List of feature outputs from the models.
    """
    features = []
    for model in models:
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            output = model(images)

            features.append(output)
    return features


def calculate_feature_size(model, input_size=(1, 3, 224, 224)):
    """
    Calculate the size of the output feature map of a model.

    Args:
        model (torch.nn.Module): The model to calculate the feature size for.
        input_size (tuple): Input tensor size (default: (1, 3, 224, 224)).

    Returns:
        int: The number of output features.
    """
    dummy_input = torch.randn(input_size)
    with torch.no_grad():
        output = model.forward(dummy_input)
    return output.shape[1]


def get_trainer(epochs):
    """
    Creates a PyTorch Lightning trainer object for training.

    Args:
        epochs (int): Number of epochs to train.

    Returns:
        pl.Trainer: Trainer object configured for training.
    """
    return pl.Trainer(
        max_epochs=epochs,
        logger=WandbLogger(project="voting_classifier_project"),
        log_every_n_steps=2,
        devices="auto",
        accelerator="gpu",
        num_sanity_val_steps=0,
        val_check_interval=1.0,
        accumulate_grad_batches=8,
    )


def train_meta_learner(
    full_dataset,
    percentage_of_dataset,
    pretrained_models,
):
    """
    Trains a meta learner on the given dataset using pre-trained models.

    Args:
        full_dataset (Dataset): The complete dataset for training.
        percentage_of_dataset (float): Percentage of the dataset to use.
        pretrained_models (dict): Dictionary of pre-trained models.
    """
    meta_learner = MetaLearner(pretrained_models)

    balanced_dataset = balance_dataset(full_dataset)

    subset_size = int(percentage_of_dataset * len(balanced_dataset))
    subset_indices = np.random.choice(len(balanced_dataset), subset_size, replace=False)
    subset_dataset = Subset(balanced_dataset, subset_indices)

    train_indices, val_indices = stratified_split(subset_dataset)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    val_dataset.dataset.transform = VAL_TEST_TRANSFORM

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=3,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
    )

    trainer = get_trainer()

    trainer.fit(meta_learner, train_loader, val_loader)


def evaluate_meta_learner(percentage_of_dataset, channels, meta_learner, device):
    """
    Evaluates the meta learner on the test dataset.

    Args:
        percentage_of_dataset (float): Percentage of dataset to evaluate.
        channels (list): List of velocity channels to use for evaluation.
        meta_learner (torch.nn.Module): Meta learner model.
        device (str): Device for computation (e.g., 'cuda' or 'cpu').
    """
    test_dataset = PlanetaryDataset(
        data_dir="/kaggle/input/gsoc-protoplanetary-disks/Test_Clean",
        csv_file="/kaggle/input/gsoc-protoplanetary-disks/test_info.csv",
        channels=channels,
        transform=VAL_TEST_TRANSFORM,
    )

    balanced_test_dataset = balance_dataset(test_dataset)
    subset_size = int(percentage_of_dataset * len(balanced_test_dataset))
    subset_indices = np.random.choice(
        len(balanced_test_dataset), subset_size, replace=False
    )
    subset_dataset = Subset(balanced_test_dataset, subset_indices)
    test_indices, _ = stratified_split(subset_dataset, val_size=0.0)
    test_dataset_small = Subset(subset_dataset, test_indices)

    print(f"Test dataset length: {len(test_dataset_small)}")
    test_loader = DataLoader(
        test_dataset_small, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            logits = meta_learner(images)
            preds = torch.argmax(logits, dim=1)

            print(f"Logits: {logits}, preds: {preds}")

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    print(f"Meta-Learner accuracy: {accuracy * 100:.2f}%")

    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Meta-Learner Confusion Matrix")
    plt.savefig("meta_learner_conf_matrix.png")
    plt.close()


def evaluate_ensemble(trained_models, percentage_of_dataset, test_dataset):
    """
    Evaluates the ensemble of trained models on a test dataset.

    Args:
        trained_models (dict): Dictionary containing the trained models.
        percentage_of_dataset (float): Percentage of the dataset to evaluate on.
        test_dataset (Dataset): Dataset to evaluate the ensemble on.

    Returns:
        float: Accuracy of the ensemble on the test dataset.
    """
    print(
        f"Started evaluating the ensemble that contains {len(trained_models)} models..."
    )

    balanced_test_dataset = balance_dataset(test_dataset)

    subset_size = int(percentage_of_dataset * len(balanced_test_dataset))
    subset_indices = np.random.choice(
        len(balanced_test_dataset), subset_size, replace=False
    )
    subset_dataset = Subset(balanced_test_dataset, subset_indices)

    test_indices, _ = stratified_split(subset_dataset, val_size=0.0)
    test_dataset_small = Subset(subset_dataset, test_indices)

    print(f"Test dataset length for ensemble: {len(test_dataset_small)}")

    test_loader = DataLoader(
        test_dataset_small, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            print(f"labels: {labels}")
            preds_list = []
            for model_list in trained_models.values():
                logits = model_list[0](images)
                preds = torch.argmax(logits, dim=1)
                preds_list.append(preds)

            final_preds = torch.min(torch.stack(preds_list), dim=0).values

            all_predictions.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    print(all_predictions)
    all_labels = np.array(all_labels)
    print(all_labels)
    accuracy = np.mean(all_predictions == all_labels)
    print(f"Ensemble accuracy on test dataset: {accuracy * 100:.2f}%")

    cm = confusion_matrix(all_labels, all_predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Ensemble Confusion Matrix")

    save_dir = "figures"
    os.makedirs(save_dir, exist_ok=True)
    conf_matrix_path = os.path.join(save_dir, "ensemble_conf_matrix_channels_final.png")
    plt.savefig(conf_matrix_path)
    plt.close()
    print(f"Confusion matrix saved to {conf_matrix_path}")

    return accuracy


def train(
    percentage_of_dataset,
    dataset,
    channels,
    seed=42,
    pre_trained_dir=None,
    model_type="steerable",
):
    """
    Trains a model on the given dataset for the specified velocity channels.

    Args:
        percentage_of_dataset (float): Percentage of dataset to use for training.
        dataset (Dataset): The dataset to train the model on.
        channels (list): List of velocity channels.
        seed (int): Random seed for reproducibility (default: 42).
        pre_trained_dir (str): Path to pre-trained model directory (default: None).
        model_type (str): Type of model to use ('steerable' or 'equivariant_hybrid').

    Returns:
        torch.nn.Module: Trained model.
    """
    print(f"Training model for channels: {channels}")

    torch.manual_seed(seed)
    random.seed(seed)

    formatted_channels = "_".join(map(str, channels))

    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_file_path = os.path.join(save_dir, f"model_channels_{formatted_channels}.pt")

    if model_type == "steerable":
        model = create_steerable_model()
    elif model_type == "equivariant_hybrid":
        model = create_equivariant_hybrid_model()

    pre_trained_dir = "/kaggle/input/first_10_epochs_steerable_balanced_50_percent_data/pytorch/default/3"

    if pre_trained_dir:
        pre_trained_model_path = os.path.join(
            pre_trained_dir, f"model_channels{formatted_channels}.pt"
        )
        if os.path.exists(pre_trained_model_path):
            print(f"Loading pre-trained model from {pre_trained_model_path}")
            model.load_state_dict(
                torch.load(pre_trained_model_path)
            )  # Load the saved model state
        else:
            print(
                f"Pre-trained model not found at {pre_trained_model_path}. Starting training from scratch."
            )
    full_dataset = PlanetaryDataset(
        data_dir="/kaggle/input/gsoc-protoplanetary-disks/Train_Clean",
        csv_file="/kaggle/input/gsoc-protoplanetary-disks/train_info_cleaned.csv",
        channels=channels,
        transform=TRAIN_TRANSFORM,
    )

    balanced_dataset = balance_dataset(full_dataset)

    subset_size = int(percentage_of_dataset * len(balanced_dataset))
    subset_indices = np.random.choice(len(balanced_dataset), subset_size, replace=False)
    subset_dataset = Subset(balanced_dataset, subset_indices)

    train_indices, val_indices = stratified_split(subset_dataset)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=max(len(channels) // 2, 2),
        shuffle=True,
        num_workers=3,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(len(channels) // 2, 2),
        shuffle=False,
        num_workers=3,
        pin_memory=True,
    )

    trainer = get_trainer()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    torch.save(model.state_dict(), model_file_path)
    print(f"Model saved to {model_file_path}")

    print("Training complete!")
    return model


def evaluate_model_on_channels(percentage_of_dataset, models, channels):
    """
    Method to evaluate a model on a test dataset.
    """
    test_dataset = PlanetaryDataset(
        data_dir="/kaggle/input/gsoc-protoplanetary-disks/Test_Clean",
        csv_file="/kaggle/input/gsoc-protoplanetary-disks/test_info.csv",
        channels=channels,
        transform=TRAIN_TRANSFORM,
    )
    balanced_test_dataset = balance_dataset(test_dataset)

    subset_size = int(percentage_of_dataset * len(balanced_test_dataset))
    subset_indices = np.random.choice(
        len(balanced_test_dataset), subset_size, replace=False
    )
    subset_dataset = Subset(balanced_test_dataset, subset_indices)

    test_indices, _ = stratified_split(subset_dataset, val_size=0.0)
    test_dataset_small = Subset(subset_dataset, test_indices)

    print(f"Test dataset length: {len(test_dataset_small)}")
    test_loader = DataLoader(
        test_dataset_small, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    if len(models) == 1:
        return plot_confusion_matrix_and_roc(models[0], test_loader, channels)


def learn(
    percentage_of_dataset,
    trained_models,
    dataset,
    channel_subsets,
    model_type="steerable",
):
    """
    Function to train a model on a dataset and evaluate it on the test dataset. If the model is better than the previous one, it is saved.
    """
    for channels in channel_subsets:
        model = train(dataset, channels, model_type)
        accuracy = evaluate_model_on_channels(percentage_of_dataset, [model], channels)
        if str(channels) not in trained_models.keys():
            trained_models[str(channels)] = [model, accuracy]
        else:
            if accuracy > trained_models[str(channels)][1]:
                print(
                    f"The accuracy of the new run of the sweep for channels {channels} is {accuracy} and is bigger than the previous one: {trained_models[str(channels)][1]}"
                )
                trained_models[str(channels)] = [model, accuracy]


def run_sweeps_for_channel_subsets(
    percentage_of_dataset,
    trained_models,
    channel_subsets,
    sweep_id,
    dataset,
    model_type="steerable",
    number_of_sweeps=1,
):
    """
    Function to run sweeps for different channel subsets.
    """
    for channels in channel_subsets:
        print(f"Starting sweep for channel subset: {channels}")
        wandb.agent(
            sweep_id,
            lambda: learn(
                percentage_of_dataset, trained_models, dataset, [channels], model_type
            ),
            count=number_of_sweeps,
        )

    print("Evaluating ensemble of all trained models...")
    test_channels = sum(channel_subsets, [])
    test_dataset = PlanetaryDataset(
        data_dir="/kaggle/input/gsoc-protoplanetary-disks/Test_Clean",
        csv_file="/kaggle/input/gsoc-protoplanetary-disks/test_info.csv",
        channels=test_channels,
        transform=TRAIN_TRANSFORM,
    )
    evaluate_ensemble(test_dataset, channel_subsets)


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def weighted_vote(predictions_list, channel_subsets):
    """
    Make a weighted vote based on the predictions of the models (middle models have more weight).
    """
    final_predictions = {}
    weights = []

    num_subsets = len(channel_subsets)
    for i in range(num_subsets):
        if i <= num_subsets // 2:
            weight = (i + 1) / (num_subsets // 2 + 1)
        else:
            weight = (num_subsets - i) / (num_subsets // 2 + 1)
        weights.append(weight)

    for run_id in predictions_list[0].keys():
        weighted_sum = 0.0
        total_weight = 0.0
        valid_count = 0
        for preds, weight in zip(predictions_list, weights):
            if run_id in preds:
                weighted_sum += weight * np.mean(preds[run_id])
                total_weight += weight
                valid_count += 1

        if valid_count > 0:
            final_predictions[run_id] = np.round(weighted_sum / total_weight).astype(
                int
            )

    return final_predictions
