import numpy as np
from pytorch_lightning import Callback
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomAffine, ToTensor
from torch.utils.data import DataLoader, random_split


from dataset import PlanetaryDataset
from models import EquivariantHybridModel, E2SteerableCNN


class ValidRunsLogger(Callback):
    def on_epoch_end(self, trainer, pl_module):
        valid_runs = sum([len(batch) > 0 for batch in trainer.train_dataloader])
        print(f"Epoch {trainer.current_epoch}: Number of valid runs: {valid_runs}")
        wandb.log({"valid_runs": valid_runs})


def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def train(model_name, dataset, channel_subsets, epochs):
    models = []
    all_run_predictions = {}
    all_run_labels = {}

    for channels in channel_subsets:
        if model_name == "EquivariantHybridModel":
            model = EquivariantHybridModel(
                num_classes=2, lr=0.001, weight_decay=0.0001, num_channels=len(channels)
            )
        elif model_name == "E2SteerableCNN":
            model = E2SteerableCNN(
                num_classes=2, lr=0.001, weight_decay=0.0001, num_channels=len(channels)
            )

        trainer = Trainer(
            max_epochs=epochs, logger=WandbLogger(), callbacks=[ValidRunsLogger()]
        )

        train_dataset = PlanetaryDataset(
            data_dir="/content/drive/MyDrive/Kinematic_Data/Train_Clean",
            csv_file="/content/drive/MyDrive/Kinematic_Data/train_info.csv",
            channels=channels,
            transform=Compose(
                [
                    RandomHorizontalFlip(p=0.5),
                    RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    ToTensor(),
                ]
            ),
        )

        test_dataset = PlanetaryDataset(
            data_dir="/content/drive/MyDrive/Kinematic_Data/Test_Clean",
            csv_file="/content/drive/MyDrive/Kinematic_Data/test_info.csv",
            channels=channels,
            transform=Compose(
                [
                    RandomHorizontalFlip(p=0.5),
                    RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    ToTensor(),
                ]
            ),
        )

        train_size = 100  # shorter training size for faster training
        test_size = 30  # shorter test size for faster training
        _, train_dataset = random_split(
            train_dataset, [len(train_dataset) - train_size, train_size]
        )
        _, test_dataset = random_split(
            test_dataset, [len(test_dataset) - test_size, test_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=len(channels) // 2,
            shuffle=True,
            num_workers=4,
            collate_fn=custom_collate,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=len(channels) // 2,
            shuffle=False,
            num_workers=4,
            collate_fn=custom_collate,
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                if batch is None:
                    continue
                images, batch_labels = batch
                run_ids = [
                    int(dataset.data.iloc[i]["run"]) for i in range(len(batch_labels))
                ]

                logits = model(images)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                batch_labels = batch_labels.cpu().numpy()

                for run_id, pred, label in zip(run_ids, preds, batch_labels):
                    if run_id not in all_run_predictions:
                        all_run_predictions[run_id] = []
                        all_run_labels[run_id] = label
                    all_run_predictions[run_id].append(pred)

        models.append(model)

    return models, all_run_predictions, all_run_labels


def active_learning(
    model_name,
    channel_subsets,
    dataset,
    initial_subset_size=5,
    n_iterations=5,
    epochs=10,
    uncertainty_strategy="entropy",
):
    def calculate_uncertainty(logits):
        if uncertainty_strategy == "entropy":
            probs = torch.nn.functional.softmax(logits, dim=1)
            log_probs = torch.log(probs)
            uncertainty = -torch.sum(probs * log_probs, dim=1)
        elif uncertainty_strategy == "margin":
            probs = torch.nn.functional.softmax(logits, dim=1)
            top2_probs, _ = torch.topk(probs, 2, dim=1)
            uncertainty = top2_probs[:, 0] - top2_probs[:, 1]
        elif uncertainty_strategy == "model":
            uncertainty = torch.std(logits, dim=1)
        else:
            raise ValueError(f"Unknown uncertainty strategy: {uncertainty_strategy}")
        return uncertainty

    def select_channels(remaining_channels, model, dataset):
        model.eval()
        uncertainties = []
        for channel in remaining_channels:
            subset = [channel]
            temp_dataset = PlanetaryDataset(
                data_dir=dataset.data_dir,
                csv_file=dataset.csv_file,
                channels=subset,
                transform=dataset.transform,
            )
            temp_loader = DataLoader(
                temp_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                collate_fn=custom_collate,
            )
            channel_uncertainty = []
            for batch in temp_loader:
                if batch is None:
                    continue
                images, _ = batch
                logits = model(images)
                uncertainty = calculate_uncertainty(logits)
                channel_uncertainty.append(uncertainty.item())
            avg_uncertainty = np.mean(channel_uncertainty)
            uncertainties.append((channel, avg_uncertainty))
        uncertainties.sort(
            key=lambda x: x[1], reverse=(uncertainty_strategy != "margin")
        )
        return [channel for channel, _ in uncertainties[:5]]

    def vote(predictions_dict):
        final_predictions = {}
        for run_id, preds in predictions_dict.items():
            final_predictions[run_id] = np.round(np.mean(preds)).astype(int)
        return final_predictions

    for _ in range(n_iterations):
        initial_channels = channel_subsets[:initial_subset_size]
        models, run_predictions, run_labels = train(
            model_name, dataset, [initial_channels], epochs
        )

        # Select channels based on uncertainty
        remaining_channels = [
            ch for ch in channel_subsets if ch not in initial_channels
        ]
        additional_channels = select_channels(remaining_channels, models[0], dataset)
        initial_channels.extend(additional_channels)

        # Retrain with the updated channel subset
        models, additional_predictions, _ = train(
            model_name, dataset, [initial_channels], epochs
        )

        # Update run_predictions with new predictions
        for run_id, preds in additional_predictions.items():
            if run_id in run_predictions:
                run_predictions[run_id].extend(preds)
            else:
                run_predictions[run_id] = preds

    final_predictions = vote(run_predictions)
    list(run_labels.values())

    accuracy = np.mean(
        [final_predictions[run_id] == run_labels[run_id] for run_id in run_labels]
    )
    print(f"Final accuracy after voting: {accuracy * 100:.2f}%")
