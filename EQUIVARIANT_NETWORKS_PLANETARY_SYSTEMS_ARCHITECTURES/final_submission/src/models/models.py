import pytorch_lightning as pl
import torch
from torch import nn
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import wandb

from torchvision.models import vgg16
from utilities.training import run_model_sequentially
from torch.optim import Adam

from e2cnn import gspaces
from e2cnn import nn as e2nn


class EquivariantHybridModel(pl.LightningModule):
    """
    A simple equivariant model that uses group-equivariant convolutions for
    detecting rotational symmetries in image data.

    This model is designed with two R2Conv layers followed by ReLU activations
    and a final fully connected layer for classification. It is specifically
    designed for rotational symmetry.

    Args:
        num_classes (int): Number of output classes for classification (default: 2).
        lr (float): Learning rate for the Adam optimizer (default: 0.001).
        weight_decay (float): Weight decay (L2 penalty) for the optimizer (default: 0.0001).
        num_channels (int): Number of input channels (default: 1).
        input_size (int): Size of the input images (default: 600).

    """

    def __init__(
        self,
        num_classes=2,
        lr=0.001,
        weight_decay=0.0001,
        num_channels=1,
        input_size=600,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.input_size = input_size

        # rotation group with N=8
        r2_act = gspaces.Rot2dOnR2(N=16)

        feat_type_in = e2nn.FieldType(r2_act, 3 * num_channels * [r2_act.trivial_repr])
        feat_type_out = e2nn.FieldType(r2_act, 10 * [r2_act.regular_repr])

        # equivariant layers
        self.conv1 = e2nn.R2Conv(feat_type_in, feat_type_out, kernel_size=5, padding=0)
        self.relu1 = e2nn.ReLU(feat_type_out)

        self.conv2 = e2nn.R2Conv(feat_type_out, feat_type_out, kernel_size=3, padding=0)
        self.relu2 = e2nn.ReLU(feat_type_out)

        spatial_dim = (
            self.input_size - 4 - 2
        )  # two convolutions without padding: input_size - (5-1) - (3-1)
        conv_out_channels = feat_type_out.size
        conv_out_size = conv_out_channels * spatial_dim * spatial_dim

        self.fc = nn.Linear(conv_out_size, num_classes)

    @torch.compile
    def forward(self, x):
        x = e2nn.GeometricTensor(x, self.conv1.in_type)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.tensor.view(x.tensor.shape[0], -1)
        x = self.fc(x)
        return x

    @torch.compile
    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        return loss

    def get_feature_maps(self, x):
        x = e2nn.GeometricTensor(x, self.conv1.in_type)
        x = self.conv1(x)
        return x

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


class EquivariantVgg16(pl.LightningModule):
    def __init__(
        self,
        num_classes=2,
        lr=0.00001,
        weight_decay=0.0001,
        num_channels=3,
        input_size=224,
    ):
        super(EquivariantVgg16, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay

        # rotation group (C4) for equivariance (90-degree rotations)
        r2_act = gspaces.rot2dOnR2(N=4)

        vgg = vgg16()

        self.features = self.modify_conv_layers(vgg.features, r2_act, num_channels)

        # from the original VGG16
        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(100352, 4096),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        self.outputs = []

    def modify_conv_layers(self, layers, r2_act, input_channels):
        """Replace every convolution layer in VGG16 with equivariant convolutions using a finite group"""
        modified_layers = []

        # Define the input FieldType with 3 trivial representations for RGB channels
        in_type = e2nn.FieldType(r2_act, 3 * [r2_act.trivial_repr])

        for idx, layer in enumerate(layers):
            if isinstance(layer, nn.Conv2d):
                print(f"Layer {idx} is a conv2d")
                out_channels = layer.out_channels
                print(
                    f"Conv2d layer: {layer} | in_channels: {input_channels} | out_channels: {out_channels}"
                )

                # regular representation of the finite group
                out_type = e2nn.FieldType(r2_act, out_channels * [r2_act.regular_repr])

                # Replace with R2Conv (equivariant convolution)
                kernel_size = (
                    layer.kernel_size[0]
                    if isinstance(layer.kernel_size, tuple)
                    else layer.kernel_size
                )
                equivariant_conv = e2nn.R2Conv(
                    in_type,
                    out_type,
                    kernel_size=kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                )
                modified_layers.append(equivariant_conv)

                print(f"After layer {idx}, out_type: {out_type}")

                # Update in_type for the next layer
                in_type = out_type

            elif isinstance(layer, nn.ReLU):
                print(f"Layer {idx} is a relu")
                relu = e2nn.ReLU(in_type)
                modified_layers.append(relu)

            elif isinstance(layer, nn.MaxPool2d):
                print(f"Layer {idx} is a maxpool")
                equivariant_pooling = e2nn.PointwiseAvgPoolAntialiased(
                    in_type, sigma=0.66, stride=layer.stride
                )
                modified_layers.append(equivariant_pooling)

            elif isinstance(layer, nn.BatchNorm2d):
                print(f"Layer {idx} is a batchnorm")
                equivariant_bn = e2nn.IIDBatchNorm2d(in_type)
                modified_layers.append(equivariant_bn)

        return e2nn.SequentialModule(*modified_layers)

    def forward(self, x):

        x = e2nn.GeometricTensor(x, self.features[0].in_type)

        x = self.features(x)

        x = x.tensor

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        print("Training step initiated.")
        images, labels = batch
        logits = self.forward(images)
        preds = torch.argmax(logits, dim=1)

        print(
            f"Training - Predictions: {preds.cpu().numpy()}, Labels: {labels.cpu().numpy()}"
        )

        loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.1, patience=3, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels)
        preds = torch.argmax(logits, dim=1)
        print(
            f"Validation - Predictions: {preds.cpu().numpy()}, Labels: {labels.cpu().numpy()}"
        )
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, logger=True, sync_dist=True)
        self.outputs.append(
            {
                "val_loss": loss.cpu(),
                "val_acc": acc.cpu(),
                "preds": preds.cpu().numpy(),
                "labels": labels.cpu().numpy(),
            }
        )
        return {
            "val_loss": loss.cpu(),
            "val_acc": acc.cpu(),
            "preds": preds.cpu().numpy(),
            "labels": labels.cpu().numpy(),
        }

    def on_validation_epoch_end(self):
        print("On validation epoch end!!!!")
        preds = np.concatenate(
            [
                (
                    output["preds"]
                    if isinstance(output["preds"], np.ndarray)
                    else output["preds"].cpu().numpy()
                )
                for output in self.outputs
            ]
        )
        labels = np.concatenate(
            [
                (
                    output["labels"]
                    if isinstance(output["labels"], np.ndarray)
                    else output["labels"].cpu().numpy()
                )
                for output in self.outputs
            ]
        )

        precision = precision_score(labels, preds, average="binary")
        recall = recall_score(labels, preds, average="binary")
        f1 = f1_score(labels, preds, average="binary")

        if len(np.unique(labels)) == 2:
            roc_auc = roc_auc_score(labels, preds)
        else:
            roc_auc = None

        metrics = {
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "val_roc_auc": roc_auc,
            "val_loss": np.mean([output["val_loss"].item() for output in self.outputs]),
            "val_acc": np.mean([output["val_acc"].item() for output in self.outputs]),
        }

        wandb.log(metrics)

        self.log("val_precision", precision, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_recall", recall, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_f1", f1, prog_bar=True, logger=True, sync_dist=True)
        if roc_auc:
            self.log("val_roc_auc", roc_auc, prog_bar=True, logger=True, sync_dist=True)

        self.outputs = []


class EquivariantSteerableModel(pl.LightningModule):
    """
    A steerable equivariant model based on the continuous rotation group SO(2).

    This model uses steerable convolutions, making it robust to any rotation
    in the plane. It is particularly well-suited for tasks where the objects of
    interest can appear at arbitrary angles.

    Args:
        num_classes (int): The number of output classes. Default is 2.
        lr (float): Learning rate for the optimizer. Default is 0.001.
        weight_decay (float): Weight decay for the optimizer. Default is 0.0001.
        input_size (int): The size of the input image (assumed square). Default is 600.
    """

    def __init__(
        self,
        num_classes=2,
        lr=0.001,
        weight_decay=0.0001,
        num_channels=1,
        input_size=600,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.input_size = input_size

        # continuous rotation group (SO(2))
        r2_act = gspaces.rot2dOnR2(N=-1)

        in_type = e2nn.FieldType(r2_act, 3 * [r2_act.trivial_repr])
        self.input_type = in_type

        activation1 = e2nn.FourierELU(
            r2_act, 8, irreps=r2_act.fibergroup.bl_irreps(3), N=16, inplace=True
        )
        out_type = activation1.in_type
        self.block1 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            e2nn.IIDBatchNorm2d(out_type),
            activation1,
        )

        activation2 = e2nn.FourierELU(
            r2_act, 16, irreps=r2_act.fibergroup.bl_irreps(3), N=16, inplace=True
        )
        out_type = activation2.in_type
        self.block2 = e2nn.SequentialModule(
            e2nn.R2Conv(
                self.block1.out_type, out_type, kernel_size=5, padding=2, bias=False
            ),
            e2nn.IIDBatchNorm2d(out_type),
            activation2,
            e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2),
        )

        activation3 = e2nn.FourierELU(
            r2_act, 32, irreps=r2_act.fibergroup.bl_irreps(3), N=16, inplace=True
        )
        out_type = activation3.in_type
        self.block3 = e2nn.SequentialModule(
            e2nn.R2Conv(
                self.block2.out_type, out_type, kernel_size=5, padding=2, bias=False
            ),
            e2nn.IIDBatchNorm2d(out_type),
            activation3,
            e2nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2),
        )

        output_invariant_type = e2nn.FieldType(r2_act, 64 * [r2_act.trivial_repr])
        self.invariant_map = e2nn.R2Conv(
            out_type, output_invariant_type, kernel_size=1, bias=False
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.c = output_invariant_type.size

        self.fully_net = nn.Sequential(
            nn.LayerNorm(self.c),
            nn.ELU(inplace=True),
            nn.Linear(self.c, num_classes),
        )
        self.outputs = []

    def forward(self, x):
        x = e2nn.GeometricTensor(x, self.input_type)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.invariant_map(x)

        x = x.tensor
        x = self.pool(x)

        x = x.view(x.shape[0], -1)
        x = self.fully_net(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("train_acc", acc, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        print("Validation step")
        images, labels = batch
        logits = self.forward(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, logger=True, sync_dist=True)
        self.outputs.append(
            {
                "val_loss": loss.cpu(),
                "val_acc": acc.cpu(),
                "preds": preds.cpu().numpy(),
                "labels": labels.cpu().numpy(),
            }
        )
        return {
            "val_loss": loss.cpu(),
            "val_acc": acc.cpu(),
            "preds": preds.cpu().numpy(),
            "labels": labels.cpu().numpy(),
        }

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def on_validation_epoch_end(self):
        print("On validation epoch end!")
        preds = np.concatenate(
            [
                (
                    output["preds"]
                    if isinstance(output["preds"], np.ndarray)
                    else output["preds"].cpu().numpy()
                )
                for output in self.outputs
            ]
        )
        labels = np.concatenate(
            [
                (
                    output["labels"]
                    if isinstance(output["labels"], np.ndarray)
                    else output["labels"].cpu().numpy()
                )
                for output in self.outputs
            ]
        )

        # metrics
        precision = precision_score(labels, preds, average="binary")
        recall = recall_score(labels, preds, average="binary")
        f1 = f1_score(labels, preds, average="binary")

        if len(np.unique(labels)) == 2:
            roc_auc = roc_auc_score(labels, preds)
        else:
            roc_auc = None

        metrics = {
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "val_roc_auc": roc_auc,
            "val_loss": np.mean([output["val_loss"].item() for output in self.outputs]),
            "val_acc": np.mean([output["val_acc"].item() for output in self.outputs]),
        }

        wandb.log(metrics)

        self.log("val_precision", precision, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_recall", recall, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_f1", f1, prog_bar=True, logger=True, sync_dist=True)
        if roc_auc:
            self.log("val_roc_auc", roc_auc, prog_bar=True, logger=True, sync_dist=True)

        self.outputs = []


class MetaLearnerWithVoting(pl.LightningModule):
    def __init__(self, pretrained_models, voting="soft"):
        super(MetaLearnerWithVoting, self).__init__()
        self.pretrained_models = pretrained_models
        self.voting = voting

    def forward(self, x):
        x = x.to(self.device)

        for model in self.pretrained_models.values():
            model = model.to(self.device)

        base_model_outputs = run_model_sequentially(
            list(self.pretrained_models.values()), x, self.device
        )

        # average probabilities
        if self.voting == "soft":
            probs = [torch.softmax(output, dim=1) for output in base_model_outputs]
            avg_probs = torch.mean(torch.stack(probs, dim=0), dim=0)
            return avg_probs

        # majority vote
        elif self.voting == "hard":
            preds = [torch.argmax(output, dim=1) for output in base_model_outputs]
            stacked_preds = torch.stack(preds, dim=0)
            majority_vote = torch.mode(stacked_preds, dim=0).values
            return majority_vote

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Voting classifiers do not require training.")

    def configure_optimizers(self):
        return None


def create_steerable_model():
    return EquivariantSteerableModel(
        num_classes=2, lr=0.001, weight_decay=0.0001, num_channels=1
    )


def create_equivariant_hybrid_model():
    return EquivariantHybridModel(
        num_classes=2, lr=0.001, weight_decay=0.0001, num_channels=1
    )
