import torch.optim as optim
import torch
from torch import nn
from e3nn.o3 import Irreps, FullyConnectedTensorProduct
from torchmetrics import Accuracy
import pytorch_lightning as pl
from torchvision.models import resnet18
from e2cnn import gspaces
from e2cnn import nn as e2nn


class EquivariantHybridModel(pl.LightningModule):
    def __init__(self, num_classes=2, lr=0.001, weight_decay=0.0001, num_channels=12):
        super().__init__()
        self.save_hyperparameters()
        self.num_channels = num_channels

        # feature extraction
        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(
            3 * num_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.resnet.fc = nn.Identity()  # Removing the fully connected layer

        irreps_in = Irreps(f"{512}x0e")
        irreps_out = Irreps(f"{1024}x0e")  # Increased output dimensionality

        self.equivariant_layer = FullyConnectedTensorProduct(
            irreps_in, irreps_in, irreps_out, internal_weights=True, shared_weights=True
        )

        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.resnet(x)
        y = torch.ones_like(x)
        x = self.equivariant_layer(x, y)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class E2SteerableCNN(pl.LightningModule):
    def __init__(self, num_classes=2, lr=0.001, weight_decay=0.0001, num_channels=12):
        super().__init__()
        self.save_hyperparameters()

        # type of symmetry (SE(2) - rotations and translations)
        self.gspace = gspaces.Rot2dOnR2(N=8)  # N rotations

        self.input_type = e2nn.FieldType(
            self.gspace, num_channels * [self.gspace.trivial_repr]
        )
        self.mid_type = e2nn.FieldType(self.gspace, 16 * [self.gspace.regular_repr])
        self.output_type = e2nn.FieldType(self.gspace, 32 * [self.gspace.trivial_repr])

        self.conv1 = e2nn.R2Conv(
            self.input_type, self.mid_type, kernel_size=5, padding=2, stride=1
        )
        self.conv2 = e2nn.R2Conv(
            self.mid_type, self.output_type, kernel_size=5, padding=2, stride=1
        )

        self.nonlin = e2nn.ReLU(self.mid_type, inplace=True)

        self.pool = e2nn.PointwiseMaxPoolAntialiased(
            self.output_type, sigma=0.6, stride=2, kernel_size=2
        )

        self.fully_connected = e2nn.GroupPooling(self.output_type)
        self.fc = nn.Linear(32, num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = e2nn.GeometricTensor(x, self.input_type)
        x = self.conv1(x)
        x = self.nonlin(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.fully_connected(x)
        x = x.tensor
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
