import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, MultiStepLR, OneCycleLR

import wandb


class MLP_Encoder(pl.LightningModule):
    def __init__(
        self,
        encoding_layers: nn.ModuleList | None = None,
        latent_dim: int = 32,
        max_seq_length: int = 75,
        lr: float = 1e-4,
        weight_decay: float = 1e-8,
        adam_eps: float = 1e-7,
        weight_init: str = "xavier",
        use_batchnorm: bool = False,
        num_mlp_layers: int = 5,
        mlp_layer_dim: int = 128,
        activation: str = "gelu",
        leaky_relu_frac: float = 0.2,
        dropout: float = 0.2,
        add_noise: bool = False,
        data_length: int = 100,
        max_lr_factor: float = 1.5,
        pct_start: float = 0.25,
        scheduler_name: str = "cosine",
        gamma: float = 0.5,
        eta_min: float = 1e-9,
        step_size: int = 5,
        # use_self_attention: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # self.transformer = AutoModel.from_pretrained(transformer_model_name)

        self.lr = lr
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.weight_init = weight_init
        self.use_batchnorm = use_batchnorm
        self.num_mlp_layers = num_mlp_layers
        self.mlp_layer_dim = mlp_layer_dim
        self.dropout = dropout
        self.add_noise = add_noise
        self.data_length = data_length
        self.max_lr_factor = max_lr_factor
        self.pct_start = pct_start
        self.scheduler_name = scheduler_name
        self.eta_min = eta_min
        self.gamma = gamma
        self.step_size = step_size

        self.test_val_metrics = {"test": [], "val": []}

        if encoding_layers is None:
            self.encoding_layers = nn.ModuleList()
            for _i in range(num_mlp_layers + 1):
                if _i == num_mlp_layers:
                    self.encoding_layers.append(nn.Linear(mlp_layer_dim, latent_dim))
                elif _i > 0:
                    self.encoding_layers.append(nn.Linear(mlp_layer_dim, mlp_layer_dim))
                else:
                    self.encoding_layers.append(nn.Linear(max_seq_length, mlp_layer_dim))

                if use_batchnorm:
                    self.encoding_layers.append(
                        nn.BatchNorm1d(mlp_layer_dim)
                    )  # Add BatchNorm layer

                self.encoding_layers.append(
                    nn.GELU()
                    if activation == "gelu"
                    else nn.LeakyReLU(leaky_relu_frac, inplace=True)
                )
                if _i != num_mlp_layers:
                    self.encoding_layers.append(nn.Dropout(dropout))
        else:
            self.encoding_layers = encoding_layers

    def forward(self, x):
        for layer in self.encoding_layers:
            x = layer(x)
        return x

    def configure_optimizers(self) -> (list, list):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_name == "none":
            return self.optimizer

        if self.scheduler_name == "cycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.max_lr_factor * self.lr,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=self.data_length,
                pct_start=self.pct_start,
                # steps_per_epoch=len(self.train_dataloader()),
            )
        elif self.scheduler_name == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.eta_min,
            )
        elif self.scheduler_name == "exp":
            self.scheduler = ExponentialLR(
                self.optimizer,
                gamma=self.gamma,
            )
        else:
            self.scheduler = MultiStepLR(
                self.optimizer,
                list(range(0, self.trainer.max_epochs, self.step_size)),
                gamma=self.gamma,
            )

        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]

    def test_val_epoch_end(self, step: str = "val"):
        all_metrics = torch.stack(self.test_val_metrics[step])
        avg_loss = torch.mean(all_metrics)
        self.log(f"avg_{step}_loss", avg_loss)
        self.test_val_metrics[step].clear()

    def on_train_epoch_end(self):
        self.logged_graph = False

    def on_validation_epoch_end(self):
        self.logged_graph = False
        self.test_val_epoch_end(step="val")

    def on_test_epoch_end(self):
        self.logged_graph = False
        self.test_val_epoch_end(step="test")

    def lr_scheduler_step(self, scheduler, metric) -> None:
        self.scheduler.step()

    def on_epoch_end(self) -> None:
        self.scheduler.step()


class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int = 32,
        max_seq_length: int = 75,
        lr: float = 1e-4,
        weight_decay: float = 1e-8,
        adam_eps: float = 1e-7,
        weight_init: str = "xavier",
        use_batchnorm: bool = False,
        num_mlp_layers: int = 5,
        mlp_layer_dim: int = 128,
        activation: str = "gelu",
        leaky_relu_frac: float = 0.2,
        dropout: float = 0.2,
        add_noise: bool = False,
        data_length: int = 0,
        # use_self_attention: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # self.transformer = AutoModel.from_pretrained(transformer_model_name)

        self.lr = lr
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.weight_init = weight_init
        self.use_batchnorm = use_batchnorm
        self.num_mlp_layers = num_mlp_layers
        self.mlp_layer_dim = mlp_layer_dim
        self.dropout = dropout
        self.add_noise = add_noise
        # self.use_self_attention = use_self_attention

        self.data_length = data_length

        self.encoding_layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()

        for _i in range(num_mlp_layers + 1):
            if _i == num_mlp_layers:
                self.encoding_layers.append(nn.Linear(mlp_layer_dim, latent_dim))
                self.output_layers.append(nn.Linear(mlp_layer_dim, max_seq_length))
            elif _i > 0:
                self.encoding_layers.append(nn.Linear(mlp_layer_dim, mlp_layer_dim))
                self.output_layers.append(nn.Linear(mlp_layer_dim, mlp_layer_dim))
            else:
                self.encoding_layers.append(nn.Linear(max_seq_length, mlp_layer_dim))
                self.output_layers.append(nn.Linear(latent_dim, mlp_layer_dim))

            if use_batchnorm:
                self.encoding_layers.append(nn.BatchNorm1d(mlp_layer_dim))  # Add BatchNorm layer
                self.output_layers.append(nn.BatchNorm1d(mlp_layer_dim))

            self.encoding_layers.append(
                nn.GELU() if activation == "gelu" else nn.LeakyReLU(leaky_relu_frac, inplace=True)
            )
            if _i != num_mlp_layers:
                self.output_layers.append(
                    nn.GELU()
                    if activation == "gelu"
                    else nn.LeakyReLU(leaky_relu_frac, inplace=True)
                )
                self.encoding_layers.append(nn.Dropout(dropout))
                self.output_layers.append(nn.Dropout(dropout))
            else:
                self.output_layers.append(nn.Identity())

        self.loss = nn.MSELoss()
        print("Encoding: ", self.encoding_layers)
        print("Decoding: ", self.output_layers)

        self.init_weights(self.encoding_layers)  # Initialize the weights
        self.init_weights(self.output_layers)  # Initialize the weights

        self.example_input_array = torch.zeros((1, max_seq_length), dtype=torch.float)

    def init_weights(self, layers):
        for layer in layers:
            if isinstance(layer, nn.Linear):
                if self.weight_init == "xavier":
                    nn.init.xavier_uniform_(layer.weight)  # Apply Xavier initialization
                elif self.weight_init == "kaiming":
                    nn.init.kaiming_uniform_(layer.weight)  # Apply Kaiming initialization
                elif self.weight_init == "orthogonal":
                    nn.init.orthogonal_(layer.weight)  # Apply orthogonal initialization

                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # print(x.size())
        for layer in self.encoding_layers:
            x = layer(x)
            # print(x.size())
        for layer in self.output_layers:
            x = layer(x)
            # print(x.size())
        return x

    def training_step(self, batch, batch_idx):
        return self.process_batch(batch, step="train")

    def validation_step(self, batch, batch_idx):
        return self.process_batch(batch, step="val")

    def test_step(self, batch, batch_idx):
        return self.process_batch(batch, step="test")

    def process_batch(self, batch, step: str = "train"):
        _, spectrum = batch

        spectrum = spectrum.to(self.device)
        # add Gaussian noise to input
        if self.add_noise:
            noise = torch.randn(spectrum.size()).float().to(self.device)
            gen_data = self(spectrum + noise)
        else:
            gen_data = self(spectrum)
        loss = self.loss(spectrum, gen_data)
        self.log(f"{step}_loss", loss)
        wandb.log({f"{step}_loss": loss})

        return loss

    def configure_optimizers(self) -> (list, list):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=self.data_length,
            # steps_per_epoch=len(self.train_dataloader()),
        )
        # return self.optimizer
        # return [self.optimizer], [self.scheduler]
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, metric) -> None:
        self.scheduler.step()

    def on_epoch_end(self) -> None:
        self.scheduler.step()
