import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, MultiStepLR, OneCycleLR

import wandb


class Multiloader_Autoencoder(pl.LightningModule):
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
        data_length: int = 100,
        max_lr_factor: float = 1.5,
        pct_start: float = 0.25,
        scheduler_name: str = "cosine",
        gamma: float = 0.5,
        eta_min: float = 1e-9,
        step_size: int = 5,
        sub_cont: bool = False,
        pad_value: float | None = None,
        center: bool = False,
        random_cycle: bool = False,
        # use_self_attention: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

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
        self.sub_cont = sub_cont
        self.center = center
        self.random_cycle = random_cycle

        self.test_val_metrics = {"test": [], "val": []}

        self.data_length = data_length

        self.logged_graph = False

        self.pad_value = pad_value

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

    def validation_step(self, batch, batch_idx, dataloader_idx):
        return self.process_batch(batch, step="val")

    def test_step(self, batch, batch_idx, dataloader_idx):
        return self.process_batch(batch, step="test")

    def process_batch(self, batch, step: str = "train"):
        avg_loss = 0
        if type(batch) == dict:
            keys = sorted(batch.keys())
            length = len(keys)

        elif type(batch) == list:
            if len(batch) == 2:
                batch = [batch]
            length = len(batch)

        for i in range(length):
            if type(batch) == dict:
                vel, spectrum = batch[keys[i]]
            else:
                vel, spectrum = batch[i]

            spectrum = spectrum.to(self.device)

            if self.add_noise:
                ## assumes normalized data (1% noise)
                noise = torch.randn(spectrum.size()).float().to(self.device) / 100.0
                gen_data = self(spectrum + noise)
            else:
                gen_data = self(spectrum)

            if self.pad_value is None:
                avg_loss += self.loss(spectrum, gen_data)

            else:
                spec = spectrum[~(vel == self.pad_value)]
                pred_spec = gen_data[~(vel == self.pad_value)]
                avg_loss += self.loss(spec, pred_spec)

        avg_loss /= length

        self.log(f"{step}_loss", avg_loss)

        if step != "train":
            self.test_val_metrics[step].append(avg_loss)
            return {f"{step}_loss", avg_loss}

        if self.trainer.current_epoch % 10 == 0 and not self.logged_graph:
            self.logged_graph = True

            try:
                vel = vel[0, :].detach().cpu().numpy()
                y_true = spectrum[0, :].detach().cpu().numpy()
                y_pred = gen_data[0, :].detach().cpu().numpy()

                if self.pad_value is not None:
                    y_true = y_true[np.where(vel != self.pad_value)]
                    y_pred = y_pred[np.where(vel != self.pad_value)]
                    vel = vel[np.where(vel != self.pad_value)]

                self.log_prediction(vel, y_true, y_pred, step=step)

            except Exception as e:
                print(f"Failed to log graph: {e}")

        return avg_loss

    def log_prediction(
        self, vel: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, step: str = "train"
    ):
        fig = plt.figure(figsize=(10.0, 7.5))

        plt.plot(vel, y_true, lw=3, color="firebrick", label="True")
        plt.plot(
            vel,
            y_pred,
            lw=3,
            color="darkviolet",
            label=f"Predicted (epoch {self.trainer.current_epoch})",
        )

        plt.legend(loc="best", fontsize=10)
        plt.xlabel("Velocity [km/s]", fontsize=12)
        plt.ylabel("Intensity", fontsize=12)

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.title(f"Epoch {self.trainer.current_epoch}", fontsize=12)

        image = wandb.Image(fig, caption=f"Epoch {self.trainer.current_epoch}")

        wandb.log({f"{step}_prediction": image})

    def test_val_epoch_end(self, step: str = "val"):
        all_metrics = torch.stack(self.test_val_metrics[step])
        avg_loss = torch.mean(all_metrics)
        self.log(f"avg_{step}_loss", avg_loss)
        self.test_val_metrics[step].clear()

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
