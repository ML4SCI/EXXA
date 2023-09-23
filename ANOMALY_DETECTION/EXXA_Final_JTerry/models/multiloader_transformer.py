import math

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, MultiStepLR, OneCycleLR

import wandb


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        device: str = "cpu",
        hid_dim: int = 16,
        n_layers: int = 5,
        n_heads: int = 4,
        pf_dim: int = 32,
        dropout: float = 0.2,
        activation: str = "gelu",
        pos_enc_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.tok_embedding = nn.Linear(input_dim, hid_dim)
        self.device = device
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim or hid_dim
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.scale = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(self.device)
        self.pos_enc_scale = pos_enc_scale

    def positional_encoding(self, seq_len, batch_size) -> torch.Tensor:
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1).to(self.device)
        div_term = torch.exp(
            torch.arange(0, self.hid_dim, 2).float() * (-math.log(10000.0) / self.hid_dim)
        ).to(self.device)
        pos = torch.zeros(seq_len, self.hid_dim).to(self.device)
        pos[:, 0::2] = torch.sin(position * div_term) * self.pos_enc_scale
        pos[:, 1::2] = torch.cos(position * div_term) * self.pos_enc_scale
        # return torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1).to(self.device)
        # return pos.unsqueeze(0)
        return pos

    def forward(self, src) -> torch.Tensor:
        batch_size = src.shape[0]
        seq_len = src.shape[1]
        pos = self.positional_encoding(seq_len, batch_size)
        return self.dropout((self.tok_embedding(src) * self.scale) + pos)


class TransformerEncoder(TransformerBlock):
    def __init__(
        self,
        input_dim: int = 1,
        device: str = "cpu",
        hid_dim: int = 16,
        n_layers: int = 5,
        n_heads: int = 4,
        pf_dim: int = 32,
        dropout: float = 0.2,
        activation: str = "gelu",
        pos_enc_scale: float = 1.0,
    ) -> None:
        super().__init__(
            input_dim,
            device,
            hid_dim,
            n_layers,
            n_heads,
            pf_dim,
            dropout,
            activation,
            pos_enc_scale,
        )
        self.input_dim = input_dim  # Adding the input_dim attribute
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    self.hid_dim,
                    n_heads,
                    pf_dim or hid_dim,
                    nn.Dropout(dropout).p,
                    activation=activation,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )

    def set_super_seq_length(self, length):
        self.seq_length = length

    def forward(self, src) -> torch.Tensor:
        # Check for empty input tensor
        if src.numel() == 0:
            raise ValueError("Expected input tensor to be non-empty")

        # Check for input tensor dimension
        if len(src.shape) != 3:
            raise ValueError("Mismatched input dimension")

        # Check if the last dimension of the input tensor matches input_dim
        if src.shape[-1] != self.input_dim:
            raise ValueError(
                f"Mismatched input dimension in Encoder. Expected last dimension to be {self.input_dim}, but got {src.shape[-1]}"
            )

        # Get embeddings with positional encodings from TransformerBlock
        embedded = super().forward(src)
        # Apply transformer encoder layers
        for layer in self.layers:
            embedded = layer(embedded)
        return embedded


class TransformerDecoder(TransformerBlock):
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        device: str = "cpu",
        hid_dim: int = 16,
        n_layers: int = 5,
        n_heads: int = 4,
        pf_dim: int = 32,
        dropout: float = 0.2,
        activation: str = "gelu",
        pos_enc_scale: float = 1.0,
    ) -> None:
        super().__init__(
            input_dim,
            device,
            hid_dim,
            n_layers,
            n_heads,
            pf_dim,
            dropout,
            activation,
            pos_enc_scale,
        )
        self.fc_out = nn.Linear(hid_dim, output_dim or hid_dim)
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    self.hid_dim,
                    n_heads,
                    pf_dim or hid_dim,
                    nn.Dropout(dropout).p,
                    activation=activation,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )

    def set_super_seq_length(self, length):
        self.seq_length = length

    def forward(self, trg, enc_src) -> torch.Tensor:
        # Check for empty input tensor
        if trg.numel() == 0 or enc_src.numel() == 0:
            raise ValueError("Expected input tensors to be non-empty")

        # Check for input tensor dimension
        if len(trg.shape) != 3 or len(enc_src.shape) != 3:
            raise ValueError("Mismatched input dimension")

        trg = super().forward(trg)
        for layer in self.layers:
            trg = layer(trg, enc_src)
        batch_size, seq_length, _ = trg.shape
        trg = trg.reshape(batch_size, seq_length, self.hid_dim)
        return self.fc_out(trg)


class MultiloaderTransformerSeq2Seq(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        lr: float = 1e-4,
        adam_eps: float = 5e-7,
        weight_decay: float = 1e-8,
        add_noise: bool = False,
        n_layers: int = 5,
        n_heads: int = 4,
        trg_eq_zero: bool = True,
        device: str = "cpu",
        hid_dim: int = 16,
        dropout: float = 0.2,
        activation="gelu",
        pf_dim: int = 32,
        max_v: float = 5.0,
        line_index: int = 1,
        data_length: int = 100,
        max_lr_factor: float = 1.5,
        pct_start: float = 0.25,
        scheduler_name: str = "cosine",
        gamma: float = 0.5,
        eta_min: float = 1e-9,
        step_size: int = 5,
        pos_enc_scale: float = 1.0,
        sub_cont: bool = False,
        # val_dataloaders: list = [],
        # train_dataloaders: list = [],
        # test_dataloaders: list = [],
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            device=device,
            hid_dim=hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            activation=activation,
            pf_dim=pf_dim,
            pos_enc_scale=pos_enc_scale,
        )
        self.decoder = TransformerDecoder(
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
            hid_dim=hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            activation=activation,
            pf_dim=pf_dim,
            pos_enc_scale=pos_enc_scale,
        )
        self.loss = nn.MSELoss()
        self.lr = lr
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.add_noise = add_noise
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.trg_eq_zero = trg_eq_zero
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.activation = activation
        self.pf_dim = pf_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_v = max_v
        self.line_index = line_index
        self.data_length = data_length
        self.max_lr_factor = max_lr_factor
        self.pct_start = pct_start
        self.scheduler_name = scheduler_name
        self.eta_min = eta_min
        self.gamma = gamma
        self.step_size = step_size
        self.pos_enc_scale = pos_enc_scale
        self.sub_cont = sub_cont

        self.test_val_metrics = {"test": [], "val": []}

        self.logged_graph = False

        # self.test_dataloaders = test_dataloaders
        # self.train_dataloaders = train_dataloaders
        # self.val_dataloaders = val_dataloaders

    def forward(self, spectrum) -> torch.Tensor:
        spectrum = spectrum.unsqueeze(-1)

        # # Assert that the reshaping worked correctly
        assert (
            spectrum.shape[-1] == self.encoder.input_dim
        ), f"Input tensor last dimension should be {self.encoder.input_dim}, but got {spectrum.shape[-1]}"
        enc_src = self.encoder(spectrum)
        trg = torch.zeros_like(spectrum).to(self.device) if self.trg_eq_zero else spectrum
        outputs = self.decoder(trg, enc_src)
        return outputs.squeeze(-1)

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
                # assuming normalized data for the time being (1% noise)
                noise = torch.randn(spectrum.size()).float().to(self.device) / 100.0
                # gen_data = self(spectrum + noise)
                spectrum += noise
            # else:
            gen_data = self(spectrum)
            ## todo: do I want to do loss with noise?
            avg_loss += self.loss(spectrum, gen_data)

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

    def on_train_epoch_end(self):
        self.logged_graph = False

    def on_validation_epoch_end(self):
        self.logged_graph = False
        self.test_val_epoch_end(step="val")

    def on_test_epoch_end(self):
        self.logged_graph = False
        self.test_val_epoch_end(step="test")

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

    def lr_scheduler_step(self, scheduler, metric) -> None:
        if self.scheduler_name != "none":
            self.scheduler.step()

    def on_epoch_end(self) -> None:
        self.logged_graph = False

        if self.scheduler_name != "none":
            self.scheduler.step()
