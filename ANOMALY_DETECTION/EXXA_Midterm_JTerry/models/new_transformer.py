import math

import pytorch_lightning as pl
import torch
import torch.nn as nn

# from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        device: str = "cpu",
        seq_length: int = 101,
        hid_dim: int = 16,
        n_layers: int = 5,
        n_heads: int = 4,
        pf_dim: int = 32,
        dropout: float = 0.2,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.tok_embedding = nn.Linear(input_dim, hid_dim)
        self.device = device
        self.seq_length = seq_length
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim or hid_dim
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.scale = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(self.device)

    def positional_encoding(self, seq_len, batch_size) -> torch.Tensor:
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1).to(self.device)
        div_term = torch.exp(
            torch.arange(0, self.hid_dim, 2).float() * (-math.log(10000.0) / self.hid_dim)
        ).to(self.device)
        pos = torch.zeros(seq_len, self.hid_dim).to(self.device)
        pos[:, 0::2] = torch.sin(position * div_term)
        pos[:, 1::2] = torch.cos(position * div_term)
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
        seq_length: int = 101,
        hid_dim: int = 16,
        n_layers: int = 5,
        n_heads: int = 4,
        pf_dim: int = 32,
        dropout: float = 0.2,
        activation: str = "gelu",
    ) -> None:
        super().__init__(
            input_dim, device, seq_length, hid_dim, n_layers, n_heads, pf_dim, dropout, activation
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
        seq_length: int = 101,
        hid_dim: int = 16,
        n_layers: int = 5,
        n_heads: int = 4,
        pf_dim: int = 32,
        dropout: float = 0.2,
        activation: str = "gelu",
    ) -> None:
        super().__init__(
            input_dim, device, seq_length, hid_dim, n_layers, n_heads, pf_dim, dropout, activation
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


class IntegratedTransformerSeq2Seq(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        lr: float = 1e-4,
        adam_eps: float = 5e-7,
        weight_decay: float = 1e-8,
        add_noise: bool = False,
        seq_length: int = 101,
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
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            device=device,
            seq_length=seq_length,
            hid_dim=hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            activation=activation,
            pf_dim=pf_dim,
        )
        self.decoder = TransformerDecoder(
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
            seq_length=seq_length,
            hid_dim=hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            activation=activation,
            pf_dim=pf_dim,
        )
        self.loss = nn.MSELoss()
        self.lr = lr
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.add_noise = add_noise
        self.seq_length = seq_length
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

    def validation_step(self, batch, batch_idx):
        return self.process_batch(batch, step="val")

    def test_step(self, batch, batch_idx):
        return self.process_batch(batch, step="test")

    def process_batch(self, batch, step: str = "train"):
        _, spectrum = batch
        spectrum = spectrum.to(self.device)

        if self.add_noise:
            noise = torch.randn(spectrum.size()).float().to(self.device)
            gen_data = self(spectrum + noise)
        else:
            gen_data = self(spectrum)

        loss = self.loss(spectrum, gen_data)
        self.log(f"{step}_loss", loss)
        return loss

    def configure_optimizers(self) -> (list, list):
        self.optimizer = torch.optim.Adam(
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
