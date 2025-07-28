import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR

import wandb


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hid_dim: int = 4,
        n_layers: int = 3,
        n_heads: int = 4,
        pf_dim: int = 48,
        dropout: float = 0.2,
        seq_length: int = 101,
        activation: str = "gelu",
        device: torch.device = "cpu",
    ) -> None:
        """Transformer encoder that takes in the original sequence"""
        super().__init__()

        self.tok_embedding = nn.Linear(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(seq_length, hid_dim)
        self.device = device

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    hid_dim, n_heads, pf_dim, dropout, activation=activation, batch_first=True
                )
                for _ in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src) -> torch.Tensor:
        # positional encoding
        pos = torch.arange(0, src.shape[1]).unsqueeze(0).repeat(src.shape[0], 1).to(self.device)

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            src = layer(src)
        return src


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 1,
        hid_dim: int = 4,
        n_layers: int = 3,
        n_heads: int = 4,
        pf_dim: int = 48,
        dropout: float = 0.2,
        seq_length: int = 101,
        activation: str = "gelu",
        device: torch.device = "cpu",
    ) -> None:
        """Transformer decoder that takes in the output of an encoder"""
        super().__init__()

        self.tok_embedding = nn.Linear(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(seq_length, hid_dim)
        self.device = device

        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    hid_dim, n_heads, pf_dim, dropout, activation=activation, batch_first=True
                )
                for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src) -> torch.Tensor:
        # positional encoding
        pos = torch.arange(0, trg.shape[1]).unsqueeze(0).repeat(trg.shape[0], 1).to(self.device)

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            trg = layer(trg, enc_src)

        return self.fc_out(trg)


class TransformerSeq2Seq(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_dim: int = 4,
        num_layers: int = 4,
        num_heads: int = 4,
        pf_dim: int = 48,
        dropout: float = 0.2,
        seq_length: int = 101,
        lr: float = 1e-4,
        weight_decay: float = 1e-8,
        adam_eps: float = 1e-7,
        activation: str = "gelu",
        add_noise: bool = False,
        device: torch.device = "cpu",
        trg_eq_zero: bool = False,
        line_index: int = 1,
        max_v: float = 5.0,
        data_length: int = 0,
    ) -> None:
        """Sequence to sequence transformer that uses a transformer encoder then transformer decoder"""
        super().__init__()

        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            hid_dim=hidden_dim,
            n_layers=num_layers,
            n_heads=num_heads,
            pf_dim=pf_dim,
            dropout=dropout,
            device=device,
            activation=activation,
            seq_length=seq_length,
        )
        self.decoder = TransformerDecoder(
            output_dim=output_dim,
            hid_dim=hidden_dim,
            n_layers=num_layers,
            n_heads=num_heads,
            pf_dim=pf_dim,
            activation=activation,
            seq_length=seq_length,
            device=device,
            dropout=dropout,
        )

        # self.final_layer = nn.Linear(hidden_dim, output_dim)

        self.loss = nn.MSELoss()

        self.lr = lr
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay

        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pf_dim = pf_dim
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.add_noise = add_noise
        self.line_index = line_index
        self.max_v = max_v

        self.trg_eq_zero = trg_eq_zero

        self.data_length = data_length

        self.example_input_array = torch.zeros((1, seq_length), dtype=torch.float)

    def forward(self, spectrum) -> torch.Tensor:
        # spectrum shape is [batch_size, sequence_length]
        spectrum = spectrum.unsqueeze(-1)

        # Pass the input through the transformer encoder
        enc_src = self.encoder(spectrum)

        # input either the raw spectrum or simply zeros
        trg = torch.zeros_like(spectrum).to(self.device) if self.trg_eq_zero else spectrum

        # through decoder
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

    def create_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones(size, size)).transpose(0, 1)
        return mask.masked_fill(mask == 1, float("-inf")).to(self.device)  # (seq_len, seq_len)

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
