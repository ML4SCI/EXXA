import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb


class GRU_Encoder(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.dropout = dropout

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, signal) -> torch.Tensor:
        # batch_size = signal.shape[0]
        # print(f"Encoder {signal.size()}")
        _, hidden = self.gru(signal)
        # print(f"First encoder hidden {hidden.size()}")
        return hidden
        # print(f"Encoder hidden {hidden.transpose(0, 1).contiguous().view(batch_size, -1).size()}")
        # return hidden.transpose(0, 1).contiguous().view(batch_size, -1)


class GRU_Decoder(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 10,
        dropout: float = 0.2,
        entire_sequence: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.input_size = input_size
        self.entire_sequence = entire_sequence

        self.gru = nn.GRU(
            # output_size,
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, signal, hidden) -> torch.Tensor:
        # signal = signal.unsqueeze(1)
        # signal = signal.squeeze(-1)
        # hidden = hidden.unsqueeze(0)
        # print(f"Decoder {hidden.size()} {signal.size()}")
        if not self.entire_sequence:
            output, _ = self.gru(signal, hidden)
        else:
            output, _ = self.gru(signal)
        return self.out(output.squeeze(1)).unsqueeze(-1)

        # input to the decoder has to be (batch_size, 1, hidden_size)
        # signal = signal.unsqueeze(1)
        # hidden = hidden.unsqueeze(0)
        # output, _ = self.gru(signal, hidden)
        # return self.out(output.squeeze(1))

    #     self.gru = nn.GRU(hidden_size + output_size, hidden_size, num_layers=num_layers, batch_first=True,)
    #     self.out = nn.Linear(hidden_size, output_size)

    # def forward(self, signal, hidden):
    #     decoder_input = torch.cat((signal, hidden), -1)
    #     output, _ = self.gru(decoder_input)
    #     output = self.out(output.squeeze(1))
    #     return output


class GRU_Seq2Seq(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 10,
        lr: float = 5e-5,
        adam_eps: float = 1e-6,
        weight_decay: float = 1e-8,
        # embedding_size: int = 1,
        sequence_length: int = 101,
        add_noise: bool = False,
        dropout: float = 0.2,
        entire_sequence: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.entire_sequence = entire_sequence

        self.encoder = GRU_Encoder(
            # input_size=sequence_length,
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = GRU_Decoder(
            # output_size=sequence_length,
            input_size=hidden_size if entire_sequence else 1,
            output_size=1,  # if not entire_sequence else sequence_length,
            hidden_size=1 if entire_sequence else hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            entire_sequence=entire_sequence,
        )

        self.lr = lr
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay

        self.add_noise = add_noise

        self.loss = nn.MSELoss()

        self.example_input_array = torch.zeros((1, sequence_length), dtype=torch.float32)

    def forward(self, source) -> torch.Tensor:
        batch_size = source.shape[0]

        source = source.unsqueeze(-1)

        hidden = self.encoder(source)
        # print(hidden.size())
        if self.entire_sequence:
            # source = source.permute(1, 0, 2)
            # hidden = hidden.permute(1, 0, 2)
            hidden = hidden[-1].unsqueeze(1).repeat(1, source.size(1), 1)
            # print(hidden.size())
            output = self.decoder(hidden, None)  # .squeeze(1)
            # print(output.size(), output.squeeze().size())
            return output.squeeze()
            # print(hidden.size(), source.size(), source.squeeze(-1).size(), source.squeeze(-1).unsqueeze(1).size())
            # return self.decoder(source.squeeze(-1).unsqueeze(1), hidden).squeeze(1)
            # return self.decoder(out, hidden).squeeze(1)

        outputs = torch.zeros(batch_size, 1, self.sequence_length).to(self.device)

        decoder_input = source[:, 0, :].unsqueeze(1)
        # print(decoder_input.size(), hidden.size())
        # iteratively generate and collect the decoder's outputs
        for t in range(self.sequence_length):
            output = self.decoder(decoder_input, hidden)
            outputs[:, :, t] = output.squeeze(-1)
            decoder_input = output
        return outputs.squeeze(1)

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

    def training_step(self, batch, batch_idx):
        return self.process_batch(batch, step="train")

    def validation_step(self, batch, batch_idx):
        return self.process_batch(batch, step="val")

    def test_step(self, batch, batch_idx):
        return self.process_batch(batch, step="test")

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, eps=self.adam_eps, weight_decay=self.weight_decay
        )
        return self.optimizer
