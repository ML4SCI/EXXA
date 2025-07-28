import pytorch_lightning as pl
import torch
import torch.nn as nn


class Discriminator(pl.LightningModule):
    def __init__(
        self,
        # encoder_model: pl.LightningModule,
        max_seq_length: int,
        output_dim: int = 1,
        #  transformer_model_name: str
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
    ) -> None:
        super().__init__()
        self.input_dim = max_seq_length
        self.output_dim = output_dim
        # self.transformer = AutoModel.from_pretrained(transformer_model_name)

        self.lr = lr
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.weight_init = weight_init
        self.use_batchnorm = use_batchnorm
        self.num_mlp_layers = num_mlp_layers
        self.mlp_layer_dim = mlp_layer_dim
        self.dropout = dropout

        # self.encoder_model = encoder_model

        self.layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()

        for _i in range(num_mlp_layers):
            if _i > 0:
                self.layers.append(nn.Linear(mlp_layer_dim, mlp_layer_dim))
            else:
                self.layers.append(nn.Linear(self.input_dim, mlp_layer_dim))

            if use_batchnorm:
                self.layers.append(nn.BatchNorm1d(mlp_layer_dim))  # Add BatchNorm layer

            self.layers.append(
                nn.GELU() if activation == "gelu" else nn.LeakyReLU(leaky_relu_frac, inplace=True)
            )
            self.layers.append(nn.Dropout(dropout))

        self.output_layers.append(nn.Linear(mlp_layer_dim, self.output_dim))
        self.output_layers.append(nn.Sigmoid())

        self.init_weights()  # Initialize the weights

        self.example_input_array = torch.zeros((1, max_seq_length), dtype=torch.float)

        self.loss = nn.BCELoss()

    def init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if self.weight_init == "xavier":
                    nn.init.xavier_uniform_(layer.weight)  # Apply Xavier initialization
                elif self.weight_init == "kaiming":
                    nn.init.kaiming_uniform_(layer.weight)  # Apply Kaiming initialization
                elif self.weight_init == "orthogonal":
                    nn.init.orthogonal_(layer.weight)  # Apply orthogonal initialization

                nn.init.constant_(layer.bias, 0.0)

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        for layer in self.output_layers:
            z = layer(z)
        return z

    # def process_batch(self, batch, step: str = "train"):

    #     clean_enc_spec, dirty_spec = batch

    #     batch_size = clean_enc_spec.shape[0]
    #     spec = spec.to(self.device)
    #     spectrum = spectrum.to(self.device)

    #     clean_output = self(clean_enc_spec)
    #     dirty_output = self(dirty_spec)

    #     clean = torch.ones(batch_size, 1).float().to(self.device)
    #     dirty = torch.zeros(batch_size, 1).float().to(self.device)

    #     clean_loss = self.loss(clean_output, clean)
    #     dirty_loss = self.loss(dirty_output, dirty)

    #     loss = 0.5 * (clean_loss + dirty_loss)

    #     self.log(f"{step}_loss", loss)
    #     wandb.log({f"{step}_loss": loss})

    # def training_step(self, batch, batch_idx):
    #     return self.process_batch(batch, step="train")

    # def validation_step(self, batch, batch_idx):
    #     return self.process_batch(batch, step="val")

    # def test_step(self, batch, batch_idx):
    #     return self.process_batch(batch, step="test")

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, eps=self.adam_eps, weight_decay=self.weight_decay
        )
        return self.optimizer


class MultiloaderADDA(pl.LightningModule):
    def __init__(
        self,
        encoder_model: pl.LightningModule,
        # classifier_model: pl.LightningModule,
        seq_length: int = 75,
        #  transformer_model_name: str
        lr: float = 1e-4,
        weight_decay: float = 1e-8,
        adam_eps: float = 1e-7,
        weight_init: str = "xavier",
        use_batchnorm: bool = False,
        num_mlp_layers: int = 5,
        mlp_layer_dim: int = 128,
        activation: str = "gelu",
        dropout: float = 0.2,
        padded_spectra: bool = False,
        encoder_name: str = "",
        # **kwargs: dict,
    ) -> None:
        super().__init__()

        self.automatic_optimization = False

        self.encoder = encoder_model
        # self.classifier = classifier_model
        self.classifier = Discriminator(
            seq_length,
            lr=lr,
            weight_decay=weight_decay,
            adam_eps=adam_eps,
            weight_init=weight_init,
            use_batchnorm=use_batchnorm,
            num_mlp_layers=num_mlp_layers,
            mlp_layer_dim=mlp_layer_dim,
            dropout=dropout,
        )

        self.seq_length = seq_length

        self.lr = lr
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.weight_init = weight_init
        self.activation = activation

        self.use_batchnorm = use_batchnorm
        self.num_mlp_layers = num_mlp_layers
        self.mlp_layer_dim = mlp_layer_dim
        self.dropout = dropout
        self.padded_spectra = padded_spectra

        self.encoder_name = encoder_name

        # self.example_input_array = torch.zeros((1, seq_length), dtype=torch.float)

    def forward(self, dirty_spec):
        return self.encoder(dirty_spec)

    def process_batch(self, batch, step: str = "train"):
        d_opt, g_opt = self.optimizers()

        avg_g_loss = 0.0
        avg_d_loss = 0.0
        if type(batch) == dict:
            keys = sorted(batch.keys())
            length = len(keys)

        elif type(batch) == list:
            if len(batch) == 2:
                batch = [batch]
            length = len(batch)
        for i in range(length):
            if not self.padded_spectra:
                if type(batch) == dict:
                    dirty_spec, clean_enc_spec = batch[keys[i]]
                else:
                    dirty_spec, clean_enc_spec = batch[i]
                mask = None
            else:
                if type(batch) == dict:
                    mask, dirty_spec, clean_enc_spec = batch[keys[i]]
                else:
                    mask, dirty_spec, clean_enc_spec = batch[i]

            batch_size = clean_enc_spec.shape[0]

            dirty_spec = dirty_spec.unsqueeze(-1)

            dirty_enc_spec = self(dirty_spec)  # , mask=mask)

            dirty_enc_spec = torch.max(dirty_enc_spec, dim=1)[0]
            # use this for the encoder
            dirty_enc_spec_copy = dirty_enc_spec.detach().clone().float().to(self.device)

            dirty_classification = self.classifier(dirty_enc_spec)
            clean_classification = self.classifier(clean_enc_spec)

            clean = torch.ones(batch_size, 1).float().to(self.device)
            dirty = torch.zeros(batch_size, 1).float().to(self.device)

            clean_d_loss = self.classifier.loss(clean_classification, clean)
            dirty_d_loss = self.classifier.loss(dirty_classification, dirty)

            ## train discriminator
            d_loss = 0.5 * (clean_d_loss + dirty_d_loss)

            if step == "train":
                d_opt.zero_grad()
                self.manual_backward(d_loss)
                d_opt.step()

            ## train encoder
            clean = torch.ones(batch_size, 1).float().to(self.device)
            dirty_classification = self.classifier(dirty_enc_spec_copy)
            g_loss = self.classifier.loss(dirty_classification, clean)

            if step == "train":
                g_opt.zero_grad()
                self.manual_backward(g_loss)
                g_opt.step()

            self.log(f"{step}_d_loss", d_loss)

            self.log(f"{step}_g_loss", g_loss)

            avg_g_loss += g_loss
            avg_d_loss += d_loss

        avg_d_loss /= length
        avg_g_loss /= length

        self.log(f"avg_{step}_d_loss", avg_d_loss)

        self.log(f"avg_{step}_g_loss", avg_g_loss)

        return {f"avg_{step}_d_loss": d_loss, f"avg_{step}_g_loss": avg_g_loss}

    def training_step(self, batch, batch_idx):
        return self.process_batch(batch, step="train")

    def validation_step(self, batch, batch_idx, dataloader_idx):
        return self.process_batch(batch, step="val")

    def test_step(self, batch, batch_idx, dataloader_idx):
        return self.process_batch(batch, step="test")

    def configure_optimizers(self):
        # self.d_opt = self.discriminator.configure_optimizers()
        # self.g_opt = self.generator.configure_optimizers()
        self.d_opt = torch.optim.Adam(
            self.parameters(), lr=self.lr, eps=self.adam_eps, weight_decay=self.weight_decay
        )
        self.g_opt = torch.optim.Adam(
            self.parameters(), lr=self.lr, eps=self.adam_eps, weight_decay=self.weight_decay
        )
        return self.d_opt, self.g_opt
