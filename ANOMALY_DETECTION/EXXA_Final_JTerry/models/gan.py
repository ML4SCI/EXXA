import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb


class SelfAttention(nn.Module):
    def __init__(self, input_dim: int = 1, num_heads: int = 1, max_sequence_length: int = 75):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.max_sequence_length = max_sequence_length

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

        # self.positional_encoding = self._get_positional_encoding(max_sequence_length, input_dim)

    def forward(self, x, vels):
        batch_size, seq_len, _ = x.size()

        query = (
            self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = (
            self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        )

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float)
        )
        attention_scores += self.positional_encoding(vels)[:, :seq_len, :seq_len]
        attention_weights = self.softmax(attention_scores)
        return (
            torch.matmul(attention_weights, value)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, -1)
        )  ## attended values

    def get_positional_encoding(self, vels):
        positional_encoding = torch.zeros(1, self.max_sequence_length, self.input_dim)
        # position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.input_dim, 2).float() * (-np.log10(10000.0) / self.input_dim)
        )
        positional_encoding[:, :, 0::2] = torch.sin(vels * div_term)
        positional_encoding[:, :, 1::2] = torch.cos(vels * div_term)
        return positional_encoding


class Generator(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int = 101,
        max_seq_length: int = 75,
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
        use_self_attention: bool = False,
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
        self.use_self_attention = use_self_attention

        self.layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()

        for _i in range(num_mlp_layers):
            if _i > 0:
                self.layers.append(nn.Linear(mlp_layer_dim, mlp_layer_dim))
            else:
                self.layers.append(nn.Linear(self.latent_dim, mlp_layer_dim))

            if use_batchnorm:
                self.layers.append(nn.BatchNorm1d(mlp_layer_dim))  # Add BatchNorm layer

            self.layers.append(
                nn.GELU() if activation == "gelu" else nn.LeakyReLU(leaky_relu_frac, inplace=True)
            )
            self.layers.append(nn.Dropout(dropout))

        if use_self_attention:
            self.self_attention = SelfAttention(mlp_layer_dim, max_sequence_length=max_seq_length)

        self.output_layers.append(nn.Linear(mlp_layer_dim, max_seq_length))
        self.output_layers.append(nn.Identity())

        self.init_weights()  # Initialize the weights

        # self.example_input_array = torch.zeros((1, latent_dim), dtype=torch.float)

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

    def forward(self, z, vels):
        for layer in self.layers:
            z = layer(z)
        if self.use_self_attention:
            z = self.self_attention(z, vels)
        for layer in self.output_layers:
            z = layer(z)
        return z

    def training_step(self, batch, batch_idx):
        pass  # Will be defined during GAN's training_step

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, eps=self.adam_eps, weight_decay=self.weight_decay
        )
        return self.optimizer


class Discriminator(pl.LightningModule):
    def __init__(
        self,
        max_seq_length: int,
        output_dim: int = 1,
        #  transformer_model_name: str
        lr: float = 1e-4,
        weight_decay: float = 1e-8,
        adam_eps: float = 1e-7,
        weight_init: str = "xavier",
        use_batchnorm: bool = True,
        num_mlp_layers: int = 5,
        mlp_layer_dim: int = 128,
        activation: str = "gelu",
        leaky_relu_frac: float = 0.2,
        dropout: float = 0.2,
        use_self_attention: bool = False,
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
        self.use_self_attention = use_self_attention

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

        if use_self_attention:
            self.self_attention = SelfAttention(mlp_layer_dim, max_sequence_length=max_seq_length)

        self.output_layers.append(nn.Linear(mlp_layer_dim, self.output_dim))
        self.output_layers.append(nn.Sigmoid())

        self.init_weights()  # Initialize the weights

        # self.example_input_array = torch.zeros((1, input_dim), dtype=torch.float)

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

    def forward(self, z, vels):
        for layer in self.layers:
            z = layer(z)
        if self.use_self_attention:
            z = self.self_attention(z, vels)
        for layer in self.output_layers:
            z = layer(z)
        return z

    def training_step(self, batch, batch_idx):
        pass  # Will be defined during GAN's training_step

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, eps=self.adam_eps, weight_decay=self.weight_decay
        )
        return self.optimizer


class GAN(pl.LightningModule):
    def __init__(
        self,
        model_hparams: dict,
        # max_seq_length: int = 75,
        # latent_dim: int = 100,
        # #  transformer_model_name: str
        # weight_init: str = "xavier",
        # gen_lr: float = 1e-4,
        # gen_weight_decay: float = 1e-8,
        # gen_adam_eps: float = 1e-7,
        # gen_use_batchnorm: bool = False,
        # gen_num_mlp_layers: int = 5,
        # gen_mlp_layer_dim: int = 128,
        # gen_activation: str = "gelu",
        # gen_dropout: float = 0.2,
        # disc_lr: float = 1e-4,
        # disc_weight_decay: float = 1e-8,
        # disc_adam_eps: float = 1e-7,
        # disc_use_batchnorm: bool = False,
        # disc_num_mlp_layers: int = 5,
        # disc_mlp_layer_dim: int = 128,
        # disc_activation: str = "gelu",
        # leaky_relu_frac: float = 0.2,
        # disc_dropout: float = 0.2,
        # use_self_attention: bool = False,
    ) -> None:
        super().__init__()

        self.automatic_optimization = False

        self.weight_init = model_hparams["weight_init"]
        self.max_seq_length = model_hparams["max_seq_length"]
        self.output_dim = model_hparams["max_seq_length"]
        self.latent_dim = model_hparams["latent_dim"]
        self.leaky_relu_frac = model_hparams["leaky_relu_frac"]

        self.gen_lr = model_hparams["gen_lr"]
        self.gen_weight_decay = model_hparams["gen_weight_decay"]
        self.gen_adam_eps = model_hparams["gen_adam_eps"]
        self.gen_use_batchnorm = model_hparams["gen_use_batchnorm"]
        self.gen_num_mlp_layers = model_hparams["gen_num_mlp_layers"]
        self.gen_mlp_layer_dim = model_hparams["gen_mlp_layer_dim"]
        self.gen_dropout = model_hparams["gen_dropout"]
        self.gen_activation = model_hparams["gen_activation"]

        self.disc_lr = model_hparams["disc_lr"]
        self.disc_weight_decay = model_hparams["disc_weight_decay"]
        self.disc_adam_eps = model_hparams["disc_adam_eps"]
        self.disc_use_batchnorm = model_hparams["disc_use_batchnorm"]
        self.disc_num_mlp_layers = model_hparams["disc_num_mlp_layers"]
        self.disc_mlp_layer_dim = model_hparams["disc_mlp_layer_dim"]
        self.disc_dropout = model_hparams["disc_dropout"]
        self.disc_activation = model_hparams["disc_activation"]

        self.use_self_attention = bool(model_hparams["use_self_attention"])

        self.add_noise = bool(model_hparams["add_noise"])

        # self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.generator = Generator(
            latent_dim=self.latent_dim,
            max_seq_length=self.max_seq_length,
            lr=self.gen_lr,
            adam_eps=self.gen_adam_eps,
            weight_decay=self.gen_weight_decay,
            activation=self.gen_activation,
            weight_init=self.weight_init,
            use_batchnorm=self.gen_use_batchnorm,
            num_mlp_layers=self.gen_num_mlp_layers,
            mlp_layer_dim=self.gen_mlp_layer_dim,
            dropout=self.gen_dropout,
            leaky_relu_frac=self.leaky_relu_frac,
            use_self_attention=self.use_self_attention,
        )
        self.discriminator = Discriminator(
            max_seq_length=self.max_seq_length,
            output_dim=1,
            lr=self.disc_lr,
            adam_eps=self.disc_adam_eps,
            weight_decay=self.disc_weight_decay,
            activation=self.disc_activation,
            weight_init=self.weight_init,
            use_batchnorm=self.disc_use_batchnorm,
            num_mlp_layers=self.disc_num_mlp_layers,
            mlp_layer_dim=self.disc_mlp_layer_dim,
            dropout=self.disc_dropout,
            leaky_relu_frac=self.leaky_relu_frac,
            use_self_attention=self.use_self_attention,
        )

        self.loss = nn.BCELoss()

    def forward(self, z, vels):
        return self.generator(z, vels)

    def training_step(self, batch, batch_idx):
        _ = self.process_batch(batch, step="train")

    def validation_step(self, batch, batch_idx):
        _ = self.process_batch(batch, step="val")

    def test_step(self, batch, batch_idx):
        return self.process_batch(batch, step="test")

    def process_batch(self, batch, step: str = "train"):
        vels, real_data = batch

        vels = vels.to(self.device)
        real_data = real_data.to(self.device)

        d_opt, g_opt = self.optimizers()

        batch_size = real_data.size(0)
        valid = torch.ones(batch_size, 1).float().to(self.device)
        fake = torch.zeros(batch_size, 1).float().to(self.device)

        # Train Generator
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        gen_data = self.generator(z, vels)
        validity = self.discriminator(gen_data, vels)
        g_loss = self.loss(validity, valid)
        self.log(f"{step}_g_loss", g_loss)
        wandb.log({f"{step}_g_loss": g_loss})
        if step == "train":
            g_opt.zero_grad()
            self.manual_backward(g_loss)
            g_opt.step()
        # return g_loss

        # Train Discriminator
        if self.add_noise:
            noise = torch.randn(real_data.size()).float().to(self.device)
            real_validity = self.discriminator(real_data + noise, vels)
        else:
            real_validity = self.discriminator(real_data, vels)
        real_loss = self.loss(real_validity, valid)

        z = torch.randn(batch_size, self.latent_dim).float().to(self.device)
        gen_data = self.generator(z, vels)
        fake_validity = self.discriminator(gen_data, vels)
        fake_loss = self.loss(fake_validity, fake)

        d_loss = (real_loss + fake_loss) / 2.0
        self.log(f"{step}_d_loss", d_loss)
        wandb.log({f"{step}_d_loss": d_loss})
        if step == "train":
            d_opt.zero_grad()
            self.manual_backward(d_loss)
            d_opt.step()

        return {f"{step}_d_loss": d_loss, f"{step}_g_loss": g_loss}
        # return d_loss

    def configure_optimizers(self):
        self.d_opt = self.discriminator.configure_optimizers()
        self.g_opt = self.generator.configure_optimizers()
        return self.d_opt, self.g_opt
