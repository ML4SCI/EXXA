import pytorch_lightning as pl
import torch

# sys.path.insert(0, "./")
from models.autoencoder import Autoencoder
from models.transformer import TransformerSeq2Seq
from utils.globals import all_model_hparams, model_types


def load_trained_model(
    model_name: str,
    model_path: str = "./trained_model/",
    use_checkpoint: bool = False,
    device: torch.device | None = None,
) -> pl.LightningModule:
    """Loads a trained model with a given name at a given path.
    Hyperparameters must be in utils.globals.all_model_hparams
    and its type (e.g. 'transformer') in utils.globals.model_types
    """
    model_type = model_types[model_name]
    model_hparams = all_model_hparams[model_name]

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if model_type == "transformer":
        model = TransformerSeq2Seq(
            input_dim=model_hparams["input_dim"],
            output_dim=model_hparams["output_dim"],
            num_heads=model_hparams["num_heads"],
            num_layers=model_hparams["num_layers"],
            hidden_dim=model_hparams["hidden_dim"],
            pf_dim=model_hparams["pf_dim"],
            seq_length=model_hparams["seq_length"],
            lr=model_hparams["lr"],
            adam_eps=model_hparams["adam_eps"],
            weight_decay=model_hparams["weight_decay"],
            activation=model_hparams["activation"],
            dropout=model_hparams["dropout"],
            add_noise=bool(model_hparams["add_noise"]),
            trg_eq_zero=bool(model_hparams["trg_eq_zero"]),
            device=device,
        )

    else:
        model = Autoencoder(
            latent_dim=model_hparams["latent_dim"],
            max_seq_length=model_hparams["max_seq_length"],
            lr=model_hparams["lr"],
            adam_eps=model_hparams["adam_eps"],
            weight_decay=model_hparams["weight_decay"],
            weight_init=model_hparams["weight_init"],
            use_batchnorm=model_hparams["use_batchnorm"],
            num_mlp_layers=model_hparams["num_mlp_layers"],
            mlp_layer_dim=model_hparams["mlp_layer_dim"],
            activation=model_hparams["activation"],
            leaky_relu_frac=model_hparams["leaky_relu_frac"],
            dropout=model_hparams["dropout"],
            add_noise=bool(model_hparams["add_noise"]),
        )

    if use_checkpoint:
        # load the training checkpoint rather than the final model
        try:
            checkpoint = torch.load(
                f"{model_path}Checkpoints/{model_name}/final_model_checkpoint_{model_name}.pyt"
            )
        except FileNotFoundError:
            # load final model if that doesn't work
            checkpoint = torch.load(f"{model_path}final_model_{model_name}.pyt")

    else:
        checkpoint = torch.load(f"{model_path}final_model_{model_name}.pyt")

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model
