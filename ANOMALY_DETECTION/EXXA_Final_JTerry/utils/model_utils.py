from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml

# sys.path.insert(0, "./")
from models.autoencoder import Autoencoder
from models.multiloader_autoencoder import Multiloader_Autoencoder
from models.multiloader_transformer import MultiloaderTransformerSeq2Seq
from models.new_transformer import IntegratedTransformerSeq2Seq
from models.transformer import TransformerSeq2Seq
from utils.globals import all_model_hparams, model_types


def load_trained_model(
    model_name: str,
    model_path: str = "./trained_models/",
    use_checkpoint: bool = False,
    device: torch.device | None = None,
    parameter_path: str = "./Parameters/",
) -> pl.LightningModule:
    """Loads a trained model with a given name at a given path.
    Hyperparameters must be in utils.globals.all_model_hparams
    and its type (e.g. 'transformer') in utils.globals.model_types
    """
    if model_name in model_types:
        model_type = model_types[model_name]
        model_hparams = all_model_hparams[model_name]
    else:
        model_hparams = yaml.safe_load(Path(f"{parameter_path}{model_name}.yaml").read_text())
        model_type = model_hparams["model_type"]

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

    elif model_type == "new_transformer":
        model = IntegratedTransformerSeq2Seq(
            input_dim=model_hparams["input_dim"],
            output_dim=model_hparams["output_dim"],
            n_heads=model_hparams["n_heads"],
            n_layers=model_hparams["n_layers"],
            hid_dim=model_hparams["hid_dim"],
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

    elif model_type == "autoencoder":
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
    elif model_type == "multiloader_autoencoder":
        model = Multiloader_Autoencoder(
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
            max_lr_factor=model_hparams["max_lr_factor"],
            pct_start=model_hparams["pct_start"],
            gamma=model_hparams["gamma"],
            eta_min=model_hparams["eta_min"],
            scheduler_name=model_hparams["scheduler_name"],
            pad_value=model_hparams["pad_value"],
            center=bool(model_hparams["center"]),
            random_cycle=bool(model_hparams["random_cycle"]),
            sub_cont=bool(model_hparams["sub_cont"]),
        )
    elif model_type == "multiloader_transformer":
        model = MultiloaderTransformerSeq2Seq(
            input_dim=model_hparams["input_dim"],
            output_dim=model_hparams["output_dim"],
            n_heads=model_hparams["n_heads"],
            n_layers=model_hparams["n_layers"],
            hid_dim=model_hparams["hid_dim"],
            pf_dim=model_hparams["pf_dim"],
            lr=model_hparams["lr"],
            adam_eps=model_hparams["adam_eps"],
            weight_decay=model_hparams["weight_decay"],
            activation=model_hparams["activation"],
            dropout=model_hparams["dropout"],
            add_noise=bool(model_hparams["add_noise"]),
            trg_eq_zero=bool(model_hparams["trg_eq_zero"]),
            device=device,
            max_v=model_hparams["max_v"],
            line_index=model_hparams["line_index"],
            max_lr_factor=model_hparams["max_lr_factor"],
            pct_start=model_hparams["pct_start"],
            pos_enc_scale=model_hparams["pos_enc_scale"],
            eta_min=model_hparams["eta_min"],
            scheduler_name=model_hparams["scheduler_name"],
            gamma=model_hparams["gamma"],
            sub_cont=model_hparams["sub_cont"],
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


def save_model_hparams(
    hparams: dict, model_name: str, model_type: str, hparam_path: str = "./Parameters/"
):
    """Saves model hyperparameters as a yaml file"""
    hparams["model_name"] = model_name
    hparams["model_type"] = model_type
    with open(f"{hparam_path}{model_name}.yaml", "w") as file:
        yaml.dump(hparams, file)
    print(f"Saved hyperparameters at {hparam_path}{model_name}.yaml")
