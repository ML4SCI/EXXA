import argparse
import copy
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, progress
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader

import wandb

sys.path.insert(0, "./")
# from models.domain_adaptation.adda import ADDA
from models.autoencoder import MLP_Encoder
from models.domain_adaptation.multiloader_adda import MultiloaderADDA
from utils.data_utils import make_adda_dataset, make_multiple_datasets
from utils.globals import (
    data_path,
    watt_to_jansky,
)
from utils.model_utils import load_trained_model, save_model_hparams

### Set seed for reproducibility
np.random.seed(123)
random.seed(123)
torch.manual_seed(123)


def check_and_make_dir(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == "__main__":
    print("Parsing arguments")

    parser = argparse.ArgumentParser(
        prog="ADDA",
        description="Adversarial domain adaptation",
    )

    parser.add_argument(
        "--add_noise", type=int, default=0, help="Add noise to the input spectrum"
    )
    parser.add_argument(
        "--trg_eq_zero", type=int, default=0, help="Make the target spectrum all zeros"
    )

    parser.add_argument("--padded_spectra", type=int, default=0, help="Pad spectra")

    parser.add_argument(
        "--num_mlp_layers", type=int, default=5, help="MLP depth for discriminator"
    )

    parser.add_argument(
        "--mlp_layer_dim", type=int, default=64, help="MLP dimension for discrimator"
    )

    parser.add_argument(
        "--encoder_name",
        type=str,
        default="brisk-sweep-18",
        help="Model to use as pretrained encoder",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="./Data/Keplerian/",
        help="Directory where data is stored",
    )

    parser.add_argument(
        "--log_wandb",
        type=int,
        default=1,
        help="Use wandb logger",
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="adda_anomaly",
        help="Name of wandb project",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./trained_adda_models/",
        help="Directory to save the models",
    )
    parser.add_argument(
        "--model_load_path",
        type=str,
        default="./trained_models/",
        help="Directory to save the models",
    )
    parser.add_argument(
        "--dirty_ext",
        type=str,
        default="Convolved/",
        help="Extension to dirty data",
    )

    parser.add_argument(
        "--scale_data", type=float, default=-1.0, help="Factor by which to scale up data"
    )

    parser.add_argument(
        "--one_per_run", type=int, default=1, help="Only use a single dump file per simulation"
    )

    parser.add_argument(
        "--line_index",
        type=int,
        default=1,
        help="Emission line index (0 = 1-0, 1 = 2-1, 2 = 3-2)",
    )
    parser.add_argument(
        "--max_v",
        type=float,
        default=5.0,
        help="Maximum (absolute value) velocity channel relative to systemic",
    )

    parser.add_argument(
        "--wandb_sweep",
        type=int,
        default=0,
        help="Do a WandB sweep",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers",
    )

    parser.add_argument(
        "--accelerator_name",
        type=str,
        default="mps",
        help="Type of accelerator to use (if available)",
    )

    parser.add_argument(
        "--ignore_zeros",
        type=int,
        default=1,
        help="Ignore spectra that are only zeros",
    )

    parser.add_argument(
        "--parameter_path",
        type=str,
        default="./Parameters/",
        help="Directory with saved model hyperparameters",
    )

    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=1000,
        help="Number of spectra to encode at once",
    )

    args = parser.parse_args()

    # data_path = args.data_path
    wandb_project_name = args.wandb_project_name
    model_save_path = args.model_save_path
    model_load_path = args.model_load_path
    scale_data = args.scale_data
    num_workers = args.num_workers
    wandb_sweep = bool(args.wandb_sweep)
    # max_seq_length = args.max_seq_length
    log_wandb = bool(args.log_wandb)
    line_index = args.line_index
    max_v = args.max_v
    encoder_name = args.encoder_name
    dirty_ext = args.dirty_ext
    one_per_run = bool(args.one_per_run)
    accelerator_name = args.accelerator_name

    # if encoder_name in model_types:
    #     model_type = model_types[encoder_name]
    #     model_hparams = all_model_hparams[encoder_name]
    # else:
    model_hparams = yaml.safe_load(Path(f"{args.parameter_path}{encoder_name}.yaml").read_text())
    model_type = model_hparams["model_type"]
    max_seq_length = model_hparams.get("max_seq_length", 1)
    if "hid_dim" in model_hparams:
        max_seq_length = model_hparams["hid_dim"]

    # model_hparams = vars(args)

    if "sub_cont" not in model_hparams:
        model_hparams["sub_cont"] = True
    if "pad_value" not in model_hparams:
        model_hparams["pad_value"] = -1.0
    if "center" not in model_hparams:
        model_hparams["center"] = True
    if "random_cycle" not in model_hparams:
        model_hparams["random_cycle"] = False
    if "num_mlp_layers" not in model_hparams:
        model_hparams["num_mlp_layers"] = args.num_mlp_layers
    if "mlp_layer_dim" not in model_hparams:
        model_hparams["mlp_layer_dim"] = args.mlp_layer_dim

    if model_hparams["scale_data"] == 0.0:
        model_hparams["scale_data"] = 1.0 / watt_to_jansky
        scale_data = model_hparams["scale_data"]

    keys_to_delete = [
        "data_path",
        "model_save_path",
        "model_load_path" "num_workers",
        "wandb_sweep",
        "log_wandb",
        "dirty_ext",
        "accelerator_name",
    ]

    for key in keys_to_delete:
        if key in model_hparams:
            del model_hparams[key]

    validation_split = 0.2
    test_split = 0.2
    batch_size = 64
    num_epochs = 50
    include_noise = True

    check_and_make_dir(f"{model_save_path}")
    check_and_make_dir(f"{model_save_path}ADDA_Checkpoints/")

    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if accelerator_name == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif accelerator_name == "cuda:0":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    trained_encoder_model = load_trained_model(
        encoder_name,
        model_path=model_load_path,
        device=device,
        parameter_path=args.parameter_path,
    )

    print("Getting data")
    # get datasets
    train_data, val_data, test_data = make_multiple_datasets(
        data_path,
        scale_data=scale_data,
        val_split=validation_split,
        test_split=test_split,
        line_index=line_index,
        max_v=max_v,
        one_per_run=one_per_run,
        accelerator_name=accelerator_name,
        ignore_zeros=bool(args.ignore_zeros),
        sub_cont=bool(model_hparams["sub_cont"]),
        pad_value=model_hparams["pad_value"],
        seq_length=max_seq_length,
        center=bool(model_hparams["center"]),
        random_cycle=bool(model_hparams["random_cycle"]),
        dirty=True,
        dirty_ext=dirty_ext,
    )

    print("Loading encoder")

    if "transformer" in model_type:
        encoder_model = copy.deepcopy(trained_encoder_model.encoder)
    elif "autoencoder" in model_type:
        encoding_layers = copy.deepcopy(trained_encoder_model.encoding_layers)
        encoder_model = MLP_Encoder(
            encoding_layers=encoding_layers,
            latent_dim=model_hparams["latent_dim"],
            max_seq_length=max_seq_length,
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
            data_length=len(train_data),
            max_lr_factor=model_hparams["max_lr_factor"],
            pct_start=model_hparams["pct_start"],
            gamma=model_hparams["gamma"],
            eta_min=model_hparams["eta_min"],
            scheduler_name=model_hparams["scheduler_name"],
        )

    trained_encoder_model = load_trained_model(
        encoder_name,
        model_path=model_load_path,
        device=device,
        parameter_path=args.parameter_path,
    )

    for run in train_data:
        train_data[run] = make_adda_dataset(
            train_data[run], trained_encoder_model, batch_size=args.inference_batch_size
        )
        val_data[run] = make_adda_dataset(
            val_data[run], trained_encoder_model, batch_size=args.inference_batch_size
        )
        test_data[run] = make_adda_dataset(
            test_data[run], trained_encoder_model, batch_size=args.inference_batch_size
        )

    run_order = sorted(train_data.keys())
    # get data loaders
    train_loaders = {
        run: DataLoader(
            train_data[run], batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        for run in run_order
    }
    run_order = sorted(test_data.keys())
    val_loaders = [
        DataLoader(val_data[run], batch_size=batch_size, num_workers=num_workers)
        for run in run_order
    ]
    run_order = sorted(test_data.keys())
    test_loaders = [
        DataLoader(test_data[run], batch_size=batch_size, num_workers=num_workers)
        for run in run_order
    ]

    combined_train_loader = CombinedLoader(train_loaders, mode="max_size_cycle")
    combined_test_loader = CombinedLoader(test_loaders, mode="sequential")
    combined_val_loader = CombinedLoader(val_loaders, mode="sequential")

    print("Setting up WandB")

    # set up wandb logger
    os.environ["WANDB_API_KEY"] = "199115cad71655dbb5640225359e90bc0a91bcca"
    # wandb entity
    entity = "chlab"
    logger_kwargs = {
        "resume": "allow",
        "config": model_hparams,
    }

    # if log_wandb:
    logger = WandbLogger(project=wandb_project_name, entity=entity, **logger_kwargs)
    run_name = logger.experiment.name
    check_and_make_dir(f"{model_save_path}ADDA_Checkpoints/{run_name}/")

    print("Setting up run")

    #### necessary for newer PTL versions
    devices = 1
    accelerator = "gpu" if devices == 1 else "cpu"

    # make the trainer
    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        max_epochs=num_epochs,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=False,
                mode="min",
                monitor="avg_val_g_loss" if "multiloader" in model_type else "val_g_loss",
                save_top_k=1,
                every_n_epochs=1,
                save_on_train_epoch_end=False,
                dirpath=f"{model_save_path}ADDA_Checkpoints/{run_name}/",
                filename=f"final_adda_model_checkpoint_{run_name}",
            ),
            LearningRateMonitor("epoch"),
            progress.TQDMProgressBar(refresh_rate=1),
            EarlyStopping(
                monitor="avg_val_g_loss" if "multiloader" in model_type else "val_g_loss",
                min_delta=0.00,
                patience=10,
                verbose=False,
                mode="min",
            ),
        ],
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    print("Creating model")
    # get the model

    # model_hparams = all_model_hparams[encoder_name]
    # model_type = model_types[encoder_name]
    # enc_input_type = model_input_dims[encoder_name]
    # enc_output_type = model_output_dims[encoder_name]
    # if "input_dim" in model_hparams and model_hparams["input_dim"] is None:
    #     model_hparams["input_dim"] = max_seq_length if type(enc_input_type) == str else enc_input_type
    #     model_hparams["output_dim"] = max_seq_length if type(enc_output_type) == str else enc_output_type

    model = MultiloaderADDA(
        encoder_model,
        seq_length=max_seq_length,
        lr=model_hparams["lr"],
        weight_decay=model_hparams["weight_decay"],
        adam_eps=model_hparams["adam_eps"],
        weight_init=model_hparams.get("weight_init", "xavier"),
        num_mlp_layers=model_hparams["num_mlp_layers"],
        mlp_layer_dim=model_hparams["mlp_layer_dim"],
        activation=model_hparams["activation"],
        dropout=model_hparams["dropout"],
    )
    encoder_model.to(device)
    model.to(device)

    # fit the model
    print("Training")
    trainer.fit(
        model, train_dataloaders=combined_train_loader, val_dataloaders=combined_val_loader
    )

    save_model_hparams(
        model_hparams, run_name, f"adda_{model_type}", hparam_path=args.parameter_path
    )

    # save the model
    print("Saving final model")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            # "optimizer_state_dict": model.optimizer.state_dict(),
        },
        f"{model_save_path}final_adda_model_{run_name}.pyt",
    )

    # clear memory
    train_data, val_data, train_loaders, val_loaders = None, None, None, None
    combined_train_loader, combined_val_loader = None, None

    # Test the model
    ## Get test metrics
    print("Testing model")
    print(trainer.test(model, combined_test_loader))

    save_model_hparams(
        model_hparams, f"adda_{run_name}", model_type, hparam_path=args.parameter_path
    )

    # access the 'test_loss' (list of dict)
    # loss_value = test_loss[0]["test_g_loss"]

    # use loss_value value for  wandb.log
    # wandb.log({"final_loss": loss_value})

    wandb.finish()
